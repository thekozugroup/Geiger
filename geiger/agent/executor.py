from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx

from geiger.types import ExecutionResult, ToolDefinition, Trace


@dataclass
class ToolResult:
    tool: str
    args: dict[str, Any]
    result: str
    success: bool = True


@dataclass
class AgentConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.7


class AgentSession:
    def __init__(
        self,
        config: AgentConfig,
        tools: list[ToolDefinition],
        session_id: str | None = None,
    ):
        self.config = config
        self.tools = tools
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self._http_client: httpx.AsyncClient | None = None

    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._http_client

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    async def send(
        self, prompt: str, system_prompt: str | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        if system_prompt:
            self.add_message("system", system_prompt)

        self.add_message("user", prompt)

        payload = self._build_payload()
        response = await self._call_api(payload)
        assistant_message = response["choices"][0]["message"]
        content = assistant_message.get("content", "")
        self.messages.append({"role": "assistant", "content": content})

        yield {"type": "message", "content": content}

        if "tool_calls" in assistant_message:
            for call in assistant_message["tool_calls"]:
                tool_name = call["function"]["name"]
                tool_args = json.loads(call["function"]["arguments"])
                self.tool_calls.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": "",
                })
                yield {"type": "tool_call", "tool": tool_name, "args": tool_args}

    def _build_payload(self) -> dict[str, Any]:
        messages = []
        for msg in self.messages:
            if msg["role"] == "system":
                messages.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "tool":
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": msg["content"],
                })

        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.tools
        ]

        return {
            "model": self.config.model,
            "messages": messages,
            "tools": tools_schema if tools_schema else None,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

    async def _call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        retry_count = 0
        max_retries = 3
        base_delay = 1.0

        while True:
            try:
                response = await self.http_client.post("/chat/completions", json=payload)
                if response.status_code == 429:
                    if retry_count >= max_retries:
                        raise RuntimeError("API rate limit exceeded after retries")
                    retry_count += 1
                    delay = base_delay * (2 ** (retry_count - 1))
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and retry_count < max_retries:
                    retry_count += 1
                    delay = base_delay * (2 ** (retry_count - 1))
                    await asyncio.sleep(delay)
                    continue
                raise

    async def execute_tool(
        self, tool_name: str, tool_args: dict[str, Any], tool_executor: dict[str, Any]
    ) -> str:
        if tool_name not in tool_executor:
            result = f"Error: Unknown tool '{tool_name}'"
            self._update_tool_call_result(tool_name, tool_args, result, success=False)
            return result

        try:
            func = tool_executor[tool_name]
            if asyncio.iscoroutinefunction(func):
                result = await func(**tool_args)
            else:
                result = func(**tool_args)
            result_str = json.dumps(result) if not isinstance(result, str) else result
        except Exception as e:
            result_str = f"Error executing {tool_name}: {str(e)}"

        self._update_tool_call_result(tool_name, tool_args, result_str)
        return result_str

    def _update_tool_call_result(
        self, tool_name: str, tool_args: dict[str, Any], result: str, success: bool = True
    ) -> None:
        for call in self.tool_calls:
            if call["tool"] == tool_name and call["args"] == tool_args and call["result"] == "":
                call["result"] = result
                break

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def get_trace(self) -> Trace:
        return Trace(
            session_id=self.session_id,
            messages=list(self.messages),
            tool_calls=list(self.tool_calls),
            timestamp=time.time(),
        )


class AgentExecutor:
    def __init__(
        self,
        config: AgentConfig | None = None,
        max_workers: int = 4,
    ):
        self.config = config or AgentConfig()
        self.max_workers = max_workers
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_workers)
        return self._semaphore

    async def execute(
        self,
        prompts: list[tuple[str, str | None, list[ToolDefinition], dict[str, Any]]],
    ) -> AsyncGenerator[Trace, None]:
        tasks = [
            self._run_session(prompt, system_prompt, tools, tool_executor)
            for prompt, system_prompt, tools, tool_executor in prompts
        ]

        for coro in asyncio.as_completed(tasks):
            task_result = await coro
            if task_result is not None:
                session, trace = task_result
                await session.close()
                yield trace

    async def _run_session(
        self,
        prompt: str,
        system_prompt: str | None,
        tools: list[ToolDefinition],
        tool_executor: dict[str, Any],
    ) -> tuple[AgentSession, Trace] | None:
        async with self.semaphore:
            session = AgentSession(self.config, tools)
            try:
                traces: list[Trace] = []
                async for event in session.send(prompt, system_prompt):
                    if event["type"] == "tool_call":
                        result = await session.execute_tool(
                            event["tool"], event["args"], tool_executor
                        )
                        session.add_message(
                            "tool",
                            f"Tool '{event['tool']}' returned: {result}",
                        )
                traces.append(session.get_trace())
                return (session, traces[-1])
            except Exception as e:
                trace = session.get_trace()
                trace.error = str(e)
                return (session, trace)

    async def execute_single(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_executor: dict[str, Any] | None = None,
    ) -> Trace:
        tools = tools or []
        tool_executor = tool_executor or {}

        async with self.semaphore:
            session = AgentSession(self.config, tools)
            try:
                async for event in session.send(prompt, system_prompt):
                    if event["type"] == "tool_call":
                        result = await session.execute_tool(
                            event["tool"], event["args"], tool_executor
                        )
                        session.add_message(
                            "tool",
                            f"Tool '{event['tool']}' returned: {result}",
                        )
                return session.get_trace()
            finally:
                await session.close()
