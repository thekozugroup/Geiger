from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional
from dataclasses import dataclass

from geiger.types import ExecutionResult


class ToolHandler:
    """Handles tool execution with concurrency control."""

    def __init__(self, max_concurrency: int = 5):
        self.max_concurrency = max_concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._handlers: dict[str, Callable] = {}

    async def __aenter__(self) -> "ToolHandler":
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._semaphore = None

    def register_handler(self, tool_name: str, handler: Callable) -> None:
        """Register a handler for a tool."""
        self._handlers[tool_name] = handler

    async def run_single(self, tool_name: str, params: dict[str, Any]) -> ExecutionResult:
        """Execute a single tool with given parameters."""
        start = time.time()

        if tool_name not in self._handlers:
            return ExecutionResult(
                stdout="",
                stderr=f"No handler for tool '{tool_name}'",
                returncode=-1,
                success=False,
                error_message=f"No handler for tool '{tool_name}'",
            )

        try:
            handler = self._handlers[tool_name]
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**params)
            else:
                result = handler(**params)
            result_str = str(result) if result is not None else ""
            return ExecutionResult(
                stdout=result_str,
                stderr="",
                returncode=0,
                success=True,
                error_message="",
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                returncode=-1,
                success=False,
                error_message=str(e),
            )

    async def run_batch(
        self, tasks: list[tuple[str, dict[str, Any]]]
    ) -> list[ExecutionResult]:
        """Execute multiple tools concurrently with semaphore control."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_with_semaphore(tool_name: str, params: dict[str, Any]) -> ExecutionResult:
            async with self._semaphore:
                return await self.run_single(tool_name, params)

        results = await asyncio.gather(
            *[run_with_semaphore(name, params) for name, params in tasks]
        )
        return list(results)