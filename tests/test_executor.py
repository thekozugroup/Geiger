import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from geiger.agent.executor import (
    AgentExecutor,
    AgentConfig,
    AgentSession,
    ToolResult,
)
from geiger.types import ToolDefinition, Trace


@pytest.fixture
def agent_config():
    return AgentConfig(
        base_url="https://api.test.com/v1",
        api_key="test-key-123",
        model="gpt-4",
        max_tokens=1000,
        temperature=0.5,
    )


@pytest.fixture
def sample_tools():
    return [
        ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Test input"}
                },
                "required": ["input"],
            },
        )
    ]


@pytest.fixture
def mock_api_response():
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'll test this tool for you.",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "test_tool",
                                "arguments": '{"input": "hello"}',
                            },
                        }
                    ],
                }
            }
        ],
        "model": "gpt-4",
    }


@pytest.fixture
def mock_api_response_no_tools():
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you?",
                }
            }
        ],
        "model": "gpt-4",
    }


class TestAgentSession:
    @pytest.mark.asyncio
    async def test_session_creation(self, agent_config, sample_tools):
        session = AgentSession(agent_config, sample_tools)
        assert session.session_id is not None
        assert len(session.session_id) == 36
        assert session.messages == []
        assert session.tool_calls == []

    @pytest.mark.asyncio
    async def test_session_custom_id(self, agent_config, sample_tools):
        custom_id = "custom-session-123"
        session = AgentSession(agent_config, sample_tools, session_id=custom_id)
        assert session.session_id == custom_id

    def test_add_message(self, agent_config, sample_tools):
        session = AgentSession(agent_config, sample_tools)
        session.add_message("user", "Hello")
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "Hello"

    def test_build_payload(self, agent_config, sample_tools):
        session = AgentSession(agent_config, sample_tools)
        session.add_message("system", "You are helpful.")
        session.add_message("user", "Test prompt")

        payload = session._build_payload()

        assert payload["model"] == "gpt-4"
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 1000
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["function"]["name"] == "test_tool"

    def test_build_payload_empty_tools(self, agent_config):
        session = AgentSession(agent_config, [])
        session.add_message("user", "Test")
        payload = session._build_payload()
        assert payload["tools"] is None

    def test_get_trace(self, agent_config, sample_tools):
        session = AgentSession(agent_config, sample_tools)
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")

        trace = session.get_trace()

        assert trace.session_id == session.session_id
        assert len(trace.messages) == 2
        assert trace.timestamp > 0
        assert isinstance(trace, Trace)


class TestAgentExecutor:
    @pytest.mark.asyncio
    async def test_executor_initialization(self):
        executor = AgentExecutor()
        assert executor.config is not None
        assert executor.max_workers == 4
        assert executor.semaphore is not None

    @pytest.mark.asyncio
    async def test_executor_custom_config(self, agent_config):
        executor = AgentExecutor(config=agent_config, max_workers=8)
        assert executor.config == agent_config
        assert executor.max_workers == 8

    @pytest.mark.asyncio
    async def test_execute_single_no_tools(self, agent_config, mock_api_response_no_tools):
        executor = AgentExecutor(config=agent_config)

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_api_response_no_tools
        mock_client.post.return_value = mock_response

        with patch.object(AgentSession, "http_client", mock_client):
            session = AgentSession(agent_config, [])
            session.add_message("user", "Hello")

            payload = session._build_payload()
            assert payload["tools"] is None


class TestTraceGeneration:
    def test_trace_to_dict(self, agent_config, sample_tools):
        session = AgentSession(agent_config, sample_tools)
        session.add_message("user", "Test message")
        session.add_message("assistant", "Response")

        trace = session.get_trace()
        trace_dict = trace.to_dict()

        assert "session_id" in trace_dict
        assert "messages" in trace_dict
        assert "tool_calls" in trace_dict
        assert "timestamp" in trace_dict
        assert trace_dict["session_id"] == session.session_id

    def test_trace_messages_are_copies(self, agent_config, sample_tools):
        session = AgentSession(agent_config, sample_tools)
        session.add_message("user", "Original")
        trace1 = session.get_trace()

        session.add_message("assistant", "Added later")
        trace2 = session.get_trace()

        assert len(trace1.messages) == 1
        assert len(trace2.messages) == 2


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_execute_tool_success(self, agent_config, sample_tools):
        session = AgentSession(agent_config, sample_tools)

        async def mock_handler(**kwargs):
            return '{"files": ["file1.txt", "file2.txt"]}'

        tool_executor = {"file_search": mock_handler}
        result = await session.execute_tool("file_search", {"pattern": "*.txt"}, tool_executor)
        assert "file1.txt" in result

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, agent_config, sample_tools):
        session = AgentSession(agent_config, sample_tools)

        async def mock_handler(**kwargs):
            raise FileNotFoundError("Tool not found")

        tool_executor = {"file_search": mock_handler}
        result = await session.execute_tool("nonexistent_tool", {}, tool_executor)
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self, agent_config, sample_tools):
        session = AgentSession(agent_config, sample_tools)

        async def mock_handler(**kwargs):
            raise RuntimeError("Execution failed")

        tool_executor = {"failing_tool": mock_handler}
        result = await session.execute_tool("failing_tool", {}, tool_executor)
        assert "Error" in result


class TestAgentSessionToolCalls:
    @pytest.mark.asyncio
    async def test_add_tool_result(self, agent_config, sample_tools):
        session = AgentSession(agent_config, sample_tools)
        session.add_message("user", "What's the weather?")
        session.add_message(
            "assistant",
            "Let me check that for you.",
        )
        session.add_tool_result("call_123", "Sunny, 72F")

        tool_msgs = [m for m in session.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "Sunny, 72F"
        assert tool_msgs[0]["tool_call_id"] == "call_123"