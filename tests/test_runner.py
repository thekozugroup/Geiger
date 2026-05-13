import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from geiger.tools.runner import ToolHandler
from geiger.types import ExecutionResult


class TestToolHandler:
    def test_register_handler(self):
        handler = ToolHandler()
        mock_fn = MagicMock()
        handler.register_handler("test_tool", mock_fn)
        assert "test_tool" in handler._handlers
        assert handler._handlers["test_tool"] is mock_fn

    @pytest.mark.asyncio
    async def test_run_single_with_handler(self):
        handler = ToolHandler()
        async def mock_tool(input: str) -> str:
            return f"processed: {input}"
        handler.register_handler("test_tool", mock_tool)

        result = await handler.run_single("test_tool", {"input": "hello"})
        assert result.success is True
        assert result.returncode == 0
        assert "processed: hello" in result.stdout

    @pytest.mark.asyncio
    async def test_run_single_unknown_tool(self):
        handler = ToolHandler()
        result = await handler.run_single("unknown_tool", {"input": "hello"})
        assert result.success is False
        assert result.returncode == -1
        assert "No handler for tool 'unknown_tool'" in result.stderr

    @pytest.mark.asyncio
    async def test_run_batch(self):
        handler = ToolHandler(max_concurrency=2)

        async def mock_tool(name: str) -> str:
            return f"done: {name}"

        handler.register_handler("tool_a", mock_tool)
        handler.register_handler("tool_b", mock_tool)

        tasks = [
            ("tool_a", {"name": "task1"}),
            ("tool_b", {"name": "task2"}),
            ("tool_a", {"name": "task3"}),
        ]
        results = await handler.run_batch(tasks)
        assert len(results) == 3
        assert all(r.success for r in results)