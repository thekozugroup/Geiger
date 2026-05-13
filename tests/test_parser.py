import pytest
from pathlib import Path
from unittest.mock import mock_open, patch, MagicMock

from geiger.tools.parser import (
    ToolParser,
    ToolDefinition,
    ArgumentDefinition,
)


@pytest.fixture
def sample_tools_md():
    return """# Tools

## tool_one

A test tool that does something useful.

### Arguments

- arg1 (string): The first argument description
- arg2 (integer): The second argument description

### Returns

Returns a string result.

## tool_two

Another tool with no arguments.

### Returns

Returns nothing useful.
"""


@pytest.fixture
def tools_md_path(tmp_path, sample_tools_md):
    path = tmp_path / "TOOLS.md"
    path.write_text(sample_tools_md)
    return path


@pytest.fixture
def parser(tools_md_path):
    return ToolParser(tools_md_path)


class TestToolParser:
    def test_parse_valid_tools_md(self, parser):
        tools = parser.parse()
        assert len(tools) == 2

    def test_tool_names_extracted(self, parser):
        tools = parser.parse()
        tool_names = [t.name for t in tools]
        assert "tool_one" in tool_names
        assert "tool_two" in tool_names

    def test_tool_descriptions_extracted(self, parser):
        tools = parser.parse()
        tool_one = next(t for t in tools if t.name == "tool_one")
        assert "test tool" in tool_one.description.lower()
        assert "useful" in tool_one.description.lower()

    def test_arguments_extracted(self, parser):
        tools = parser.parse()
        tool_one = next(t for t in tools if t.name == "tool_one")
        assert len(tool_one.arguments) == 2

        arg1 = next(a for a in tool_one.arguments if a.name == "arg1")
        assert arg1.type == "string"
        assert "first" in arg1.description.lower()

        arg2 = next(a for a in tool_one.arguments if a.name == "arg2")
        assert arg2.type == "integer"

    def test_tool_with_no_arguments(self, parser):
        tools = parser.parse()
        tool_two = next(t for t in tools if t.name == "tool_two")
        assert len(tool_two.arguments) == 0

    def test_returns_extracted(self, parser):
        tools = parser.parse()
        tool_one = next(t for t in tools if t.name == "tool_one")
        assert "string result" in tool_one.returns.lower()

    def test_handle_missing_file(self, tmp_path):
        nonexistent = tmp_path / "nonexistent.md"
        with pytest.raises(FileNotFoundError):
            ToolParser(nonexistent)

    def test_empty_file(self, tmp_path):
        empty_path = tmp_path / "empty.md"
        empty_path.write_text("")
        parser = ToolParser(empty_path)
        tools = parser.parse()
        assert tools == []

    def test_tool_definition_dataclass(self):
        arg = ArgumentDefinition(name="test_arg", type="string", description="Test desc")
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            arguments=(arg,),
            returns="test result",
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert len(tool.arguments) == 1
        assert tool.arguments[0].name == "test_arg"
        assert tool.returns == "test result"

    def test_tool_definition_empty_name_raises(self):
        with pytest.raises(ValueError):
            ToolDefinition(name="", description="Test")

    def test_argument_definition_empty_name_raises(self):
        with pytest.raises(ValueError):
            ArgumentDefinition(name="", type="string", description="Test")

    def test_argument_definition_empty_type_raises(self):
        with pytest.raises(ValueError):
            ArgumentDefinition(name="arg", type="", description="Test")


class TestToolParserEdgeCases:
    def test_tool_with_only_description(self, tmp_path):
        content = """# Tools

## solo_tool

Just a tool with a description.

"""
        path = tmp_path / "TOOLS.md"
        path.write_text(content)
        parser = ToolParser(path)
        tools = parser.parse()
        assert len(tools) == 1
        assert tools[0].name == "solo_tool"
        assert len(tools[0].arguments) == 0

    def test_arguments_without_returns(self, tmp_path):
        content = """# Tools

## arg_only_tool

Description here.

### Arguments

- param1 (number): A numeric parameter
"""
        path = tmp_path / "TOOLS.md"
        path.write_text(content)
        parser = ToolParser(path)
        tools = parser.parse()
        assert len(tools) == 1
        assert len(tools[0].arguments) == 1
        assert tools[0].returns == ""


class TestToolRunner:
    @pytest.fixture
    def tools_bin_path(self, tmp_path):
        tool_path = tmp_path / "test_tool"
        tool_path.write_text("#!/bin/bash\necho hello")
        tool_path.chmod(0o755)
        return tmp_path

    @pytest.mark.asyncio
    async def test_execute_success(self, tools_bin_path):
        from geiger.tools.parser import ToolRunner
        runner = ToolRunner(tools_bin_path)
        result = await runner.execute("test_tool")
        assert result.success is True
        assert result.returncode == 0
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_with_arguments(self, tmp_path):
        tool_path = tmp_path / "echo_args"
        tool_path.write_text("#!/bin/bash\necho $1 $2")
        tool_path.chmod(0o755)
        from geiger.tools.parser import ToolRunner
        runner = ToolRunner(tmp_path)
        result = await runner.execute("echo_args", {"arg1": "val1", "arg2": "val2"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, tools_bin_path):
        from geiger.tools.parser import ToolRunner
        runner = ToolRunner(tools_bin_path)
        with pytest.raises(FileNotFoundError):
            await runner.execute("nonexistent_tool")

    @pytest.mark.asyncio
    async def test_execute_invalid_tool_name(self, tools_bin_path):
        from geiger.tools.parser import ToolRunner
        runner = ToolRunner(tools_bin_path)
        with pytest.raises(ValueError):
            await runner.execute("invalid/name")

    @pytest.mark.asyncio
    async def test_execute_timeout(self, tmp_path):
        tool_path = tmp_path / "slow_tool"
        tool_path.write_text("#!/bin/bash\nsleep 10")
        tool_path.chmod(0o755)
        from geiger.tools.parser import ToolRunner
        runner = ToolRunner(tmp_path)
        result = await runner.execute("slow_tool", timeout=0.1)
        assert result.success is False
        assert "Timeout" in result.error_message

    def test_execute_sync(self, tools_bin_path):
        from geiger.tools.parser import ToolRunner
        runner = ToolRunner(tools_bin_path)
        result = runner.execute_sync("test_tool")
        assert result.success is True