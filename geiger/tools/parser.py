"""Tool parser module for parsing TOOLS.md definitions and executing tools."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from geiger.types import ExecutionResult, ToolDefinition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArgumentDefinition:
    """Definition of a tool argument."""

    name: str
    type: str
    description: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Argument name cannot be empty")
        if not self.type:
            raise ValueError("Argument type cannot be empty")


class ToolParser:
    """Parser for TOOLS.md markdown files."""

    TOOL_PATTERN = re.compile(r"^##\s+(\w+)\s*$", re.MULTILINE)
    ARG_SECTION_PATTERN = re.compile(r"^###\s+Arguments\s*$", re.MULTILINE)
    RETURNS_SECTION_PATTERN = re.compile(r"^###\s+Returns\s*$", re.MULTILINE)
    ARGUMENT_PATTERN = re.compile(r"^- (\w+)\s+\((\w+)\):\s*(.+)$")

    def __init__(self, tools_path: Path | str) -> None:
        """Initialize parser with path to TOOLS.md file.

        Args:
            tools_path: Path to the TOOLS.md file
        """
        self.tools_path = Path(tools_path)
        if not self.tools_path.exists():
            raise FileNotFoundError(f"Tools file not found: {self.tools_path}")

    def parse(self) -> list[ToolDefinition]:
        """Parse the TOOLS.md file and return tool definitions.

        Returns:
            List of ToolDefinition objects

        Raises:
            ValueError: If the markdown format is invalid
        """
        content = self.tools_path.read_text(encoding="utf-8")
        tools: list[ToolDefinition] = []

        tool_matches = list(self.TOOL_PATTERN.finditer(content))
        if not tool_matches:
            logger.warning("No tools found in %s", self.tools_path)
            return []

        for i, match in enumerate(tool_matches):
            tool_name = match.group(1)
            start = match.end()
            end = tool_matches[i + 1].start() if i + 1 < len(tool_matches) else len(content)
            tool_content = content[start:end]

            tool_def = self._parse_tool_content(tool_name, tool_content)
            tools.append(tool_def)

        return tools

    def _parse_tool_content(self, name: str, content: str) -> ToolDefinition:
        """Parse a single tool's content block.

        Args:
            name: Tool name
            content: Markdown content for the tool

        Returns:
            ToolDefinition
        """
        arg_section_match = self.ARG_SECTION_PATTERN.search(content)
        returns_section_match = self.RETURNS_SECTION_PATTERN.search(content)

        desc_end = len(content)
        args_start = len(content)
        returns_start = len(content)

        if arg_section_match:
            args_start = arg_section_match.end()
        if returns_section_match:
            returns_start = returns_section_match.start()
            if arg_section_match:
                args_start = min(args_start, returns_start)

        description = content[:args_start].strip()
        if arg_section_match and returns_section_match:
            if arg_section_match.start() > returns_section_match.start():
                description = content[:returns_section_match.start()].strip()
            else:
                desc_end = args_start

        description = description.strip()

        desc_lines = description.split("\n")
        clean_desc_lines = []
        for line in desc_lines:
            line = line.strip()
            if line.startswith("###") or line.startswith("##"):
                break
            if line:
                clean_desc_lines.append(line)
        description = " ".join(clean_desc_lines)

        arguments: list[ArgumentDefinition] = []
        if arg_section_match and returns_section_match:
            if arg_section_match.start() < returns_section_match.start():
                args_content = content[args_start:returns_start]
            else:
                args_content = ""
        elif arg_section_match:
            args_content = content[args_start:len(content)]
        else:
            args_content = ""

        if arg_section_match:
            args_content = content[arg_section_match.end():returns_start]
            arguments = self._parse_arguments(args_content)

        returns = ""
        if returns_section_match:
            returns = content[returns_section_match.end():].strip()
            returns = re.sub(r"\n+", " ", returns).strip()

        return ToolDefinition(
            name=name,
            description=description,
            arguments=tuple(arguments),
            returns=returns,
        )

    def _parse_arguments(self, content: str) -> list[ArgumentDefinition]:
        """Parse argument definitions from content.

        Args:
            content: Content containing argument definitions

        Returns:
            List of ArgumentDefinition objects
        """
        arguments: list[ArgumentDefinition] = []
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            arg_match = self.ARGUMENT_PATTERN.match(line)
            if arg_match:
                arguments.append(
                    ArgumentDefinition(
                        name=arg_match.group(1),
                        type=arg_match.group(2),
                        description=arg_match.group(3),
                    )
                )

        return arguments


class ToolRunner:
    """Executes tools with given arguments asynchronously."""

    def __init__(self, tools_bin_path: Path | str) -> None:
        """Initialize tool runner.

        Args:
            tools_bin_path: Path to directory containing tool binaries/scripts
        """
        self.tools_bin_path = Path(tools_bin_path)
        if not self.tools_bin_path.exists():
            raise NotADirectoryError(f"Tools bin directory not found: {self.tools_bin_path}")

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> ExecutionResult:
        """Execute a tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of argument names to values
            timeout: Maximum time to wait for execution in seconds

        Returns:
            ExecutionResult with stdout, stderr, returncode, and success status

        Raises:
            FileNotFoundError: If the tool binary does not exist
            asyncio.TimeoutError: If execution exceeds timeout
        """
        arguments = arguments or {}
        tool_path = self._resolve_tool_path(tool_name)

        cmd = self._build_command(tool_path, arguments)

        logger.debug("Executing tool: %s with args %s", tool_name, arguments)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.tools_bin_path,
            )

            try:
                await asyncio.wait_for(process.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ExecutionResult(
                    stdout="",
                    stderr=f"Execution timed out after {timeout} seconds",
                    returncode=-1,
                    success=False,
                    error_message=f"Timeout after {timeout}s",
                )

            stdout_bytes, stderr_bytes = await process.communicate()

            return ExecutionResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                returncode=process.returncode,
                success=process.returncode == 0,
            )

        except Exception as e:
            logger.exception("Tool execution failed: %s", tool_name)
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                returncode=-1,
                success=False,
                error_message=str(e),
            )

    def _resolve_tool_path(self, tool_name: str) -> Path:
        """Resolve tool binary path.

        Args:
            tool_name: Name of the tool

        Returns:
            Path to the tool binary

        Raises:
            FileNotFoundError: If tool binary does not exist
        """
        if not re.match(r"^[a-zA-Z0-9_-]+$", tool_name):
            raise ValueError(f"Invalid tool name: {tool_name}")
        tool_path = self.tools_bin_path / tool_name
        if not tool_path.exists():
            raise FileNotFoundError(f"Tool not found: {tool_path}")

        if not tool_path.is_file():
            raise ValueError(f"Tool is not a file: {tool_path}")

        return tool_path

    def _build_command(
        self,
        tool_path: Path,
        arguments: dict[str, Any],
    ) -> list[str]:
        """Build command list for subprocess execution.

        Args:
            tool_path: Path to tool binary
            arguments: Arguments to pass to the tool

        Returns:
            List of command arguments
        """
        cmd = [str(tool_path)]
        for name, value in arguments.items():
            if value is None:
                continue

            if isinstance(value, bool):
                if value:
                    cmd.extend(["--", name.lstrip("-")])
            elif isinstance(value, (list, tuple)):
                for item in value:
                    cmd.extend(["--", name.lstrip("-"), str(item)])
            else:
                cmd.extend(["--", name.lstrip("-"), str(value)])

        return cmd

    def execute_sync(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> ExecutionResult:
        """Synchronous wrapper for execute.

        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of argument names to values
            timeout: Maximum time to wait for execution in seconds

        Returns:
            ExecutionResult with stdout, stderr, returncode, and success status
        """
        return asyncio.run(self.execute(tool_name, arguments, timeout))


def parse_tools_file(tools_path: Path | str) -> list[ToolDefinition]:
    """Parse a TOOLS.md file and return tool definitions.

    Convenience function that creates a ToolParser and parses the file.

    Args:
        tools_path: Path to TOOLS.md file

    Returns:
        List of ToolDefinition objects
    """
    parser = ToolParser(tools_path)
    return parser.parse()