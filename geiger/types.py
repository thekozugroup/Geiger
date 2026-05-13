from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExecutionResult:
    """Result of a tool execution."""

    stdout: str
    stderr: str
    returncode: int
    success: bool
    error_message: str = ""

    @classmethod
    def from_completed_process(cls, process: Any) -> ExecutionResult:
        """Create ExecutionResult from a completed subprocess."""
        return cls(
            stdout=process.stdout.decode("utf-8", errors="replace") if process.stdout else "",
            stderr=process.stderr.decode("utf-8", errors="replace") if process.stderr else "",
            returncode=process.returncode,
            success=process.returncode == 0,
        )


@dataclass(frozen=True)
class ToolDefinition:
    """Definition of a tool parsed from TOOLS.md."""

    name: str
    description: str
    arguments: tuple[Any, ...] = field(default_factory=tuple)
    returns: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tool name cannot be empty")


@dataclass
class Trace:
    """Canonical trace from agent execution."""

    session_id: str
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    timestamp: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "timestamp": self.timestamp,
            "error": self.error,
        }


@dataclass
class DatasetTrace:
    """Trace for dataset generation with graded outputs."""

    messages: list[Any]
    grade: float
    tool_definitions: list[dict[str, Any]] | None = None
