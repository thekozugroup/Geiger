from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ShareGPTMessage:
    from_: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {"from": self.from_, "value": self.value}


@dataclass
class ShareGPTConversation:
    conversations: list[ShareGPTMessage]
    system: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "conversations": [msg.to_dict() for msg in self.conversations]
        }
        if self.system is not None:
            result["system"] = self.system
        return result


def format_human_message(content: str) -> ShareGPTMessage:
    return ShareGPTMessage(from_="human", value=content)


def format_gpt_message(content: str) -> ShareGPTMessage:
    return ShareGPTMessage(from_="gpt", value=content)


def format_tool_message(content: str) -> ShareGPTMessage:
    return ShareGPTMessage(from_="tool", value=content)


def build_system_prompt(
    tool_definitions: list[dict[str, Any]] | None = None,
) -> str:
    if not tool_definitions:
        return "You are a helpful AI assistant with access to tools."

    tools_json = "\n".join(
        f"- {t['function']['name']}: {t['function']['description']}"
        for t in tool_definitions
    )
    return (
        "You are a helpful AI assistant with access to tools.\n"
        "Available tools:\n"
        f"{tools_json}\n"
        "Use tools when appropriate to help the user."
    )