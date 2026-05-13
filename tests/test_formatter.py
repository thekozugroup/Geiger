import pytest

from geiger.dataset.formatter import (
    ShareGPTMessage,
    ShareGPTConversation,
    format_human_message,
    format_gpt_message,
    format_tool_message,
    build_system_prompt,
)


class TestShareGPTMessages:
    def test_format_human_message(self):
        msg = format_human_message("Hello, world!")
        assert msg.from_ == "human"
        assert msg.value == "Hello, world!"
        assert msg.to_dict() == {"from": "human", "value": "Hello, world!"}

    def test_format_gpt_message(self):
        msg = format_gpt_message("I can help with that.")
        assert msg.from_ == "gpt"
        assert msg.value == "I can help with that."
        assert msg.to_dict() == {"from": "gpt", "value": "I can help with that."}

    def test_format_tool_message(self):
        msg = format_tool_message('{"result": "success"}')
        assert msg.from_ == "tool"
        assert msg.value == '{"result": "success"}'
        assert msg.to_dict() == {"from": "tool", "value": '{"result": "success"}'}


class TestShareGPTConversation:
    def test_conversation_to_dict(self):
        conv = ShareGPTConversation(
            conversations=[
                format_human_message("Hi"),
                format_gpt_message("Hello!"),
            ]
        )
        result = conv.to_dict()
        assert "conversations" in result
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["from"] == "human"

    def test_conversation_with_system(self):
        conv = ShareGPTConversation(
            conversations=[format_human_message("Hi")],
            system="You are helpful.",
        )
        result = conv.to_dict()
        assert result["system"] == "You are helpful."


class TestBuildSystemPrompt:
    def test_build_system_prompt_with_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                },
            }
        ]
        prompt = build_system_prompt(tools)
        assert "test_tool" in prompt
        assert "A test tool" in prompt
        assert "Available tools" in prompt

    def test_build_system_prompt_without_tools(self):
        prompt = build_system_prompt(None)
        assert "helpful AI assistant" in prompt
        assert "tools" not in prompt or "Available tools" not in prompt

    def test_build_system_prompt_empty_list(self):
        prompt = build_system_prompt([])
        assert "helpful AI assistant" in prompt