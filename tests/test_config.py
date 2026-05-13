import pytest

from geiger.config import Config


class TestConfigValidation:
    def test_validate_missing_tools_md_path(self):
        config = Config(
            tools_md_path="",
            tool_binaries_path="/tmp/bin",
            output_dir="/tmp/out",
            api_key="test-key",
        )
        with pytest.raises(ValueError, match="tools_md_path"):
            config.validate()

    def test_validate_missing_output_dir(self):
        config = Config(
            tools_md_path="/tmp/tools.md",
            tool_binaries_path="/tmp/bin",
            output_dir="",
            api_key="test-key",
        )
        with pytest.raises(ValueError, match="output_dir"):
            config.validate()

    def test_validate_api_key_not_required_by_default(self):
        config = Config(
            tools_md_path="/tmp/tools.md",
            tool_binaries_path="/tmp/bin",
            output_dir="/tmp/out",
            api_key="",
        )
        with pytest.raises(ValueError, match="api_key"):
            config.validate()

    def test_validate_success(self):
        config = Config(
            tools_md_path="/tmp/tools.md",
            tool_binaries_path="/tmp/bin",
            output_dir="/tmp/out",
            api_key="test-key",
        )
        config.validate()

    def test_validate_invalid_threshold(self):
        config = Config(
            tools_md_path="/tmp/tools.md",
            tool_binaries_path="/tmp/bin",
            output_dir="/tmp/out",
            api_key="test-key",
            min_grade_threshold=1.5,
        )
        with pytest.raises(ValueError, match="min_grade_threshold"):
            config.validate()

    def test_validate_invalid_workers(self):
        config = Config(
            tools_md_path="/tmp/tools.md",
            tool_binaries_path="/tmp/bin",
            output_dir="/tmp/out",
            api_key="test-key",
            max_agent_workers=0,
        )
        with pytest.raises(ValueError, match="max_agent_workers"):
            config.validate()