from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4"
    max_agent_workers: int = 5
    max_reviewer_workers: int = 3
    min_grade_threshold: float = 0.5
    temperature: float = 0.7
    max_tokens: int = 2048
    tools_md_path: str = ""
    tool_binaries_path: str = ""
    input_files_dir: Optional[str] = None
    output_dir: str = ""

    def validate(self) -> None:
        if not self.tools_md_path:
            raise ValueError("tools_md_path is required")
        if not self.tool_binaries_path:
            raise ValueError("tool_binaries_path is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if self.max_agent_workers < 1:
            raise ValueError("max_agent_workers must be >= 1")
        if self.max_reviewer_workers < 1:
            raise ValueError("max_reviewer_workers must be >= 1")
        if not 0.0 <= self.min_grade_threshold <= 1.0:
            raise ValueError("min_grade_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
