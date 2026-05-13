from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles

from geiger.dataset.formatter import (
    ShareGPTConversation,
    ShareGPTMessage,
    build_system_prompt,
    format_gpt_message,
    format_human_message,
    format_tool_message,
)
from geiger.types import DatasetTrace


@dataclass(frozen=True)
class TraceStep:
    """A single step in a trace conversation."""
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_result: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset generation."""
    output_dir: Path = field(default_factory=lambda: Path("output"))
    min_grade_threshold: float = 0.0
    filename_template: str = "data_{index}.json"

    def ensure_output_dir(self) -> None:
        """Create the output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetStats:
    total_traces: int = 0
    filtered_traces: int = 0
    min_grade: float = 0.0
    max_grade: float = 0.0
    avg_grade: float = 0.0
    tool_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_traces": self.total_traces,
            "filtered_traces": self.filtered_traces,
            "min_grade": self.min_grade,
            "max_grade": self.max_grade,
            "avg_grade": round(self.avg_grade, 4),
            "tool_count": self.tool_count,
        }


class DatasetGenerator:
    """Generates datasets from traces with grading."""

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()

    def filter_traces(self, traces: list[DatasetTrace]) -> list[DatasetTrace]:
        return [
            t for t in traces if t.grade >= self.config.min_grade_threshold
        ]

    def _trace_to_conversation(self, trace: DatasetTrace) -> ShareGPTConversation:
        messages: list[ShareGPTMessage] = []

        for step in trace.messages:
            if step.role == "human":
                messages.append(format_human_message(step.content))
            elif step.role == "assistant":
                if step.tool_calls:
                    tool_calls_str = json.dumps(step.tool_calls)
                    messages.append(
                        format_gpt_message(
                            f"{step.content}\n<tool_calls>{tool_calls_str}</tool_calls>"
                        )
                    )
                else:
                    messages.append(format_gpt_message(step.content))
            elif step.role == "tool":
                messages.append(format_tool_message(step.tool_result or ""))

        system = build_system_prompt(trace.tool_definitions)

        return ShareGPTConversation(conversations=messages, system=system)

    async def generate(
        self,
        traces: list[DatasetTrace],
        output_dir: Path | None = None,
    ) -> DatasetStats:
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        filtered_traces = self.filter_traces(traces)

        grades = [t.grade for t in filtered_traces] if filtered_traces else []
        tool_names: set[str] = set()
        for t in filtered_traces:
            if t.tool_definitions:
                for td in t.tool_definitions:
                    if 'function' in td and 'name' in td['function']:
                        tool_names.add(td['function']['name'])

        stats = DatasetStats(
            total_traces=len(traces),
            filtered_traces=len(filtered_traces),
            min_grade=min(grades) if grades else 0.0,
            max_grade=max(grades) if grades else 0.0,
            avg_grade=sum(grades) / len(grades) if grades else 0.0,
            tool_count=len(tool_names),
        )

        write_tasks: list[asyncio.Task[None]] = []
        for idx, trace in enumerate(filtered_traces):
            conversation = self._trace_to_conversation(trace)
            filename = self.config.filename_template.format(index=idx)
            filepath = output_dir / filename
            tmp_path = filepath.with_suffix('.tmp')

            async def write_file(path: Path, data: str) -> None:
                async with aiofiles.open(path, "w") as f:
                    await f.write(data)

            write_tasks.append(
                asyncio.create_task(
                    write_file(tmp_path, json.dumps(conversation.to_dict(), indent=2))
                )
            )

        await asyncio.gather(*write_tasks)

        for idx, trace in enumerate(filtered_traces):
            filename = self.config.filename_template.format(index=idx)
            filepath = output_dir / filename
            tmp_path = filepath.with_suffix('.tmp')
            if tmp_path.exists():
                tmp_path.rename(filepath)

        manifest_path = output_dir / "manifest.json"
        manifest_tmp_path = manifest_path.with_suffix('.json.tmp')
        async with aiofiles.open(manifest_tmp_path, "w") as f:
            await f.write(json.dumps(stats.to_dict(), indent=2))
        if manifest_tmp_path.exists():
            manifest_tmp_path.rename(manifest_path)

        return stats