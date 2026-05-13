import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from geiger.dataset.generator import (
    DatasetGenerator,
    DatasetConfig,
    DatasetStats,
    TraceStep,
)
from geiger.types import DatasetTrace


@pytest.fixture
def sample_traces():
    return [
        DatasetTrace(
            messages=[
                TraceStep(role="human", content="Hello"),
                TraceStep(
                    role="assistant",
                    content="Hi there!",
                    tool_calls=[
                        {"id": "call_1", "function": {"name": "test_tool", "arguments": {}}}
                    ],
                ),
                TraceStep(role="tool", content="Tool result", tool_result="success"),
            ],
            grade=0.8,
            tool_definitions=[
                {"function": {"name": "test_tool", "description": "A test tool", "parameters": {}}}
            ],
        ),
        DatasetTrace(
            messages=[
                TraceStep(role="human", content="Hi"),
                TraceStep(role="assistant", content="Hello!"),
            ],
            grade=0.3,
            tool_definitions=[],
        ),
        DatasetTrace(
            messages=[
                TraceStep(role="human", content="Goodbye"),
                TraceStep(role="assistant", content="See you!"),
            ],
            grade=0.9,
            tool_definitions=[],
        ),
    ]


class TestDatasetGenerator:
    def test_initialization(self):
        config = DatasetConfig(
            output_dir=Path("/tmp/test_output"),
            min_grade_threshold=0.5,
            filename_template="trace_{index}.json",
        )
        generator = DatasetGenerator(config)
        assert generator.config.min_grade_threshold == 0.5
        assert generator.config.output_dir == Path("/tmp/test_output")

    def test_filter_traces_by_grade(self, sample_traces):
        config = DatasetConfig(min_grade_threshold=0.5)
        generator = DatasetGenerator(config)
        filtered = generator.filter_traces(sample_traces)
        assert len(filtered) == 2
        grades = [t.grade for t in filtered]
        assert min(grades) >= 0.5

    def test_filter_traces_with_zero_threshold(self, sample_traces):
        config = DatasetConfig(min_grade_threshold=0.0)
        generator = DatasetGenerator(config)
        filtered = generator.filter_traces(sample_traces)
        assert len(filtered) == 3

    def test_filter_traces_with_high_threshold(self, sample_traces):
        config = DatasetConfig(min_grade_threshold=1.0)
        generator = DatasetGenerator(config)
        filtered = generator.filter_traces(sample_traces)
        assert len(filtered) == 0


class TestTraceToConversation:
    def test_trace_to_conversation_human_assistant(self):
        config = DatasetConfig()
        generator = DatasetGenerator(config)
        trace = DatasetTrace(
            messages=[
                TraceStep(role="human", content="Hello world"),
                TraceStep(role="assistant", content="Hi! How can I help?"),
            ],
            grade=0.7,
            tool_definitions=None,
        )
        conv = generator._trace_to_conversation(trace)
        assert "helpful AI assistant" in conv.system
        assert len(conv.conversations) == 2

    def test_trace_to_conversation_with_tool_calls(self):
        config = DatasetConfig()
        generator = DatasetGenerator(config)
        trace = DatasetTrace(
            messages=[
                TraceStep(
                    role="assistant",
                    content="Let me check that",
                    tool_calls=[{"id": "c1", "function": {"name": "search", "arguments": {}}}],
                ),
                TraceStep(role="tool", content="Search result", tool_result="Found it"),
            ],
            grade=0.8,
            tool_definitions=[{"function": {"name": "search", "description": "Search tool", "parameters": {}}}],
        )
        conv = generator._trace_to_conversation(trace)
        assert len(conv.conversations) == 2
        assert "search" in conv.system

    def test_trace_to_conversation_system_prompt(self):
        config = DatasetConfig()
        generator = DatasetGenerator(config)
        trace = DatasetTrace(
            messages=[
                TraceStep(role="human", content="Hello"),
                TraceStep(role="assistant", content="Hi"),
            ],
            grade=0.6,
            tool_definitions=[{"function": {"name": "tool1", "description": "A tool", "parameters": {}}}],
        )
        conv = generator._trace_to_conversation(trace)
        assert "tool1" in conv.system


class TestDatasetStats:
    def test_stats_to_dict(self):
        stats = DatasetStats(
            total_traces=10,
            filtered_traces=8,
            min_grade=0.5,
            max_grade=0.9,
            avg_grade=0.75,
            tool_count=3,
        )
        d = stats.to_dict()
        assert d["total_traces"] == 10
        assert d["filtered_traces"] == 8
        assert d["min_grade"] == 0.5
        assert d["max_grade"] == 0.9
        assert d["avg_grade"] == 0.75
        assert d["tool_count"] == 3

    def test_stats_avg_grade_rounding(self):
        stats = DatasetStats(
            total_traces=2,
            filtered_traces=2,
            min_grade=0.3,
            max_grade=0.7,
            avg_grade=0.666666,
            tool_count=0,
        )
        d = stats.to_dict()
        assert d["avg_grade"] == 0.6667


class TestDatasetConfig:
    def test_ensure_output_dir(self, tmp_path):
        config = DatasetConfig(output_dir=tmp_path / "new_dir")
        config.ensure_output_dir()
        assert (tmp_path / "new_dir").exists()

    def test_filename_template(self):
        config = DatasetConfig(filename_template="trace_{index}.json")
        assert config.filename_template == "trace_{index}.json"
        formatted = config.filename_template.format(index=5)
        assert formatted == "trace_5.json"


class TestGenerateAsync:
    @pytest.mark.asyncio
    async def test_generate_with_mocked_io(self, sample_traces, tmp_path):
        config = DatasetConfig(min_grade_threshold=0.5)
        generator = DatasetGenerator(config)

        mock_file = MagicMock()
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=None)
        mock_file.write = AsyncMock()

        with patch("geiger.dataset.generator.aiofiles.open", return_value=mock_file):
            stats = await generator.generate(sample_traces, output_dir=tmp_path)

        assert stats.total_traces == 3
        assert stats.filtered_traces == 2
        assert stats.min_grade == 0.8
        assert stats.max_grade == 0.9

    @pytest.mark.asyncio
    async def test_generate_empty_traces(self, tmp_path):
        config = DatasetConfig()
        generator = DatasetGenerator(config)

        mock_file = MagicMock()
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=None)
        mock_file.write = AsyncMock()

        with patch("geiger.dataset.generator.aiofiles.open", return_value=mock_file):
            stats = await generator.generate([], output_dir=tmp_path)

        assert stats.total_traces == 0
        assert stats.filtered_traces == 0
        assert stats.avg_grade == 0.0

    @pytest.mark.asyncio
    async def test_generate_no_matching_grade(self, sample_traces, tmp_path):
        config = DatasetConfig(min_grade_threshold=1.0)
        generator = DatasetGenerator(config)

        mock_file = MagicMock()
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=None)
        mock_file.write = AsyncMock()

        with patch("geiger.dataset.generator.aiofiles.open", return_value=mock_file):
            stats = await generator.generate(sample_traces, output_dir=tmp_path)

        assert stats.filtered_traces == 0