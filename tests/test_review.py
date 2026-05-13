import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock

from geiger.agent.review import Reviewer, GradeBreakdown, ReviewStatus, TraceData


class TestReviewerGradeComputation:
    def test_grade_computation_bounds(self):
        reviewer = Reviewer(base_url="http://test.com", api_key="test-key")

        breakdown = GradeBreakdown(
            tool_usage=1.0,
            argument_accuracy=1.0,
            relevance=1.0,
            coherence=1.0,
        )
        grade = reviewer._compute_grade(breakdown)
        assert grade == 1.0

        breakdown = GradeBreakdown(
            tool_usage=0.0,
            argument_accuracy=0.0,
            relevance=0.0,
            coherence=0.0,
        )
        grade = reviewer._compute_grade(breakdown)
        assert grade == 0.0

        breakdown = GradeBreakdown(
            tool_usage=0.5,
            argument_accuracy=0.5,
            relevance=0.5,
            coherence=0.5,
        )
        grade = reviewer._compute_grade(breakdown)
        assert 0.0 <= grade <= 1.0

    def test_grade_computation_clamped(self):
        reviewer = Reviewer(base_url="http://test.com", api_key="test-key")

        breakdown = GradeBreakdown(
            tool_usage=2.0,
            argument_accuracy=-1.0,
            relevance=0.5,
            coherence=0.5,
        )
        grade = reviewer._compute_grade(breakdown)
        assert 0.0 <= grade <= 1.0


class TestReviewerStatus:
    def test_determine_status_thresholds(self):
        reviewer = Reviewer(base_url="http://test.com", api_key="test-key", min_score=0.7)

        assert reviewer._determine_status(0.9) == ReviewStatus.APPROVED
        assert reviewer._determine_status(0.7) == ReviewStatus.APPROVED
        assert reviewer._determine_status(0.69) == ReviewStatus.NEEDS_REVISION
        assert reviewer._determine_status(0.4) == ReviewStatus.NEEDS_REVISION
        assert reviewer._determine_status(0.39) == ReviewStatus.REJECTED
        assert reviewer._determine_status(0.0) == ReviewStatus.REJECTED


class TestReviewerParsing:
    def test_parse_response_valid_json(self):
        reviewer = Reviewer(base_url="http://test.com", api_key="test-key")
        content = '{"tool_usage": 0.8, "argument_accuracy": 0.9, "relevance": 0.7, "coherence": 0.6, "reasoning": "good"}'
        result = reviewer._parse_response(content)
        assert result is not None
        assert result["tool_usage"] == 0.8
        assert result["argument_accuracy"] == 0.9

    def test_parse_response_invalid_json(self):
        reviewer = Reviewer(base_url="http://test.com", api_key="test-key")
        content = "not valid json at all"
        result = reviewer._parse_response(content)
        assert result is None

    def test_parse_response_with_code_fence(self):
        reviewer = Reviewer(base_url="http://test.com", api_key="test-key")
        content = '```json\n{"tool_usage": 0.8, "argument_accuracy": 0.9, "relevance": 0.7, "coherence": 0.6}\n```'
        result = reviewer._parse_response(content)
        assert result is not None
        assert result["tool_usage"] == 0.8

    @pytest.mark.asyncio
    async def test_review_result_result(self):
        reviewer = Reviewer(base_url="http://test.com", api_key="test-key")
        reviewer._call_api = AsyncMock(return_value='{"tool_usage": 0.8, "argument_accuracy": 0.9, "relevance": 0.7, "coherence": 0.6, "reasoning": "good"}')
        result = await reviewer.review_result({"prompt": "test", "messages": [{"role": "user", "content": "hello"}]})
        assert result.grade > 0
        assert result.status in [ReviewStatus.APPROVED, ReviewStatus.REJECTED, ReviewStatus.NEEDS_REVISION]


class TestReviewerAsync:
    @pytest.mark.asyncio
    async def test_review_trace_success(self):
        reviewer = Reviewer(base_url="http://test.com", api_key="test-key")
        reviewer._call_api = AsyncMock(return_value='{"tool_usage": 0.8, "argument_accuracy": 0.9, "relevance": 0.7, "coherence": 0.6, "reasoning": "good"}')
        trace = TraceData(prompt="test prompt", messages=[{"role": "user", "content": "hello"}])
        result = await reviewer.review_trace(trace)
        assert result.grade > 0
        assert result.status in [ReviewStatus.APPROVED, ReviewStatus.REJECTED, ReviewStatus.NEEDS_REVISION]
        assert result.breakdown.tool_usage == 0.8

    @pytest.mark.asyncio
    async def test_review_trace_api_failure(self):
        reviewer = Reviewer(base_url="http://test.com", api_key="test-key")
        reviewer._call_api = AsyncMock(side_effect=httpx.HTTPStatusError("error", request=MagicMock(), response=MagicMock()))
        trace = TraceData(prompt="test prompt", messages=[{"role": "user", "content": "hello"}])
        result = await reviewer.review_trace(trace)
        assert result.grade == 0.0
        assert result.status == ReviewStatus.REJECTED

    @pytest.mark.asyncio
    async def test_batch_review_empty(self):
        reviewer = Reviewer(base_url="http://test.com", api_key="test-key")
        results = await reviewer.batch_review([])
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_review_success(self):
        reviewer = Reviewer(base_url="http://test.com", api_key="test-key", max_workers=2)
        reviewer._call_api = AsyncMock(return_value='{"tool_usage": 0.8, "argument_accuracy": 0.9, "relevance": 0.7, "coherence": 0.6, "reasoning": "good"}')
        traces = [
            TraceData(prompt="test1", messages=[{"role": "user", "content": "hello"}]),
            TraceData(prompt="test2", messages=[{"role": "user", "content": "world"}]),
        ]
        results = await reviewer.batch_review(traces)
        assert len(results) == 2
        for result in results:
            assert result.grade > 0

    @pytest.mark.asyncio
    async def test_batch_review_partial_failure(self):
        reviewer = Reviewer(base_url="http://test.com", api_key="test-key", max_workers=2)
        call_count = 0
        async def mock_call(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return '{"tool_usage": 0.8, "argument_accuracy": 0.9, "relevance": 0.7, "coherence": 0.6, "reasoning": "good"}'
            raise httpx.RequestError("failed")
        reviewer._call_api = AsyncMock(side_effect=mock_call)
        traces = [
            TraceData(prompt="test1", messages=[{"role": "user", "content": "hello"}]),
            TraceData(prompt="test2", messages=[{"role": "user", "content": "world"}]),
        ]
        results = await reviewer.batch_review(traces)
        assert len(results) == 2
        assert results[0].status == ReviewStatus.APPROVED
        assert results[1].status == ReviewStatus.REJECTED
        assert results[1].reasoning == "Request error: review service unavailable"