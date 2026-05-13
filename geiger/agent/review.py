from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import asyncio
import json
import logging

import httpx

logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """Status of a trace review."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


@dataclass(frozen=True)
class GradeBreakdown:
    """Breakdown of grades by criterion."""
    tool_usage: float = 0.0
    argument_accuracy: float = 0.0
    relevance: float = 0.0
    coherence: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "tool_usage": self.tool_usage,
            "argument_accuracy": self.argument_accuracy,
            "relevance": self.relevance,
            "coherence": self.coherence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GradeBreakdown":
        return cls(
            tool_usage=max(0.0, min(1.0, float(data.get("tool_usage", 0.0)))),
            argument_accuracy=max(0.0, min(1.0, float(data.get("argument_accuracy", 0.0)))),
            relevance=max(0.0, min(1.0, float(data.get("relevance", 0.0)))),
            coherence=max(0.0, min(1.0, float(data.get("coherence", 0.0)))),
        )


@dataclass(frozen=True)
class ReviewResult:
    """Result of a trace review."""
    status: ReviewStatus
    grade: float
    reasoning: str
    breakdown: GradeBreakdown = field(default_factory=GradeBreakdown)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceData:
    """Data for a trace to be reviewed."""
    prompt: str
    messages: list[dict[str, Any]]
    tools: Optional[list[dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "prompt": self.prompt,
            "messages": self.messages,
            "tools": self.tools or [],
        }


class Reviewer:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-4o",
        max_workers: int = 4,
        min_score: float = 0.7,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.min_score = min_score
        self.timeout = timeout
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._client: httpx.AsyncClient | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_workers)
        return self._semaphore

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the reviewer and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _create_review_prompt(self, trace: TraceData) -> str:
        """Create a review prompt for the given trace."""
        return f"""Review the following agent trace and grade it on a 0-1 scale.

## Prompt
{trace.prompt}

## Messages
{json.dumps(trace.messages, indent=2)}

## Grading Criteria
Grade each criterion on a 0-1 scale:
- tool_usage (0-1): Correct tool selection and usage
- argument_accuracy (0-1): Correct argument passing
- relevance (0-1): Response relevance to prompt
- coherence (0-1): Overall conversation coherence

## Output Format
Return a JSON object with this exact structure:
{{
    "tool_usage": <float 0-1>,
    "argument_accuracy": <float 0-1>,
    "relevance": <float 0-1>,
    "coherence": <float 0-1>,
    "reasoning": "<brief explanation of grades>"
}}

Be strict in your evaluation. Scores below 0.5 should reflect genuine issues."""

    async def _call_api(self, prompt: str) -> str:
        """Call the review API with the given prompt."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }

        try:
            suffix = "/v1/chat/completions" if not self.base_url.endswith("/v1") else "/chat/completions"
            url = f"{self.base_url}{suffix}"
            response = await self.client.post(
                url,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during API call: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error during API call: {e}")
            raise

        choices = data.get("choices", [])
        if not choices:
            return ""
        content = choices[0].get("message", {}).get("content", "")
        return content

    def _parse_response(self, content: str) -> Optional[dict[str, Any]]:
        try:
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            return json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return None

    def _compute_grade(self, breakdown: GradeBreakdown) -> float:
        grade = (
            breakdown.tool_usage * 0.3
            + breakdown.argument_accuracy * 0.3
            + breakdown.relevance * 0.2
            + breakdown.coherence * 0.2
        )
        return max(0.0, min(1.0, grade))

    def _determine_status(self, grade: float) -> ReviewStatus:
        if grade >= self.min_score:
            return ReviewStatus.APPROVED
        elif grade >= 0.4:
            return ReviewStatus.NEEDS_REVISION
        else:
            return ReviewStatus.REJECTED

    async def review_trace(self, trace: TraceData) -> ReviewResult:
        prompt = self._create_review_prompt(trace)

        try:
            content = await self._call_api(prompt)
            parsed = self._parse_response(content)

            if parsed is None:
                breakdown = GradeBreakdown()
                reasoning = "Failed to parse model response; defaulting to 0.5 grades"
            else:
                breakdown = GradeBreakdown.from_dict(parsed)
                reasoning = str(parsed.get("reasoning", ""))

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during review: {e}")
            breakdown = GradeBreakdown()
            reasoning = "HTTP error: review service unavailable"
        except httpx.TimeoutException as e:
            logger.error(f"Timeout during review: {e}")
            breakdown = GradeBreakdown()
            reasoning = "Request timeout"
        except httpx.RequestError as e:
            logger.error(f"Request error during review: {e}")
            breakdown = GradeBreakdown()
            reasoning = "Request error: review service unavailable"
        except Exception as e:
            logger.error(f"Unexpected error during review: {e}")
            breakdown = GradeBreakdown()
            reasoning = "Unexpected error during review"

        grade = self._compute_grade(breakdown)

        return ReviewResult(
            status=self._determine_status(grade),
            grade=grade,
            reasoning=reasoning,
            breakdown=breakdown,
            metadata={"model": self.model},
        )

    async def review_result(self, result: dict[str, Any]) -> ReviewResult:
        messages = result.get("messages", [])
        prompt = result.get("prompt", "")
        if not isinstance(messages, list):
            messages = []
        if not isinstance(prompt, str):
            prompt = ""
        trace_data = TraceData(
            prompt=prompt,
            messages=messages,
            tools=result.get("tools"),
        )
        return await self.review_trace(trace_data)

    async def batch_review(self, traces: list[TraceData]) -> list[ReviewResult]:
        if not traces:
            return []

        async def _review_with_semaphore(trace: TraceData) -> ReviewResult:
            async with self.semaphore:
                return await self.review_trace(trace)

        try:
            tasks = [_review_with_semaphore(t) for t in traces]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await self.close()

        processed_results: list[ReviewResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Review task {i} failed with exception: {result}")
                processed_results.append(
                    ReviewResult(
                        status=ReviewStatus.REJECTED,
                        grade=0.0,
                        reasoning="Task failed",
                        breakdown=GradeBreakdown(),
                    )
                )
            else:
                processed_results.append(result)

        return processed_results
