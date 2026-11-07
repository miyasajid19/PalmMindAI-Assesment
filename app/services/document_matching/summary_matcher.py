
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any


@dataclass(frozen=True, slots=True)
class SummaryMatchCandidate:
    document_id: str
    version_id: str
    version_number: int
    title: str
    status: str
    summary: str
    similarity_score: float


@dataclass(frozen=True, slots=True)
class SummaryMatchResult:
    is_match: bool
    candidates: tuple[SummaryMatchCandidate, ...]

    @property
    def match_count(self) -> int:
        return len(self.candidates)


class DocumentSummaryMatcher:
    """Find potentially related document versions by summary similarity."""

    def __init__(
        self,
        summary_store: Any,
        threshold: float = 0.80,
        candidate_limit: int = 20,
    ) -> None:
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError(
                "threshold must be a finite number between 0.0 and 1.0."
            )

        if (
            isinstance(candidate_limit, bool)
            or not isinstance(candidate_limit, int)
            or candidate_limit < 1
        ):
            raise ValueError("candidate_limit must be a positive integer.")

        self.summary_store = summary_store
        self.threshold = threshold
        self.candidate_limit = candidate_limit

    def match(
        self,
        incoming_summary: str,
    ) -> SummaryMatchResult:
        """Return all retrieved candidates meeting the similarity threshold."""

        if not isinstance(incoming_summary, str) or not incoming_summary.strip():
            raise ValueError("incoming_summary must be a non-empty string.")

        search_results = self.summary_store.search_similar_summaries(
            summary=incoming_summary.strip(),
            limit=self.candidate_limit,
        )

        candidates_by_version: dict[str, SummaryMatchCandidate] = {}

        for result in search_results:
            score = float(result["score"])

            if score < self.threshold:
                continue

            version_id = str(result["document_version_id"])

            candidate = SummaryMatchCandidate(
                document_id=str(result["document_id"]),
                version_id=version_id,
                version_number=int(result["version_number"]),
                title=str(result["title"]),
                status=str(result["status"]),
                summary=str(result["summary"]),
                similarity_score=score,
            )

            # Avoid returning the same version twice.
            existing = candidates_by_version.get(version_id)

            if (
                existing is None
                or candidate.similarity_score > existing.similarity_score
            ):
                candidates_by_version[version_id] = candidate

        candidates = tuple(
            sorted(
                candidates_by_version.values(),
                key=lambda item: item.similarity_score,
                reverse=True,
            )
        )

        return SummaryMatchResult(
            is_match=bool(candidates),
            candidates=candidates,
        )
