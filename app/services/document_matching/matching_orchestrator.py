from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .hash_matcher import (
    DocumentHashMatcher,
    HashMatchCandidate,
    HashMatchResult,
)
from .summary_matcher import (
    DocumentSummaryMatcher,
    SummaryMatchCandidate,
    SummaryMatchResult,
)


MatchClassification = Literal[
    "exact_file_match",
    "exact_content_match",
    "potential_match",
    "no_match",
]


@dataclass(frozen=True, slots=True)
class DocumentMatchingResult:
    classification: MatchClassification

    exact_file_matches: tuple[HashMatchCandidate, ...]
    exact_content_matches: tuple[HashMatchCandidate, ...]

    semantic_candidates: tuple[SummaryMatchCandidate, ...]

    requires_user_review: bool

    @property
    def has_exact_match(self) -> bool:
        return bool(
            self.exact_file_matches
            or self.exact_content_matches
        )

    @property
    def has_potential_match(self) -> bool:
        return bool(self.semantic_candidates)

    @property
    def is_new_document(self) -> bool:
        return self.classification == "no_match"

    @property
    def exact_match_count(self) -> int:
        return (
            len(self.exact_file_matches)
            + len(self.exact_content_matches)
        )

    @property
    def semantic_match_count(self) -> int:
        return len(self.semantic_candidates)


class DocumentMatchingOrchestrator:
    """
    Combines exact hash matching and semantic summary matching.

    Matching precedence:
        1. Exact original PDF hash.
        2. Exact normalized content hash.
        3. Semantic summary similarity.
        4. No match.

    This class is read-only.

    It does not:
        - Create database records.
        - Modify document versions.
        - Upload files.
        - Modify Qdrant.
        - Register or activate documents.
    """

    def __init__(
        self,
        hash_matcher: DocumentHashMatcher,
        summary_matcher: DocumentSummaryMatcher,
    ) -> None:
        self.hash_matcher = hash_matcher
        self.summary_matcher = summary_matcher

    def match(
        self,
        file_hash: str | None,
        content_hash: str | None,
        summary: str,
    ) -> DocumentMatchingResult:

        if not file_hash and not content_hash:
            raise ValueError(
                "At least one hash must be provided."
            )

        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(
                "summary must be a non-empty string."
            )

        # --------------------------------------------
        # 1. Exact hash matching
        # --------------------------------------------

        hash_result: HashMatchResult = self.hash_matcher.match(
            file_hash=file_hash,
            content_hash=content_hash,
        )

        exact_file_matches = tuple(
            candidate
            for candidate in hash_result.candidates
            if candidate.matched_by == "file_hash"
        )

        exact_content_matches = tuple(
            candidate
            for candidate in hash_result.candidates
            if candidate.matched_by == "content_hash"
        )

        # --------------------------------------------
        # 2. Semantic summary matching
        # --------------------------------------------

        # Exact hashes conclusively identify an existing version.
        # Skip semantic search for duplicates; it adds latency without
        # changing the classification or review decision.
        if exact_file_matches or exact_content_matches:
            classification: MatchClassification = (
                "exact_file_match"
                if exact_file_matches
                else "exact_content_match"
            )
            return DocumentMatchingResult(
                classification=classification,
                exact_file_matches=exact_file_matches,
                exact_content_matches=exact_content_matches,
                semantic_candidates=(),
                requires_user_review=True,
            )

        summary_result: SummaryMatchResult = (
            self.summary_matcher.match(
                summary.strip(),
            )
        )

        semantic_candidates = summary_result.candidates

        # --------------------------------------------
        # 3. Determine overall classification
        # --------------------------------------------

        if exact_file_matches:
            classification: MatchClassification = "exact_file_match"

        elif exact_content_matches:
            classification = "exact_content_match"

        elif semantic_candidates:
            classification = "potential_match"

        else:
            classification = "no_match"

        # --------------------------------------------
        # 4. Build unified result
        # --------------------------------------------

        requires_user_review = classification != "no_match"

        return DocumentMatchingResult(
            classification=classification,
            exact_file_matches=exact_file_matches,
            exact_content_matches=exact_content_matches,
            semantic_candidates=semantic_candidates,
            requires_user_review=requires_user_review,
        )