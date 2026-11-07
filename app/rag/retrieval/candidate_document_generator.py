from __future__ import annotations

import asyncio

import logging
import math
from numbers import Real
from typing import Any


from pydantic import BaseModel, ConfigDict, Field

from app.agents.Multi_query_generator.schemas import (
    GeneratedQuery,
    QueryVersionScope,
)
from app.rag.vector_stores.qdrant_summary import (
    QdrantSummaryStore,
)

logger = logging.getLogger(__name__)


class CandidateDocument(BaseModel):
    """A shared-corpus document version selected by summary similarity."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_version_id: str
    document_title: str
    document_summary: str
    version_year: int
    version_status: str
    similarity_score: float
    source: str | None = None


class QueryCandidateResult(BaseModel):
    """Candidate documents generated for one contextualized query."""

    model_config = ConfigDict(extra="forbid")

    query: str
    version_scope: QueryVersionScope
    target_versions: list[int] = Field(default_factory=list)
    candidates: list[CandidateDocument] = Field(default_factory=list)



class CandidateDocumentGenerationError(Exception):
    """Raised when candidate generation cannot complete."""


class CandidateDocumentGenerator:
    """
    Generate document-version candidates for one contextualized query.

    The query performs a summary similarity search constrained
    by exact document-version IDs from the shared catalogue and requested lifecycle scope.
    PostgreSQL remains the authority for catalogue and version metadata;
    Qdrant applies the allowlist before top-K ranking to avoid crowding by
    out-of-catalogue or out-of-scope summaries. Results are still validated
    against the shared catalogue after retrieval.
    """

    def __init__(
        self,
        summary_store: QdrantSummaryStore,
        similarity_threshold: float = 0.40,
        search_limit: int = 20,
        max_search_limit: int = 200,
        target_candidate_count: int = 20,
    ) -> None:
        if (
            isinstance(similarity_threshold, bool)
            or not isinstance(similarity_threshold, Real)
            or not math.isfinite(float(similarity_threshold))
            or not 0.0 <= float(similarity_threshold) <= 1.0
        ):
            raise ValueError(
                "similarity_threshold must be a finite number between 0 and 1."
            )

        if isinstance(search_limit, bool) or not isinstance(search_limit, int) or search_limit < 1:
            raise ValueError("search_limit must be a positive integer.")

        if isinstance(max_search_limit, bool) or not isinstance(max_search_limit, int) or max_search_limit < search_limit:
            raise ValueError("max_search_limit must be an integer at least search_limit.")
        if isinstance(target_candidate_count, bool) or not isinstance(target_candidate_count, int) or target_candidate_count < 1:
            raise ValueError("target_candidate_count must be a positive integer.")

        self.summary_store = summary_store
        self.similarity_threshold = similarity_threshold
        self.search_limit = search_limit
        self.max_search_limit = max_search_limit
        self.target_candidate_count = target_candidate_count

    async def generate_candidate(
        self,
        query: GeneratedQuery,
        document_catalogue: list[dict[str, Any]],
    ) -> QueryCandidateResult:
        """Generate candidates for the single contextualized query."""

        catalogue_by_version = self._build_catalogue_index(document_catalogue)
        if not catalogue_by_version:
            return self._empty_result(query)

        try:
            search_results = await self._search_until_enough_candidates(
                query=query,
                catalogue_by_version=catalogue_by_version,
            )
        except Exception as exc:
            logger.exception(
                "Summary similarity search failed for contextualized query: %s",
                query.query,
            )
            raise CandidateDocumentGenerationError(
                "Summary search failed for the contextualized query."
            ) from exc

        candidates = self._filter_and_build_candidates(
            search_results=search_results,
            query=query,
            catalogue_by_version=catalogue_by_version,
        )
        return QueryCandidateResult(
            query=query.query,
            version_scope=query.version_scope,
            target_versions=query.target_versions,
            candidates=candidates,
        )

    async def _search_until_enough_candidates(
        self,
        *,
        query: GeneratedQuery,
        catalogue_by_version: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Search only catalogue-listed, in-scope versions and page if needed.

        PostgreSQL remains the catalogue and lifecycle authority. The
        exact version allowlist is pushed into Qdrant before top-K truncation;
        returned hits are still checked against the catalogue afterward.
        """
        eligible_by_version: dict[str, CandidateDocument] = {}
        eligible_version_ids = [
            version_id
            for version_id, catalogue_entry in catalogue_by_version.items()
            if self._matches_version_scope(
                catalogue_entry=catalogue_entry,
                query=query,
            )
        ]

        # Push the eligible version/scope allowlist into Qdrant so
        # out-of-catalogue and out-of-scope summaries never consume top-K slots.
        if not eligible_version_ids:
            return []

        offset = 0

        while offset < self.max_search_limit:
            page_limit = min(
                self.search_limit,
                self.max_search_limit - offset,
            )
            page = await asyncio.to_thread(
                self.summary_store.search_similar_summaries,
                query.query,
                page_limit,
                offset=offset,
                document_version_ids=eligible_version_ids,
            )
            if not page:
                break

            eligible = self._filter_and_build_candidates(
                search_results=page,
                query=query,
                catalogue_by_version=catalogue_by_version,
            )
            for candidate in eligible:
                existing = eligible_by_version.get(candidate.document_version_id)
                if existing is None or candidate.similarity_score > existing.similarity_score:
                    eligible_by_version[candidate.document_version_id] = candidate

            # Qdrant returns descending similarity. Once the page's final
            # score is below threshold, later pages cannot qualify either.
            last_score = page[-1].get("similarity_score")
            if (
                isinstance(last_score, (int, float))
                and not isinstance(last_score, bool)
                and math.isfinite(float(last_score))
                and float(last_score) < self.similarity_threshold
            ):
                break

            offset += len(page)
            if len(eligible_by_version) >= self.target_candidate_count:
                break
            if len(page) < page_limit:
                break

        # Keep the highest-scoring eligible versions, bounded to the target.
        ranked = sorted(
            eligible_by_version.values(),
            key=lambda item: (-item.similarity_score, item.document_version_id),
        )
        return [
            {
                "document_id": item.document_id,
                "document_version_id": item.document_version_id,
                "similarity_score": item.similarity_score,
            }
            for item in ranked[: self.target_candidate_count]
        ]

    def _filter_and_build_candidates(
        self,
        search_results: list[dict[str, Any]],
        query: GeneratedQuery,
        catalogue_by_version: dict[str, dict[str, Any]],
    ) -> list[CandidateDocument]:
        """
        Filter Qdrant hits using authoritative catalogue information.

        Qdrant lifecycle status and title/summary are deliberately
        not used as authoritative metadata.
        """

        candidates_by_version: dict[str, CandidateDocument] = {}

        for hit in search_results:
            similarity_score = hit.get("similarity_score")

            if similarity_score is None:
                continue

            similarity_score = float(similarity_score)

            if not math.isfinite(similarity_score):
                continue

            if similarity_score < self.similarity_threshold:
                continue

            version_id = str(
                hit.get("document_version_id") or ""
            ).strip()

            document_id = str(
                hit.get("document_id") or ""
            ).strip()

            if not version_id or not document_id:
                continue

            # This lookup enforces the supplied catalogue because the
            # supplied catalogue contains only eligible versions.
            catalogue_entry = catalogue_by_version.get(version_id)

            if catalogue_entry is None:
                continue

            # Check both identifiers to avoid accepting inconsistent hits.
            if document_id != catalogue_entry["document_id"]:
                logger.warning(
                    "Ignoring Qdrant hit with mismatched document ID "
                    "for version %s.",
                    version_id,
                )
                continue

            if not self._matches_version_scope(
                catalogue_entry=catalogue_entry,
                query=query,
            ):
                continue

            candidate_data = {
                "document_id": catalogue_entry["document_id"],
                "document_version_id": version_id,
                "document_title": catalogue_entry["document_title"],
                "document_summary": catalogue_entry["document_summary"],
                "version_year": catalogue_entry["version_year"],
                "version_status": catalogue_entry["version_status"],
                "similarity_score": similarity_score,
            }

            # Source is optional because the current PostgreSQL
            # catalogue does not guarantee that field.
            source = catalogue_entry.get("source")
            if source is not None:
                candidate_data["source"] = source

            candidate = CandidateDocument(**candidate_data)

            # Defensive deduplication within this query only.
            # Cross-query duplicates are intentionally preserved.
            existing = candidates_by_version.get(version_id)

            if (
                existing is None
                or candidate.similarity_score > existing.similarity_score
            ):
                candidates_by_version[version_id] = candidate

        return list(candidates_by_version.values())

    @staticmethod
    def _matches_version_scope(
        catalogue_entry: dict[str, Any],
        query: GeneratedQuery,
    ) -> bool:
        """Apply lifecycle scope and optional target-year filtering."""

        version_status = catalogue_entry["version_status"]

        if (
            query.version_scope == QueryVersionScope.ACTIVE
            and version_status != "active"
        ):
            return False

        if (
            query.version_scope == QueryVersionScope.SUPERSEDED
            and version_status != "superseded"
        ):
            return False

        if (
            query.target_versions
            and catalogue_entry["version_year"] not in query.target_versions
        ):
            return False

        return True

    @staticmethod
    def _build_catalogue_index(
        document_catalogue: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """
        Validate and index the shared catalogue by version ID.

        The catalogue is the authority for version lifecycle and
        candidate metadata.
        """

        if not isinstance(document_catalogue, list):
            raise ValueError("document_catalogue must be a list.")

        catalogue_by_version: dict[str, dict[str, Any]] = {}

        required_fields = {
            "document_id",
            "document_version_id",
            "document_title",
            "document_summary",
            "version_year",
            "version_status",
        }

        for entry in document_catalogue:
            if not isinstance(entry, dict):
                raise ValueError(
                    "Every catalogue entry must be a dictionary."
                )

            if not required_fields.issubset(entry):
                missing = required_fields - entry.keys()
                raise ValueError(
                    f"Catalogue entry is missing fields: {sorted(missing)}"
                )

            document_id = str(entry["document_id"]).strip()
            version_id = str(entry["document_version_id"]).strip()

            if not document_id or not version_id:
                raise ValueError(
                    "Catalogue document IDs cannot be empty."
                )

            if version_id in catalogue_by_version:
                raise ValueError(
                    f"Duplicate document_version_id in catalogue: {version_id}"
                )

            version_status = entry["version_status"]

            if version_status not in {"active", "superseded"}:
                raise ValueError(
                    f"Unsupported catalogue version status: {version_status}"
                )

            version_year = entry["version_year"]

            if (
                isinstance(version_year, bool)
                or not isinstance(version_year, int)
                or version_year < 1
            ):
                raise ValueError(
                    f"Invalid version_year for version {version_id}."
                )

            title = entry["document_title"]
            summary = entry["document_summary"]

            if not isinstance(title, str) or not title.strip():
                raise ValueError(
                    f"Invalid document_title for version {version_id}."
                )

            if not isinstance(summary, str):
                raise ValueError(
                    f"Invalid document_summary for version {version_id}."
                )

            catalogue_by_version[version_id] = {
                "document_id": document_id,
                "document_version_id": version_id,
                "document_title": title,
                "document_summary": summary,
                "version_year": version_year,
                "version_status": version_status,
                **(
                    {"source": entry["source"]}
                    if entry.get("source") is not None
                    else {}
                ),
            }

        return catalogue_by_version

    @staticmethod
    def _empty_result(
        query: GeneratedQuery,
    ) -> QueryCandidateResult:
        """Return an empty successful result when no eligible versions are present."""

        return QueryCandidateResult(
            query=query.query,
            version_scope=query.version_scope,
            target_versions=query.target_versions,
            candidates=[],
        )