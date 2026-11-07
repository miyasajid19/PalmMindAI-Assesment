from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from app.repositories.document_version_repositories import (
    DocumentVersionRepository,
)


MatchType = Literal["file_hash", "content_hash"]


@dataclass(frozen=True, slots=True)
class HashMatchCandidate:
    document_id: UUID
    version_id: UUID
    version_number: int
    status: str
    original_filename: str
    matched_by: MatchType


@dataclass(frozen=True, slots=True)
class HashMatchResult:
    is_duplicate: bool
    candidates: list[HashMatchCandidate]

    @property
    def match_count(self) -> int:
        return len(self.candidates)


class DocumentHashMatcher:
    """
    Detect exact document duplicates using SHA-256 hashes.

    Matching priority:
        1. Original PDF file hash.
        2. Normalized extracted content hash.

    This class performs read-only matching.
    It does not register or modify documents.
    """

    def __init__(
        self,
        repository: DocumentVersionRepository,
    ) -> None:
        self.repository = repository

    def match(
        self,
        file_hash: str | None,
        content_hash: str | None,
    ) -> HashMatchResult:

        if not file_hash and not content_hash:
            raise ValueError(
                "At least one hash must be provided."
            )

        versions = self.repository.get_by_hash(
            file_hash=file_hash,
            content_hash=content_hash,
        )

        candidates_by_id: dict[UUID, HashMatchCandidate] = {}

        for version in versions:

            # Original PDF hash has higher priority.
            if file_hash and version.file_hash == file_hash:
                matched_by: MatchType = "file_hash"

            elif (
                content_hash
                and version.content_hash == content_hash
            ):
                matched_by = "content_hash"

            else:
                # Defensive check in case repository behavior changes.
                continue

            candidate = HashMatchCandidate(
                document_id=version.document_id,
                version_id=version.id,
                version_number=version.version_number,
                status=version.status,
                original_filename=version.original_filename,
                matched_by=matched_by,
            )

            candidates_by_id[version.id] = candidate

        candidates = list(candidates_by_id.values())

        return HashMatchResult(
            is_duplicate=bool(candidates),
            candidates=candidates,
        )