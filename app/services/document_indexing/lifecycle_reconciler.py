from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.documents import Document
from app.models.document_versions import DocumentVersion


@dataclass(frozen=True, slots=True)
class LifecycleVersionSnapshot:
    document_id: UUID
    document_version_id: UUID
    version_number: int
    title: str
    summary: str | None
    status: str


@dataclass(frozen=True, slots=True)
class LifecycleReconciliationResult:
    dry_run: bool
    planned_versions: int
    chunk_versions_synced: int
    summary_versions_synced: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


class QdrantLifecycleReconciler:
    """Rebuild Qdrant lifecycle projections from authoritative PostgreSQL state.

    This is intentionally replayable and idempotent. It does not persist a
    second lifecycle source of truth: every run reads current PostgreSQL
    statuses and reapplies them to existing chunk payloads and summary points.
    """

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        chunk_store,
        summary_store,
    ) -> None:
        self.session_factory = session_factory
        self.chunk_store = chunk_store
        self.summary_store = summary_store

    def _load_snapshots(self) -> list[LifecycleVersionSnapshot]:
        statement = (
            select(
                Document.id,
                DocumentVersion.id,
                DocumentVersion.version_number,
                Document.title,
                DocumentVersion.summary,
                DocumentVersion.status,
            )
            .join(
                DocumentVersion,
                DocumentVersion.document_id == Document.id,
            )
            .where(
                DocumentVersion.status.in_(("active", "superseded"))
            )
            .order_by(
                Document.id,
                DocumentVersion.version_number,
            )
        )

        with self.session_factory() as db:
            rows = db.execute(statement).all()

        return [
            LifecycleVersionSnapshot(
                document_id=row[0],
                document_version_id=row[2],
                version_number=row[3],
                title=row[4],
                summary=row[5],
                status=row[6],
            )
            for row in rows
        ]

    def reconcile(self, *, dry_run: bool = True) -> LifecycleReconciliationResult:
        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be a boolean.")

        snapshots = self._load_snapshots()
        if dry_run:
            return LifecycleReconciliationResult(
                dry_run=True,
                planned_versions=len(snapshots),
                chunk_versions_synced=0,
                summary_versions_synced=0,
                errors=(),
            )

        chunk_versions_synced = 0
        summary_versions_synced = 0
        errors: list[str] = []

        for item in snapshots:
            version_id = str(item.document_version_id)

            try:
                self.chunk_store.update_version_status(
                    document_version_id=version_id,
                    status=item.status,
                )
                chunk_versions_synced += 1
            except Exception as exc:
                errors.append(
                    f"chunk lifecycle sync failed for {version_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

            if not isinstance(item.summary, str) or not item.summary.strip():
                errors.append(
                    f"summary sync skipped for {version_id}: "
                    "PostgreSQL summary is empty."
                )
                continue

            try:
                self.summary_store.upsert_summary(
                    document_id=item.document_id,
                    document_version_id=item.document_version_id,
                    version_number=item.version_number,
                    title=item.title,
                    summary=item.summary,
                    status=item.status,
                )
                summary_versions_synced += 1
            except Exception as exc:
                errors.append(
                    f"summary lifecycle sync failed for {version_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

        return LifecycleReconciliationResult(
            dry_run=False,
            planned_versions=len(snapshots),
            chunk_versions_synced=chunk_versions_synced,
            summary_versions_synced=summary_versions_synced,
            errors=tuple(errors),
        )
