from datetime import date
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models.documents import Document
from app.models.document_versions import DocumentVersion


class DocumentVersionRepository:
    """Persistence operations for versions in the shared document corpus."""

    VALID_STATUSES = {"pending", "active", "superseded"}

    def __init__(self, db: Session) -> None:
        self.db = db

    @classmethod
    def _validate_status(cls, status: str) -> None:
        if status not in cls.VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Allowed statuses: "
                f"{sorted(cls.VALID_STATUSES)}"
            )

    def create(
        self,
        document_id: UUID,
        version_number: int,
        original_filename: str,
        storage_key: str,
        version_label: str | None = None,
        file_hash: str | None = None,
        content_hash: str | None = None,
        summary: str | None = None,
        status: str = "pending",
        effective_from: date | None = None,
        effective_until: date | None = None,
        supersedes_version_id: UUID | None = None,
        page_count: int | None = None,
        version_id: UUID | None = None,
    ) -> DocumentVersion:
        self._validate_status(status)
        if self.db.scalar(select(Document.id).where(Document.id == document_id)) is None:
            raise ValueError(f"Document not found: {document_id}")

        if supersedes_version_id is not None:
            previous = self.db.scalar(
                select(DocumentVersion).where(
                    DocumentVersion.id == supersedes_version_id,
                    DocumentVersion.document_id == document_id,
                )
            )
            if previous is None:
                raise ValueError(
                    "supersedes_version_id must belong to the same document."
                )

        version = DocumentVersion(
            id=version_id,
            document_id=document_id,
            version_number=version_number,
            version_label=version_label,
            original_filename=original_filename,
            storage_key=storage_key,
            file_hash=file_hash,
            content_hash=content_hash,
            summary=summary,
            status=status,
            effective_from=effective_from,
            effective_until=effective_until,
            supersedes_version_id=supersedes_version_id,
            page_count=page_count,
        )
        self.db.add(version)
        self.db.flush()
        self.db.refresh(version)
        return version

    def get_by_id(self, version_id: UUID) -> DocumentVersion | None:
        return self.db.scalar(
            select(DocumentVersion).where(DocumentVersion.id == version_id)
        )

    def get_by_document(self, document_id: UUID) -> list[DocumentVersion]:
        statement = (
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_number.desc())
        )
        return list(self.db.scalars(statement).all())

    def get_active_version(self, document_id: UUID) -> DocumentVersion | None:
        statement = (
            select(DocumentVersion)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.status == "active",
                Document.archived.is_(False),
            )
        )
        return self.db.scalar(statement)

    def get_by_hash(
        self,
        file_hash: str | None = None,
        content_hash: str | None = None,
    ) -> list[DocumentVersion]:
        if not file_hash and not content_hash:
            raise ValueError("Provide at least one hash to search.")
        conditions = []
        if file_hash:
            conditions.append(DocumentVersion.file_hash == file_hash)
        if content_hash:
            conditions.append(DocumentVersion.content_hash == content_hash)
        statement = (
            select(DocumentVersion)
            .where(or_(*conditions))
            .order_by(DocumentVersion.created_at.desc())
        )
        return list(self.db.scalars(statement).all())

    def get_shared_retrieval_catalog(self) -> list[dict[str, object]]:
        """Return the shared corpus catalogue of non-archived active/history versions."""
        statement = (
            select(Document, DocumentVersion)
            .join(DocumentVersion, DocumentVersion.document_id == Document.id)
            .where(
                Document.archived.is_(False),
                DocumentVersion.status.in_(("active", "superseded")),
            )
            .order_by(Document.id, DocumentVersion.version_number.desc())
        )

        catalogue: list[dict[str, object]] = []
        for document, version in self.db.execute(statement):
            effective_year = (
                version.effective_from.year
                if version.effective_from is not None
                else version.created_at.year
            )
            catalogue.append(
                {
                    "document_id": str(document.id),
                    "document_version_id": str(version.id),
                    "document_title": document.title,
                    "document_summary": version.summary or "",
                    "version_year": effective_year,
                    "version_status": version.status,
                    "last_updated": version.created_at.isoformat(),
                }
            )
        return catalogue

    def get_retrieval_eligible_versions(
        self,
        document_id: UUID,
        include_historical: bool = True,
    ) -> list[DocumentVersion]:
        statement = (
            select(DocumentVersion)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                DocumentVersion.document_id == document_id,
                Document.archived.is_(False),
            )
        )
        statuses = ["active", "superseded"] if include_historical else ["active"]
        statement = statement.where(DocumentVersion.status.in_(statuses))
        statement = statement.order_by(DocumentVersion.version_number.desc())
        return list(self.db.scalars(statement).all())

    def update(self, version_id: UUID, **kwargs) -> DocumentVersion | None:
        version = self.get_by_id(version_id)
        if version is None:
            return None
        allowed_fields = {
            "version_label", "file_hash", "content_hash", "summary",
            "effective_from", "effective_until", "supersedes_version_id",
            "page_count", "original_filename",
        }
        invalid_fields = set(kwargs) - allowed_fields
        if invalid_fields:
            raise ValueError(f"Invalid version fields: {invalid_fields}")
        if (
            "supersedes_version_id" in kwargs
            and kwargs["supersedes_version_id"] is not None
        ):
            previous = self.db.scalar(
                select(DocumentVersion).where(
                    DocumentVersion.id == kwargs["supersedes_version_id"],
                    DocumentVersion.document_id == version.document_id,
                )
            )
            if previous is None:
                raise ValueError(
                    "supersedes_version_id must belong to the same document."
                )
        for field, value in kwargs.items():
            setattr(version, field, value)
        self.db.flush()
        self.db.refresh(version)
        return version

    def change_status(
        self,
        version_id: UUID,
        status: str,
    ) -> DocumentVersion | None:
        self._validate_status(status)
        version = self.get_by_id(version_id)
        if version is None:
            return None
        if status == "active":
            raise ValueError("Use activate_pending_version() to activate a version.")
        version.status = status
        self.db.flush()
        self.db.refresh(version)
        return version

    def activate_pending_version(
        self,
        version_id: UUID,
    ) -> tuple[DocumentVersion, DocumentVersion | None] | None:
        target = self.db.scalar(
            select(DocumentVersion).where(DocumentVersion.id == version_id)
        )
        if target is None:
            return None

        document = self.db.scalar(
            select(Document)
            .where(Document.id == target.document_id)
            .with_for_update()
        )
        if document is None:
            raise ValueError(f"Parent document not found: {target.document_id}")

        target = self.db.scalar(
            select(DocumentVersion)
            .where(
                DocumentVersion.id == version_id,
                DocumentVersion.document_id == document.id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if target is None:
            return None
        if document.archived:
            raise ValueError("An archived document cannot be activated.")
        if target.status != "pending":
            raise ValueError(
                f"Only pending versions can be activated. Current status: {target.status}"
            )

        previous = self.db.scalar(
            select(DocumentVersion)
            .where(
                DocumentVersion.document_id == document.id,
                DocumentVersion.status == "active",
                DocumentVersion.id != target.id,
            )
            .with_for_update()
        )
        if previous is not None:
            previous.status = "superseded"
            self.db.flush()
        target.status = "active"
        self.db.flush()
        return target, previous

    def delete(self, version_id: UUID) -> bool:
        version = self.get_by_id(version_id)
        if version is None:
            return False
        self.db.delete(version)
        self.db.flush()
        return True
