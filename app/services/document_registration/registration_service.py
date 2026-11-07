from dataclasses import dataclass
import re
from datetime import date
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.documents import Document
from app.models.document_versions import DocumentVersion
from app.repositories import DocumentRepository, DocumentVersionRepository


@dataclass(frozen=True, slots=True)
class DocumentRegistrationData:
    title: str
    document_type: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class VersionRegistrationData:
    original_filename: str
    storage_key: str
    file_hash: str | None = None
    content_hash: str | None = None
    summary: str | None = None
    version_label: str | None = None
    effective_from: date | None = None
    effective_until: date | None = None
    page_count: int | None = None


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    document_id: UUID
    document_version_id: UUID
    version_number: int
    status: str
    archived: bool


class DocumentRegistrationService:
    """Register logical documents and version rows in the caller transaction."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.document_repository = DocumentRepository(db)
        self.version_repository = DocumentVersionRepository(db)

    @staticmethod
    def _validate_document_data(data: DocumentRegistrationData) -> None:
        if not isinstance(data, DocumentRegistrationData):
            raise TypeError("data must be a DocumentRegistrationData instance.")
        if not data.title or not data.title.strip():
            raise ValueError("Document title cannot be empty.")
        if not data.document_type or not data.document_type.strip():
            raise ValueError("Document type cannot be empty.")

    @staticmethod
    def _validate_version_data(data: VersionRegistrationData) -> None:
        if not isinstance(data, VersionRegistrationData):
            raise TypeError("data must be a VersionRegistrationData instance.")
        if not data.original_filename or not data.original_filename.strip():
            raise ValueError("Original filename cannot be empty.")
        if not data.storage_key or not data.storage_key.strip():
            raise ValueError("Storage key cannot be empty.")
        if data.page_count is not None and (
            not isinstance(data.page_count, int)
            or isinstance(data.page_count, bool)
            or data.page_count <= 0
        ):
            raise ValueError("Page count must be a positive integer.")
        if (
            data.effective_from is not None
            and data.effective_until is not None
            and data.effective_until < data.effective_from
        ):
            raise ValueError("effective_until cannot be earlier than effective_from.")
        if not isinstance(data.summary, str) or not data.summary.strip():
            raise ValueError("Document summary cannot be empty.")
        if not isinstance(data.file_hash, str) or re.fullmatch(
            r"[0-9a-fA-F]{64}", data.file_hash.strip()
        ) is None:
            raise ValueError("File SHA-256 hash must contain exactly 64 hexadecimal characters.")

    def _create_version(
        self,
        document_id: UUID,
        version_number: int,
        data: VersionRegistrationData,
        supersedes_version_id: UUID | None = None,
        version_id: UUID | None = None,
    ) -> DocumentVersion:
        return self.version_repository.create(
            document_id=document_id,
            version_number=version_number,
            original_filename=data.original_filename,
            storage_key=data.storage_key,
            version_label=data.version_label,
            file_hash=data.file_hash,
            content_hash=data.content_hash,
            summary=data.summary,
            status="pending",
            effective_from=data.effective_from,
            effective_until=data.effective_until,
            supersedes_version_id=supersedes_version_id,
            page_count=data.page_count,
            version_id=version_id,
        )

    @staticmethod
    def _build_result(document: Document, version: DocumentVersion) -> RegistrationResult:
        return RegistrationResult(
            document_id=document.id,
            document_version_id=version.id,
            version_number=version.version_number,
            status=version.status,
            archived=document.archived,
        )

    def register_new_document(
        self,
        document_data: DocumentRegistrationData,
        version_data: VersionRegistrationData,
        *,
        document_id: UUID | None = None,
        version_id: UUID | None = None,
    ) -> RegistrationResult:
        self._validate_document_data(document_data)
        self._validate_version_data(version_data)
        with self.db.begin_nested():
            document = self.document_repository.create(
                title=document_data.title.strip(),
                document_type=document_data.document_type.strip(),
                description=document_data.description,
                document_id=document_id,
            )
            version = self._create_version(
                document_id=document.id,
                version_number=1,
                data=version_data,
                version_id=version_id,
            )
            result = self._build_result(document, version)
        return result

    def register_new_version(
        self,
        document_id: UUID,
        version_data: VersionRegistrationData,
        *,
        version_id: UUID | None = None,
    ) -> RegistrationResult:
        if not isinstance(document_id, UUID):
            raise TypeError("document_id must be a UUID.")
        self._validate_version_data(version_data)
        with self.db.begin_nested():
            document = self.db.scalar(
                select(Document).where(Document.id == document_id).with_for_update()
            )
            if document is None:
                raise ValueError(f"Document not found: {document_id}")
            if document.archived:
                raise ValueError("Cannot register a new version for an archived document.")

            existing_versions = self.version_repository.get_by_document(document_id)
            next_version_number = (
                max(version.version_number for version in existing_versions) + 1
                if existing_versions
                else 1
            )
            active_version = self.version_repository.get_active_version(document_id)
            version = self._create_version(
                document_id=document_id,
                version_number=next_version_number,
                data=version_data,
                supersedes_version_id=active_version.id if active_version else None,
                version_id=version_id,
            )
            result = self._build_result(document, version)
        return result
