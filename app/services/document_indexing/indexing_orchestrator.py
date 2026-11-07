from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db_session
from app.models.documents import Document
from app.models.document_versions import DocumentVersion
from app.repositories import DocumentVersionRepository
from app.rag.pdf_reader.loader import (
    DocumentLoader,
    pdf_loader,
    text_loader as default_text_loader,
)
from app.rag.ingestion.hash_service import PDFHashService
from app.rag.vector_stores.qdrant_summary import summary_store
from app.services.document_indexing.indexing_service import (
    DocumentIndexingService,
    IndexingResult,
)
from app.storage.file_storage import FileStorage


@dataclass(frozen=True, slots=True)
class IndexingOrchestrationResult:
    document_id: str
    document_version_id: str
    version_number: int
    page_count: int
    ocr_page_count: int
    chunk_count: int
    qdrant_point_ids: list[str]
    summary_point_id: str
    status: str
    summary_sync_pending: bool = False
    lifecycle_sync_pending: bool = False
    sync_errors: list[str] | None = None
    chunking_strategy: str = "recursive"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "version_number": self.version_number,
            "page_count": self.page_count,
            "ocr_page_count": self.ocr_page_count,
            "chunk_count": self.chunk_count,
            "qdrant_point_ids": self.qdrant_point_ids,
            "summary_point_id": self.summary_point_id,
            "status": self.status,
            "summary_sync_pending": self.summary_sync_pending,
            "lifecycle_sync_pending": self.lifecycle_sync_pending,
            "sync_errors": self.sync_errors or [],
            "chunking_strategy": self.chunking_strategy,
        }


class DocumentIndexingOrchestrationError(RuntimeError):
    """Raised when document indexing orchestration fails."""

    def __init__(
        self,
        message: str,
        cleanup_errors: list[str] | None = None,
    ) -> None:
        self.cleanup_errors = cleanup_errors or []

        if self.cleanup_errors:
            message += (
                "\nCleanup encountered errors:\n- "
                + "\n- ".join(self.cleanup_errors)
            )

        super().__init__(message)


class DocumentIndexingOrchestrator:
    """
    Coordinates indexing and activation of an existing pending version.

    This service does not:
      - Register a document or version.
      - Upload or delete the original document.
      - Create FastAPI endpoints.
      - Perform document duplicate detection.

    PostgreSQL is the source of truth for version lifecycle.
    """

    def __init__(
        self,
        *,
        session_factory=None,
        file_storage=None,
        loader=None,
        text_loader=None,
        document_loader=None,
        indexing_service=None,
        summary_storage=None,
    ) -> None:

        self.session_factory = session_factory or get_db_session
        self.file_storage = file_storage or FileStorage()
        self.loader = loader or pdf_loader
        self.text_loader = text_loader or default_text_loader
        self.document_loader = document_loader or DocumentLoader(
            pdf_loader=self.loader,
            text_loader=self.text_loader,
        )
        self.indexing_service = (
            indexing_service or DocumentIndexingService()
        )
        self.summary_storage = summary_storage or summary_store

    
    @staticmethod
    def _load_version(
        db: Session,
        version_id: UUID,
    ) -> tuple[DocumentVersion, Document]:
        version = db.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.id == version_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if version is None:
            raise ValueError(f"Document version not found: {version_id}")

        document = db.scalar(
            select(Document)
            .where(Document.id == version.document_id)
            .with_for_update()
        )
        if document is None:
            raise ValueError(f"Parent document not found: {version.document_id}")

        if document.archived:
            raise ValueError("An archived document cannot be indexed.")
        if version.status != "pending":
            raise ValueError(
                "Only pending versions can be indexed. "
                f"Current status: {version.status}"
            )
        if not version.storage_key:
            raise ValueError("Document version has no storage key.")
        if (
            not isinstance(version.file_hash, str)
            or len(version.file_hash.strip()) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in version.file_hash.strip()
            )
        ):
            raise ValueError(
                "Document version has no valid registered file SHA-256 hash."
            )
        if not version.summary or not version.summary.strip():
            raise ValueError("Document version has no usable summary.")
        if not document.title or not document.title.strip():
            raise ValueError("Parent document has no usable title.")
        return version, document

    def _sync_chunk_status(
        self,
        document_version_id: str,
        status: str,
    ) -> list[str]:
        """Synchronize one version's lifecycle status across chunk stores."""

        errors: list[str] = []

        try:
            self.indexing_service.qdrant_store.update_version_status(
                document_version_id=document_version_id,
                status=status,
            )
        except Exception as exc:
            errors.append(
                f"Qdrant status update failed for version "
                f"{document_version_id}: {type(exc).__name__}: {exc}"
            )

        return errors
    
    def _cleanup(
        self,
        indexing_result: IndexingResult | None,
        version_id: UUID,
        summary_upsert_attempted: bool,
    ) -> list[str]:
        """
        Compensate for indexing operations after a failed activation.

        The original PDF in B2 is intentionally never deleted.
        """

        errors: list[str] = []

        if indexing_result is not None:

            try:
                self.indexing_service.qdrant_store.delete_points(
                    indexing_result.qdrant_point_ids
                )
            except Exception as exc:
                errors.append(
                    f"Qdrant cleanup failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        if summary_upsert_attempted:
            try:
                self.summary_storage.delete_summary(version_id)
            except Exception as exc:
                errors.append(
                    f"Summary cleanup failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        return errors

    def index_pending_version(
        self,
        version_id: UUID,
        chunking_strategy: str = "recursive",
    ) -> IndexingOrchestrationResult:

        if not isinstance(chunking_strategy, str):
            raise ValueError("chunking_strategy must be 'recursive' or 'semantic'.")
        chunking_strategy = chunking_strategy.strip().lower()
        if chunking_strategy not in {"recursive", "semantic"}:
            raise ValueError(
                "Unsupported chunking_strategy. Choose 'recursive' or 'semantic'."
            )

        indexing_result: IndexingResult | None = None
        summary_upsert_attempted = False
        committed = False
        commit_attempted = False

        with self.session_factory() as db:

            try:
                # ------------------------------------------
                # 1. Load and validate the pending version.
                # ------------------------------------------

                version, document = self._load_version(
                    db,
                    version_id,
                )

                document_id = version.document_id
                version_number = version.version_number
                storage_key = version.storage_key
                original_filename = Path(
                    version.original_filename
                ).name
                file_extension = Path(original_filename).suffix.lower()
                if file_extension not in {".pdf", ".txt"}:
                    raise ValueError(
                        "Unsupported document type for indexing. "
                        "Only PDF and TXT files are supported."
                    )

                # ------------------------------------------
                # 2. Download the original document from B2.
                # ------------------------------------------

                with tempfile.TemporaryDirectory(
                    prefix="voice_ai_indexing_"
                ) as temporary_directory:

                    local_document = (
                        Path(temporary_directory)
                        / original_filename
                    )

                    self.file_storage.download_file(
                        storage_key=storage_key,
                        destination=local_document,
                    )

                    if (
                        not local_document.exists()
                        or local_document.stat().st_size == 0
                    ):
                        raise IOError(
                            "B2 download produced an empty or missing document."
                        )

                    if not version.file_hash or not version.file_hash.strip():
                        raise ValueError(
                            "Document version has no registered file SHA-256 hash."
                        )

                    downloaded_file_hash = PDFHashService.calculate_file_hash(
                        local_document
                    )
                    if downloaded_file_hash.lower() != version.file_hash.strip().lower():
                        raise ValueError(
                            "Downloaded document integrity verification failed: "
                            "SHA-256 does not match the registered file hash."
                        )

                    # --------------------------------------
                    # 3. Extract document content.
                    # --------------------------------------

                    page_documents = self.document_loader.load_document(
                        local_document
                    )

                    if not page_documents:
                        raise ValueError(
                            "Document extraction returned no content."
                        )

                    # --------------------------------------
                    # 4. Index all pages in Qdrant.
                    # --------------------------------------

                    indexing_result = (
                        self.indexing_service.index_document_version(
                            page_documents=page_documents,
                            document_id=document_id,
                            document_version_id=version_id,
                            version_number=version_number,
                            source_filename=original_filename,
                            chunking_strategy=chunking_strategy,
                        )
                    )

                # The temporary PDF is now removed.
                # The original B2 object remains untouched.

                # ------------------------------------------
                # 5. Upsert the summary as pending.
                # ------------------------------------------

                summary_upsert_attempted = True
                summary_point_id = (
                    self.summary_storage.upsert_summary(
                        document_id=document_id,
                        document_version_id=version_id,
                        version_number=version_number,
                        title=document.title,
                        summary=version.summary,
                        status="pending",
                    )
                )

                # ------------------------------------------
                # 6. Update actual extracted page count.
                # ------------------------------------------

                version.page_count = indexing_result.page_count
                db.flush()

                # ------------------------------------------
                # 7. Atomically activate the version.
                # ------------------------------------------

                repository = DocumentVersionRepository(db)

                activation = repository.activate_pending_version(
                    version_id,
                )

                if activation is None:
                    raise ValueError(
                        f"Version disappeared during activation: {version_id}"
                    )

                activated_version, previous_version = activation

                previous_version_id = (
                    str(previous_version.id)
                    if previous_version is not None
                    else None
                )

                # Capture values before commit/session close.
                activated_document_id = str(activated_version.document_id)
                activated_version_id = str(activated_version.id)
                activated_version_number = activated_version.version_number
                activated_summary = activated_version.summary
                document_title = document.title

                # ------------------------------------------
                # 8. Commit PostgreSQL lifecycle changes.
                # ------------------------------------------

                commit_attempted = True
                db.commit()
                committed = True

            except Exception as exc:

                if not committed:
                    if commit_attempted:
                        rollback_errors = []
                        try:
                            db.rollback()
                        except Exception as rollback_exc:
                            rollback_errors.append(
                                "PostgreSQL rollback after uncertain "
                                "commit failed: "
                                f"{type(rollback_exc).__name__}: "
                                f"{rollback_exc}"
                            )
                        raise DocumentIndexingOrchestrationError(
                            "PostgreSQL commit outcome is uncertain. "
                            "External indexes were preserved to avoid "
                            "deleting data belonging to a possibly "
                            "committed version.",
                            cleanup_errors=rollback_errors,
                        ) from exc

                    # Clean external stores while PostgreSQL locks
                    # are still held.
                    cleanup_errors = self._cleanup(
                        indexing_result=indexing_result,
                        version_id=version_id,
                        summary_upsert_attempted=summary_upsert_attempted,
                    )

                    # Release locks only after compensation finishes.
                    try:
                        db.rollback()
                    except Exception as rollback_exc:
                        cleanup_errors.append(
                            "PostgreSQL rollback after cleanup failed: "
                            f"{type(rollback_exc).__name__}: "
                            f"{rollback_exc}"
                        )

                    raise DocumentIndexingOrchestrationError(
                        "Document indexing orchestration failed. "
                        "The PostgreSQL transaction was rolled back.",
                        cleanup_errors=cleanup_errors,
                    ) from exc

                raise

                # ----------------------------------------------
        # 9. Synchronize chunk lifecycle after commit.
        # ----------------------------------------------

        sync_errors: list[str] = []

        # Newly activated version.
        sync_errors.extend(
            self._sync_chunk_status(
                document_version_id=str(version_id),
                status="active",
            )
        )

        # Previously active version, if one existed.
        if previous_version_id is not None:
            sync_errors.extend(
                self._sync_chunk_status(
                    document_version_id=previous_version_id,
                    status="superseded",
                )
            )

        # ----------------------------------------------
        # 10. Synchronize summary lifecycle after commit.
        # ----------------------------------------------

        summary_sync_pending = False

        try:
            self.summary_storage.upsert_summary(
                document_id=activated_document_id,
                document_version_id=activated_version_id,
                version_number=activated_version_number,
                title=document_title,
                summary=activated_summary,
                status="active",
            )

        except Exception as exc:
            summary_sync_pending = True
            sync_errors.append(
                f"Summary status update failed for version "
                f"{version_id}: {type(exc).__name__}: {exc}"
            )

        lifecycle_sync_pending = any(
            error.startswith("Qdrant status update")
            for error in sync_errors
        )

        return IndexingOrchestrationResult(
            document_id=str(document_id),
            document_version_id=str(version_id),
            version_number=version_number,
            page_count=indexing_result.page_count,
            ocr_page_count=indexing_result.ocr_page_count,
            chunk_count=indexing_result.chunk_count,
            qdrant_point_ids=indexing_result.qdrant_point_ids,
            summary_point_id=str(summary_point_id),
            status="active",
            summary_sync_pending=summary_sync_pending,
            lifecycle_sync_pending=lifecycle_sync_pending,
            sync_errors=sync_errors,
            chunking_strategy=chunking_strategy,
        )