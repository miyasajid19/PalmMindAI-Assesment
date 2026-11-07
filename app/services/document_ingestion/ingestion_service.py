
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable
import mimetypes
from uuid import uuid4

from langchain_core.documents import Document
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.database import get_db_session

from app.rag.pdf_reader.validator import PDFValidator, TextValidator
from app.rag.pdf_reader.loader import (
    DocumentLoader,
    pdf_loader,
    text_loader as default_text_loader,
)
from app.rag.ingestion.hash_service import PDFHashService
from app.rag.ingestion.text_normalizer import TextNormalizer
from app.rag.ingestion.pdf_sampler import PDFSampler
from app.agents.Document_summarizer.agent import (
    DocumentSummarizerAgent,
    DocumentSummarizerOutput,
)
from app.repositories import DocumentVersionRepository
from app.services.document_registration.registration_service import (
    DocumentRegistrationData,
    DocumentRegistrationService,
    VersionRegistrationData,
    RegistrationResult,
)
from app.services.document_matching.hash_matcher import (
    DocumentHashMatcher,
)
from app.services.document_matching.summary_matcher import (
    DocumentSummaryMatcher,
)
from app.services.document_matching.matching_orchestrator import (
    DocumentMatchingOrchestrator,
    DocumentMatchingResult,
)
from app.rag.vector_stores.qdrant_summary import summary_store
from app.storage.file_storage import FileStorage


@dataclass(frozen=True, slots=True)
class IngestionResult:
    status: str
    file_path: str
    original_filename: str
    file_hash: str
    content_hash: str
    page_count: int
    summary_output: DocumentSummarizerOutput
    matching_result: DocumentMatchingResult
    registration_result: RegistrationResult | None = None
    storage_key: str | None = None
    validation_warnings: tuple[str, ...] = ()


class DocumentIngestionError(RuntimeError):
    """Raised when ingestion fails."""

    def __init__(
        self,
        message: str,
        cleanup_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.cleanup_error = cleanup_error


class DocumentIngestionService:
    """
    Coordinates PDF validation, extraction, normalization,
    summarization, duplicate detection, B2 upload and registration.

    This initial implementation ends at pending registration.
    It does not index chunks or activate the document version.
    """

    def __init__(
        self,
        *,
        validator: Any | None = None,
        text_validator: Any | None = None,
        loader: Any | None = None,
        text_loader: Any | None = None,
        document_loader: Any | None = None,
        hash_service: Any | None = None,
        normalizer: Any | None = None,
        sampler: Any | None = None,
        summarizer: Any | None = None,
        file_storage: Any | None = None,
        session_factory: Callable | None = None,
        matching_orchestrator_factory: Callable[
            [Session], DocumentMatchingOrchestrator
        ] | None = None,
        registration_service_factory: Callable[
            [Session], DocumentRegistrationService
        ] | None = None,
    ) -> None:

        self.validator = validator or PDFValidator()
        self.text_validator = text_validator or TextValidator()
        self.loader = loader or pdf_loader
        self.text_loader = text_loader or default_text_loader
        self.document_loader = document_loader or DocumentLoader(
            pdf_loader=self.loader,
            text_loader=self.text_loader,
        )
        self.hash_service = hash_service or PDFHashService()
        self.normalizer = normalizer or TextNormalizer()
        self.sampler = sampler or PDFSampler(max_pages=5)
        self.summarizer = summarizer or DocumentSummarizerAgent()
        self.file_storage = file_storage or FileStorage()

        self.session_factory = session_factory or get_db_session

        self.matching_orchestrator_factory = (
            matching_orchestrator_factory
            or self._default_matching_orchestrator
        )

        self.registration_service_factory = (
            registration_service_factory
            or DocumentRegistrationService
        )

    @staticmethod
    def _default_matching_orchestrator(
        db: Session,
    ) -> DocumentMatchingOrchestrator:

        version_repository = DocumentVersionRepository(db)

        hash_matcher = DocumentHashMatcher(
            repository=version_repository
        )

        summary_matcher = DocumentSummaryMatcher(
            summary_store=summary_store,
            threshold=0.80,
            candidate_limit=20,
        )

        return DocumentMatchingOrchestrator(
            hash_matcher=hash_matcher,
            summary_matcher=summary_matcher,
        )

    @staticmethod
    def _duplicate_lock_keys(
        file_hash: str | None,
        content_hash: str | None,
    ) -> list[int]:
        """Return deterministic PostgreSQL advisory-lock keys for input hashes."""
        lock_keys = set()
        for hash_type, hash_value in (
            ("file", file_hash),
            ("content", content_hash),
        ):
            if not hash_value:
                continue
            material = (
                f"voice-ai:duplicate:{hash_type}:{hash_value}"
            ).encode("utf-8")
            digest = hashlib.sha256(material).digest()
            lock_keys.add(int.from_bytes(digest[:8], byteorder="big", signed=True))
        return sorted(lock_keys)

    @classmethod
    def _acquire_duplicate_locks(
        cls,
        db: Session,
        file_hash: str | None,
        content_hash: str | None,
    ) -> list[int]:
        """Serialize same-hash ingestion transactions on PostgreSQL.

        Transaction-scoped locks are released automatically at commit,
        rollback, or session close. Sorted lock acquisition avoids deadlocks
        when two uploads share different subsets of file/content hashes.
        Other dialects skip the PostgreSQL-specific lock; production uses
        PostgreSQL and the hash lookup remains global.
        """
        bind = db.get_bind()
        if bind.dialect.name != "postgresql":
            return []

        lock_keys = cls._duplicate_lock_keys(
            file_hash,
            content_hash,
        )
        statement = text("SELECT pg_advisory_xact_lock(:lock_key)")
        for lock_key in lock_keys:
            db.execute(statement, {"lock_key": lock_key})
        return lock_keys

    def _upload_original(
        self,
        file_path: Path,
        storage_key: str,
        file_hash: str,
    ) -> None:

        self.file_storage.upload_file(
            local_path=file_path,
            storage_key=storage_key,
            content_type=(
                mimetypes.guess_type(file_path.name)[0]
                or "application/octet-stream"
            ),
            metadata={
                "file-hash": file_hash,
            },
        )

    def _cleanup_uploaded_file(
        self,
        storage_key: str,
    ) -> Exception | None:

        try:
            if self.file_storage.exists(storage_key):
                self.file_storage.delete_file(storage_key)

            return None

        except Exception as exc:
            return exc

    def ingest_file(
        self,
        file_path: str | Path,
    ) -> IngestionResult:

        path = Path(file_path).expanduser().resolve()
        extension = path.suffix.lower()
        if extension not in {".pdf", ".txt"}:
            raise DocumentIngestionError(
                "Unsupported document type. Only PDF and TXT files are supported."
            )

        # --------------------------------------------------
        # 1. Validate uploaded document
        # --------------------------------------------------

        validator = self.validator if extension == ".pdf" else self.text_validator
        validation = validator.validate(path)

        if not validation.is_valid:
            label = "PDF" if extension == ".pdf" else "TXT"
            raise DocumentIngestionError(
                f"{label} validation failed: "
                + "; ".join(validation.errors)
            )

        # --------------------------------------------------
        # 2. Hash original file bytes
        # --------------------------------------------------

        file_hash = self.hash_service.calculate_file_hash(
            path
        )

        # --------------------------------------------------
        # 3. Extract document text
        # --------------------------------------------------

        extracted_pages = self.document_loader.load_document(path)

        if extension == ".pdf" and len(extracted_pages) != validation.page_count:
            raise DocumentIngestionError(
                "Extracted page count does not match "
                "the validated PDF page count. "
                f"Expected {validation.page_count}, "
                f"received {len(extracted_pages)}."
            )

        # --------------------------------------------------
        # 4. Normalize page content
        # --------------------------------------------------
        
        normalized_texts = self.normalizer.normalize_pages(
            [page.page_content or "" for page in extracted_pages]
        )
        
        normalized_pages = [
            Document(
                page_content=normalized_text,
                metadata=dict(page.metadata or {}),
            )
            for page, normalized_text in zip(extracted_pages, normalized_texts)
        ]

        if not any(
            page.page_content.strip()
            for page in normalized_pages
        ):
            raise DocumentIngestionError(
                "Document extraction produced no usable text."
            )
        
        normalized_text = self.normalizer.combine_pages(
            [page.page_content for page in normalized_pages]
        )

        content_hash = (
            self.hash_service.calculate_content_hash(
                normalized_text
            )
        )

        # --------------------------------------------------
        # 5. Sample pages for summarization
        # --------------------------------------------------

        sampled_pages = self.sampler.sample(
            normalized_pages
        )

        if not sampled_pages:
            raise DocumentIngestionError(
                "Document sampling returned no content."
            )

        # --------------------------------------------------
        # 6. Generate structured metadata and summary
        # --------------------------------------------------

        summary_output = self.summarizer.summarize(
            sampled_pages
        )

        if not isinstance(
            summary_output,
            DocumentSummarizerOutput,
        ):
            raise TypeError(
                "Summarizer must return DocumentSummarizerOutput."
            )

        # --------------------------------------------------
        # 7. Match against existing documents
        # --------------------------------------------------

        with self.session_factory() as db:

            # Hold the global hash locks until the version is
            # registered and this transaction commits. A competing ingestion
            # then re-runs duplicate detection against the committed version.
            self._acquire_duplicate_locks(
                db,
                file_hash,
                content_hash,
            )

            matching_orchestrator = (
                self.matching_orchestrator_factory(db)
            )

            matching_result = matching_orchestrator.match(
                file_hash=file_hash,
                content_hash=content_hash,
                summary=summary_output.summary,
            )

            # Exact duplicate: do not upload or register.
            if matching_result.classification in {
                "exact_file_match",
                "exact_content_match",
            }:
                return IngestionResult(
                    status="duplicate",
                    file_path=str(path),
                    original_filename=path.name,
                    file_hash=file_hash,
                    content_hash=content_hash,
                    page_count=len(normalized_pages),
                    summary_output=summary_output,
                    matching_result=matching_result,
                    validation_warnings=tuple(
                        validation.warnings
                    ),
                )

            # Potential revision: require an explicit decision.
            if matching_result.classification == "potential_match":
                return IngestionResult(
                    status="review_required",
                    file_path=str(path),
                    original_filename=path.name,
                    file_hash=file_hash,
                    content_hash=content_hash,
                    page_count=len(normalized_pages),
                    summary_output=summary_output,
                    matching_result=matching_result,
                    validation_warnings=tuple(
                        validation.warnings
                    ),
                )

            if matching_result.classification != "no_match":
                raise DocumentIngestionError(
                    "Unexpected matching classification: "
                    f"{matching_result.classification}"
                )

            # --------------------------------------------------
            # 8. Allocate IDs and build B2 object key
            # --------------------------------------------------

            document_id = uuid4()
            version_id = uuid4()

            storage_key = (
                self.file_storage.build_document_version_key(
                    document_id=document_id,
                    version_id=version_id,
                    file_extension=extension,
                )
            )

            # --------------------------------------------------
            # 9. Upload the original document unchanged to B2
            # --------------------------------------------------

            upload_attempted = False

            try:
                upload_attempted = True

                self._upload_original(
                    file_path=path,
                    storage_key=storage_key,
                    file_hash=file_hash,
                )

                # --------------------------------------------------
                # 10. Register a pending document/version
                # --------------------------------------------------

                registration_service = (
                    self.registration_service_factory(db)
                )

                registration_result = (
                    registration_service.register_new_document(
                        document_data=DocumentRegistrationData(
                            title=summary_output.title,
                            document_type=summary_output.document_type,
                            description=summary_output.description,
                        ),
                        version_data=VersionRegistrationData(
                            original_filename=path.name,
                            storage_key=storage_key,
                            file_hash=file_hash,
                            content_hash=content_hash,
                            summary=summary_output.summary,
                            page_count=len(normalized_pages),
                        ),
                        document_id=document_id,
                        version_id=version_id,
                    )
                )

                # Caller owns the outer transaction.
                db.commit()

            except Exception as exc:

                db.rollback()

                cleanup_error = None

                if upload_attempted:
                    cleanup_error = self._cleanup_uploaded_file(
                        storage_key
                    )

                raise DocumentIngestionError(
                    "Document ingestion failed during upload "
                    "or database registration.",
                    cleanup_error=cleanup_error,
                ) from exc

        return IngestionResult(
            status="registered_pending",
            file_path=str(path),
            original_filename=path.name,
            file_hash=file_hash,
            content_hash=content_hash,
            page_count=len(normalized_pages),
            summary_output=summary_output,
            matching_result=matching_result,
            registration_result=registration_result,
            storage_key=storage_key,
            validation_warnings=tuple(
                validation.warnings
            ),
        )


    def ingest_pdf(self, file_path: str | Path) -> IngestionResult:
        """Backward-compatible alias; accepts the existing PDF input type."""
        path = Path(file_path).expanduser().resolve()
        if path.suffix.lower() != ".pdf":
            raise DocumentIngestionError("Expected a PDF file.")
        return self.ingest_file(path)
