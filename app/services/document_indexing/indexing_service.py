
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from langchain_core.documents import Document


@dataclass(frozen=True, slots=True)
class IndexingResult:
    document_id: str
    document_version_id: str
    version_number: int
    page_count: int
    ocr_page_count: int
    chunk_count: int
    qdrant_point_ids: list[str]
    status: str = "completed"
    chunking_strategy: str = "recursive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "version_number": self.version_number,
            "page_count": self.page_count,
            "ocr_page_count": self.ocr_page_count,
            "chunk_count": self.chunk_count,
            "qdrant": {
                "indexed_chunks": len(self.qdrant_point_ids),
                "point_ids": self.qdrant_point_ids,
            },
            "status": self.status,
            "chunking_strategy": self.chunking_strategy,
        }


class IndexingError(RuntimeError):
    """Raised when indexing fails or compensating cleanup is incomplete."""

    def __init__(
        self,
        message: str,
        cleanup_errors: list[str] | None = None,
    ) -> None:
        self.cleanup_errors = cleanup_errors or []

        if self.cleanup_errors:
            message += (
                "\nCompensating cleanup also encountered errors:\n- "
                + "\n- ".join(self.cleanup_errors)
            )

        super().__init__(message)


class DocumentIndexingService:
    """
    Index all extracted pages belonging to one document version.

    Indexes chunks into Qdrant for dense and payload full-text retrieval.

    This service does not:
      - Register documents or versions in PostgreSQL.
      - Activate or supersede versions.
      - Index document summaries.
      - Upload PDFs to object storage.

    It is intended for a newly registered pending version.
    """

    def __init__(
        self,
        text_splitter=None,
        qdrant_store=None,
        semantic_splitter=None,
    ) -> None:

        if text_splitter is None:
            from ...rag.splitters.recurrsive_text_splitter import (
                text_splitter as default_splitter,
            )
            text_splitter = default_splitter

        if qdrant_store is None:
            from ...rag.vector_stores.qdrant import client
            qdrant_store = client

        self.text_splitter = text_splitter
        self.semantic_splitter = semantic_splitter
        self.qdrant_store = qdrant_store

    def _select_splitter(self, chunking_strategy: str):
        if not isinstance(chunking_strategy, str):
            raise ValueError("chunking_strategy must be 'recursive' or 'semantic'.")

        strategy = chunking_strategy.strip().lower()
        if strategy == "recursive":
            return self.text_splitter, strategy
        if strategy == "semantic":
            if self.semantic_splitter is None:
                from ...rag.splitters.semantic_text_splitter import SemanticTextSplitter
                self.semantic_splitter = SemanticTextSplitter()
            return self.semantic_splitter, strategy

        raise ValueError(
            "Unsupported chunking_strategy. Choose 'recursive' or 'semantic'."
        )

    @staticmethod
    def _normalize_id(value: UUID | str, field_name: str) -> str:
        if isinstance(value, UUID):
            return str(value)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty UUID or string.")

        return value.strip()

    @staticmethod
    def _validate_inputs(
        page_documents: list[Document],
        version_number: int,
        source_filename: str,
    ) -> None:

        if not isinstance(page_documents, list) or not page_documents:
            raise ValueError("page_documents must be a non-empty list.")

        if any(
            not isinstance(document, Document)
            for document in page_documents
        ):
            raise TypeError(
                "Every page_documents entry must be a LangChain Document."
            )

        if (
            isinstance(version_number, bool)
            or not isinstance(version_number, int)
            or version_number < 1
        ):
            raise ValueError("version_number must be a positive integer.")

        if (
            not isinstance(source_filename, str)
            or not source_filename.strip()
        ):
            raise ValueError("source_filename must be non-empty.")

    
    def _attach_metadata(
        self,
        chunks: list[Document],
        document_id: str,
        document_version_id: str,
        version_number: int,
        source_filename: str,
        chunking_strategy: str,
    ) -> None:

        for chunk_index, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata or {})

            metadata.update({
                "document_id": document_id,
                "document_version_id": document_version_id,
                "version_number": version_number,
                "source": source_filename,
                "chunk_index": chunk_index,
                "status": "pending",
                "chunking_strategy": chunking_strategy,
            })

            chunk.metadata = metadata

    def _cleanup(
        self,
        qdrant_point_ids: list[str],
    ) -> list[str]:
        """
        Attempt cleanup in Qdrant.

        Qdrant IDs are generated before insertion, so this also handles
        partial Qdrant insertion where add_documents() raises an exception
        before returning its IDs.
        """

        cleanup_errors: list[str] = []

        try:
            self.qdrant_store.delete_points(qdrant_point_ids)
        except Exception as exc:
            cleanup_errors.append(
                f"Qdrant cleanup failed: {type(exc).__name__}: {exc}"
            )

        return cleanup_errors

    def index_document_version(
        self,
        *,
        page_documents: list[Document],
        document_id: UUID | str,
        document_version_id: UUID | str,
        version_number: int,
        source_filename: str,
        chunking_strategy: str = "recursive",
    ) -> IndexingResult:

        document_id = self._normalize_id(document_id, "document_id")
        document_version_id = self._normalize_id(
            document_version_id,
            "document_version_id",
        )
        self._validate_inputs(
            page_documents=page_documents,
            version_number=version_number,
            source_filename=source_filename,
        )

        # 1. Select strategy and split all extracted pages, not sampled pages.
        splitter, normalized_strategy = self._select_splitter(chunking_strategy)
        chunks = splitter.split_documents(page_documents)

        chunks = [
            chunk
            for chunk in chunks
            if chunk.page_content and chunk.page_content.strip()
        ]

        if not chunks:
            raise ValueError(
                "No non-empty chunks were produced from the extracted pages."
            )

        # 2. Attach metadata used by Qdrant retrieval.
        self._attach_metadata(
            chunks=chunks,
            document_id=document_id,
            document_version_id=document_version_id,
            version_number=version_number,
            source_filename=source_filename,
            chunking_strategy=normalized_strategy,
        )

        # 3. Generate Qdrant IDs before insertion.
        # This makes partial insertion cleanup possible.
        qdrant_point_ids = [ str( uuid5( NAMESPACE_URL, f"voice-ai:{document_version_id}:{chunk_index}", ) ) for chunk_index in range(len(chunks)) ]

        try:
            # 4. Dense indexing.
            inserted_point_ids = self.qdrant_store.add_documents(
                chunks,
                ids=qdrant_point_ids,
            )

            if inserted_point_ids is not None:
                if len(inserted_point_ids) != len(chunks):
                    raise RuntimeError(
                        "Qdrant returned a different number of point IDs "
                        "than the number of submitted chunks."
                    )

        except Exception as exc:
            cleanup_errors = self._cleanup(
                qdrant_point_ids=qdrant_point_ids,
            )

            raise IndexingError(
                message=(
                    "Document version indexing failed. "
                    "Compensating cleanup was attempted."
                ),
                cleanup_errors=cleanup_errors,
            ) from exc

        # 5. Calculate extraction statistics.
        ocr_page_count = sum(
            1
            for page in page_documents
            if page.metadata.get("ocr_used") is True
        )

        # 6. Return indexing result.
        return IndexingResult(
            document_id=document_id,
            document_version_id=document_version_id,
            version_number=version_number,
            page_count=len(page_documents),
            ocr_page_count=ocr_page_count,
            chunk_count=len(chunks),
            qdrant_point_ids=qdrant_point_ids,
            chunking_strategy=normalized_strategy,
        )
