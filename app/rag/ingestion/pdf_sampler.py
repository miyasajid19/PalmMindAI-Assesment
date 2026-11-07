"""Deterministic sampling of representative pages from an extracted PDF."""

from __future__ import annotations

from langchain_core.documents import Document


class PDFSampler:
    """Select representative pages from extracted PDF documents."""

    def __init__(self, max_pages: int = 5) -> None:
        if not isinstance(max_pages, int) or isinstance(max_pages, bool):
            raise TypeError("max_pages must be an integer.")
        if max_pages <= 0:
            raise ValueError("max_pages must be greater than zero.")
        self.max_pages = max_pages

    @staticmethod
    def _validate_documents(documents: list[Document]) -> None:
        if not isinstance(documents, list):
            raise TypeError("documents must be a list.")
        for index, document in enumerate(documents):
            if not isinstance(document, Document):
                raise TypeError(
                    f"Item at index {index} must be a LangChain Document."
                )

    @staticmethod
    def _get_sample_indices(total_pages: int, sample_count: int) -> list[int]:
        """Return evenly distributed zero-based indices, including endpoints."""
        if total_pages <= 0 or sample_count <= 0:
            return []
        if sample_count >= total_pages:
            return list(range(total_pages))
        if sample_count == 1:
            return [0]
        return [
            (index * (total_pages - 1)) // (sample_count - 1)
            for index in range(sample_count)
        ]

    def sample(self, documents: list[Document]) -> list[Document]:
        """Return sampled original documents without modifying them."""
        self._validate_documents(documents)
        indices = self._get_sample_indices(len(documents), self.max_pages)
        return [documents[index] for index in indices]

    def sample_with_page_numbers(
        self,
        documents: list[Document],
    ) -> list[tuple[int, Document]]:
        """Return sampled documents with one-based page numbers.

        PDFLoader's public output contract is one-based page metadata.
        When page metadata is absent, the original list position is used.
        """
        self._validate_documents(documents)
        indices = self._get_sample_indices(len(documents), self.max_pages)

        sampled: list[tuple[int, Document]] = []
        for index in indices:
            raw_page = (documents[index].metadata or {}).get("page")
            if raw_page is None:
                page_number = index + 1
            elif (
                isinstance(raw_page, bool)
                or not isinstance(raw_page, int)
                or raw_page < 1
            ):
                raise ValueError(
                    f"Page metadata at index {index} must be a positive "
                    "one-based integer."
                )
            else:
                page_number = raw_page
            sampled.append((page_number, documents[index]))

        return sampled
