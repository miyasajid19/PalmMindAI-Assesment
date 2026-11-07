"""
PyMuPDFClient.py

Native PDF text extraction service using LangChain's PyMuPDFLoader.

This class only loads/extracts PDF content. It does not apply text thresholds,
decide whether OCR is needed, or invoke any OCR engine. Those decisions belong
to the document-processing/orchestration layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root when executed directly.
# Expected location: <project_root>/app/.../PyMuPDFClient.py
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except IndexError:
    pass

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


class PyMuPDFClient:
    """
    Black-box native PDF extraction interface.

    Methods:
        load_pdf(pdf_path) -> list[Document]
        load_pdf_page(pdf_path, page_number) -> Document

    Page numbers accepted by load_pdf_page are 1-indexed.
    LangChain's PyMuPDFLoader metadata commonly stores page indices as
    0-indexed values; the returned metadata is left unchanged.
    """

    @staticmethod
    def _validate_pdf_path(pdf_path: str | Path) -> Path:
        path = Path(pdf_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"PDF does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a file path, got: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path}")

        return path

    def load_pdf(self, pdf_path: str | Path) -> list[Document]:
        """
        Extract native text from every page of a PDF.

        No OCR or threshold logic is performed.
        """
        path = self._validate_pdf_path(pdf_path)
        loader = PyMuPDFLoader(str(path))
        documents = loader.load()

        # Preserve loader metadata and add a consistent source field.
        for document in documents:
            document.metadata.setdefault("source", str(path))
            document.metadata["extraction_method"] = "pymupdf"

        return documents

    def load_pdf_page(
        self,
        pdf_path: str | Path,
        page_number: int,
    ) -> Document:
        """
        Extract native text from one PDF page.

        Args:
            pdf_path: Path to the PDF.
            page_number: 1-indexed page number.
        """
        if not isinstance(page_number, int) or isinstance(page_number, bool):
            raise TypeError("page_number must be an integer (1-indexed).")

        path = self._validate_pdf_path(pdf_path)
        loader = PyMuPDFLoader(str(path))
        documents = loader.load()

        if not 1 <= page_number <= len(documents):
            raise ValueError(
                f"page_number must be between 1 and {len(documents)}; "
                f"got {page_number}."
            )

        document = documents[page_number - 1]
        document.metadata.setdefault("source", str(path))
        document.metadata["extraction_method"] = "pymupdf"
        return document

# Initialized once when this module is imported.
native_client = PyMuPDFClient()
