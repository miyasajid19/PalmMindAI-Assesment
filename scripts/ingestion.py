"""Ingest a PDF into the shared corpus and optionally index its pending version.

Example:
    uv run python scripts/ingestion.py F:\\resume.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.document_ingestion import DocumentIngestionService
from app.services.document_indexing.indexing_orchestrator import DocumentIndexingOrchestrator


def ingest_pdf(pdf_path: Path, *, skip_index: bool = False) -> None:
    """Register the PDF in the shared corpus and optionally index its version."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

    result = DocumentIngestionService().ingest_pdf(pdf_path)
    indexing_result = None

    if result.status == "registered_pending":
        registration = result.registration_result
        if registration is None:
            raise RuntimeError(
                "Ingestion returned registered_pending without registration data."
            )
        if not skip_index:
            indexing_result = DocumentIndexingOrchestrator().index_pending_version(
                registration.document_version_id
            )

    print(f"Ingestion status: {result.status}")
    print(f"File: {result.file_path}")
    if result.registration_result is not None:
        print(f"Document ID: {result.registration_result.document_id}")
        print(f"Document version ID: {result.registration_result.document_version_id}")
    if result.storage_key:
        print(f"Storage key: {result.storage_key}")
    if indexing_result is not None:
        print(f"Indexing status: {indexing_result.status}")
        print(f"Qdrant chunks: {indexing_result.chunk_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into the shared corpus."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        type=Path,
        default=Path(r"F:\resume.pdf"),
        help=r"PDF to ingest (default: F:\resume.pdf).",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Stop after pending registration instead of indexing.",
    )
    args = parser.parse_args()
    ingest_pdf(args.pdf_path.expanduser().resolve(), skip_index=args.skip_index)


if __name__ == "__main__":
    main()
