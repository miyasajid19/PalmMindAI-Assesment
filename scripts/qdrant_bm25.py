"""Search the shared Qdrant BM25 index.

Run from the project root:
    uv run python scripts/qdrant_bm25.py "leave policy"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.retrieval.qdrant_bm25 import QdrantBM25Retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search indexed shared-corpus chunks using Qdrant BM25."
    )
    parser.add_argument("query", help="Text to search for.")
    parser.add_argument("--document-id")
    parser.add_argument("--status", default="active")
    parser.add_argument("-k", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = QdrantBM25Retriever().search(
        query=args.query,
        k=args.k,
        document_id=args.document_id,
        status=args.status,
    )

    print(f"Results: {len(results)}")
    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata or {}
        print("\\n" + "=" * 80)
        print(f"Rank: {rank}")
        print(f"Score: {score}")
        print(f"Document ID: {metadata.get('document_id')}")
        print(f"Version ID: {metadata.get('document_version_id')}")
        print(f"Source: {metadata.get('source')}")
        print(f"Page: {metadata.get('page')}")
        print(f"Content:\\n{document.page_content}")


if __name__ == "__main__":
    main()
