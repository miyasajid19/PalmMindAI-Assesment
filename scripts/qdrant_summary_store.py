"""Inspect document-version summaries in the shared Qdrant summary collection.

Run from the project root:
    uv run python scripts/qdrant_summary_store.py count
    uv run python scripts/qdrant_summary_store.py list
    uv run python scripts/qdrant_summary_store.py list --status active
    uv run python scripts/qdrant_summary_store.py show --document-id <ID>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient, models

from app.core.config import settings


def _client() -> QdrantClient:
    if not settings.QDRANT_SUMMARY_COLLECTION_NAME:
        raise ValueError("QDRANT_SUMMARY_COLLECTION_NAME is not configured.")
    return QdrantClient(url=settings.QDRANT_API_ENDPOINT, api_key=settings.QDRANT_API_KEY)


def _collection_name() -> str:
    collection_name = settings.QDRANT_SUMMARY_COLLECTION_NAME
    if not collection_name:
        raise ValueError("QDRANT_SUMMARY_COLLECTION_NAME is not configured.")
    return collection_name


def _scroll_points(
    client: QdrantClient,
    *,
    document_id: str | None = None,
    document_version_id: str | None = None,
    status: str | None = None,
    page_size: int = 256,
) -> Iterator[Any]:
    conditions = []
    for key, value in (
        ("document_id", document_id),
        ("document_version_id", document_version_id),
        ("status", status),
    ):
        if value is not None:
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )

    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=_collection_name(),
            limit=page_size,
            offset=offset,
            scroll_filter=models.Filter(must=conditions) if conditions else None,
            with_payload=True,
            with_vectors=False,
        )
        yield from points
        if next_offset is None:
            return
        offset = next_offset


def count_points(client: QdrantClient) -> int:
    result = client.count(collection_name=_collection_name(), exact=True)
    print(f"Collection: {_collection_name()}")
    print(f"Summary points: {result.count}")
    return result.count


def list_points(
    client: QdrantClient,
    *,
    document_id: str | None = None,
    document_version_id: str | None = None,
    status: str | None = None,
    summary_limit: int = 2000,
) -> int:
    found = 0
    print(f"Collection: {_collection_name()}")
    for point in _scroll_points(
        client,
        document_id=document_id,
        document_version_id=document_version_id,
        status=status,
    ):
        found += 1
        payload = point.payload or {}
        summary = str(payload.get("summary", ""))
        print("\\n" + "=" * 90)
        print(f"Point ID:             {point.id}")
        print(f"Document ID:          {payload.get('document_id')}")
        print(f"Document version ID:  {payload.get('document_version_id')}")
        print(f"Version number:       {payload.get('version_number')}")
        print(f"Title:                {payload.get('title')}")
        print(f"Status:               {payload.get('status')}")
        print(f"Summary:\\n{summary[:summary_limit]}")
        if len(summary) > summary_limit:
            print(f"... [{len(summary) - summary_limit} more characters]")
    print(f"Summary points found: {found}")
    return found


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect document summaries in the Qdrant summary collection."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("count")

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--document-id")
    list_parser.add_argument("--document-version-id")
    list_parser.add_argument("--status")
    list_parser.add_argument("--summary-limit", type=int, default=2000)

    show_parser = sub.add_parser("show")
    show_parser.add_argument("--document-id")
    show_parser.add_argument("--document-version-id")
    show_parser.add_argument("--status")
    show_parser.add_argument("--summary-limit", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = _client()
    if args.command == "count":
        count_points(client)
    else:
        list_points(
            client,
            document_id=args.document_id,
            document_version_id=args.document_version_id,
            status=args.status,
            summary_limit=args.summary_limit,
        )


if __name__ == "__main__":
    main()
