"""Qdrant collection management CLI.

Run from project root:
    python scripts/qdrant.py list
    python scripts/qdrant.py count
    python scripts/qdrant.py show --document-id <ID>
    python scripts/qdrant.py delete-id <ID>
    python scripts/qdrant.py delete-source organized.pdf
    python scripts/qdrant.py wipe --confirm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import models
from langchain_qdrant import QdrantVectorStore
from app.rag.embeddings.jina_embeddings import embedding_model
from app.rag.vector_stores.qdrant import client as qdrant_client


def _metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    value = payload.get("metadata", {})
    return value if isinstance(value, dict) else {}


def _scroll_points(
    document_id: str | None = None,
    source: str | None = None,
    page_size: int = 256,
):
    """Yield all matching points using Qdrant scroll pagination."""
    offset = None
    while True:
        points, next_offset = qdrant_client.client.scroll(
            collection_name=qdrant_client.collection_name,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            md = _metadata(point.payload)
            if document_id is not None and str(md.get("document_id")) != str(document_id):
                continue
            if source is not None and md.get("source") != source:
                continue
            yield point

        if next_offset is None:
            break
        offset = next_offset


def count_points() -> int:
    result = qdrant_client.client.count(
        collection_name=qdrant_client.collection_name,
        exact=True,
    )
    print(f"Collection: {qdrant_client.collection_name}")
    print(f"Qdrant points: {result.count}")
    return result.count


def list_points(
    document_id: str | None = None,
    source: str | None = None,
    content_limit: int = 500,
) -> int:
    found = 0
    print(f"Collection: {qdrant_client.collection_name}")

    for point in _scroll_points(document_id=document_id, source=source):
        found += 1
        payload = point.payload or {}
        md = _metadata(payload)
        content = str(payload.get("page_content", ""))

        print("\n" + "=" * 90)
        print(f"Point ID:    {point.id}")
        print(f"Document ID: {md.get('document_id')}")
        print(f"Chunk index: {md.get('chunk_index')}")
        print(f"Source:      {md.get('source')}")
        print(f"Page:        {md.get('page')}")
        print(f"Metadata:    {md}")
        print(f"Content:\n{content[:content_limit]}")
        if len(content) > content_limit:
            print(f"... [{len(content) - content_limit} more characters]")

    print(f"Qdrant points found: {found}")
    return found


def _delete_ids(point_ids: list[str | int]) -> int:
    if not point_ids:
        return 0
    qdrant_client.client.delete(
        collection_name=qdrant_client.collection_name,
        points_selector=models.PointIdsList(points=point_ids),
        wait=True,
    )
    return len(point_ids)


def delete_by_document_id(document_id: str) -> int:
    ids = [p.id for p in _scroll_points(document_id=document_id)]
    deleted = _delete_ids(ids)
    print(f"Deleted {deleted} Qdrant points for document_id={document_id}")
    return deleted


def delete_by_source(source: str) -> int:
    ids = [p.id for p in _scroll_points(source=source)]
    deleted = _delete_ids(ids)
    print(f"Deleted {deleted} Qdrant points for source={source!r}")
    return deleted


def wipe_all(confirm: bool) -> None:
    if not confirm:
        raise SystemExit("Wipe cancelled. Pass --confirm to delete the entire Qdrant collection.")

    qdrant_client.delete_collection()
    qdrant_client._create_collection_if_not_exists()
    qdrant_client.vector_store = QdrantVectorStore(
        client=qdrant_client.client,
        collection_name=qdrant_client.collection_name,
        embedding=embedding_model,
    )
    print(f"Qdrant collection wiped and recreated: {qdrant_client.collection_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and manage the Qdrant collection.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--document-id")
    list_parser.add_argument("--source")
    list_parser.add_argument("--content-limit", type=int, default=500)

    sub.add_parser("count")

    show_parser = sub.add_parser("show")
    show_parser.add_argument("--document-id", required=True)
    show_parser.add_argument("--content-limit", type=int, default=2000)

    delete_id = sub.add_parser("delete-id")
    delete_id.add_argument("document_id")

    delete_source = sub.add_parser("delete-source")
    delete_source.add_argument("source")

    wipe = sub.add_parser("wipe")
    wipe.add_argument("--confirm", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list":
        list_points(args.document_id, args.source, args.content_limit)
    elif args.command == "count":
        count_points()
    elif args.command == "show":
        list_points(document_id=args.document_id, content_limit=args.content_limit)
    elif args.command == "delete-id":
        delete_by_document_id(args.document_id)
    elif args.command == "delete-source":
        delete_by_source(args.source)
    elif args.command == "wipe":
        wipe_all(args.confirm)


if __name__ == "__main__":
    main()
