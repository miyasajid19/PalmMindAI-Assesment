"""Delete every collection from the Qdrant endpoint configured for this project.

Usage from the project root:
    uv run python scripts/delete_all_qdrant_collections.py --dry-run
    uv run python scripts/delete_all_qdrant_collections.py --confirm "DELETE ALL COLLECTIONS"

This script deletes collections permanently. It does not delete or alter the Qdrant
server itself, credentials, or any other endpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv(PROJECT_ROOT / ".env")

from app.core.config import settings  # noqa: E402

CONFIRMATION_PHRASE = "DELETE ALL COLLECTIONS"


def _endpoint_label(endpoint: str) -> str:
    """Return a host label without exposing URL credentials or query strings."""
    parsed = urlsplit(endpoint)
    if parsed.hostname:
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{parsed.hostname}{port}"
    return "(configured endpoint)"


def _connect() -> QdrantClient:
    endpoint = settings.QDRANT_API_ENDPOINT
    if not endpoint:
        raise SystemExit(
            "QDRANT_API_ENDPOINT is missing. Configure it in the project .env file."
        )

    return QdrantClient(url=endpoint, api_key=settings.QDRANT_API_KEY)


def _collection_names(client: QdrantClient) -> list[str]:
    return sorted(collection.name for collection in client.get_collections().collections)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Permanently delete every collection at the configured Qdrant endpoint."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="List all collections that would be deleted without deleting them.",
    )
    mode.add_argument(
        "--confirm",
        metavar="PHRASE",
        help=f"Delete all listed collections; phrase must be exactly: {CONFIRMATION_PHRASE}",
    )
    args = parser.parse_args()

    if args.confirm is not None and args.confirm != CONFIRMATION_PHRASE:
        parser.error(f"Confirmation phrase must exactly equal: {CONFIRMATION_PHRASE}")

    client = _connect()
    endpoint = settings.QDRANT_API_ENDPOINT or ""
    names = _collection_names(client)

    print(f"Qdrant endpoint: {_endpoint_label(endpoint)}")
    print(f"Collections found: {len(names)}")
    if names:
        for name in names:
            print(f"  - {name}")
    else:
        print("No collections exist at this endpoint.")

    if args.dry_run:
        print("Dry run only; nothing was deleted.")
        return 0

    if not names:
        print("Nothing to delete.")
        return 0

    deleted: list[str] = []
    failed: list[tuple[str, str]] = []

    for name in names:
        try:
            client.delete_collection(collection_name=name)
            deleted.append(name)
            print(f"Deleted: {name}")
        except Exception as exc:  # report individual collection failures
            failed.append((name, str(exc)))
            print(f"FAILED: {name}: {exc}", file=sys.stderr)

    remaining = _collection_names(client)
    print(f"Deleted collections: {len(deleted)}")
    print(f"Collections remaining: {len(remaining)}")
    if remaining:
        for name in remaining:
            print(f"  - {name}")

    if failed or remaining:
        return 1

    print("Verified: all collections at the configured Qdrant endpoint are deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
