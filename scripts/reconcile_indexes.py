"""Reconcile Qdrant lifecycle payloads from PostgreSQL.

Dry-run (default):
    python scripts/reconcile_indexes.py

Apply corrections:
    python scripts/reconcile_indexes.py --apply

PostgreSQL is authoritative. Re-running --apply is safe and idempotent.
This repairs lifecycle projections and re-upserts summary points; it does not
rebuild missing chunk vectors or re-run PDF extraction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import engine, get_db_session
from app.services.document_indexing.lifecycle_reconciler import (
    QdrantLifecycleReconciler,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile Qdrant lifecycle metadata from PostgreSQL."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply lifecycle repairs. Without this flag, only preview.",
    )
    args = parser.parse_args()

    try:
        if args.apply:
            # Connect to external stores only for an explicit write operation.
            from app.rag.vector_stores.qdrant import client as chunk_store
            from app.rag.vector_stores.qdrant_summary import summary_store

            reconciler = QdrantLifecycleReconciler(
                session_factory=get_db_session,
                chunk_store=chunk_store,
                summary_store=summary_store,
            )
        else:
            reconciler = QdrantLifecycleReconciler(
                session_factory=get_db_session,
                chunk_store=None,
                summary_store=None,
            )

        result = reconciler.reconcile(dry_run=not args.apply)
        print(json.dumps(result.to_dict(), indent=2, default=str))
        if result.dry_run:
            print("Dry-run only. Add --apply to write Qdrant lifecycle updates.")
        elif result.errors:
            print("Reconciliation completed with errors; review the report.")
            return 1
        else:
            print("Qdrant lifecycle projections are synchronized.")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
