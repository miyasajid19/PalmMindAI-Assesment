from contextlib import contextmanager
from uuid import uuid4
from unittest.mock import MagicMock

import pytest

from app.services.document_indexing.lifecycle_reconciler import (
    QdrantLifecycleReconciler,
)


def _reconciler(rows):
    db = MagicMock()
    db.execute.return_value.all.return_value = rows

    @contextmanager
    def session_factory():
        yield db

    chunk_store = MagicMock()
    summary_store = MagicMock()
    reconciler = QdrantLifecycleReconciler(
        session_factory=session_factory,
        chunk_store=chunk_store,
        summary_store=summary_store,
    )
    return reconciler, chunk_store, summary_store


def _row(*, summary="Version summary", status="active"):
    return (
        uuid4(),
        uuid4(),
        uuid4(),
        2,
        "Policy document",
        summary,
        status,
    )


def test_lifecycle_reconciler_dry_run_does_not_write():
    reconciler, chunk_store, summary_store = _reconciler([
        _row(),
        _row(status="superseded"),
    ])

    result = reconciler.reconcile()

    assert result.dry_run is True
    assert result.planned_versions == 2
    assert result.chunk_versions_synced == 0
    assert result.summary_versions_synced == 0
    chunk_store.update_version_status.assert_not_called()
    summary_store.upsert_summary.assert_not_called()


def test_lifecycle_reconciler_reprojects_postgres_statuses():
    active = _row(status="active")
    historical = _row(status="superseded")
    reconciler, chunk_store, summary_store = _reconciler([
        active,
        historical,
    ])

    result = reconciler.reconcile(dry_run=False)

    assert result.dry_run is False
    assert result.planned_versions == 2
    assert result.chunk_versions_synced == 2
    assert result.summary_versions_synced == 2
    assert result.errors == ()
    assert chunk_store.update_version_status.call_args_list[0].kwargs == {
        "document_version_id": str(active[2]),
        "status": "active",
    }
    assert chunk_store.update_version_status.call_args_list[1].kwargs == {
        "document_version_id": str(historical[2]),
        "status": "superseded",
    }
    assert summary_store.upsert_summary.call_count == 2


def test_lifecycle_reconciler_continues_after_one_projection_fails():
    valid = _row()
    missing_summary = _row(summary="   ", status="superseded")
    reconciler, chunk_store, summary_store = _reconciler([
        valid,
        missing_summary,
    ])
    chunk_store.update_version_status.side_effect = [
        RuntimeError("Qdrant unavailable"),
        None,
    ]

    result = reconciler.reconcile(dry_run=False)

    assert result.planned_versions == 2
    assert result.chunk_versions_synced == 1
    assert result.summary_versions_synced == 1
    assert len(result.errors) == 2
    assert "Qdrant unavailable" in result.errors[0]
    assert "summary is empty" in result.errors[1]
    assert summary_store.upsert_summary.call_count == 1


def test_lifecycle_reconciler_requires_boolean_dry_run():
    reconciler, _, _ = _reconciler([])

    with pytest.raises(TypeError, match="dry_run"):
        reconciler.reconcile(dry_run=1)
