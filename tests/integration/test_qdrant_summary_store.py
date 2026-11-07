import uuid

import pytest

from app.core.config import settings
from app.rag.vector_stores.qdrant_summary import QdrantSummaryStore


@pytest.fixture
def summary_store():
    return QdrantSummaryStore()


def test_summary_collection_exists(summary_store):
    assert summary_store.client.collection_exists(
        summary_store.collection_name
    )

    assert summary_store.collection_name == "voice_ai_summaries"

    # The existing chunk collection must remain separate and untouched.
    assert summary_store.collection_name != settings.QDRANT_COLLECTION_NAME


def test_summary_upsert_search_and_delete(summary_store):
    document_id = str(uuid.uuid4())
    version_id = str(uuid.uuid4())

    summary = (
        "Integration test document about employee leave policies, "
        "annual leave entitlement, sick leave, and approval procedures."
    )

    try:
        # 1. Insert a temporary summary.
        summary_store.upsert_summary(
            document_id=document_id,
            document_version_id=version_id,
            version_number=999,
            title="Temporary Qdrant Integration Test",
            summary=summary,
            status="pending",
        )

        # 2. Search using the exact same summary.
        results = summary_store.search_similar_summaries(
            summary=summary,
            limit=10,
        )

        matching_results = [
            result
            for result in results
            if result["document_version_id"] == version_id
        ]

        assert matching_results, "Inserted summary was not found."

        match = matching_results[0]

        assert match["document_id"] == document_id
        assert match["title"] == "Temporary Qdrant Integration Test"
        assert match["status"] == "pending"
        assert match["similarity_score"] >= 0.99

    finally:
        # 3. Always remove the temporary point, even if an assertion fails.
        summary_store.delete_summary(version_id)

    # 4. Verify that deletion succeeded.
    results_after_delete = summary_store.search_similar_summaries(
        summary=summary,
        limit=10,
    )

    assert not any(
        result["document_version_id"] == version_id
        for result in results_after_delete
    )

def test_summary_search_applies_document_version_allowlist(summary_store):
    allowed_document_id = str(uuid.uuid4())
    allowed_version_id = str(uuid.uuid4())
    excluded_document_id = str(uuid.uuid4())
    excluded_version_id = str(uuid.uuid4())
    summary = (
        "Integration test summary about employee leave policies, "
        "annual leave entitlement, sick leave, and approval procedures."
    )

    try:
        for document_id, version_id, title in (
            (allowed_document_id, allowed_version_id, "Allowed summary"),
            (excluded_document_id, excluded_version_id, "Excluded summary"),
        ):
            summary_store.upsert_summary(
                document_id=document_id,
                document_version_id=version_id,
                version_number=1,
                title=title,
                summary=summary,
                status="active",
            )

        results = summary_store.search_similar_summaries(
            summary=summary,
            limit=10,
            document_version_ids=[allowed_version_id],
        )

        assert [item["document_version_id"] for item in results] == [
            allowed_version_id
        ]
    finally:
        summary_store.delete_summary(allowed_version_id)
        summary_store.delete_summary(excluded_version_id)
