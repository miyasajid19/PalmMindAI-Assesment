import pytest

from app.rag.retrieval.chunk_exclusion import build_excluded_chunk_filters


def test_builds_exact_chunk_filters_for_version_and_legacy_document_ids():
    filters = build_excluded_chunk_filters(["version-1:4", "document-2:7"])

    assert len(filters) == 4

    first_version_filter = filters[0]
    first_document_filter = filters[1]

    version_conditions = {
        condition.key: condition.match.value
        for condition in first_version_filter.must
    }
    document_conditions = {
        condition.key: condition.match.value
        for condition in first_document_filter.must
    }

    assert version_conditions == {
        "metadata.document_version_id": "version-1",
        "metadata.chunk_index": 4,
    }
    assert document_conditions == {
        "metadata.document_id": "version-1",
        "metadata.chunk_index": 4,
    }


def test_deduplicates_excluded_chunk_ids():
    filters = build_excluded_chunk_filters(["version-1:4", "version-1:4"])

    assert len(filters) == 2


@pytest.mark.parametrize("chunk_id", ["", "missing-index", "version-1:not-an-int", ":2"])
def test_rejects_malformed_chunk_ids(chunk_id):
    with pytest.raises(ValueError, match="chunk ID|non-empty"):
        build_excluded_chunk_filters([chunk_id])
