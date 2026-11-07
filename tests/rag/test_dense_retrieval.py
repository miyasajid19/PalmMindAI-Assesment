import pytest


from app.rag.retrieval.dense import DenseRetriever
from app.rag.vector_stores.qdrant import QdrantVectorStoreClient


class FakeQdrantClient:
    def __init__(self):
        self.received = None

    def similarity_search_with_score(self, **kwargs):
        self.received = kwargs
        return []


def get_filter_conditions(query_filter):
    if query_filter is None:
        return {}
    return {
        condition.key: (
            condition.match.value
            if hasattr(condition.match, "value")
            else condition.match.any
        )
        for condition in query_filter.must
    }


def test_dense_search_filters_active_by_default_without_identity_fields():
    client = FakeQdrantClient()
    DenseRetriever(client).search("machine learning")
    assert client.received["query"] == "machine learning"
    assert client.received["k"] == 20
    assert get_filter_conditions(client.received["query_filter"]) == {
        "metadata.status": "active"
    }


def test_dense_search_supports_document_and_version_allowlists():
    client = FakeQdrantClient()
    DenseRetriever(client).search(
        "machine learning",
        k=10,
        document_id="doc-123",
        candidate_document_version_ids=["version-1", "version-2"],
    )
    conditions = get_filter_conditions(client.received["query_filter"])
    assert conditions["metadata.document_id"] == "doc-123"
    assert conditions["metadata.status"] == "active"
    version_condition = next(
        c for c in client.received["query_filter"].must
        if c.key == "metadata.document_version_id"
    )
    assert version_condition.match.any == ["version-1", "version-2"]


def test_dense_status_none_disables_lifecycle_filter():
    client = FakeQdrantClient()
    DenseRetriever(client).search("query", status=None)
    assert client.received["query_filter"] is None


@pytest.mark.parametrize("query", ["", "  "])
def test_dense_search_rejects_empty_query(query):
    with pytest.raises(ValueError, match="query cannot be empty"):
        DenseRetriever(FakeQdrantClient()).search(query)


def test_dense_search_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        DenseRetriever(FakeQdrantClient()).search("query", k=0)


def test_dense_search_rejects_empty_status():
    with pytest.raises(ValueError, match="status cannot be empty"):
        DenseRetriever(FakeQdrantClient()).search("query", status=" ")


def test_dense_search_excludes_previous_chunks_inside_qdrant():
    client = FakeQdrantClient()
    DenseRetriever(client).search(
        "query", exclude=True, excluded_chunk_ids=["version-1:3"]
    )
    query_filter = client.received["query_filter"]
    version_conditions = {
        condition.key: condition.match.value
        for condition in query_filter.must_not[0].must
    }
    assert version_conditions == {
        "metadata.document_version_id": "version-1",
        "metadata.chunk_index": 3,
    }


def test_dense_search_ignores_excluded_ids_when_exclude_is_false():
    client = FakeQdrantClient()
    DenseRetriever(client).search(
        "query", exclude=False, excluded_chunk_ids=["version-1:3"]
    )
    assert client.received["query_filter"].must_not in (None, [])


def test_qdrant_wrapper_forwards_query_filter():
    class FakeVectorStore:
        def __init__(self):
            self.received = None

        def similarity_search_with_score(self, **kwargs):
            self.received = kwargs
            return []

    wrapper = QdrantVectorStoreClient.__new__(QdrantVectorStoreClient)
    fake_store = FakeVectorStore()
    wrapper.vector_store = fake_store
    query_filter = object()
    wrapper.similarity_search_with_score(
        query="machine learning", k=15, query_filter=query_filter
    )
    assert fake_store.received == {
        "query": "machine learning", "k": 15, "query_filter": query_filter
    }
