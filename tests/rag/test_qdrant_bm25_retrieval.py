
from langchain_core.documents import Document

from app.rag.retrieval.qdrant_bm25 import QdrantBM25Retriever


class FakeQdrantClient:
    def __init__(self):
        self.received = None

    def bm25_search_with_score(self, **kwargs):
        self.received = kwargs
        return [(Document(page_content="matched"), 0.75)]


def get_filter_conditions(query_filter):
    return {condition.key: condition.match.value for condition in query_filter.must}


def test_bm25_search_filters_shared_corpus_by_document_and_status():
    client = FakeQdrantClient()
    results = QdrantBM25Retriever(client).search(
        query="retrieval", document_id="doc-1", status="active"
    )
    assert isinstance(results[0][0], Document)
    assert results[0][1] == 0.75
    assert get_filter_conditions(client.received["query_filter"]) == {
        "metadata.document_id": "doc-1",
        "metadata.status": "active",
    }


def test_bm25_status_none_disables_status_filter():
    client = FakeQdrantClient()
    QdrantBM25Retriever(client).search(query="retrieval", status=None)
    assert client.received["query_filter"] is None


def test_bm25_search_excludes_previous_chunks_inside_qdrant():
    client = FakeQdrantClient()
    QdrantBM25Retriever(client).search(
        query="retrieval", exclude=True, excluded_chunk_ids=["version-1:5"]
    )
    query_filter = client.received["query_filter"]
    version_conditions = {
        condition.key: condition.match.value
        for condition in query_filter.must_not[0].must
    }
    assert version_conditions == {
        "metadata.document_version_id": "version-1",
        "metadata.chunk_index": 5,
    }
