from langchain_core.documents import Document

from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.pipeline import HybridRetrievalPipeline
from app.rag.retrieval.qdrant_bm25 import QdrantBM25Retriever


class FakeVectorStore:
    def __init__(self):
        self.dense_calls = []
        self.bm25_calls = []

    def similarity_search_with_score(self, **kwargs):
        self.dense_calls.append(kwargs)
        return []

    def bm25_search_with_score(self, **kwargs):
        self.bm25_calls.append(kwargs)
        return []


def _filter_payload(query_filter):
    return query_filter.model_dump(exclude_none=True)


def test_dense_retriever_filters_candidate_versions():
    store = FakeVectorStore()
    DenseRetriever(store).search(
        "policy",
        status=None,
        candidate_document_version_ids=["version-1", "version-1", "version-2"],
    )
    conditions = _filter_payload(store.dense_calls[0]["query_filter"])["must"]
    assert {
        "key": "metadata.document_version_id",
        "match": {"any": ["version-1", "version-2"]},
    } in conditions


def test_bm25_retriever_uses_the_same_candidate_filter():
    store = FakeVectorStore()
    QdrantBM25Retriever(store).search(
        "policy",
        status=None,
        candidate_document_version_ids=["version-1", "version-2"],
    )
    conditions = _filter_payload(store.bm25_calls[0]["query_filter"])["must"]
    assert {
        "key": "metadata.document_version_id",
        "match": {"any": ["version-1", "version-2"]},
    } in conditions


def test_empty_candidate_list_does_not_search_qdrant():
    store = FakeVectorStore()
    assert DenseRetriever(store).search("policy", candidate_document_version_ids=[]) == []
    assert QdrantBM25Retriever(store).search("policy", candidate_document_version_ids=[]) == []
    assert store.dense_calls == []
    assert store.bm25_calls == []


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.documents


def _document(version_id: str, chunk_index: int):
    return Document(
        page_content=f"content-{version_id}-{chunk_index}",
        metadata={"document_version_id": version_id, "chunk_index": chunk_index},
    )


def test_pipeline_passes_candidates_without_active_status_and_limits_rrf(monkeypatch):
    dense = FakeRetriever([(_document("version-1", index), 1.0) for index in range(6)])
    bm25 = FakeRetriever([(_document("version-1", index), 1.0) for index in range(6, 12)])
    monkeypatch.setattr("app.rag.retrieval.pipeline.dense_retriever", dense)
    monkeypatch.setattr("app.rag.retrieval.pipeline.bm25_retriever", bm25)

    result = HybridRetrievalPipeline().search(
        query="policy",
        candidate_document_version_ids=["version-1"],
        status=None,
    )

    assert len(result["results"]) == 10
    assert dense.calls[0]["candidate_document_version_ids"] == ["version-1"]
    assert bm25.calls[0]["candidate_document_version_ids"] == ["version-1"]
    assert dense.calls[0]["status"] is None
    assert bm25.calls[0]["status"] is None
