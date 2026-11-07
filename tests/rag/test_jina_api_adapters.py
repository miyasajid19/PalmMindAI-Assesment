import json

import httpx
from app.rag.embeddings.jina_embeddings import JinaEmbeddingClient
from app.rag.retrieval.cross_encoder_reranker import CrossEncoderReranker


def test_jina_embeddings_uses_langchain_contract_and_query_passage_tasks():
    requests_seen = []

    def handler(request):
        payload = json.loads(request.content)
        requests_seen.append(payload)
        texts = payload["input"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [float(index + 1), 0.0, 0.0]}
                    for index, _ in enumerate(texts)
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = JinaEmbeddingClient(
        model_name="jina-embeddings-v4",
        dimensions=3,
        batch_size=2,
        api_key="test-key",
        http_client=http_client,
    )

    vectors = client.embed_documents(["passage one", "passage two"])
    query_vector = client.embed_query("query")

    assert vectors == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert query_vector == [1.0, 0.0, 0.0]
    assert requests_seen[0]["model"] == "jina-embeddings-v4"
    assert requests_seen[0]["task"] == "retrieval.passage"
    assert requests_seen[1]["task"] == "retrieval.query"
    assert client.get_embedding_model() is client


def test_jina_reranker_maps_remote_indices_and_preserves_candidates():
    def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == "jina-reranker-v3.5"
        assert payload["return_documents"] is False
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.2},
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    reranker = CrossEncoderReranker(api_key="test-key", http_client=http_client)
    candidates = [
        {"chunk_id": "chunk-a", "rank": 1, "score": 0.1, "page_content": "first", "metadata": {"k": "a"}},
        {"chunk_id": "chunk-b", "rank": 2, "score": 0.2, "page_content": "second", "metadata": {"k": "b"}},
    ]

    result = reranker.rerank("query", candidates)

    assert [item["chunk_id"] for item in result] == ["chunk-b", "chunk-a"]
    assert [item["reranker_score"] for item in result] == [0.91, 0.2]
    assert "reranker_score" not in candidates[0]
    assert result[0]["metadata"] is not candidates[1]["metadata"]
