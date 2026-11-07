import pytest

from app.workflows.rag.nodes.retriever import create_retriever_node


class FakeRetrievalOrchestrator:
    def __init__(self, selected_candidates):
        self.selected_candidates = selected_candidates
        self.received = None

    async def retrieve(self, **kwargs):
        self.received = kwargs
        count = len(self.selected_candidates)
        return {
            "selected_candidates": self.selected_candidates,
            "selected_count": count,
            "eligible_count": count,
            "fused_count": count,
            "reranked_count": count,
            "retrieval_reason": None,
            "candidate_generation_error": None,
        }


@pytest.mark.asyncio
async def test_retriever_runs_once_using_original_query_and_returns_selected_documents():
    selected = [{"chunk_id": "doc-a:0", "page_content": "Evidence A"}]
    orchestrator = FakeRetrievalOrchestrator(selected)
    node = create_retriever_node(orchestrator)

    updates = await node({"query": "Original question", "context": []})

    assert orchestrator.received == {
        "current_query": "Original question",
        "conversation_history": [],
    }
    assert updates["retrieval_attempt"] == 1
    assert updates["retrieved_documents"] is selected
    assert updates["retrieval_diagnostics"]["total_attempts"] == 1
    assert updates["retrieval_diagnostics"]["selected_count"] == 1
    assert "retry" not in updates
    assert "rewritten_query" not in updates
    assert "hypothetical_answer" not in updates


@pytest.mark.asyncio
async def test_retriever_handles_empty_selected_documents_in_one_pass():
    orchestrator = FakeRetrievalOrchestrator([])
    node = create_retriever_node(orchestrator)

    updates = await node({"query": "Question with no matching documents"})

    assert updates["retrieved_documents"] == []
    assert updates["retrieval_attempt"] == 1
    assert updates["retrieval_diagnostics"]["total_attempts"] == 1
