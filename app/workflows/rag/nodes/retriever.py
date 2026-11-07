"""LangGraph Retriever node backed by the RetrievalOrchestrator."""

from collections.abc import Callable
from typing import Any

from app.services.retrieval.retrieval_service import RetrievalOrchestrator
from app.workflows.rag.state import RAGWorkflowState


def create_retriever_node(
    retriever: RetrievalOrchestrator,
) -> Callable[[RAGWorkflowState], Any]:
    """Create a single-pass retrieval node."""

    async def retriever_node(state: RAGWorkflowState) -> dict[str, Any]:
        retrieval_result = await retriever.retrieve(
            current_query=state["query"],
            conversation_history=state.get("context"),
        )
        documents = retrieval_result.get("selected_candidates") or []
        return {
            "retrieved_documents": documents,
            "retrieval_attempt": 1,
            "retrieval_diagnostics": {
                "total_attempts": 1,
                "selected_count": retrieval_result.get("selected_count", len(documents)),
                "eligible_count": retrieval_result.get("eligible_count"),
                "fused_count": retrieval_result.get("fused_count"),
                "reranked_count": retrieval_result.get("reranked_count"),
                "retrieval_reason": retrieval_result.get("retrieval_reason"),
                "candidate_generation_error": retrieval_result.get("candidate_generation_error"),
            },
        }

    return retriever_node
