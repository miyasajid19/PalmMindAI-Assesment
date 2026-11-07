"""Conditional routing for input validation in the streamlined RAG graph."""

from app.workflows.rag.state import RAGWorkflowState


def route_after_validation(state: RAGWorkflowState) -> str:
    """Route valid queries to retrieval and invalid queries directly to END."""
    return "valid" if state.get("is_valid") is True else "invalid"
