"""LangGraph Static Validation node backed by StaticValidationLayer."""

from collections.abc import Callable
from typing import Any

from app.validation import StaticValidationLayer
from app.workflows.rag.state import RAGWorkflowState


def create_static_validation_node(
    validator: StaticValidationLayer,
) -> Callable[[RAGWorkflowState], dict[str, Any]]:
    """Validate citations against the exact documents passed to Generation."""

    def static_validation_node(state: RAGWorkflowState) -> dict[str, Any]:
        generation_output = state.get("generation_output")
        if generation_output is None:
            raise ValueError("generation_output is required before static validation.")

        validation_result = validator.validate_generation_output(
            output=generation_output,
            retrieved_documents=state.get("retrieved_documents") or [],
        )
        return {"validation_result": validation_result}

    return static_validation_node
