"""Shared state contract for the streamlined one-pass RAG workflow."""

from typing import Any, NotRequired, Required, TypedDict

from langchain_core.messages import BaseMessage

from app.agents.Generation.schemas import GenerationOutput
from app.validation.schemas import GenerationValidationResult


class RAGWorkflowState(TypedDict):
    """State passed between input validation, retrieval, generation and validation."""

    query: Required[str]
    context: NotRequired[list[BaseMessage] | None]

    is_valid: NotRequired[bool]
    validation_error: NotRequired[str | None]

    streaming: NotRequired[bool]
    retrieved_documents: NotRequired[list[dict[str, Any]]]
    retrieval_attempt: NotRequired[int]
    retrieval_diagnostics: NotRequired[dict[str, Any] | None]

    generation_output: NotRequired[GenerationOutput | None]
    validation_result: NotRequired[GenerationValidationResult | None]
    errors: NotRequired[list[str]]
