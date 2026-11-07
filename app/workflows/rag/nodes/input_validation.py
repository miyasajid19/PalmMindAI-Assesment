"""Input normalization and validation for the RAG workflow."""

import re
from typing import Any

from app.workflows.rag.state import RAGWorkflowState


_WHITESPACE_PATTERN = re.compile(r"\s+")
_LETTER_PATTERN = re.compile(r"[^\W\d_]", re.UNICODE)

_INVALID_QUERY_MESSAGE = (
    "Your query appears to be empty or does not contain meaningful text.\n"
    "Please enter a question or describe the information you need.\n"
    "You can also refer to something from your previous conversation.\n"
    "Please rephrase your query and try again."
)


def validate_query(query: str) -> tuple[str, bool, str | None]:
    """Normalize whitespace and reject empty, numeric-only, or symbol-only input.

    The function deliberately does not assess query meaning or length because
    short conversational replies may depend on the conversation history.
    """
    if not isinstance(query, str):
        return "", False, _INVALID_QUERY_MESSAGE

    normalized_query = _WHITESPACE_PATTERN.sub(" ", query).strip()

    if not normalized_query or not _LETTER_PATTERN.search(normalized_query):
        return normalized_query, False, _INVALID_QUERY_MESSAGE

    return normalized_query, True, None


def input_validation_node(state: RAGWorkflowState) -> dict[str, Any]:
    """Validate the user's query and return only the state fields to update."""
    normalized_query, is_valid, error_message = validate_query(state["query"])

    return {
        "query": normalized_query,
        "is_valid": is_valid,
        "validation_error": error_message,
    }
