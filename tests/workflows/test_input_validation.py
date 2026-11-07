import pytest

from app.workflows.rag.nodes.input_validation import (
    input_validation_node,
    validate_query,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("    What    is     our leave policy?    ", "What is our leave policy?"),
        ("\tWhat\nabout\t2025?\n", "What about 2025?"),
        ("No", "No"),
        ("Why?", "Why?"),
        ("No, I disagree", "No, I disagree"),
    ],
)
def test_validate_query_normalizes_whitespace_and_preserves_text(query, expected):
    normalized_query, is_valid, error_message = validate_query(query)

    assert normalized_query == expected
    assert is_valid is True
    assert error_message is None


@pytest.mark.parametrize("query", ["", "   ", "\t\n  ", "123456", "12 34", "!!!", "@@##"])
def test_validate_query_rejects_empty_numeric_only_and_symbol_only_input(query):
    normalized_query, is_valid, error_message = validate_query(query)

    assert is_valid is False
    assert error_message
    assert normalized_query == " ".join(query.split())


def test_input_validation_node_returns_graph_state_updates():
    state = {
        "query": "   What    is our policy?  ",
        
        
    }

    updates = input_validation_node(state)

    assert updates == {
        "query": "What is our policy?",
        "is_valid": True,
        "validation_error": None,
    }


def test_input_validation_node_returns_error_for_invalid_query():
    state = {
        "query": "123456",
        
        
    }

    updates = input_validation_node(state)

    assert updates["query"] == "123456"
    assert updates["is_valid"] is False
    assert updates["validation_error"]
