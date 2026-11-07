from app.workflows.rag.routing import route_after_validation


def test_invalid_query_routes_to_end():
    assert route_after_validation({"is_valid": False}) == "invalid"


def test_valid_query_routes_to_retriever():
    assert route_after_validation({"is_valid": True}) == "valid"
