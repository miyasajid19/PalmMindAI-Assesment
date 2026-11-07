import pytest

from app.agents.Generation.schemas import AnswerSegment, GenerationOutput, GenerationStatus
from app.validation.schemas import GenerationValidationResult
from app.workflows.rag.graph import create_rag_graph


class FakeRetriever:
    def __init__(self):
        self.calls = []

    async def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "selected_candidates": [{
                "chunk_id": "doc-version:0",
                "page_content": "The company provides 20 days of annual leave.",
                "metadata": {"document_id": "doc"},
            }],
            "selected_count": 1,
            "eligible_count": 1,
            "fused_count": 1,
            "reranked_count": 1,
        }


class FakeGenerationAgent:
    def __init__(self):
        self.received = None

    async def agenerate(self, **kwargs):
        self.received = kwargs
        return GenerationOutput(
            status=GenerationStatus.ANSWERED,
            segments=[AnswerSegment(text="The annual leave entitlement is 20 days.", citations=["E1"])],
        )


class FakeValidator:
    def validate_generation_output(self, output, retrieved_documents):
        return GenerationValidationResult(is_valid=True, output=output, citations=[])


def build_graph(retriever=None, generation=None):
    return create_rag_graph(
        retriever=retriever or FakeRetriever(),
        generation_agent=generation or FakeGenerationAgent(),
        validator=FakeValidator(),
    )


@pytest.mark.asyncio
async def test_graph_is_single_pass_retrieval_then_generation():
    retriever = FakeRetriever()
    generation = FakeGenerationAgent()
    graph = build_graph(retriever, generation)

    result = await graph.ainvoke({"query": "How many annual leave days are provided?"})

    assert result["is_valid"] is True
    assert result["retrieval_attempt"] == 1
    assert result["generation_output"].status == GenerationStatus.ANSWERED
    assert result["validation_result"].is_valid is True
    assert len(retriever.calls) == 1
    assert retriever.calls[0]["current_query"] == "How many annual leave days are provided?"
    assert len(generation.received["retrieved_documents"]) == 1
    assert generation.received["retrieved_documents"][0]["chunk_id"] == "doc-version:0"


@pytest.mark.asyncio
async def test_graph_ends_after_invalid_input_without_retrieval():
    retriever = FakeRetriever()
    graph = build_graph(retriever)

    result = await graph.ainvoke({"query": "!!!"})

    assert result["is_valid"] is False
    assert result["validation_error"]
    assert retriever.calls == []
    assert "generation_output" not in result
