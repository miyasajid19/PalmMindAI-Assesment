import pytest

from app.agents.Generation import (
    AnswerSegment,
    GenerationOutput,
    GenerationStatus,
)
from app.workflows.rag.nodes.generation import create_generation_node


class FakeGenerationAgent:
    def __init__(self, output):
        self.output = output
        self.received = None

    async def agenerate(
        self,
        current_query,
        retrieved_documents,
        conversation_history=None,
    ):
        self.received = {
            "current_query": current_query,
            "retrieved_documents": retrieved_documents,
            "conversation_history": conversation_history,
        }
        return self.output


@pytest.mark.asyncio
async def test_generation_node_injects_agent_and_maps_output():
    documents = [
        {
            "chunk_id": "chunk-1",
            "page_content": "Annual leave entitlement is 20 days.",
        },
        {
            "chunk_id": "chunk-2",
            "page_content": "Unused leave carry-forward is limited to 5 days.",
        },
    ]
    context = []
    output = GenerationOutput(
        status=GenerationStatus.ANSWERED,
        segments=[
            AnswerSegment(
                text="Employees receive 20 days of annual leave.",
                citations=["E1"],
            )
        ],
    )
    agent = FakeGenerationAgent(output)
    node = create_generation_node(agent)

    updates = await node(
        {
            "query": "How many annual leave days do employees receive?",
            "organization_id": "org-id",
            "user_id": "user-id",
            "context": context,
            "retrieved_documents": documents,
        }
    )

    assert agent.received == {
        "current_query": "How many annual leave days do employees receive?",
        "retrieved_documents": documents,
        "conversation_history": context,
    }
    assert updates == {"generation_output": output}


@pytest.mark.asyncio
async def test_generation_node_defaults_documents_and_history_to_empty():
    output = GenerationOutput(
        status=GenerationStatus.INSUFFICIENT_EVIDENCE,
        segments=[
            AnswerSegment(
                text="There is not enough evidence to answer.",
                citations=[],
            )
        ],
    )
    agent = FakeGenerationAgent(output)
    node = create_generation_node(agent)

    updates = await node({"query": "What is the company leave policy?"})

    assert agent.received == {
        "current_query": "What is the company leave policy?",
        "retrieved_documents": [],
        "conversation_history": None,
    }
    assert updates == {"generation_output": output}


@pytest.mark.asyncio
async def test_generation_node_preserves_partial_output_status():
    output = GenerationOutput(
        status=GenerationStatus.PARTIAL,
        segments=[
            AnswerSegment(
                text="The documents confirm the annual leave entitlement.",
                citations=["E1"],
            )
        ],
    )
    node = create_generation_node(FakeGenerationAgent(output))

    updates = await node(
        {
            "query": "Explain the leave policy.",
            "retrieved_documents": [
                {"page_content": "Employees receive annual leave."}
            ],
        }
    )

    assert updates["generation_output"].status == GenerationStatus.PARTIAL


@pytest.mark.asyncio
async def test_generation_uses_retrieved_documents_directly_without_compression():
    original = [{"chunk_id": "doc:0", "page_content": "Long original passage."}]
    compressed = [{"chunk_id": "doc:0", "page_content": "Short compressed passage."}]
    agent = FakeGenerationAgent(
        GenerationOutput(
            status=GenerationStatus.ANSWERED,
            segments=[AnswerSegment(text="Answer.", citations=["E1"])],
        )
    )
    node = create_generation_node(agent)

    await node({
        "query": "Question?",
        "retrieved_documents": original,
        "context_documents": compressed,
    })

    assert agent.received["retrieved_documents"] is original
    assert original[0]["page_content"] == "Long original passage."
    assert compressed[0]["page_content"] == "Short compressed passage."
