import pytest
from pydantic import ValidationError

from app.agents.Generation import GenerationOutput
from app.validation import StaticValidationLayer


def test_generation_output_accepts_answer_only_payload():
    output = GenerationOutput.model_validate(
        {"answer": "The available documents do not contain relevant information."}
    )

    assert output.answer == "The available documents do not contain relevant information."
    assert output.model_dump() == {"answer": output.answer}


def test_generation_output_rejects_legacy_fields_and_empty_answer():
    with pytest.raises(ValidationError):
        GenerationOutput.model_validate({"status": "answered", "segments": []})

    with pytest.raises(ValidationError):
        GenerationOutput.model_validate({"answer": "   "})


def test_static_validator_accepts_answer_only_output():
    output = GenerationOutput(answer="Supported answer.")
    result = StaticValidationLayer().validate_generation_output(output, [])

    assert result.is_valid is True
    assert result.output == output
    assert result.errors == []


def test_generation_agent_accepts_plain_markdown_provider_output():
    from langchain_core.messages import AIMessage

    from app.agents.Generation import GenerationAgent

    output = GenerationAgent._parse_output(
        AIMessage(content="**HiPPA** means a hierarchical pronunciation assessment framework.")
    )

    assert output.answer == "**HiPPA** means a hierarchical pronunciation assessment framework."


def test_generation_agent_extracts_answer_from_json_envelope():
    from langchain_core.messages import AIMessage

    from app.agents.Generation import GenerationAgent

    output = GenerationAgent._parse_output(
        AIMessage(content='{"answer":"Supported answer."}')
    )

    assert output.answer == "Supported answer."


@pytest.mark.asyncio
async def test_streaming_generation_accepts_plain_markdown_chunks():
    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.runnables import RunnableLambda

    from app.agents.Generation import GenerationAgent, GenerationStreamEventType

    class FakeStreamingChain:
        async def astream(self, _payload, config=None):
            yield AIMessageChunk(content="**HiPPA** ")
            yield AIMessageChunk(content="is a pronunciation assessment framework.")

    agent = GenerationAgent(model=RunnableLambda(lambda _: AIMessage(content="")))
    agent.streaming_chain = FakeStreamingChain()
    evidence = [{"page_content": "HiPPA is a pronunciation assessment framework.", "metadata": {"source": "paper.pdf"}}]

    events = [
        event async for event in agent.astream_generate(
            current_query="What is HiPPA?",
            retrieved_documents=evidence,
        )
    ]

    deltas = "".join(
        event.delta for event in events
        if event.event_type == GenerationStreamEventType.CONTENT_DELTA
    )
    completed = [event for event in events if event.event_type == GenerationStreamEventType.COMPLETED]

    assert deltas == "**HiPPA** is a pronunciation assessment framework."
    assert len(completed) == 1
    assert completed[0].output.answer == deltas
