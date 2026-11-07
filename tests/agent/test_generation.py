import pytest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app.agents.Generation import (
    AnswerSegment,
    CitationValidationError,
    GenerationAgent,
    GenerationConfig,
    GenerationOutput,
    GenerationStatus,
    GenerationStreamEventType,
)


@pytest.fixture
def sample_candidates():
    return [
        {
            "chunk_id": "internal-chunk-1",
            "page_content": (
                "Employees are entitled to 20 days of annual leave "
                "per calendar year."
            ),
            "metadata": {
                "document_id": "internal-document-id",
                "document_version_id": "internal-version-id",
                "source": "Employee Leave Policy.pdf",
                "page": 4,
                "chunk_index": 3,
                "organization_id": "internal-organization-id",
            },
            "reranker_score": 0.91,
        },
        {
            "chunk_id": "internal-chunk-2",
            "page_content": (
                "Unused annual leave can be carried forward up to 5 days."
            ),
            "metadata": {
                "source_filename": "Employee Leave Policy.pdf",
                "page": 5,
                "section_heading": "Leave Carry Forward",
            },
            "reranker_score": 0.82,
        },
    ]


@pytest.fixture
def expected_output():
    return GenerationOutput(
        status=GenerationStatus.ANSWERED,
        segments=[
            AnswerSegment(
                text="Employees receive 20 days of annual leave per year.",
                citations=["E1"],
            ),
            AnswerSegment(
                text="Up to 5 unused days can be carried forward.",
                citations=["E2"],
            ),
        ],
    )


def make_agent():
    return GenerationAgent(
        model=RunnableLambda(lambda _: GenerationOutput(
            status=GenerationStatus.ANSWERED,
            segments=[
                AnswerSegment(
                    text="The retrieved documents support this answer.",
                    citations=["E1"],
                )
            ],
        ))
    )


def test_generate_returns_structured_output(sample_candidates, expected_output):
    agent = make_agent()
    agent.chain = RunnableLambda(lambda _: expected_output)

    result = agent.generate(
        current_query="How many annual leave days do employees receive?",
        retrieved_documents=sample_candidates,
    )

    assert isinstance(result, GenerationOutput)
    assert result.status == GenerationStatus.ANSWERED
    assert result.segments[0].citations == ["E1"]


def test_prompt_uses_ranked_evidence_and_selected_metadata(sample_candidates):
    agent = make_agent()
    evidence = agent._prepare_evidence(sample_candidates)

    rendered = agent.prompt.invoke(
        {
            "current_query_json": '"How many leave days?"',
            "conversation_history": [],
            "retrieved_evidence_json": __import__("json").dumps(evidence),
        }
    ).to_string()

    assert '"evidence_id": "E1"' in rendered
    assert '"evidence_id": "E2"' in rendered
    assert "Employee Leave Policy.pdf" in rendered
    assert '"page": 4' in rendered
    assert "Leave Carry Forward" in rendered
    assert "20 days of annual leave" in rendered

    # Backend-only identifiers and retrieval scores must not enter the prompt.
    assert "internal-document-id" not in rendered
    assert "internal-version-id" not in rendered
    assert "internal-organization-id" not in rendered
    assert "reranker_score" not in rendered
    assert "internal-chunk-1" not in rendered


def test_history_is_passed_without_modification(sample_candidates):
    agent = make_agent()
    history = [
        HumanMessage(content="What is the leave policy?"),
        AIMessage(content="The documents describe annual leave."),
    ]
    original = history.copy()
    captured = {}

    def fake_chain(payload):
        captured.update(payload)
        return GenerationOutput(
            status=GenerationStatus.ANSWERED,
            segments=[
                AnswerSegment(text="Supported answer.", citations=["E1"])
            ],
        )

    agent.chain = RunnableLambda(fake_chain)
    agent.generate(
        current_query="How many days?",
        retrieved_documents=sample_candidates,
        conversation_history=history,
    )

    assert captured["conversation_history"] == original
    assert history == original


def test_empty_retrieval_returns_insufficient_evidence_without_llm_call():
    agent = make_agent()
    called = False

    def unexpected_call(_):
        nonlocal called
        called = True
        raise AssertionError("LLM should not be called without evidence.")

    agent.chain = RunnableLambda(unexpected_call)

    result = agent.generate(
        current_query="What is the annual leave policy?",
        retrieved_documents=[],
    )

    assert result.status == GenerationStatus.INSUFFICIENT_EVIDENCE
    assert result.segments[0].citations == []
    assert "don't have enough information" in result.segments[0].text
    assert called is False


@pytest.mark.asyncio
async def test_async_generation(sample_candidates):
    agent = make_agent()
    expected = GenerationOutput(
        status=GenerationStatus.PARTIAL,
        segments=[
            AnswerSegment(
                text="The available chunk confirms annual leave entitlement.",
                citations=["E1"],
            )
        ],
    )
    agent.chain = RunnableLambda(lambda _: expected)

    result = await agent.agenerate(
        current_query="Explain annual leave.",
        retrieved_documents=sample_candidates,
    )

    assert result == expected


@pytest.mark.asyncio
async def test_streaming_emits_text_then_complete_validated_output(
    sample_candidates,
    expected_output,
):
    agent = make_agent()

    class FakeStreamingChain:
        async def astream(self, _payload, config=None):
            yield {
                "status": "answered",
                "segments": [
                    {"text": "Employees receive", "citations":[]},
                ],
            }
            yield {
                "status": "answered",
                "segments": [
                    {
                        "text": "Employees receive 20 days of annual leave per year.",
                        "citations": ["E1"],
                    },
                    {
                        "text": "Up to 5 unused days can be carried forward.",
                        "citations": ["E2"],
                    },
                ],
            }

    agent.streaming_chain = FakeStreamingChain()

    events = [
        event
        async for event in agent.astream_generate(
            current_query="Summarize annual leave.",
            retrieved_documents=sample_candidates,
        )
    ]

    content_events = [
        event for event in events
        if event.event_type == GenerationStreamEventType.CONTENT_DELTA
    ]
    completed_events = [
        event for event in events
        if event.event_type == GenerationStreamEventType.COMPLETED
    ]

    first_segment_text = "".join(
        event.delta
        for event in content_events
        if event.segment_index == 0
    )
    second_segment_text = "".join(
        event.delta
        for event in content_events
        if event.segment_index == 1
    )

    assert first_segment_text == expected_output.segments[0].text
    assert second_segment_text == expected_output.segments[1].text
    assert len(completed_events) == 1
    assert completed_events[0].output == expected_output
    assert events[-1].event_type == GenerationStreamEventType.COMPLETED


@pytest.mark.asyncio
async def test_streaming_rejects_unknown_citation_after_content_was_streamed(
    sample_candidates,
):
    agent = make_agent()

    class FakeStreamingChain:
        async def astream(self, _payload, config=None):
            yield {
                "status": "answered",
                "segments": [
                    {"text": "This answer cites an unknown source.", "citations":[]},
                ],
            }
            yield {
                "status": "answered",
                "segments": [
                    {
                        "text": "This answer cites an unknown source.",
                        "citations": ["E99"],
                    },
                ],
            }

    agent.streaming_chain = FakeStreamingChain()

    events = []
    with pytest.raises(CitationValidationError) as exc_info:
        async for event in agent.astream_generate(
            current_query="Question?",
            retrieved_documents=sample_candidates,
        ):
            events.append(event)

    assert any(
        event.event_type == GenerationStreamEventType.CONTENT_DELTA
        for event in events
    )
    assert exc_info.value.invalid_citations == ("E99",)
    assert not any(
        event.event_type == GenerationStreamEventType.COMPLETED
        for event in events
    )


def test_non_streaming_generation_rejects_unknown_citation(sample_candidates):
    agent = make_agent()
    agent.chain = RunnableLambda(
        lambda _: GenerationOutput(
            status=GenerationStatus.ANSWERED,
            segments=[
                AnswerSegment(
                    text="This answer cites an unknown source.",
                    citations=["E99"],
                )
            ],
        )
    )

    with pytest.raises(CitationValidationError) as exc_info:
        agent.generate(
            current_query="Question?",
            retrieved_documents=sample_candidates,
        )

    assert exc_info.value.invalid_citations == ("E99",)


@pytest.mark.asyncio
async def test_streaming_without_evidence_emits_only_completed_event():
    agent = make_agent()

    events = [
        event
        async for event in agent.astream_generate(
            current_query="Question?",
            retrieved_documents=[],
        )
    ]

    assert len(events) == 1
    assert events[0].event_type == GenerationStreamEventType.COMPLETED
    assert events[0].output.status == GenerationStatus.INSUFFICIENT_EVIDENCE


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        GenerationConfig(temperature=2.1)

    with pytest.raises(ValueError):
        GenerationConfig(max_output_tokens=0)


def test_output_schema_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        GenerationOutput.model_validate(
            {
                "status": "answered",
                "segments": [{
                    "text": "Answer",
                    "citations": [],
                    "url": "https://example.com",
                }],
            }
        )
