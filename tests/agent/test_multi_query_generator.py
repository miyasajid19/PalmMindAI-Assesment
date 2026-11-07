import pytest

from pydantic import ValidationError
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from app.agents.Multi_query_generator import (
    AvailableDocuments,
    EmptyQueryError,
    GeneratedQuery,
    MultiQueryGeneratorAgent,
    MultiQueryGeneratorConfig,
    MultiQueryGeneratorOutput,
    QueryVersionScope,
)


def query(
    text: str,
    scope: QueryVersionScope = QueryVersionScope.ACTIVE,
    years: list[int] | None = None,
) -> GeneratedQuery:
    return GeneratedQuery(
        query=text,
        version_scope=scope,
        target_versions=years or [],
    )


def catalogue() -> AvailableDocuments:
    return AvailableDocuments(
        active_documents=[{
            "document_title": "Leave Policy",
            "document_summary": "Current leave rules.",
            "version_year": 2026,
            "version_status": "active",
        }],
        superseded_documents=[{
            "document_title": "Leave Policy",
            "document_summary": "Historical leave rules.",
            "version_year": 2024,
            "version_status": "superseded",
        }, {
            "document_title": "Leave Policy",
            "document_summary": "Prior historical leave rules.",
            "version_year": 2025,
            "version_status": "superseded",
        }],
    )


@pytest.fixture
def make_agent(monkeypatch):
    def _make(config=None):
        model = RunnableLambda(
            lambda _, **kwargs: '{"queries": [{"query": "Default", '
            '"version_scope": "active", "target_versions": []}]}'
        )
        monkeypatch.setattr(
            "app.agents.Multi_query_generator.agent.multi_query_llm_client.get_llm",
            lambda **_: model,
        )
        return MultiQueryGeneratorAgent(config=config)

    return _make


def test_default_configuration_limits_queries_to_one():
    assert MultiQueryGeneratorConfig().max_queries == 1


def test_successful_query_generation(make_agent):
    agent = make_agent(MultiQueryGeneratorConfig(max_queries=2))
    expected = [
        query("RAG retrieval architecture"),
        query("Techniques for improving RAG retrieval accuracy"),
    ]
    agent.chain = RunnableLambda(
        lambda _: MultiQueryGeneratorOutput(queries=expected)
    )

    result = agent.generate("How can I improve RAG retrieval?")

    assert result == expected


def test_active_historical_and_comparison_queries_are_validated(make_agent):
    agent = make_agent(MultiQueryGeneratorConfig(max_queries=3))
    expected = [
        query("Current leave entitlements", QueryVersionScope.ACTIVE, [2026]),
        query("2024 leave entitlements", QueryVersionScope.SUPERSEDED, [2024]),
        query("Compare leave policies", QueryVersionScope.BOTH, [2024, 2026]),
    ]
    agent.chain = RunnableLambda(
        lambda _: MultiQueryGeneratorOutput(queries=expected)
    )

    assert agent.generate("Compare current and 2024 leave policy", available_documents=catalogue()) == expected


def test_multiple_superseded_years_are_supported(make_agent):
    agent = make_agent()
    expected = query(
        "Compare historical leave policies",
        QueryVersionScope.SUPERSEDED,
        [2024, 2025],
    )
    agent.chain = RunnableLambda(
        lambda _: MultiQueryGeneratorOutput(queries=[expected])
    )

    assert agent.generate(
        "Compare 2024 and 2025",
        available_documents=catalogue(),
    ) == [expected]


@pytest.mark.parametrize("year", [2023, 9999])
def test_unknown_years_are_rejected(make_agent, year):
    agent = make_agent()
    agent.chain = RunnableLambda(lambda _: MultiQueryGeneratorOutput(
        queries=[query("Unavailable", QueryVersionScope.BOTH, [year])]
    ))

    with pytest.raises(Exception, match="unavailable"):
        agent.generate("Historical policy", catalogue())


def test_scope_and_target_year_must_match(make_agent):
    agent = make_agent()
    agent.chain = RunnableLambda(lambda _: MultiQueryGeneratorOutput(
        queries=[query("Wrong scope", QueryVersionScope.ACTIVE, [2024])]
    ))

    with pytest.raises(Exception, match="scope"):
        agent.generate(
            "Current policy",
            available_documents=catalogue(),
        )


def test_empty_catalogue_allows_unspecified_scope(make_agent):
    agent = make_agent()
    expected = query("What is cosine similarity?", QueryVersionScope.ACTIVE)
    agent.chain = RunnableLambda(
        lambda _: MultiQueryGeneratorOutput(queries=[expected])
    )

    assert agent.generate("What is cosine similarity?") == [expected]


def test_contextual_history_is_preserved(make_agent):
    agent = make_agent()
    history = [
        HumanMessage(content="What is RAG?"),
        AIMessage(content="Retrieval-Augmented Generation."),
    ]
    captured = {}

    def generate(payload):
        captured.update(payload)
        return MultiQueryGeneratorOutput(
            queries=[query("RAG retrieval architecture")]
        )

    agent.chain = RunnableLambda(generate)
    agent.generate("Explain its architecture", history)

    assert captured["conversation_history"] == history
    assert all(
        received is original
        for received, original in zip(captured["conversation_history"], history)
    )


def test_history_is_not_modified(make_agent):
    agent = make_agent()
    history = [HumanMessage(content="What is BM25?")]
    original = history.copy()
    agent.generate("Explain scoring", history)
    assert history == original


def test_empty_query_is_rejected(make_agent):
    with pytest.raises(EmptyQueryError):
        make_agent().generate("  ")


def test_maximum_query_count_is_enforced(make_agent):
    agent = make_agent(MultiQueryGeneratorConfig(max_queries=3))
    generated = [
        query(f"Query {index}") for index in range(1, 8)
    ]
    agent.chain = RunnableLambda(
        lambda _: MultiQueryGeneratorOutput(queries=generated)
    )

    result = agent.generate("Explain RAG")
    assert [item.query for item in result] == ["Query 1", "Query 2", "Query 3"]


def test_duplicate_queries_are_rejected():
    with pytest.raises(ValidationError):
        MultiQueryGeneratorOutput(queries=[
            query("Improve retrieval"),
            query("improve retrieval"),
        ])


def test_target_versions_are_unique_and_positive():
    with pytest.raises(ValidationError):
        GeneratedQuery(
            query="Invalid",
            version_scope=QueryVersionScope.BOTH,
            target_versions=[2024, 2024],
        )


def test_prompt_contains_catalogue_and_configured_maximum(make_agent):
    agent = make_agent(MultiQueryGeneratorConfig(max_queries=4))
    formatted = agent.prompt.format(
        current_query="Explain RAG.",
        conversation_history=[],
        available_documents=catalogue().model_dump(),
        format_instructions=agent.parser.get_format_instructions(),
    )
    assert "maximum number of generated queries is 4" in formatted.lower()
    assert "Leave Policy" in formatted
    assert "target_versions" in formatted


@pytest.mark.asyncio
async def test_async_query_generation(make_agent):
    agent = make_agent(MultiQueryGeneratorConfig(max_queries=2))
    expected = [
        query("BM25 retrieval mechanism"),
        query("BM25 scoring and ranking"),
    ]
    agent.chain = RunnableLambda(
        lambda _: MultiQueryGeneratorOutput(queries=expected)
    )

    assert await agent.agenerate("Explain BM25 retrieval") == expected


@pytest.mark.parametrize("field,value", [
    ("max_queries", 0),
    ("max_output_tokens", 0),
])
def test_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        MultiQueryGeneratorConfig(**{field: value})
