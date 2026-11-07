import pytest

from langchain_core.messages import HumanMessage, AIMessage

from app.agents.Multi_query_generator import (
    AvailableDocuments,
    GeneratedQuery,
    MultiQueryGeneratorAgent,
    MultiQueryGeneratorConfig,
)


AVAILABLE_DOCUMENTS = AvailableDocuments(
    active_documents=[{
        "document_title": "RAG Guide",
        "document_summary": "Current retrieval architecture guidance.",
        "version_year": 2026,
        "version_status": "active",
    }],
    superseded_documents=[],
)


@pytest.mark.integration
def test_real_multi_query_generation():

    config = MultiQueryGeneratorConfig(
        max_queries=6,
        temperature=0.2,
        max_output_tokens=1500,
    )

    agent = MultiQueryGeneratorAgent(config=config)

    history = [
        HumanMessage(content="What is Retrieval-Augmented Generation?"),
        AIMessage(
            content=(
                "RAG combines information retrieval with "
                "language generation to ground model responses."
            )
        ),
        HumanMessage(content="Explain its retrieval architecture."),
        AIMessage(
            content=(
                "A typical retrieval architecture includes "
                "query processing, retrieval, and context construction."
            )
        ),
    ]

    result = agent.generate(
        current_query="How can we improve its retrieval accuracy?",
        conversation_history=history,
        available_documents=AVAILABLE_DOCUMENTS,
    )

    assert isinstance(result, list)
    assert len(result) >= 1
    assert len(result) <= config.max_queries

    assert all(isinstance(item, GeneratedQuery) for item in result)
    queries = [item.query for item in result]

    assert all(query.strip() for query in queries)
    assert len(queries) == len(set(query.casefold() for query in queries))


@pytest.mark.integration
def test_real_simple_query_generation():

    agent = MultiQueryGeneratorAgent()

    result = agent.generate(
        current_query="What is cosine similarity?",
        conversation_history=[],
        available_documents=AVAILABLE_DOCUMENTS,
    )

    assert isinstance(result, list)
    assert 1 <= len(result) <= agent.config.max_queries

    assert all(isinstance(item, GeneratedQuery) for item in result)
    assert all(item.query.strip() for item in result)