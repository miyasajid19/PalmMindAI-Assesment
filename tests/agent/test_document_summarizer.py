
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from langchain_core.documents import Document

from app.agents.Document_summarizer.agent import (
    DocumentSummarizerAgent,
    DocumentSummarizerError,
    EmptyDocumentInputError,
)
from app.agents.Document_summarizer.config import DocumentSummarizerConfig
from app.agents.Document_summarizer.schemas import DocumentSummarizerOutput


# =========================================================
# Fixtures
# =========================================================

@pytest.fixture
def sample_pages():
    return [
        Document(
            page_content=(
                "Employee Leave Policy\n"
                "This policy explains annual leave, sick leave, "
                "and employee leave approval procedures."
            ),
            metadata={"page": 1},
        ),
        Document(
            page_content=(
                "Employees must submit leave requests to their "
                "manager before taking planned leave."
            ),
            metadata={"page": 5},
        ),
    ]


@pytest.fixture
def sample_output():
    return {
        "title": "Employee Leave Policy",
        "document_type": "Policy",
        "description": (
            "A policy defining employee leave entitlements "
            "and approval procedures."
        ),
        "summary": (
            "This document describes employee leave entitlements, "
            "including annual and sick leave. It explains the "
            "procedures employees must follow when requesting leave "
            "and the approval responsibilities of managers."
        ),
    }


@pytest.fixture
def mock_client():
    return Mock()


def configure_mock_response(client, content):
    message = SimpleNamespace(
        content=content,
        refusal=None,
    )

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=message)
        ]
    )

    client.chat.completions.create.return_value = response


@pytest.fixture
def agent(mock_client):
    return DocumentSummarizerAgent(
        client=mock_client,
        model="test-model",
    )


# =========================================================
# Successful Execution
# =========================================================

def test_successful_summarization(
    agent,
    mock_client,
    sample_pages,
    sample_output,
):
    configure_mock_response(
        mock_client,
        json.dumps(sample_output),
    )

    result = agent.summarize(sample_pages)
    assert isinstance(result, DocumentSummarizerOutput)

    assert result.title == "Employee Leave Policy"
    assert result.document_type == "Policy"

    assert result.description
    assert result.summary

    mock_client.chat.completions.create.assert_called_once()


def test_llm_receives_sampled_page_content(
    agent,
    mock_client,
    sample_pages,
    sample_output,
):
    configure_mock_response(
        mock_client,
        json.dumps(sample_output),
    )

    agent.summarize(sample_pages)

    call_args = mock_client.chat.completions.create.call_args

    messages = call_args.kwargs["messages"]

    user_message = messages[1]["content"]

    assert "Employee Leave Policy" in user_message
    assert "leave approval procedures" in user_message
    assert "[SAMPLED PAGE 1]" in user_message
    assert "[SAMPLED PAGE 2]" in user_message


# =========================================================
# Input Validation
# =========================================================

def test_empty_pages(agent, mock_client):

    with pytest.raises(EmptyDocumentInputError):
        agent.summarize([])

    mock_client.chat.completions.create.assert_not_called()


def test_all_pages_empty(agent, mock_client):

    pages = [
        Document(page_content="   "),
        Document(page_content=""),
    ]

    with pytest.raises(EmptyDocumentInputError):
        agent.summarize(pages)

    mock_client.chat.completions.create.assert_not_called()


def test_invalid_page_type(agent):

    pages = ["This is not a LangChain Document"]

    with pytest.raises(TypeError):
        agent.summarize(pages)


# =========================================================
# Output Validation
# =========================================================

def test_invalid_json_response(
    agent,
    mock_client,
    sample_pages,
):
    configure_mock_response(
        mock_client,
        '{"title": "Incomplete response"}',
    )

    with pytest.raises(DocumentSummarizerError):
        agent.summarize(sample_pages)


# =========================================================
# Configuration Tests
# =========================================================

def test_custom_configuration(
    mock_client,
    sample_pages,
    sample_output,
):
    configure_mock_response(
        mock_client,
        json.dumps(sample_output),
    )

    config = DocumentSummarizerConfig(
        summary_length=300,
        summary_style="concise",
        language="English",
        temperature=0.1,
        max_output_tokens=1500,
    )

    agent = DocumentSummarizerAgent(
        client=mock_client,
        model="test-model",
        config=config,
    )

    result = agent.summarize(sample_pages)

    assert result.title == "Employee Leave Policy"

    call_args = mock_client.chat.completions.create.call_args

    assert call_args.kwargs["temperature"] == 0.1
    assert call_args.kwargs["max_tokens"] == 1500


def test_prompt_contains_configured_values(mock_client):

    config = DocumentSummarizerConfig(
        summary_length=300,
        summary_style="concise",
        language="English",
    )

    agent = DocumentSummarizerAgent(
        client=mock_client,
        model="test-model",
        config=config,
    )

    rendered_prompt = agent.prompt.invoke(
        {"document_text": "Sample document content"}
    ).to_string()

    assert "300 words" in rendered_prompt
    assert "concise" in rendered_prompt
    assert "English" in rendered_prompt
    assert "Sample document content" in rendered_prompt


# =========================================================
# LLM Error Handling
# =========================================================

def test_llm_empty_response(
    agent,
    mock_client,
    sample_pages,
):
    message = SimpleNamespace(
        content=None,
        refusal=None,
    )

    mock_client.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(message=message)
            ]
        )
    )

    with pytest.raises(DocumentSummarizerError):
        agent.summarize(sample_pages)


def test_llm_refusal(
    agent,
    mock_client,
    sample_pages,
):
    message = SimpleNamespace(
        content=None,
        refusal="Request refused.",
    )

    mock_client.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(message=message)
            ]
        )
    )

    with pytest.raises(DocumentSummarizerError):
        agent.summarize(sample_pages)
