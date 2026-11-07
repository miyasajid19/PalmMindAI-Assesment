import pytest
from app.agents.Generation import AnswerSegment, GenerationOutput, GenerationStatus
from app.validation import StaticValidationLayer


@pytest.fixture
def candidates():
    return [
        {
            "page_content": "Annual leave is 20 days.",
            "metadata": {
                "document_id": "doc-101",
                "document_version_id": "version-3",
                "source_filename": "Employee Handbook.pdf",
                "page": 12,
                "section_heading": "Annual Leave",
                "storage_key": "private/should-never-be-returned.pdf",
            },
        },
        {
            "page_content": "   ",
            "metadata": {"document_id": "ignored"},
        },
        {
            "page_content": "Carry forward is limited to 5 days.",
            "metadata": {
                "document_id": "doc-101",
                "document_version_id": "version-3",
                "source": "Employee Handbook.pdf",
                "page": 14,
            },
        },
    ]


@pytest.fixture
def validator():
    return StaticValidationLayer()


def test_validates_output_and_resolves_cited_metadata(validator, candidates):
    output = GenerationOutput(
        status=GenerationStatus.ANSWERED,
        segments=[
            AnswerSegment(
                text="Annual leave is 20 days.",
                citations=["E1"],
            ),
            AnswerSegment(
                text="Carry forward is limited to 5 days.",
                citations=["E2"],
            ),
        ],
    )

    result = validator.validate_generation_output(output, candidates)

    assert result.is_valid is True
    assert result.errors == []
    assert [
        citation.evidence_id for citation in result.citations
    ] == ["E1", "E2"]
    assert result.citations[0].document_id == "doc-101"
    assert result.citations[0].document_version_id == "version-3"
    assert result.citations[0].document_title == "Employee Handbook.pdf"
    assert result.citations[0].page == 12
    assert result.citations[0].section == "Annual Leave"
    assert "storage_key" not in result.citations[0].model_dump()


def test_rejects_unknown_citation_ids(validator, candidates):
    output = GenerationOutput(
        status=GenerationStatus.ANSWERED,
        segments=[
            AnswerSegment(text="Unsupported source.", citations=["E99"])
        ],
    )

    result = validator.validate_generation_output(output, candidates)

    assert result.is_valid is False
    assert "E99" in result.errors[0]
    assert result.citations == []


def test_rejects_malformed_generation_schema(validator, candidates):
    result = validator.validate_generation_output(
        {
            "status": "answered",
            "segments": [{"text": "", "citations": []}],
        },
        candidates,
    )

    assert result.is_valid is False
    assert result.output is None
    assert result.errors


def test_insufficient_evidence_cannot_carry_citations(validator, candidates):
    output = GenerationOutput(
        status=GenerationStatus.INSUFFICIENT_EVIDENCE,
        segments=[
            AnswerSegment(text="Not enough information.", citations=["E1"])
        ],
    )

    result = validator.validate_generation_output(output, candidates)

    assert result.is_valid is False
    assert any("must not contain citations" in error for error in result.errors)


def test_answer_without_citations_is_warning_not_semantic_failure(
    validator,
    candidates,
):
    output = GenerationOutput(
        status=GenerationStatus.ANSWERED,
        segments=[AnswerSegment(text="A response.", citations=[])],
    )

    result = validator.validate_generation_output(output, candidates)

    assert result.is_valid is True
    assert result.warnings
    assert result.citations == []


def test_citation_ids_follow_generation_evidence_order(validator, candidates):
    output = GenerationOutput(
        status=GenerationStatus.ANSWERED,
        segments=[AnswerSegment(text="Carry forward.", citations=["E2"])],
    )

    result = validator.validate_generation_output(output, candidates)

    assert result.is_valid is True
    assert result.citations[0].page == 14
