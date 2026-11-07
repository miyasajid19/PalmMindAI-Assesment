import pytest

from app.agents.Generation import AnswerSegment, GenerationOutput, GenerationStatus
from app.validation import GenerationValidationResult
from app.workflows.rag.nodes.static_validation import (
    create_static_validation_node,
)


class FakeStaticValidationLayer:
    def __init__(self, result):
        self.result = result
        self.received = None

    def validate_generation_output(self, output, retrieved_documents):
        self.received = {
            "output": output,
            "retrieved_documents": retrieved_documents,
        }
        return self.result


def test_static_validation_node_injects_validator_and_maps_result():
    generation_output = GenerationOutput(
        status=GenerationStatus.ANSWERED,
        segments=[
            AnswerSegment(
                text="Annual leave is 20 days.",
                citations=["E1"],
            )
        ],
    )
    documents = [
        {
            "page_content": "Annual leave is 20 days.",
            "metadata": {"document_id": "doc-1", "page": 4},
        }
    ]
    validation_result = GenerationValidationResult(
        is_valid=True,
        output=generation_output,
    )
    validator = FakeStaticValidationLayer(validation_result)
    node = create_static_validation_node(validator)

    updates = node(
        {
            "query": "What is the annual leave entitlement?",
            "retrieved_documents": documents,
            "generation_output": generation_output,
        }
    )

    assert validator.received == {
        "output": generation_output,
        "retrieved_documents": documents,
    }
    assert updates == {"validation_result": validation_result}


def test_static_validation_node_defaults_retrieved_documents_to_empty():
    generation_output = GenerationOutput(
        status=GenerationStatus.INSUFFICIENT_EVIDENCE,
        segments=[
            AnswerSegment(
                text="There is not enough evidence to answer.",
                citations=[],
            )
        ],
    )
    validation_result = GenerationValidationResult(
        is_valid=True,
        output=generation_output,
    )
    validator = FakeStaticValidationLayer(validation_result)
    node = create_static_validation_node(validator)

    updates = node({"generation_output": generation_output})

    assert validator.received == {
        "output": generation_output,
        "retrieved_documents": [],
    }
    assert updates == {"validation_result": validation_result}


def test_static_validation_node_requires_generation_output():
    node = create_static_validation_node(
        FakeStaticValidationLayer(
            GenerationValidationResult(is_valid=False)
        )
    )

    with pytest.raises(
        ValueError,
        match="generation_output is required",
    ):
        node({"query": "What is the company policy?"})


def test_static_validation_uses_retrieved_documents_directly_without_compression():
    generation_output = GenerationOutput(
        status=GenerationStatus.ANSWERED,
        segments=[AnswerSegment(text="Answer.", citations=["E1"])],
    )
    original = [{"chunk_id": "doc:0", "page_content": "Original passage."}]
    compressed = [{"chunk_id": "doc:0", "page_content": "Compressed passage."}]
    validator = FakeStaticValidationLayer(
        GenerationValidationResult(is_valid=True, output=generation_output)
    )
    node = create_static_validation_node(validator)

    node({
        "generation_output": generation_output,
        "retrieved_documents": original,
        "context_documents": compressed,
    })

    assert validator.received["retrieved_documents"] is original
