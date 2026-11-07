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
