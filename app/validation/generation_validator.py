from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from app.agents.Generation.schemas import GenerationOutput
from .schemas import GenerationValidationResult


class InvalidCitationReferenceError(ValueError):
    """Legacy exception retained for compatibility with older imports."""


class StaticValidationLayer:
    """Validate the generation contract: one non-empty answer string."""

    def validate_generation_output(
        self,
        output: GenerationOutput | dict[str, Any],
        retrieved_documents: Sequence[dict[str, Any]],
    ) -> GenerationValidationResult:
        """Check schema shape only; answer content remains LLM-generated."""
        try:
            parsed = (
                output
                if isinstance(output, GenerationOutput)
                else GenerationOutput.model_validate(output)
            )
        except ValidationError as exc:
            return GenerationValidationResult(is_valid=False, errors=[str(exc)])

        # Keep the retrieved documents parameter for the workflow interface.
        # Source/document details are included by the generation agent inside
        # its Markdown answer rather than returned as separate schema fields.
        return GenerationValidationResult(
            is_valid=True,
            output=parsed,
            citations=[],
        )


static_validation_layer = StaticValidationLayer()
