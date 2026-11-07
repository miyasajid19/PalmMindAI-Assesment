from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from app.agents.Generation.schemas import (
    GenerationOutput,
    GenerationStatus,
)

from .schemas import CitationReference, GenerationValidationResult


_EVIDENCE_ID_PATTERN = re.compile(r"^E[1-9][0-9]*$")


class InvalidCitationReferenceError(ValueError):
    """Raised when an answer cites an evidence ID not present in its request."""

    def __init__(self, invalid_citations: Sequence[str]) -> None:
        self.invalid_citations = tuple(dict.fromkeys(invalid_citations))
        super().__init__(
            "Unknown evidence citation(s): " + ", ".join(self.invalid_citations)
        )


class StaticValidationLayer:
    """Deterministic validation and citation metadata mapping.

    This component does not call an LLM and does not perform semantic
    claim-to-source verification. It validates the response contract,
    checks request-scoped citation IDs, and maps cited IDs to trusted
    metadata from the original retrieval candidates.
    """

    @staticmethod
    def validate_citation_ids(
        output: GenerationOutput,
        evidence: Sequence[dict[str, Any]],
    ) -> GenerationOutput:
        """Ensure every citation references evidence supplied to this request."""
        available_ids = {
            item["evidence_id"]
            for item in evidence
            if isinstance(item, dict)
            and isinstance(item.get("evidence_id"), str)
        }
        invalid_ids = [
            citation
            for segment in output.segments
            for citation in segment.citations
            if not isinstance(citation, str)
            or not _EVIDENCE_ID_PATTERN.fullmatch(citation)
            or citation not in available_ids
        ]
        if invalid_ids:
            raise InvalidCitationReferenceError(invalid_ids)
        return output

    @staticmethod
    def _build_citation_registry(
        retrieved_documents: Sequence[dict[str, Any]],
    ) -> dict[str, CitationReference]:
        """Build the same E1..En ordering used by GenerationAgent.

        Empty chunks are skipped exactly as they are during prompt evidence
        preparation. Only allowlisted, non-sensitive metadata is retained.
        """
        registry: dict[str, CitationReference] = {}
        evidence_index = 0

        for candidate in retrieved_documents:
            if not isinstance(candidate, dict):
                raise TypeError("Every retrieved candidate must be a dictionary.")

            content = candidate.get("page_content")
            if not isinstance(content, str) or not content.strip():
                continue

            metadata = candidate.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise TypeError("Retrieved candidate metadata must be a dictionary.")

            evidence_index += 1
            evidence_id = f"E{evidence_index}"

            document_title = (
                metadata.get("document_title")
                or metadata.get("source_filename")
                or metadata.get("source")
            )
            section = metadata.get("section_heading") or metadata.get("section")

            registry[evidence_id] = CitationReference(
                evidence_id=evidence_id,
                document_id=(
                    str(metadata["document_id"])
                    if metadata.get("document_id") is not None
                    else None
                ),
                document_version_id=(
                    str(metadata["document_version_id"])
                    if metadata.get("document_version_id") is not None
                    else None
                ),
                document_title=(
                    document_title.strip()
                    if isinstance(document_title, str) and document_title.strip()
                    else None
                ),
                page=metadata.get("page"),
                section=(
                    section.strip()
                    if isinstance(section, str) and section.strip()
                    else None
                ),
            )

        return registry

    def validate_generation_output(
        self,
        output: GenerationOutput | dict[str, Any],
        retrieved_documents: Sequence[dict[str, Any]],
    ) -> GenerationValidationResult:
        """Validate output structure and return trusted citation metadata.

        Invalid results are returned with errors instead of being silently
        repaired. Warnings identify conditions that cannot be proven by
        deterministic checks alone.
        """
        try:
            parsed_output = (
                output
                if isinstance(output, GenerationOutput)
                else GenerationOutput.model_validate(output)
            )
        except ValidationError as exc:
            return GenerationValidationResult(
                is_valid=False,
                errors=[str(exc)],
            )

        try:
            registry = self._build_citation_registry(retrieved_documents)
        except (TypeError, ValueError) as exc:
            return GenerationValidationResult(
                is_valid=False,
                output=parsed_output,
                errors=[str(exc)],
            )

        evidence = [
            {"evidence_id": evidence_id}
            for evidence_id in registry
        ]
        errors: list[str] = []
        warnings: list[str] = []

        try:
            self.validate_citation_ids(parsed_output, evidence)
        except InvalidCitationReferenceError as exc:
            errors.append(str(exc))

        citation_ids = list(
            dict.fromkeys(
                citation
                for segment in parsed_output.segments
                for citation in segment.citations
                if isinstance(citation, str)
            )
        )
        citation_references = [
            registry[citation_id]
            for citation_id in citation_ids
            if citation_id in registry
        ]

        has_citations = any(segment.citations for segment in parsed_output.segments)
        if (
            parsed_output.status == GenerationStatus.INSUFFICIENT_EVIDENCE
            and has_citations
        ):
            errors.append(
                "An insufficient_evidence response must not contain citations."
            )
        elif (
            parsed_output.status in {
                GenerationStatus.ANSWERED,
                GenerationStatus.PARTIAL,
            }
            and not has_citations
        ):
            warnings.append(
                "The response has no citations. Static validation cannot "
                "determine whether its claims require supporting evidence."
            )

        return GenerationValidationResult(
            is_valid=not errors,
            output=parsed_output,
            errors=errors,
            warnings=warnings,
            citations=citation_references if not errors else [],
        )


static_validation_layer = StaticValidationLayer()
