from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agents.Generation.schemas import GenerationOutput


class CitationReference(BaseModel):
    """Trusted backend metadata for one request-scoped evidence citation.

    This model deliberately contains no URL, storage key, or permission data.
    A future API layer can resolve these identifiers through an authorized
    document-viewing service.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    document_id: str | None = None
    document_version_id: str | None = None
    document_title: str | None = None
    page: int | str | None = None
    section: str | None = None


class GenerationValidationResult(BaseModel):
    """Result of deterministic validation for one generated response."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    output: GenerationOutput | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    citations: list[CitationReference] = Field(default_factory=list)
