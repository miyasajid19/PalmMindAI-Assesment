from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class GenerationStatus(str, Enum):
    """Legacy status enum retained for import compatibility."""

    ANSWERED = "answered"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AnswerSegment(BaseModel):
    """Legacy segment model retained for import compatibility."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)


class GenerationOutput(BaseModel):
    """Single-string answer contract produced by the generation agent."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(
        min_length=1,
        description=(
            "The complete user-facing Markdown answer, including any concise "
            "Sources footer when supported by retrieved evidence."
        ),
    )


class GenerationStreamEventType(str, Enum):
    """Kinds of events emitted by the generation streaming interface."""

    CONTENT_DELTA = "content_delta"
    COMPLETED = "completed"


class GenerationStreamEvent(BaseModel):
    """One event from a streaming generation request."""

    model_config = ConfigDict(extra="forbid")

    event_type: GenerationStreamEventType
    delta: str | None = None
    segment_index: int | None = None
    output: GenerationOutput | None = None
