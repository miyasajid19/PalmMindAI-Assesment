from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryVersionScope(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    BOTH = "both"


class AvailableDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_title: str
    document_summary: str
    version_year: int
    version_status: Literal["active", "superseded"]

    @field_validator("version_year")
    @classmethod
    def validate_version_year(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("version_year must be a positive integer.")
        return value


class AvailableDocuments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_documents: list[AvailableDocument] = Field(default_factory=list)
    superseded_documents: list[AvailableDocument] = Field(
        default_factory=list
    )

    @field_validator("active_documents")
    @classmethod
    def validate_active_documents(
        cls,
        documents: list[AvailableDocument],
    ) -> list[AvailableDocument]:
        if any(document.version_status != "active" for document in documents):
            raise ValueError("active_documents must contain active versions.")
        return documents

    @field_validator("superseded_documents")
    @classmethod
    def validate_superseded_documents(
        cls,
        documents: list[AvailableDocument],
    ) -> list[AvailableDocument]:
        if any(
            document.version_status != "superseded"
            for document in documents
        ):
            raise ValueError(
                "superseded_documents must contain superseded versions."
            )
        return documents


class GeneratedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1)
    version_scope: QueryVersionScope
    target_versions: list[int] = Field(default_factory=list)

    @field_validator("target_versions")
    @classmethod
    def validate_target_versions(cls, years: list[int]) -> list[int]:
        if any(isinstance(year, bool) or year < 1 for year in years):
            raise ValueError(
                "target_versions must contain positive integer years."
            )
        if len(years) != len(set(years)):
            raise ValueError("target_versions must contain unique years.")
        return years


class MultiQueryGeneratorOutput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    queries: list[GeneratedQuery] = Field(
        description="Distinct contextualized search queries.",
        min_length=1,
    )

    @field_validator("queries")
    @classmethod
    def validate_queries(
        cls,
        queries: list[GeneratedQuery],
    ) -> list[GeneratedQuery]:
        if any(not query.query for query in queries):
            raise ValueError("Queries cannot be empty.")

        normalized = [query.query.casefold() for query in queries]

        if len(normalized) != len(set(normalized)):
            raise ValueError("Queries must be unique.")

        return queries