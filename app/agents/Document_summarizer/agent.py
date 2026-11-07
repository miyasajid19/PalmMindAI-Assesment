from __future__ import annotations

import json
from typing import Any
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser

from ...llm.summarizer_client import summarizer_llm_client

from .config import DocumentSummarizerConfig
from .prompts import get_document_summarizer_prompt
from .schemas import DocumentSummarizerOutput


class DocumentSummarizerError(Exception):
    """Base exception for document summarization errors."""


class EmptyDocumentInputError(DocumentSummarizerError):
    """Raised when no usable document content is provided."""


class DocumentSummarizerAgent:
    """
    Stateless Document Summarizer Agent.

    Responsibilities:
        1. Analyze sampled document pages.
        2. Extract document metadata.
        3. Generate a document description.
        4. Generate a comprehensive document summary.

    Does not interact with PostgreSQL, Qdrant, or file storage.
    """

    def __init__(
        self,
        config: DocumentSummarizerConfig | None = None,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:

        self.config = config or DocumentSummarizerConfig()
        self.client = client
        self.model = model

        self.llm = (
            None
            if client is not None
            else summarizer_llm_client.get_llm(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            )
        )

        # Structured output parser
        self.parser = PydanticOutputParser(
            pydantic_object=DocumentSummarizerOutput
        )

        # Prompt template
        self.prompt = get_document_summarizer_prompt(
            summary_length=self.config.summary_length,
            summary_style=self.config.summary_style,
            language=self.config.language,
            format_instructions=self.parser.get_format_instructions(),
        )

        if self.client is not None:
            self.chain = None
        else:
            self.chain = self.prompt | self.llm | self.parser

    # =====================================================
    # Input Preparation
    # =====================================================

    @staticmethod
    def _prepare_pages(
        pages: Sequence[Document],
    ) -> str:
        """
        Combine sampled LangChain Documents into one text input.
        """

        if not pages:
            raise EmptyDocumentInputError(
                "No sampled pages were provided."
            )

        formatted_pages = []

        for index, page in enumerate(pages, start=1):

            if not isinstance(page, Document):
                raise TypeError(
                    f"Expected LangChain Document at position {index}, "
                    f"received {type(page).__name__}."
                )

            page_text = page.page_content.strip()

            if not page_text:
                continue

            formatted_pages.append(
                f"[SAMPLED PAGE {index}]\n{page_text}"
            )

        if not formatted_pages:
            raise EmptyDocumentInputError(
                "All supplied sampled pages are empty."
            )

        return "\n\n".join(formatted_pages)

    # =====================================================
    # Summarization
    # =====================================================

    def summarize(
        self,
        pages: Sequence[Document],
    ) -> DocumentSummarizerOutput:
        """
        Generate structured document metadata and summary.

        Args:
            pages:
                Sampled LangChain Documents returned by PDFSampler.

        Returns:
            DocumentSummarizerOutput:
                Validated structured document information.
        """

        # Prepare sampled page contents
        document_text = self._prepare_pages(pages)

        if self.client is None:
            return self.chain.invoke({"document_text": document_text})

        rendered_prompt = self.prompt.invoke(
            {"document_text": document_text}
        ).to_string()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only the JSON object requested by the user."
                    ),
                },
                {"role": "user", "content": rendered_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
        )

        try:
            message = response.choices[0].message
            if getattr(message, "refusal", None):
                raise DocumentSummarizerError(
                    f"LLM refused the request: {message.refusal}"
                )

            content = getattr(message, "content", None)
            if not isinstance(content, str) or not content.strip():
                raise DocumentSummarizerError(
                    "LLM returned an empty response."
                )

            return DocumentSummarizerOutput.model_validate(
                json.loads(content)
            )
        except DocumentSummarizerError:
            raise
        except (IndexError, AttributeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise DocumentSummarizerError(
                "LLM response could not be parsed or validated."
            ) from exc