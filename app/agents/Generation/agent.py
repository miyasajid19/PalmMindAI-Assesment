from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser

from app.llm.generator_client import generator_llm_client
from app.tools import book_interview
from app.validation import (
    InvalidCitationReferenceError,
    static_validation_layer,
)

from .config import GenerationConfig
from .prompts import get_generation_prompt
from .schemas import (
    AnswerSegment,
    GenerationOutput,
    GenerationStatus,
    GenerationStreamEvent,
    GenerationStreamEventType,
)


class GenerationAgentError(Exception):
    """Base exception for generation-agent errors."""


class CitationValidationError(GenerationAgentError):
    """Raised when the model cites an evidence ID absent from this request."""

    def __init__(self, invalid_citations: Sequence[str]) -> None:
        self.invalid_citations = tuple(dict.fromkeys(invalid_citations))
        citation_list = ", ".join(self.invalid_citations)
        super().__init__(
            f"Generation returned unknown evidence citation(s): {citation_list}"
        )


class GenerationAgent:
    """
    Stateless company-grounded Generation Agent.

    Responsibilities:
        1. Receive the current query, optional conversation history,
           and the final ranked retrieval candidates.
        2. Prepare prompt-visible evidence with request-scoped evidence IDs.
        3. Invoke the configured LLM once.
        4. Parse the response into the GenerationOutput schema.
        5. Validate citation IDs against the evidence for this request.

    This agent does not perform retrieval, reranking, semantic citation
    verification, source URL resolution, document access, or orchestration.
    """

    INSUFFICIENT_EVIDENCE_MESSAGE = (
        "I don't have enough information in the available company documents "
        "to answer that. Please ask me something related to the available "
        "company information."
    )

    def __init__(
        self,
        config: GenerationConfig | None = None,
        model: Any | None = None,
    ) -> None:
        # Registry for the conversational tool-calling integration; the current
        # structured-output chain does not invoke tools yet.
        self.available_tools = [book_interview]
        self.config = config or GenerationConfig()
        self.llm = model or generator_llm_client.get_llm(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )

        self.parser = PydanticOutputParser(
            pydantic_object=GenerationOutput,
        )
        self.streaming_parser = JsonOutputParser()
        self.prompt = get_generation_prompt(
            format_instructions=self.parser.get_format_instructions(),
        )
        self.chain = self.prompt | self.llm | self.parser
        self.streaming_chain = self.prompt | self.llm | self.streaming_parser

    @staticmethod
    def _prepare_evidence(
        retrieved_documents: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Prepare a minimal prompt representation of ranked retrieval results.

        The order supplied by the retrieval pipeline is preserved. Internal
        storage details, database identifiers, scores, and permissions are
        deliberately excluded from the prompt.
        """
        evidence: list[dict[str, Any]] = []

        for candidate in retrieved_documents:
            if not isinstance(candidate, dict):
                raise TypeError(
                    "Every retrieved candidate must be a dictionary."
                )

            content = candidate.get("page_content")
            if not isinstance(content, str) or not content.strip():
                continue

            metadata = candidate.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise TypeError(
                    "Retrieved candidate metadata must be a dictionary."
                )

            item: dict[str, Any] = {
                "evidence_id": f"E{len(evidence) + 1}",
                "content": content.strip(),
            }

            document_title = (
                metadata.get("document_title")
                or metadata.get("source_filename")
                or metadata.get("source")
            )
            if isinstance(document_title, str) and document_title.strip():
                item["document"] = document_title.strip()

            page = metadata.get("page")
            if page is not None:
                item["page"] = page

            section = (
                metadata.get("section_heading")
                or metadata.get("section")
            )
            if isinstance(section, str) and section.strip():
                item["section"] = section.strip()

            evidence.append(item)

        return evidence

    @staticmethod
    def _insufficient_evidence_output() -> GenerationOutput:
        """Return the deterministic no-evidence response without an LLM call."""
        return GenerationOutput(
            status=GenerationStatus.INSUFFICIENT_EVIDENCE,
            segments=[
                AnswerSegment(
                    text=GenerationAgent.INSUFFICIENT_EVIDENCE_MESSAGE,
                    citations=[],
                )
            ],
        )

    @staticmethod
    def validate_citations(
        output: GenerationOutput,
        evidence: Sequence[dict[str, Any]],
    ) -> GenerationOutput:
        """
        Validate that every cited evidence ID belongs to this request.

        This is an existence check only. It does not determine whether a
        particular chunk semantically supports the claim that cites it.
        """
        try:
            return static_validation_layer.validate_citation_ids(
                output,
                evidence,
            )
        except InvalidCitationReferenceError as exc:
            raise CitationValidationError(
                exc.invalid_citations
            ) from exc

    @staticmethod
    def _parse_output(result: Any) -> GenerationOutput:
        if isinstance(result, GenerationOutput):
            return result
        return GenerationOutput.model_validate(result)

    @staticmethod
    def _build_payload(
        current_query: str,
        evidence: Sequence[dict[str, Any]],
        conversation_history: Sequence[BaseMessage] | None,
    ) -> dict[str, Any]:
        return {
            "current_query_json": json.dumps(
                current_query,
                ensure_ascii=False,
            ),
            "conversation_history": list(conversation_history or []),
            "retrieved_evidence_json": json.dumps(
                evidence,
                ensure_ascii=False,
            ),
        }

    def generate(
        self,
        current_query: str,
        retrieved_documents: Sequence[dict[str, Any]],
        conversation_history: Sequence[BaseMessage] | None = None,
    ) -> GenerationOutput:
        """
        Generate a company-grounded answer from ranked retrieval candidates.

        Citation IDs are checked against this request's prepared evidence
        before the result is returned.
        """
        evidence = self._prepare_evidence(retrieved_documents)
        if not evidence:
            return self._insufficient_evidence_output()

        result = self.chain.invoke(
            self._build_payload(
                current_query,
                evidence,
                conversation_history,
            ),
            config={"metadata": {"agent": "generation"}},
        )
        output = self._parse_output(result)
        return self.validate_citations(output, evidence)

    async def agenerate(
        self,
        current_query: str,
        retrieved_documents: Sequence[dict[str, Any]],
        conversation_history: Sequence[BaseMessage] | None = None,
    ) -> GenerationOutput:
        """Asynchronous counterpart to generate()."""
        evidence = self._prepare_evidence(retrieved_documents)
        if not evidence:
            return self._insufficient_evidence_output()

        result = await self.chain.ainvoke(
            self._build_payload(
                current_query,
                evidence,
                conversation_history,
            ),
            config={"metadata": {"agent": "generation"}},
        )
        output = self._parse_output(result)
        return self.validate_citations(output, evidence)

    async def astream_generate(
        self,
        current_query: str,
        retrieved_documents: Sequence[dict[str, Any]],
        conversation_history: Sequence[BaseMessage] | None = None,
    ) -> AsyncIterator[GenerationStreamEvent]:
        """
        Stream answer text, then emit the complete validated result.

        CONTENT_DELTA events are provisional: citations have not yet been
        validated. After the model finishes, the complete structured response
        is parsed and its citation IDs are checked. The final COMPLETED event
        is emitted only after those checks succeed.

        If citation validation or final parsing fails, the stream raises an
        exception and no COMPLETED event is emitted.
        """
        evidence = self._prepare_evidence(retrieved_documents)
        if not evidence:
            yield GenerationStreamEvent(
                event_type=GenerationStreamEventType.COMPLETED,
                output=self._insufficient_evidence_output(),
            )
            return

        latest_payload: dict[str, Any] | None = None
        streamed_text: dict[int, str] = {}

        try:
            async for partial in self.streaming_chain.astream(
                self._build_payload(
                    current_query,
                    evidence,
                    conversation_history,
                ),
                config={"metadata": {"agent": "generation"}},
            ):
                if not isinstance(partial, dict):
                    raise GenerationAgentError(
                        "The streaming parser returned an unexpected value."
                    )

                latest_payload = partial
                partial_segments = partial.get("segments", [])
                if not isinstance(partial_segments, list):
                    continue

                for index, segment in enumerate(partial_segments):
                    if not isinstance(segment, dict):
                        continue

                    current_text = segment.get("text")
                    if not isinstance(current_text, str):
                        continue

                    previous_text = streamed_text.get(index, "")
                    if current_text == previous_text:
                        continue
                    if not current_text.startswith(previous_text):
                        raise GenerationAgentError(
                            "The streaming parser revised previously emitted "
                            "answer text; refusing to emit duplicated or "
                            "inconsistent content."
                        )

                    delta = current_text[len(previous_text):]
                    streamed_text[index] = current_text
                    if delta:
                        yield GenerationStreamEvent(
                            event_type=GenerationStreamEventType.CONTENT_DELTA,
                            segment_index=index,
                            delta=delta,
                        )

            if latest_payload is None:
                raise GenerationAgentError(
                    "The model stream completed without a structured response."
                )

            output = self._parse_output(latest_payload)

            # Flush any final text not surfaced by the last partial parser event.
            for index, segment in enumerate(output.segments):
                previous_text = streamed_text.get(index, "")
                if segment.text == previous_text:
                    continue
                if not segment.text.startswith(previous_text):
                    raise GenerationAgentError(
                        "The final parsed answer differs from text already "
                        "emitted during streaming."
                    )

                delta = segment.text[len(previous_text):]
                streamed_text[index] = segment.text
                if delta:
                    yield GenerationStreamEvent(
                        event_type=GenerationStreamEventType.CONTENT_DELTA,
                        segment_index=index,
                        delta=delta,
                    )

            validated_output = self.validate_citations(output, evidence)
            yield GenerationStreamEvent(
                event_type=GenerationStreamEventType.COMPLETED,
                output=validated_output,
            )

        except CitationValidationError:
            raise
        except GenerationAgentError:
            raise
        except Exception as exc:
            raise GenerationAgentError(
                "Streaming generation failed before a validated response "
                "could be produced."
            ) from exc
