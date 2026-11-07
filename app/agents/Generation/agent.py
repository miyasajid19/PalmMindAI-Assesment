from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser

from app.llm.generator_client import generator_llm_client
from app.tools import book_interview

from .config import GenerationConfig
from .prompts import get_generation_prompt
from .schemas import GenerationOutput, GenerationStreamEvent, GenerationStreamEventType


class GenerationAgentError(Exception):
    """Base exception for generation-agent errors."""


class CitationValidationError(GenerationAgentError):
    """Legacy exception retained for import compatibility."""


class GenerationAgent:
    """Generate one complete Markdown answer from ranked retrieval evidence."""

    INSUFFICIENT_EVIDENCE_MESSAGE = (
        "I couldn't find enough relevant information in the available company "
        "documents to answer this reliably. Some supporting evidence or details "
        "may be missing. If you share more context or ask about a topic covered "
        "by the company documents, I'll do my best to help."
    )

    def __init__(self, config: GenerationConfig | None = None, model: Any | None = None):
        self.available_tools = [book_interview]
        self.config = config or GenerationConfig()
        self.llm = model or generator_llm_client.get_llm(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        self.parser = PydanticOutputParser(pydantic_object=GenerationOutput)
        self.streaming_parser = JsonOutputParser()
        self.prompt = get_generation_prompt(
            format_instructions=self.parser.get_format_instructions(),
        )
        self.chain = self.prompt | self.llm | self.parser
        self.streaming_chain = self.prompt | self.llm | self.streaming_parser

    @staticmethod
    def _prepare_evidence(retrieved_documents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for candidate in retrieved_documents:
            if not isinstance(candidate, dict):
                raise TypeError("Every retrieved candidate must be a dictionary.")
            content = candidate.get("page_content")
            if not isinstance(content, str) or not content.strip():
                continue
            metadata = candidate.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise TypeError("Retrieved candidate metadata must be a dictionary.")
            item: dict[str, Any] = {
                "evidence_id": f"E{len(evidence) + 1}",
                "content": content.strip(),
            }
            title = metadata.get("document_title") or metadata.get("source_filename") or metadata.get("source")
            if isinstance(title, str) and title.strip():
                item["document"] = title.strip()
            page = metadata.get("page")
            if page is not None:
                item["page"] = page
            section = metadata.get("section_heading") or metadata.get("section")
            if isinstance(section, str) and section.strip():
                item["section"] = section.strip()
            evidence.append(item)
        return evidence

    @staticmethod
    def _insufficient_evidence_output() -> GenerationOutput:
        return GenerationOutput(answer=GenerationAgent.INSUFFICIENT_EVIDENCE_MESSAGE)

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
            "current_query_json": json.dumps(current_query, ensure_ascii=False),
            "conversation_history": list(conversation_history or []),
            "retrieved_evidence_json": json.dumps(evidence, ensure_ascii=False),
        }

    def generate(
        self,
        current_query: str,
        retrieved_documents: Sequence[dict[str, Any]],
        conversation_history: Sequence[BaseMessage] | None = None,
    ) -> GenerationOutput:
        evidence = self._prepare_evidence(retrieved_documents)
        if not evidence:
            return self._insufficient_evidence_output()
        result = self.chain.invoke(
            self._build_payload(current_query, evidence, conversation_history),
            config={"metadata": {"agent": "generation"}},
        )
        return self._parse_output(result)

    async def agenerate(
        self,
        current_query: str,
        retrieved_documents: Sequence[dict[str, Any]],
        conversation_history: Sequence[BaseMessage] | None = None,
    ) -> GenerationOutput:
        evidence = self._prepare_evidence(retrieved_documents)
        if not evidence:
            return self._insufficient_evidence_output()
        result = await self.chain.ainvoke(
            self._build_payload(current_query, evidence, conversation_history),
            config={"metadata": {"agent": "generation"}},
        )
        return self._parse_output(result)

    async def astream_generate(
        self,
        current_query: str,
        retrieved_documents: Sequence[dict[str, Any]],
        conversation_history: Sequence[BaseMessage] | None = None,
    ) -> AsyncIterator[GenerationStreamEvent]:
        evidence = self._prepare_evidence(retrieved_documents)
        if not evidence:
            yield GenerationStreamEvent(
                event_type=GenerationStreamEventType.COMPLETED,
                output=self._insufficient_evidence_output(),
            )
            return

        latest_payload: dict[str, Any] | None = None
        streamed_answer = ""
        try:
            async for partial in self.streaming_chain.astream(
                self._build_payload(current_query, evidence, conversation_history),
                config={"metadata": {"agent": "generation"}},
            ):
                if not isinstance(partial, dict):
                    raise GenerationAgentError("The streaming parser returned an unexpected value.")
                latest_payload = partial
                current_answer = partial.get("answer")
                if not isinstance(current_answer, str) or current_answer == streamed_answer:
                    continue
                if not current_answer.startswith(streamed_answer):
                    raise GenerationAgentError("The streaming parser revised previously emitted answer text.")
                delta = current_answer[len(streamed_answer):]
                streamed_answer = current_answer
                if delta:
                    yield GenerationStreamEvent(
                        event_type=GenerationStreamEventType.CONTENT_DELTA,
                        segment_index=0,
                        delta=delta,
                    )

            if latest_payload is None:
                raise GenerationAgentError("The model stream completed without a structured response.")
            output = self._parse_output(latest_payload)
            if output.answer != streamed_answer:
                if not output.answer.startswith(streamed_answer):
                    raise GenerationAgentError("The final parsed answer differs from streamed text.")
                delta = output.answer[len(streamed_answer):]
                if delta:
                    yield GenerationStreamEvent(
                        event_type=GenerationStreamEventType.CONTENT_DELTA,
                        segment_index=0,
                        delta=delta,
                    )
            yield GenerationStreamEvent(
                event_type=GenerationStreamEventType.COMPLETED,
                output=output,
            )
        except GenerationAgentError:
            raise
        except Exception as exc:
            raise GenerationAgentError(
                "Streaming generation failed before a response could be produced."
            ) from exc
