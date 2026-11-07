"""Conversation orchestration boundary: route booking turns or run RAG, then persist."""

import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.services.conversation_history import ConversationHistoryService
from app.services.intent_classifier import IntentClassifier, IntentClassifierError, intent_classifier

logger = logging.getLogger(__name__)


_BOOKING_INTENT_PATTERN = re.compile(
    r"\b(?:book|booking|schedule|scheduling|reserve|reservation|arrange|set up)\b",
    re.IGNORECASE,
)
_INTERVIEW_PATTERN = re.compile(r"\binterview\b", re.IGNORECASE)
_GREETING_PATTERN = re.compile(r"^(?:hi|hello|hey|good morning|good afternoon|good evening)(?:[!. ,]*)$", re.IGNORECASE)


class ConversationOrchestrator:
    """Run one conversational turn with Redis-backed context."""

    def __init__(
        self,
        history_service: ConversationHistoryService | None = None,
        interview_booking_conversation: Any | None = None,
        classifier: IntentClassifier | Any | None = None,
    ) -> None:
        self.history_service = history_service or ConversationHistoryService()
        self.interview_booking_conversation = interview_booking_conversation
        self.classifier = classifier or intent_classifier

    async def run_turn(
        self,
        *,
        graph: Any,
        session_id: str,
        query: str,
        graph_config: dict[str, Any] | None = None,
        intent: str | None = None,
    ) -> dict[str, Any]:
        """Load history, route intent, invoke the flow, and persist its answer."""
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        history_started = time.perf_counter()
        history = await self.history_service.load_history(session_id)
        logger.info("chat_stage=history_load duration_seconds=%.3f", time.perf_counter() - history_started)

        intent = intent or await self.detect_intent(query=query, history=history)

        if intent in {"interview_booking", "booking_followup"}:
            booking_conversation = self.interview_booking_conversation
            if booking_conversation is None:
                from app.services.interview_booking_conversation import (
                    InterviewBookingConversation,
                )

                booking_conversation = InterviewBookingConversation()
                self.interview_booking_conversation = booking_conversation

            answer = await booking_conversation.respond(query=query, history=history)
            result: dict[str, Any] = {"booking_response": answer}
        elif intent == "greeting":
            answer = "Hi! I can help answer questions from the available documents or help you book an interview. What would you like to do?"
            result = {"booking_response": answer}
        else:
            graph_input: dict[str, Any] = {"query": query, "context": history}
            graph_started = time.perf_counter()
            if graph_config is not None:
                result = await graph.ainvoke(graph_input, config=graph_config)
            else:
                result = await graph.ainvoke(graph_input)
            logger.info("chat_stage=rag_graph duration_seconds=%.3f", time.perf_counter() - graph_started)
            answer = self._extract_answer(result)

        if answer:
            save_started = time.perf_counter()
            try:
                await self.history_service.save_turn(session_id, query, answer)
            finally:
                logger.info("chat_stage=redis_save duration_seconds=%.3f", time.perf_counter() - save_started)
        return result

    async def detect_intent(self, *, query: str, history: list[BaseMessage]) -> str:
        """Use the configured LLM classifier with a safe deterministic fallback."""
        booking_context_active = self._is_interview_booking_turn(query, history)
        try:
            return await self.classifier.classify(
                query=query, booking_context_active=booking_context_active
            )
        except IntentClassifierError:
            logger.warning("intent_classifier_unavailable; using deterministic fallback")
            if booking_context_active:
                return "booking_followup" if history else "interview_booking"
            if _GREETING_PATTERN.match(query.strip()):
                return "greeting"
            return "rag"

    @classmethod
    def _is_interview_booking_turn(
        cls,
        query: str,
        history: list[BaseMessage],
    ) -> bool:
        """Detect a scheduling request or an active booking dialogue for context."""
        scheduling_target = _INTERVIEW_PATTERN.search(query) or re.search(
            r"\b(?:appointment|hr|human resources)\b", query, re.IGNORECASE
        )
        if scheduling_target and _BOOKING_INTENT_PATTERN.search(query):
            return True

        recent = history[-8:]
        assistant_prompts = [
            (index, message.content.lower())
            for index, message in enumerate(recent)
            if isinstance(message, AIMessage) and isinstance(message.content, str)
        ]
        for assistant_index, lowered in reversed(assistant_prompts):
            markers = sum(
                marker in lowered
                for marker in ("name", "email", "date", "time", "confirm")
            )
            if ("interview" in lowered or "appointment" in lowered) and markers >= 1:
                return True

            is_booking_field_prompt = any(
                phrase in lowered
                for phrase in (
                    "full name",
                    "email address",
                    "exact date",
                    "preferred time",
                    "what date",
                    "what time",
                )
            )
            if not is_booking_field_prompt:
                continue
            for message in recent[:assistant_index]:
                if not isinstance(message, HumanMessage) or not isinstance(message.content, str):
                    continue
                user_target = _INTERVIEW_PATTERN.search(message.content) or re.search(
                    r"\b(?:appointment|hr|human resources)\b",
                    message.content,
                    re.IGNORECASE,
                )
                if user_target and _BOOKING_INTENT_PATTERN.search(message.content):
                    return True
        return False

    @staticmethod
    def _extract_answer(result: dict[str, Any]) -> str:
        booking_response = result.get("booking_response")
        if isinstance(booking_response, str):
            return booking_response

        output = result.get("generation_output")
        if output is None:
            validation_result = result.get("validation_result")
            output = getattr(validation_result, "output", None)
        if output is None:
            return ""

        segments = getattr(output, "segments", None)
        if segments is None and isinstance(output, dict):
            segments = output.get("segments", [])
        parts: list[str] = []
        for segment in segments or []:
            text = getattr(segment, "text", None)
            if text is None and isinstance(segment, dict):
                text = segment.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts)
