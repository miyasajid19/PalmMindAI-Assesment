"""LLM-backed conversational intent classification using OpenCode Zen SystemOne."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

_SYSTEMONE_URL = "https://opencode.ai/zen/v1/systemone"
_MODEL = "jev-1.13-free"
_ALLOWED_INTENTS = {"greeting", "interview_booking", "booking_followup", "rag", "general"}


class IntentClassifierError(RuntimeError):
    """Raised when the intent classification service cannot be used."""


class IntentClassifier:
    """Classify a user message into a SaviraVoice conversational route."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str = _SYSTEMONE_URL,
        model: str = _MODEL,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENCODE_API_KEY")
        self.url = url
        self.model = model
        self.timeout = timeout

    async def classify(self, *, query: str, booking_context_active: bool = False) -> str:
        """Call SystemOne asynchronously without blocking the event loop."""
        state = json.dumps(
            {
                "user_message": query,
                "booking_context_active": booking_context_active,
                "task": (
                    "Classify the user's message for a conversational backend. "
                    "Choose exactly one intent from greeting, interview_booking, "
                    "booking_followup, rag, general. If booking_context_active is true "
                    "and this message supplies or changes booking details or confirms/"
                    "declines a booking, choose booking_followup. A fresh request to "
                    "book/schedule/reserve an interview or appointment is "
                    "interview_booking. Greetings and small talk are greeting. "
                    "Questions asking for information grounded in uploaded documents "
                    "are rag. Other requests are general. Do not treat a message as "
                    "booking merely because it contains a name, date, or email unless "
                    "booking_context_active is true."
                ),
            },
            ensure_ascii=False,
        )
        return await asyncio.to_thread(self._classify_sync, state)

    def _classify_sync(self, state: str) -> str:
        if not self.api_key:
            raise IntentClassifierError(
                "OPENCODE_API_KEY is not configured for intent classification."
            )

        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": (
                        "Return the single best intent label only, using one of the "
                        "allowed criteria keys."
                    ),
                    "criteria": {
                        "greeting": "Greeting, hello, hi, or casual small talk.",
                        "interview_booking": (
                            "A new request to book, schedule, reserve, or arrange an "
                            "interview or appointment, including with HR."
                        ),
                        "booking_followup": (
                            "A reply within an active booking dialogue providing or "
                            "changing name/email/date/time or confirming/declining."
                        ),
                        "rag": (
                            "A factual question that should be answered from indexed "
                            "uploaded documents or company/resume knowledge."
                        ),
                        "general": "Other conversational intent not listed above.",
                    },
                }
            },
        }

        try:
            response = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.exception("intent_classification_request_failed")
            raise IntentClassifierError("Intent classification request failed.") from exc

        label = self._extract_intent(result)
        if label not in _ALLOWED_INTENTS:
            logger.warning("intent_classifier_unrecognized_result=%r", result)
            return "rag"
        return label

    @classmethod
    def _extract_intent(cls, response: Any) -> str:
        """Handle common SystemOne response shapes without trusting arbitrary text."""
        if isinstance(response, str):
            candidate: Any = response
        elif isinstance(response, dict):
            for key in ("intent", "answer", "result", "answers", "data", "output"):
                if key in response:
                    candidate = cls._extract_intent(response[key])
                    if candidate:
                        return candidate
            candidate = response.get("label", response.get("value", response.get("choice", "")))
        elif isinstance(response, list):
            for item in response:
                candidate = cls._extract_intent(item)
                if candidate:
                    return candidate
            return ""
        else:
            return ""

        if isinstance(candidate, dict):
            candidate = candidate.get("label", candidate.get("value", candidate.get("choice", "")))
        if not isinstance(candidate, str):
            return ""
        normalized = candidate.strip().lower().replace(" ", "_").replace("-", "_")
        for intent in _ALLOWED_INTENTS:
            if normalized == intent or normalized.startswith(f"{intent}:"):
                return intent
        return ""

intent_classifier = IntentClassifier()
