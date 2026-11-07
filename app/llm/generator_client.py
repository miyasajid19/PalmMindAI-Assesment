from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import settings


class GeneratorLLMClient:
    """Creates the chat model used exclusively by the generation agent."""

    def get_llm(
        self,
        *,
        temperature: float = 0.1,
        max_output_tokens: int = 5000,
    ) -> ChatOpenAI:
        return ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )


generator_llm_client = GeneratorLLMClient()
