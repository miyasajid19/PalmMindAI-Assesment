from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import settings


class RetrievalLLMClient:
    """Creates the chat model for retrieval-specific LLM operations."""

    def get_llm(
        self,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1500,
    ) -> ChatOpenAI:
        return ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )


retrieval_llm_client = RetrievalLLMClient()
