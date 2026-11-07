"""Jina API-backed LangChain embedding adapter for SaviraVoice."""

from __future__ import annotations

from typing import Sequence

import httpx

from langchain_core.embeddings import Embeddings

from app.core.config import settings


class JinaEmbeddingClient(Embeddings):
    """LangChain Embeddings-compatible client backed by Jina's hosted API."""

    DEFAULT_MODEL = "jina-embeddings-v4"
    DEFAULT_DIMENSIONS = 2048

    def __init__(
        self,
        model_name: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        timeout_seconds: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        model_name = model_name or settings.JINA_EMBEDDING_MODEL or self.DEFAULT_MODEL
        dimensions = dimensions if dimensions is not None else settings.JINA_EMBEDDING_DIMENSIONS
        batch_size = batch_size if batch_size is not None else settings.JINA_EMBEDDING_BATCH_SIZE
        timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.JINA_API_TIMEOUT_SECONDS
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("dimensions must be a positive integer.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        self.model_name = model_name.strip()
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.api_key = api_key if api_key is not None else settings.JINA_API_KEY
        self.base_url = (base_url or settings.JINA_API_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client

    def _client(self) -> httpx.Client:
        if self._http_client is None:
            if not self.api_key:
                raise RuntimeError("JINA_API_KEY is required for Jina embeddings.")
            self._http_client = httpx.Client(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout_seconds,
            )
        return self._http_client

    def _embed_batch(self, texts: list[str], task: str) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self.model_name,
            "input": texts,
            "task": task,
            "dimensions": self.dimensions,
            "truncate": True,
        }
        response = self._client().post(f"{self.base_url}/embeddings", json=payload)
        response.raise_for_status()
        data = response.json().get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("Jina embeddings API returned an unexpected item count.")

        indexed: list[tuple[int, list[float]]] = []
        for fallback_index, item in enumerate(data):
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise RuntimeError("Jina embeddings API returned an invalid embedding item.")
            index = item.get("index", fallback_index)
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(texts):
                raise RuntimeError("Jina embeddings API returned an invalid embedding index.")
            vector = item["embedding"]
            if len(vector) != self.dimensions:
                raise RuntimeError(
                    f"Jina embedding dimension mismatch: expected {self.dimensions}, got {len(vector)}."
                )
            indexed.append((index, [float(value) for value in vector]))

        indexed.sort(key=lambda pair: pair[0])
        if [index for index, _ in indexed] != list(range(len(texts))):
            raise RuntimeError("Jina embeddings API returned duplicate or missing indices.")
        return [vector for _, vector in indexed]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed non-empty texts as retrieval passages, preserving their order."""
        cleaned = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
        vectors: list[list[float]] = []
        for start in range(0, len(cleaned), self.batch_size):
            vectors.extend(self._embed_batch(cleaned[start : start + self.batch_size], "retrieval.passage"))
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed strings as retrieval passages through LangChain's standard interface."""
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a query with Jina's retrieval.query task adapter."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        return self._embed_batch([query.strip()], "retrieval.query")[0]

    def get_embedding_model(self) -> Embeddings:
        """Return this object through the standard LangChain Embeddings interface."""
        return self


embedding_client = JinaEmbeddingClient()
embedding_model = embedding_client.get_embedding_model()
