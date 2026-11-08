"""Jina API-backed reranker preserving SaviraVoice's candidate contract."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Sequence

import httpx

from app.core.config import settings


class CrossEncoderReranker:
    """Rerank retrieved chunks with Jina's hosted listwise reranker API.

    The public class/method contract is retained for existing orchestration and
    tests. Passing a predict-compatible model remains supported as a test seam.
    """

    DEFAULT_MODEL = "jina-reranker-v3.5"

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        device: str | None = None,
        model: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 90.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        model_name = model_name or settings.JINA_RERANKER_MODEL or self.DEFAULT_MODEL
        timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.JINA_API_TIMEOUT_SECONDS
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be a positive integer.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        self.model_name = model_name.strip()
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = model
        self.api_key = api_key if api_key is not None else settings.JINA_API_KEY
        self.base_url = (base_url or settings.JINA_API_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client

    def _client(self) -> httpx.Client:
        if self._http_client is None:
            if not self.api_key:
                raise RuntimeError("JINA_API_KEY is required for Jina reranking.")
            self._http_client = httpx.Client(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout_seconds,
            )
        return self._http_client

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidates ordered by Jina relevance scores."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty.")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise ValueError("candidates must be a sequence.")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer.")
        if not candidates:
            return []

        normalized = [
            self._validate_candidate(candidate, index)
            for index, candidate in enumerate(candidates)
        ]
        if self.model is not None:
            return self._rerank_with_predict_model(query.strip(), normalized, top_k)

        payload = {
            "model": self.model_name,
            "query": query.strip(),
            "documents": [candidate["page_content"] for candidate in normalized],
            "top_n": top_k if top_k is not None else len(normalized),
            "return_documents": False,
        }
        response = self._client().post(f"{self.base_url}/rerank", json=payload)
        response.raise_for_status()
        results = response.json().get("results")
        if not isinstance(results, list):
            raise RuntimeError("Jina reranker API returned an invalid results list.")

        reranked: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                raise RuntimeError("Jina reranker API returned an invalid result item.")
            index = item.get("index")
            score = item.get("relevance_score")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(normalized):
                raise RuntimeError("Jina reranker API returned an invalid candidate index.")
            if index in seen_indices:
                raise RuntimeError("Jina reranker API returned duplicate candidate indices.")
            if isinstance(score, bool) or not isinstance(score, Real):
                raise RuntimeError("Jina reranker API returned a non-numeric relevance score.")
            score = float(score)
            if not math.isfinite(score):
                raise RuntimeError("Jina reranker API returned a non-finite relevance score.")
            seen_indices.add(index)

            enriched = dict(normalized[index])
            enriched["metadata"] = dict(normalized[index].get("metadata") or {})
            enriched["raw_reranker_score"] = score
            enriched["reranker_score"] = self._normalize_score(score)
            reranked.append(enriched)
        if top_k is None and len(reranked) != len(normalized):
            raise RuntimeError("Jina reranker API returned a score count different from the candidate count.")
        reranked= reranked[:top_k] if top_k is not None else reranked
        return reranked
    @staticmethod
    def _normalize_score(score: float) -> float:
        """Keep probability-like API scores; sigmoid-transform raw logits to [0, 1]."""
        if 0.0 <= score <= 1.0:
            return score
        if score >= 0.0:
            return 1.0 / (1.0 + math.exp(-score))
        exp_score = math.exp(score)
        return exp_score / (1.0 + exp_score)

    def _rerank_with_predict_model(
        self,
        query: str,
        normalized: list[dict[str, Any]],
        top_k: int | None,
    ) -> list[dict[str, Any]]:
        pairs = [(query, candidate["page_content"]) for candidate in normalized]
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        try:
            scores = list(raw_scores)
        except TypeError as exc:
            raise ValueError("Cross-encoder returned invalid scores.") from exc
        if len(scores) != len(normalized):
            raise ValueError("Cross-encoder returned a score count different from the candidate count.")

        reranked = []
        for candidate, score in zip(normalized, scores):
            if isinstance(score, bool) or not isinstance(score, Real):
                raise ValueError("Cross-encoder scores must be numeric.")
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("Cross-encoder scores must be finite.")
            enriched = dict(candidate)
            enriched["metadata"] = dict(candidate.get("metadata") or {})
            enriched["raw_reranker_score"] = score
            enriched["reranker_score"] = self._normalize_score(score)
            reranked.append(enriched)

        reranked.sort(
            key=lambda candidate: (
                -candidate["reranker_score"],
                candidate.get("rank", float("inf")),
                str(candidate.get("chunk_id", "")),
            )
        )
        return reranked[:top_k] if top_k is not None else reranked

    @staticmethod
    def _validate_candidate(candidate: dict[str, Any], index: int) -> dict[str, Any]:
        if not isinstance(candidate, dict):
            raise ValueError(f"Candidate {index} must be a dictionary.")

        required = {"rank", "score", "page_content", "metadata"}
        missing = required - candidate.keys()
        if missing:
            raise ValueError(f"Candidate {index} is missing fields: {sorted(missing)}")

        rank = candidate["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError(f"Candidate {index} rank must be a positive integer.")
        if not isinstance(candidate["page_content"], str) or not candidate["page_content"].strip():
            raise ValueError(f"Candidate {index} page_content must be a non-empty string.")
        if not isinstance(candidate["metadata"], dict):
            raise ValueError(f"Candidate {index} metadata must be a dictionary.")

        normalized = dict(candidate)
        normalized["metadata"] = dict(candidate["metadata"])
        metadata = normalized["metadata"]
        if "chunk_id" not in normalized:
            version_id = metadata.get("document_version_id") or metadata.get("document_id") or metadata.get("source", "")
            chunk_index = metadata.get("chunk_index")
            normalized["chunk_id"] = (
                f"{version_id}:{chunk_index}"
                if chunk_index is not None
                else f"{version_id}:{metadata.get('page', '')}:{candidate['page_content']}"
            )
        return normalized
