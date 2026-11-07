from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage

from app.agents.Multi_query_generator.agent import MultiQueryGeneratorAgent
from app.agents.Multi_query_generator.schemas import AvailableDocument, AvailableDocuments, GeneratedQuery
from app.rag.retrieval.adaptive_k_selector import AdaptiveKSelector
from app.rag.retrieval.candidate_document_generator import CandidateDocumentGenerationError, CandidateDocumentGenerator
from app.rag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from app.rag.retrieval.pipeline import HybridRetrievalPipeline
from app.repositories.document_version_repositories import DocumentVersionRepository

logger = logging.getLogger(__name__)


class RetrievalOrchestrationError(Exception):
    """Raised when retrieval orchestration cannot produce a valid result."""


class RetrievalOrchestrator:
    """Coordinate one contextualized query, retrieval, one reranker, and adaptive selection."""

    def __init__(
        self,
        *,
        document_version_repository: DocumentVersionRepository,
        multi_query_agent: MultiQueryGeneratorAgent,
        candidate_document_generator: CandidateDocumentGenerator,
        hybrid_retrieval_pipeline: HybridRetrievalPipeline,
        cross_encoder_reranker: CrossEncoderReranker,
        adaptive_k_selector: AdaptiveKSelector,
    ) -> None:
        self.document_version_repository = document_version_repository
        self.multi_query_agent = multi_query_agent
        self.candidate_document_generator = candidate_document_generator
        self.hybrid_retrieval_pipeline = hybrid_retrieval_pipeline
        self.cross_encoder_reranker = cross_encoder_reranker
        self.adaptive_k_selector = adaptive_k_selector

    async def retrieve(
        self,
        *,
        current_query: str,
        conversation_history: Sequence[BaseMessage] | None = None,
        top_k: int = 25,
        candidate_k: int = 25,
        mode: str = "hybrid",
        rrf_k: int = 60,
        exclude: bool = False,
        excluded_chunk_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate one contextualized query, retrieve up to 25 chunks, rerank, and select 3–10."""
        if not isinstance(exclude, bool):
            raise ValueError("exclude must be a boolean.")
        if excluded_chunk_ids is not None and not isinstance(excluded_chunk_ids, list):
            raise ValueError("excluded_chunk_ids must be a list of strings or None.")
        self._validate_request(current_query=current_query, top_k=top_k, candidate_k=candidate_k, mode=mode, rrf_k=rrf_k)

        history = list(conversation_history or [])
        document_catalogue = self.document_version_repository.get_shared_retrieval_catalog()
        if not document_catalogue:
            return self._empty_response(current_query=current_query, top_k=top_k, candidate_k=candidate_k, mode=mode, reason="no_retrievable_document_versions")

        available_documents = self._build_available_documents(document_catalogue)
        generated_queries = await self.multi_query_agent.agenerate(
            current_query=current_query,
            conversation_history=history,
            available_documents=available_documents,
        )
        from rich import print
        print(f"\n\n\n\nGenerated Queries: {generated_queries}")
        if not generated_queries:
            raise RetrievalOrchestrationError(
                "The contextualized query generator returned no usable query."
            )
        generated_query = generated_queries[0]
        if not generated_query.query.strip():
            raise RetrievalOrchestrationError(
                "The contextualized query generator returned an empty query."
            )
        generated_queries = [generated_query]

        try:
            candidate_result = await self.candidate_document_generator.generate_candidate(
                query=generated_query,
                document_catalogue=document_catalogue,
            )
            print("\n\n\n\nCandidate Generation Result:", candidate_result)
        except CandidateDocumentGenerationError as exc:
            logger.exception("Candidate generation failed.")
            response = self._empty_response(current_query=current_query, top_k=top_k, candidate_k=candidate_k, mode=mode, reason="candidate_generation_failed")
            response["candidate_generation_error"] = str(exc)
            response["generated_queries"] = [self._query_to_dict(query) for query in generated_queries]
            response["expansion"]["stop_reason"] = "candidate_generation_failed"
            return response

        version_ids = [
            candidate.document_version_id
            for candidate in candidate_result.candidates
        ]


        print("\n\n\n\nCandidate Document Version IDs:", version_ids)
        if not version_ids:
            retrieved = []
        else:
            retrieval_result = await asyncio.to_thread(
                self.hybrid_retrieval_pipeline.search,
                query=candidate_result.query,
                top_k=10,
                candidate_k=10,
                mode=mode,
                rrf_k=rrf_k,
                status=None,
                candidate_document_version_ids=version_ids,
                exclude=exclude,
                excluded_chunk_ids=excluded_chunk_ids,
            )
            from rich import print
            print("\n\n\n\nRetrieval Result:", retrieval_result)
            retrieved = retrieval_result["results"]
        reranked = await asyncio.to_thread(self.cross_encoder_reranker.rerank, current_query, retrieved) if retrieved else []
        selection = self.adaptive_k_selector.select(reranked, expansion_round=0)
        print("\n\n\n\nAdaptive Selection Result:", selection)
        ans= {
            "query": current_query,
            "mode": mode,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "selected_candidates": selection["selected_candidates"],
            "selected_count": selection["selected_count"],
            "eligible_count": selection["eligible_count"],
            "fused_count": len(retrieved),
            "reranked_count": len(reranked),
            "generated_queries": [self._query_to_dict(query) for query in generated_queries],
            "expansion": {"rounds_performed": 0, "stop_reason": "single_pass"},
            "retrieval_reason": None,
            "candidate_generation_error": None,
        }
        print("Retrieval Orchestrator Response:", ans)
        return ans

    @staticmethod
    def _build_available_documents(document_catalogue: list[dict[str, Any]]) -> AvailableDocuments:
        active: list[AvailableDocument] = []
        superseded: list[AvailableDocument] = []
        for entry in document_catalogue:
            item = AvailableDocument(document_title=entry["document_title"], document_summary=entry["document_summary"], version_year=entry["version_year"], version_status=entry["version_status"])
            if item.version_status == "active":
                active.append(item)
            elif item.version_status == "superseded":
                superseded.append(item)
            else:
                raise ValueError(f"Unsupported document version status: {item.version_status}")
        return AvailableDocuments(active_documents=active, superseded_documents=superseded)

    @staticmethod
    def _query_to_dict(query: GeneratedQuery) -> dict[str, Any]:
        return {"query": query.query, "version_scope": query.version_scope.value, "target_versions": list(query.target_versions)}

    @staticmethod
    def _validate_request(*, current_query: str, top_k: int, candidate_k: int, mode: str, rrf_k: int) -> None:
        if not isinstance(current_query, str) or not current_query.strip():
            raise ValueError("current_query cannot be empty.")
        for name, value in (("top_k", top_k), ("candidate_k", candidate_k), ("rrf_k", rrf_k)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if mode not in {"hybrid", "bm25", "dense"}:
            raise ValueError("mode must be hybrid, bm25, or dense.")

    @staticmethod
    def _empty_response(*, current_query: str, top_k: int, candidate_k: int, mode: str, reason: str) -> dict[str, Any]:
        return {
            "query": current_query, "mode": mode, "top_k": top_k, "candidate_k": candidate_k,
            "selected_candidates": [], "selected_count": 0, "eligible_count": 0,
            "fused_count": 0, "reranked_count": 0, "generated_queries": [],
            "expansion": {"rounds_performed": 0, "stop_reason": reason},
            "retrieval_reason": reason, "candidate_generation_error": None,
        }
