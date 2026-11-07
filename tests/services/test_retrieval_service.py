import inspect

import pytest

from app.agents.Multi_query_generator.schemas import GeneratedQuery, QueryVersionScope
from app.rag.retrieval.candidate_document_generator import (
    CandidateDocument,
    CandidateDocumentGenerationError,
    QueryCandidateResult,
)
from app.services.retrieval.retrieval_service import RetrievalOrchestrator


class FakeDocumentVersionRepository:
    def __init__(self, catalogue):
        self.catalogue = catalogue

    def get_shared_retrieval_catalog(self):
        return self.catalogue


class FakeMultiQueryAgent:
    def __init__(self):
        self.calls = 0

    async def agenerate(self, **kwargs):
        self.calls += 1
        return [GeneratedQuery(query="contextualized query", version_scope=QueryVersionScope.ACTIVE)]


class FakeCandidateGenerator:
    async def generate_candidate(self, *, query, document_catalogue):
        return QueryCandidateResult(
            query=query.query,
            version_scope=query.version_scope,
            target_versions=query.target_versions,
            candidates=[
                CandidateDocument(
                    document_id="document-1",
                    document_version_id=f"version-{query.query}",
                    document_title="Document",
                    document_summary="Summary",
                    version_year=2026,
                    version_status="active",
                    similarity_score=0.9,
                )
            ],
        )


class FakeHybridPipeline:
    def __init__(self):
        self.calls = []

    def search(self, *, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        index = 1 if query == "initial query" else 2
        return {
            "result_count": 1,
            "results": [{
                "rank": 1,
                "score": 0.9,
                "page_content": f"content {index}",
                "metadata": {
                    "document_version_id": f"version-{query}",
                    "chunk_index": index,
                },
            }],
        }


class FakeReranker:
    def rerank(self, query, candidates):
        return [dict(candidate, reranker_score=0.9) for candidate in candidates]


class FakeSelector:
    k_min = 2
    k_max = 5
    score_threshold = 0.6
    max_expansion_rounds = 1

    def select(self, candidates, expansion_round=0):
        selected = list(candidates[: self.k_max])
        return {
            "selected_candidates": selected,
            "selected_count": len(selected),
            "eligible_count": len(selected),
            "expansion_required": len(selected) < self.k_min and expansion_round < self.max_expansion_rounds,
            "expansion_round": expansion_round,
            "max_expansion_rounds": self.max_expansion_rounds,
            "score_threshold": self.score_threshold,
        }


def build_orchestrator(catalogue, candidate_generator=None, hybrid_pipeline=None):
    return RetrievalOrchestrator(
        document_version_repository=FakeDocumentVersionRepository(catalogue),
        multi_query_agent=FakeMultiQueryAgent(),
        candidate_document_generator=candidate_generator or FakeCandidateGenerator(),
        hybrid_retrieval_pipeline=hybrid_pipeline or FakeHybridPipeline(),
        cross_encoder_reranker=FakeReranker(),
        adaptive_k_selector=FakeSelector(),
    )


def sample_catalogue():
    return [{
        "document_id": "document-1",
        "document_version_id": "version-1",
        "document_title": "Document",
        "document_summary": "Summary",
        "version_year": 2026,
        "version_status": "active",
        "last_updated": None,
    }]


@pytest.mark.asyncio
async def test_orchestrator_retrieves_once_without_query_expansion():
    agent = FakeMultiQueryAgent()
    orchestrator = build_orchestrator(sample_catalogue())
    orchestrator.multi_query_agent = agent

    result = await orchestrator.retrieve(current_query="What is the policy?")

    assert agent.calls == 1
    assert result["expansion"]["rounds_performed"] == 0
    assert result["expansion"]["stop_reason"] == "single_pass"
    assert result["selected_count"] == 1
    assert len(result["generated_queries"]) == 1


def test_catalogue_is_split_into_active_and_superseded_models():
    catalogue = [
        {"document_id":"d1","document_version_id":"v1","document_title":"Active","document_summary":"Active summary","version_year":2026,"version_status":"active"},
        {"document_id":"d1","document_version_id":"v0","document_title":"Old","document_summary":"Old summary","version_year":2025,"version_status":"superseded"},
    ]
    available = RetrievalOrchestrator._build_available_documents(catalogue)
    assert len(available.active_documents) == 1
    assert len(available.superseded_documents) == 1
    assert available.active_documents[0].version_year == 2026


@pytest.mark.asyncio
async def test_candidate_generation_failure_is_reported_explicitly():
    class FailingCandidateGenerator:
        async def generate_candidate(self, **kwargs):
            raise CandidateDocumentGenerationError("summary search unavailable")

    result = await build_orchestrator(
        sample_catalogue(), candidate_generator=FailingCandidateGenerator()
    ).retrieve(current_query="What is the policy?")
    assert result["selected_candidates"] == []
    assert result["retrieval_reason"] == "candidate_generation_failed"
    assert result["candidate_generation_error"] == "summary search unavailable"


@pytest.mark.asyncio
async def test_orchestrator_forwards_chunk_exclusions():
    pipeline = FakeHybridPipeline()
    await build_orchestrator(sample_catalogue(), hybrid_pipeline=pipeline).retrieve(
        current_query="What is the policy?",
        exclude=True,
        excluded_chunk_ids=["previous:12"],
    )
    assert pipeline.calls
    assert all(call["exclude"] is True for call in pipeline.calls)
    assert all(call["excluded_chunk_ids"] == ["previous:12"] for call in pipeline.calls)


def test_retrieval_orchestrator_has_no_query_worker_fanout():
    assert not hasattr(RetrievalOrchestrator, "_retrieve_candidate_branches")
    assert "retrieval_concurrency" not in inspect.signature(
        RetrievalOrchestrator.__init__
    ).parameters
