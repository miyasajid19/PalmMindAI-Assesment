from __future__ import annotations

import pytest

from app.agents.Multi_query_generator.schemas import GeneratedQuery, QueryVersionScope
from app.rag.retrieval.adaptive_k_selector import AdaptiveKSelector
from app.rag.retrieval.candidate_document_generator import CandidateDocument, QueryCandidateResult
from app.services.retrieval.retrieval_service import RetrievalOrchestrator


def catalogue_entry(version_id: str) -> dict:
    return {
        "document_id": "doc-1",
        "document_version_id": version_id,
        "document_title": "Shared policy",
        "document_summary": "A test policy document.",
        "version_year": 2026,
        "version_status": "active",
        "last_updated": None,
    }


def candidate(version_id: str) -> CandidateDocument:
    return CandidateDocument(
        document_id="doc-1",
        document_version_id=version_id,
        document_title="Shared policy",
        document_summary="A test policy document.",
        version_year=2026,
        version_status="active",
        similarity_score=0.9,
    )


def generated(query: str) -> GeneratedQuery:
    return GeneratedQuery(query=query, version_scope=QueryVersionScope.ACTIVE)


class FakeDocumentVersionRepository:
    def __init__(self, catalogue):
        self.catalogue = catalogue
        self.calls = 0

    def get_shared_retrieval_catalog(self):
        self.calls += 1
        return self.catalogue


class ScriptedAgent:
    def __init__(self, batches):
        self.batches, self.calls = list(batches), []

    async def agenerate(self, **kwargs):
        self.calls.append(kwargs)
        return self.batches.pop(0) if self.batches else []


class ScriptedCandidateGenerator:
    def __init__(self, batches):
        self.batches, self.calls = list(batches), []

    async def generate_candidate(self, *, query, document_catalogue):
        self.calls.append({
            "query": query,
            "document_catalogue": list(document_catalogue),
        })
        batch = self.batches.pop(0) if self.batches else []
        matching = next(
            (item for item in batch if item.query == query.query),
            None,
        )
        return matching or (batch[0] if batch else None)


class RecordingHybridPipeline:
    def __init__(self, results_by_query):
        self.results_by_query, self.calls = results_by_query, []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.results_by_query[kwargs["query"]]


class DeterministicReranker:
    def rerank(self, query, candidates):
        return [
            dict(item, reranker_score=item["metadata"].get("test_reranker_score", 0.95))
            for item in candidates
        ]


def retrieval_result(version_id: str, chunk_index: int, content: str):
    return {
        "result_count": 1,
        "results": [{
            "rank": 1,
            "score": 0.8,
            "page_content": content,
            "metadata": {
                "document_id": "doc-1",
                "document_version_id": version_id,
                "chunk_index": chunk_index,
                "test_reranker_score": 0.95,
            },
        }],
    }


def make_orchestrator(*, catalogue, query_batches, candidate_batches, results_by_query):
    repository = FakeDocumentVersionRepository(catalogue)
    agent = ScriptedAgent(query_batches)
    candidate_generator = ScriptedCandidateGenerator(candidate_batches)
    pipeline = RecordingHybridPipeline(results_by_query)
    orchestrator = RetrievalOrchestrator(
        document_version_repository=repository,
        multi_query_agent=agent,
        candidate_document_generator=candidate_generator,
        hybrid_retrieval_pipeline=pipeline,
        cross_encoder_reranker=DeterministicReranker(),
        adaptive_k_selector=AdaptiveKSelector(
            k_min=2, k_max=5, score_threshold=0.6, max_expansion_rounds=1,
        ),
    )
    return orchestrator, repository, agent, candidate_generator, pipeline


@pytest.mark.asyncio
async def test_shared_catalogue_is_passed_to_candidate_generation():
    shared_catalogue = [catalogue_entry("shared-v1")]
    orchestrator, repository, _, candidate_gen, pipeline = make_orchestrator(
        catalogue=shared_catalogue,
        query_batches=[[generated("policy query")]],
        candidate_batches=[[
            QueryCandidateResult(
                query="policy query", version_scope=QueryVersionScope.ACTIVE,
                target_versions=[], candidates=[candidate("shared-v1")],
            )
        ]],
        results_by_query={"policy query": retrieval_result("shared-v1", 0, "Shared content")},
    )

    result = await orchestrator.retrieve(
        current_query="What does the policy say?", top_k=2,
    )

    assert repository.calls == 1
    assert candidate_gen.calls[0]["document_catalogue"] == shared_catalogue
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["candidate_document_version_ids"] == ["shared-v1"]
    assert pipeline.calls[0]["status"] is None
    assert result["selected_count"] == 1
    assert result["selected_candidates"][0]["metadata"]["document_version_id"] == "shared-v1"


@pytest.mark.asyncio
async def test_empty_shared_catalogue_short_circuits_downstream_components():
    orchestrator, _, agent, candidate_gen, pipeline = make_orchestrator(
        catalogue=[], query_batches=[[generated("must not run")]],
        candidate_batches=[[]], results_by_query={},
    )
    result = await orchestrator.retrieve(current_query="Find a document")

    assert result["retrieval_reason"] == "no_retrievable_document_versions"
    assert result["selected_candidates"] == []
    assert agent.calls == []
    assert candidate_gen.calls == []
    assert pipeline.calls == []


@pytest.mark.asyncio
async def test_retrieval_pipeline_uses_only_one_contextualized_query():
    version_id = "shared-v1"
    branch_results = [
        QueryCandidateResult(
            query=query, version_scope=QueryVersionScope.ACTIVE,
            target_versions=[], candidates=[candidate(version_id)],
        )
        for query in ("policy wording", "policy requirements")
    ]
    orchestrator, _, _, _, pipeline = make_orchestrator(
        catalogue=[catalogue_entry(version_id)],
        query_batches=[[generated("policy wording"), generated("policy requirements")]],
        candidate_batches=[branch_results],
        results_by_query={
            "policy wording": retrieval_result(version_id, 3, "Same chunk"),
            "policy requirements": retrieval_result(version_id, 3, "Same chunk"),
        },
    )

    result = await orchestrator.retrieve(current_query="Explain the policy", top_k=2)

    assert len(pipeline.calls) == 1
    assert result["fused_count"] == 1
    assert result["reranked_count"] == 1
    assert result["selected_count"] == 1
    assert len(result["generated_queries"]) == 1


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_adaptive_selector_does_not_trigger_additional_query_generation():
    version_id = "shared-v1"
    initial = [QueryCandidateResult(
        query="initial query", version_scope=QueryVersionScope.ACTIVE,
        target_versions=[], candidates=[candidate(version_id)],
    )]
    orchestrator, _, agent, candidate_gen, pipeline = make_orchestrator(
        catalogue=[catalogue_entry(version_id)],
        query_batches=[[generated("initial query")], [generated("must not be used")]],
        candidate_batches=[initial, []],
        results_by_query={"initial query": retrieval_result(version_id, 0, "First evidence")},
    )

    result = await orchestrator.retrieve(
        current_query="Explain the policy comprehensively", top_k=2,
    )

    assert len(agent.calls) == 1
    assert len(candidate_gen.calls) == 1
    assert len(pipeline.calls) == 1
    assert result["expansion"]["rounds_performed"] == 0
    assert result["expansion"]["stop_reason"] == "single_pass"
    assert result["selected_count"] == 1
    assert result["selected_candidates"][0]["metadata"]["chunk_index"] == 0


@pytest.mark.asyncio
async def test_unexpected_hybrid_pipeline_exception_propagates():
    class BrokenPipeline:
        def search(self, **kwargs):
            raise RuntimeError("Qdrant unavailable")

    version_id = "shared-v1"
    orchestrator, _, _, _, _ = make_orchestrator(
        catalogue=[catalogue_entry(version_id)],
        query_batches=[[generated("policy query")]],
        candidate_batches=[[
            QueryCandidateResult(
                query="policy query", version_scope=QueryVersionScope.ACTIVE,
                target_versions=[], candidates=[candidate(version_id)],
            )
        ]],
        results_by_query={},
    )
    orchestrator.hybrid_retrieval_pipeline = BrokenPipeline()

    with pytest.raises(RuntimeError, match="Qdrant unavailable"):
        await orchestrator.retrieve(current_query="Explain the policy", top_k=1)
