import pytest

from app.agents.Multi_query_generator.schemas import (
    GeneratedQuery,
    QueryVersionScope,
)
from app.rag.retrieval.candidate_document_generator import (
    CandidateDocumentGenerator,
)


class FakeSummaryStore:

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search_similar_summaries(
        self,
        summary,
        limit=20,
        *,
        offset=0,
        document_version_ids=None,
    ):
        self.calls.append({
            "query": summary,
            "limit": limit,
            "offset": offset,
            "document_version_ids": document_version_ids,
        })

        results = self.results.get(summary, [])
        if document_version_ids is not None:
            allowed = {str(value) for value in document_version_ids}
            results = [
                item for item in results
                if item.get("document_version_id") in allowed
            ]
        return results[offset:offset + limit]


@pytest.mark.asyncio
async def test_candidate_generation_filters_catalogue_and_scope():

    store = FakeSummaryStore({
        "active policy": [
            {
                "document_id": "doc-1",
                "document_version_id": "version-1",
                "status": "superseded",  # Intentionally stale.
                "similarity_score": 0.85,
            },
            {
                "document_id": "doc-2",
                "document_version_id": "unauthorized-version",
                "similarity_score": 0.95,
            },
            {
                "document_id": "doc-3",
                "document_version_id": "version-3",
                "similarity_score": 0.30,
            },
        ]
    })

    generator = CandidateDocumentGenerator(
        summary_store=store,
        similarity_threshold=0.40,
    )

    catalogue = [
        {
            "document_id": "doc-1",
            "document_version_id": "version-1",
            "document_title": "Privacy Policy",
            "document_summary": "Privacy requirements.",
            "version_year": 2026,
            "version_status": "active",
            "last_updated": "2026-01-01",
        }
    ]

    query = GeneratedQuery(
        query="active policy",
        version_scope=QueryVersionScope.ACTIVE,
        target_versions=[],
    )

    results = await generator.generate_candidate(
        query=query,
        document_catalogue=catalogue,
    )

    assert len(results.candidates) == 1

    candidate = results.candidates[0]

    assert candidate.document_version_id == "version-1"
    assert candidate.version_status == "active"
    assert candidate.similarity_score == 0.85


@pytest.mark.asyncio
async def test_target_version_filter():

    store = FakeSummaryStore({
        "policy": [
            {
                "document_id": "doc-1",
                "document_version_id": "version-2025",
                "similarity_score": 0.90,
            },
            {
                "document_id": "doc-1",
                "document_version_id": "version-2026",
                "similarity_score": 0.85,
            },
        ]
    })

    generator = CandidateDocumentGenerator(store)

    catalogue = [
        {
            "document_id": "doc-1",
            "document_version_id": "version-2025",
            "document_title": "Policy",
            "document_summary": "Older policy.",
            "version_year": 2025,
            "version_status": "superseded",
            "last_updated": "2025-01-01",
        },
        {
            "document_id": "doc-1",
            "document_version_id": "version-2026",
            "document_title": "Policy",
            "document_summary": "Current policy.",
            "version_year": 2026,
            "version_status": "active",
            "last_updated": "2026-01-01",
        },
    ]

    query = GeneratedQuery(
        query="policy",
        version_scope=QueryVersionScope.SUPERSEDED,
        target_versions=[2025],
    )

    results = await generator.generate_candidate(
        query=query,
        document_catalogue=catalogue,
    )

    assert len(results.candidates) == 1
    assert results.candidates[0].version_year == 2025


@pytest.mark.asyncio
async def test_candidate_generation_prefilters_catalogue_versions_before_top_k():
    hits = [
        {
            "document_id": f"private-doc-{index}",
            "document_version_id": f"private-version-{index}",
            "similarity_score": 0.99 - index * 0.001,
        }
        for index in range(20)
    ]
    hits.extend([
        {
            "document_id": "authorized-doc",
            "document_version_id": "authorized-version",
            "similarity_score": 0.80,
        },
        {
            "document_id": "authorized-doc-2",
            "document_version_id": "authorized-version-2",
            "similarity_score": 0.79,
        },
    ])
    store = FakeSummaryStore({"query": hits})
    generator = CandidateDocumentGenerator(
        summary_store=store,
        search_limit=20,
        max_search_limit=100,
        target_candidate_count=2,
    )
    catalogue = [
        {
            "document_id": "authorized-doc",
            "document_version_id": "authorized-version",
            "document_title": "Authorized Policy",
            "document_summary": "Relevant authorized policy.",
            "version_year": 2026,
            "version_status": "active",
        },
        {
            "document_id": "authorized-doc-2",
            "document_version_id": "authorized-version-2",
            "document_title": "Second Authorized Policy",
            "document_summary": "Another relevant policy.",
            "version_year": 2026,
            "version_status": "active",
        },
    ]
    query = GeneratedQuery(
        query="query",
        version_scope=QueryVersionScope.ACTIVE,
    )

    results = await generator.generate_candidate(
        query=query,
        document_catalogue=catalogue,
    )

    assert [c.document_version_id for c in results.candidates] == [
        "authorized-version",
        "authorized-version-2",
    ]
    # Qdrant receives the exact catalogue-scoped version allowlist before
    # applying its top-K limit; unauthorized summaries never occupy slots.
    assert len(store.calls) == 1
    assert set(store.calls[0]["document_version_ids"]) == {
        "authorized-version",
        "authorized-version-2",
    }
    assert store.calls[0]["offset"] == 0


@pytest.mark.asyncio
async def test_summary_search_allowlist_excludes_out_of_catalogue_and_wrong_scope_versions():
    hits = [
        {
            "document_id": f"unauthorized-doc-{index}",
            "document_version_id": f"unauthorized-version-{index}",
            "similarity_score": 0.99 - index * 0.001,
        }
        for index in range(15)
    ]
    hits.extend([
        {
            "document_id": f"old-doc-{index}",
            "document_version_id": f"superseded-version-{index}",
            "similarity_score": 0.90 - index * 0.01,
        }
        for index in range(3)
    ])
    hits.extend([
        {
            "document_id": f"allowed-doc-{index}",
            "document_version_id": f"allowed-version-{index}",
            "similarity_score": 0.80 - index * 0.01,
        }
        for index in range(2)
    ])

    catalogue = [
        {
            "document_id": f"old-doc-{index}",
            "document_version_id": f"superseded-version-{index}",
            "document_title": "Old policy",
            "document_summary": "Superseded summary",
            "version_year": 2025,
            "version_status": "superseded",
        }
        for index in range(3)
    ]
    catalogue.extend([
        {
            "document_id": f"allowed-doc-{index}",
            "document_version_id": f"allowed-version-{index}",
            "document_title": "Active policy",
            "document_summary": "Current summary",
            "version_year": 2026,
            "version_status": "active",
        }
        for index in range(2)
    ])

    store = FakeSummaryStore({"active policy": hits})
    generator = CandidateDocumentGenerator(
        summary_store=store,
        target_candidate_count=20,
    )
    query = GeneratedQuery(
        query="active policy",
        version_scope=QueryVersionScope.ACTIVE,
    )

    results = await generator.generate_candidate(
        query=query,
        document_catalogue=catalogue,
    )

    assert {
        candidate.document_version_id
        for candidate in results.candidates
    } == {"allowed-version-0", "allowed-version-1"}
    assert store.calls[0]["document_version_ids"] == [
        "allowed-version-0",
        "allowed-version-1",
    ]

@pytest.mark.parametrize(
    "threshold",
    [True, False, float("nan"), float("inf"), 1.5, "0.4"],
)
def test_candidate_generator_rejects_invalid_similarity_threshold(threshold):
    with pytest.raises(ValueError, match="similarity_threshold"):
        CandidateDocumentGenerator(
            summary_store=FakeSummaryStore({}),
            similarity_threshold=threshold,
        )
