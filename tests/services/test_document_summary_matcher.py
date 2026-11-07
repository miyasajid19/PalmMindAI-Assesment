
from unittest.mock import MagicMock
import pytest

from app.services.document_matching import DocumentSummaryMatcher


@pytest.fixture
def summary_store():
    return MagicMock()


@pytest.fixture
def matcher(summary_store):
    return DocumentSummaryMatcher(
        summary_store=summary_store,
        threshold=0.80,
        candidate_limit=20,
    )



def make_result(
    version_id="version-1",
    score=0.90,
    status="active",
):
    return {
        "document_id": "document-1",
        "document_version_id": version_id,
        "version_number": 1,
        "title": "Employee Leave Policy",
        "summary": "Policy describing employee leave rules.",
        "status": status,
        "score": score,
    }


def test_similar_summary_is_matched(matcher, summary_store):
    summary_store.search_similar_summaries.return_value = [
        make_result(score=0.91)
    ]

    result = matcher.match(
        "Employee leave policy and eligibility.",
    )

    assert result.is_match is True
    assert result.match_count == 1
    assert result.candidates[0].similarity_score == 0.91


def test_result_below_threshold_is_ignored(matcher, summary_store):
    summary_store.search_similar_summaries.return_value = [
        make_result(score=0.79)
    ]

    result = matcher.match(
        "A completely different document.",
    )

    assert result.is_match is False
    assert result.match_count == 0


def test_result_at_exact_threshold_is_matched(matcher, summary_store):
    summary_store.search_similar_summaries.return_value = [
        make_result(score=0.80)
    ]

    result = matcher.match(
        "Employee leave policy.",
    )

    assert result.is_match is True
    assert result.match_count == 1


def test_multiple_candidates_are_returned(matcher, summary_store):
    summary_store.search_similar_summaries.return_value = [
        make_result("version-1", 0.91),
        make_result("version-2", 0.87),
        make_result("version-3", 0.82),
    ]

    result = matcher.match(
        "Employee leave policy.",
    )

    assert result.match_count == 3
    assert [c.version_id for c in result.candidates] == [
        "version-1",
        "version-2",
        "version-3",
    ]


def test_candidates_are_sorted_by_similarity(matcher, summary_store):
    summary_store.search_similar_summaries.return_value = [
        make_result("version-1", 0.82),
        make_result("version-2", 0.95),
        make_result("version-3", 0.88),
    ]

    result = matcher.match(
        "Employee leave policy.",
    )

    scores = [c.similarity_score for c in result.candidates]

    assert scores == [0.95, 0.88, 0.82]


def test_archived_document_version_is_still_a_match(matcher, summary_store):
    summary_store.search_similar_summaries.return_value = [
        make_result(score=0.91, status="superseded")
    ]

    result = matcher.match(
        "Employee leave policy.",
    )

    assert result.is_match is True
    assert result.candidates[0].status == "superseded"


def test_duplicate_version_ids_are_deduplicated(matcher, summary_store):
    summary_store.search_similar_summaries.return_value = [
        make_result("version-1", 0.85),
        make_result("version-1", 0.93),
    ]

    result = matcher.match(
        "Employee leave policy.",
    )

    assert result.match_count == 1
    assert result.candidates[0].similarity_score == 0.93


def test_empty_search_results(matcher, summary_store):
    summary_store.search_similar_summaries.return_value = []

    result = matcher.match(
        "New document summary.",
    )

    assert result.is_match is False
    assert result.candidates == ()


def test_empty_summary_is_rejected(matcher):
    with pytest.raises(ValueError):
        matcher.match("   ")


def test_invalid_threshold_is_rejected(summary_store):
    with pytest.raises(ValueError):
        DocumentSummaryMatcher(summary_store, threshold=1.5)


def test_invalid_candidate_limit_is_rejected(summary_store):
    with pytest.raises(ValueError):
        DocumentSummaryMatcher(summary_store, candidate_limit=0)


def test_search_limit_is_passed_to_store(matcher, summary_store):
    summary_store.search_similar_summaries.return_value = []

    matcher.match(
        "Employee leave policy.",
    )

    summary_store.search_similar_summaries.assert_called_once_with(
        summary="Employee leave policy.",
        limit=20,
    )

@pytest.mark.parametrize(
    "threshold",
    [True, False, float("nan"), float("inf"), 1.5, "0.8"],
)
def test_summary_matcher_rejects_invalid_threshold_values(
    summary_store,
    threshold,
):
    with pytest.raises(ValueError, match="threshold"):
        DocumentSummaryMatcher(summary_store, threshold=threshold)
