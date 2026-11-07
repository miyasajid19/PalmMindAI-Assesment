from uuid import uuid4

import pytest

from app.services.document_matching import (
    DocumentMatchingOrchestrator,
    HashMatchCandidate,
    HashMatchResult,
    SummaryMatchCandidate,
    SummaryMatchResult,
)


class FakeHashMatcher:

    def __init__(self, result):
        self.result = result
        self.called = False

    def match(self, file_hash, content_hash):
        self.called = True
        return self.result


class FakeSummaryMatcher:

    def __init__(self, result):
        self.result = result
        self.called = False
        self.received_summary = None

    def match(self, incoming_summary):
        self.called = True
        self.received_summary = incoming_summary
        return self.result


def create_hash_candidate(matched_by):
    return HashMatchCandidate(
        document_id=uuid4(),
        version_id=uuid4(),
        version_number=1,
        status="active",
        original_filename="policy.pdf",
        matched_by=matched_by,
    )


def create_summary_candidate(score=0.91):
    return SummaryMatchCandidate(
        document_id=str(uuid4()),
        version_id=str(uuid4()),
        version_number=1,
        title="Employee Leave Policy",
        status="active",
        summary="Employee leave and absence policy.",
        similarity_score=score,
    )


def create_orchestrator(
    hash_candidates=None,
    summary_candidates=None,
):
    hash_result = HashMatchResult(
        is_duplicate=bool(hash_candidates),
        candidates=hash_candidates or [],
    )

    summary_result = SummaryMatchResult(
        is_match=bool(summary_candidates),
        candidates=tuple(summary_candidates or []),
    )

    hash_matcher = FakeHashMatcher(hash_result)
    summary_matcher = FakeSummaryMatcher(summary_result)

    orchestrator = DocumentMatchingOrchestrator(
        hash_matcher=hash_matcher,
        summary_matcher=summary_matcher,
    )

    return orchestrator, hash_matcher, summary_matcher


def test_exact_file_match_has_highest_priority():
    file_candidate = create_hash_candidate("file_hash")
    content_candidate = create_hash_candidate("content_hash")
    semantic_candidate = create_summary_candidate()

    orchestrator, _, summary_matcher = create_orchestrator(
        hash_candidates=[
            file_candidate,
            content_candidate,
        ],
        summary_candidates=[semantic_candidate],
    )

    result = orchestrator.match(
        file_hash="file123",
        content_hash="content123",
        summary="Employee leave policy.",
    )

    assert result.classification == "exact_file_match"
    assert result.has_exact_match is True
    assert result.requires_user_review is True

    assert result.exact_file_matches == (file_candidate,)
    assert result.exact_content_matches == (content_candidate,)

    # Exact matches short-circuit the semantic Qdrant search.
    assert result.semantic_candidates == ()
    assert summary_matcher.called is False


def test_exact_content_match_when_file_hash_does_not_match():
    content_candidate = create_hash_candidate("content_hash")

    orchestrator, _, summary_matcher = create_orchestrator(
        hash_candidates=[content_candidate],
    )

    result = orchestrator.match(
        file_hash="different_file_hash",
        content_hash="matching_content_hash",
        summary="Employee leave policy.",
    )

    assert result.classification == "exact_content_match"
    assert result.has_exact_match is True
    assert result.requires_user_review is True
    assert summary_matcher.called is False


def test_potential_match_when_no_exact_hash_match_exists():
    semantic_candidate = create_summary_candidate()

    orchestrator, _, _ = create_orchestrator(
        summary_candidates=[semantic_candidate],
    )

    result = orchestrator.match(
        file_hash="new_file_hash",
        content_hash="new_content_hash",
        summary="Employee leave and absence policy.",
    )

    assert result.classification == "potential_match"
    assert result.has_exact_match is False
    assert result.has_potential_match is True
    assert result.requires_user_review is True


def test_no_match_when_no_candidates_exist():
    orchestrator, _, _ = create_orchestrator()

    result = orchestrator.match(
        file_hash="new_file_hash",
        content_hash="new_content_hash",
        summary="Completely unrelated document.",
    )

    assert result.classification == "no_match"
    assert result.is_new_document is True
    assert result.requires_user_review is False
    assert result.exact_match_count == 0
    assert result.semantic_match_count == 0


def test_exact_match_takes_precedence_over_semantic_match():
    file_candidate = create_hash_candidate("file_hash")
    semantic_candidate = create_summary_candidate()

    orchestrator, _, summary_matcher = create_orchestrator(
        hash_candidates=[file_candidate],
        summary_candidates=[semantic_candidate],
    )

    result = orchestrator.match(
        file_hash="matching_file_hash",
        content_hash="matching_content_hash",
        summary="Employee leave policy.",
    )

    assert result.classification == "exact_file_match"
    assert result.semantic_match_count == 0
    assert result.semantic_candidates == ()
    assert summary_matcher.called is False


def test_matchers_receive_correct_inputs():
    orchestrator, hash_matcher, summary_matcher = create_orchestrator()

    result = orchestrator.match(
        file_hash="file_hash_value",
        content_hash="content_hash_value",
        summary="  Employee leave policy.  ",
    )

    assert hash_matcher.called is True
    assert summary_matcher.called is True
    assert summary_matcher.received_summary == "Employee leave policy."

    assert result.classification == "no_match"


def test_matching_requires_at_least_one_hash():
    orchestrator, hash_matcher, summary_matcher = create_orchestrator()

    with pytest.raises(ValueError):
        orchestrator.match(
            file_hash=None,
            content_hash=None,
            summary="Employee leave policy.",
        )

    assert hash_matcher.called is False
    assert summary_matcher.called is False


def test_matching_rejects_empty_summary():
    orchestrator, hash_matcher, summary_matcher = create_orchestrator()

    with pytest.raises(ValueError):
        orchestrator.match(
            file_hash="file_hash",
            content_hash="content_hash",
            summary="   ",
        )

    assert hash_matcher.called is False
    assert summary_matcher.called is False


def test_multiple_exact_file_matches_are_preserved():
    first = create_hash_candidate("file_hash")
    second = create_hash_candidate("file_hash")

    orchestrator, _, _ = create_orchestrator(
        hash_candidates=[first, second],
    )

    result = orchestrator.match(
        file_hash="matching_hash",
        content_hash="different_content",
        summary="Employee leave policy.",
    )

    assert result.classification == "exact_file_match"
    assert result.exact_match_count == 2
    assert result.exact_file_matches == (first, second)


def test_result_properties():
    file_candidate = create_hash_candidate("file_hash")
    content_candidate = create_hash_candidate("content_hash")
    semantic_candidate = create_summary_candidate()

    orchestrator, _, summary_matcher = create_orchestrator(
        hash_candidates=[file_candidate, content_candidate],
        summary_candidates=[semantic_candidate],
    )

    result = orchestrator.match(
        file_hash="file_hash",
        content_hash="content_hash",
        summary="Employee leave policy.",
    )

    assert result.has_exact_match is True
    assert result.has_potential_match is False
    assert result.is_new_document is False
    assert result.exact_match_count == 2
    assert result.semantic_match_count == 0
    assert summary_matcher.called is False

