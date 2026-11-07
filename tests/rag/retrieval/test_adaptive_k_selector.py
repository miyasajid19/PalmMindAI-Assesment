import pytest

from app.rag.retrieval.adaptive_k_selector import AdaptiveKSelector


def _candidate(chunk_id, score, rank=1):
    return {
        "chunk_id": chunk_id,
        "rank": rank,
        "page_content": f"content for {chunk_id}",
        "metadata": {"document_version_id": "version-1"},
        "reranker_score": score,
    }


def test_selects_only_candidates_at_or_above_threshold():
    selector = AdaptiveKSelector(k_min=2, k_max=4, score_threshold=0.6)
    candidates = [
        _candidate("chunk-1", 0.59, 1),
        _candidate("chunk-2", 0.60, 2),
        _candidate("chunk-3", 0.91, 3),
    ]

    result = selector.select(candidates)

    assert [item["chunk_id"] for item in result["selected_candidates"]] == [
        "chunk-3",
        "chunk-2",
    ]
    assert result["eligible_count"] == 2
    assert result["expansion_required"] is False


def test_caps_selected_candidates_at_k_max():
    selector = AdaptiveKSelector(k_min=2, k_max=3, score_threshold=0.6)
    candidates = [_candidate(f"chunk-{i}", 0.9 - i * 0.01, i) for i in range(5)]

    result = selector.select(candidates)

    assert result["selected_count"] == 3
    assert result["eligible_count"] == 5


def test_exactly_k_min_qualifying_candidates_do_not_expand():
    selector = AdaptiveKSelector(k_min=2, k_max=5, score_threshold=0.6)

    result = selector.select([_candidate("a", 0.8), _candidate("b", 0.7)])

    assert result["selected_count"] == 2
    assert result["expansion_required"] is False


def test_fewer_than_k_min_qualifying_candidates_request_expansion():
    selector = AdaptiveKSelector(k_min=3, k_max=5, score_threshold=0.6)

    result = selector.select(
        [_candidate("a", 0.9), _candidate("b", 0.59), _candidate("c", 0.5)]
    )

    assert result["selected_count"] == 1
    assert result["expansion_required"] is True


def test_zero_qualifying_candidates_request_expansion():
    selector = AdaptiveKSelector(k_min=2, k_max=5, score_threshold=0.6)

    result = selector.select([_candidate("a", 0.4), _candidate("b", 0.2)])

    assert result["selected_candidates"] == []
    assert result["expansion_required"] is True


def test_expansion_stops_at_max_rounds_without_weak_fallback():
    selector = AdaptiveKSelector(
        k_min=3, k_max=5, score_threshold=0.6, max_expansion_rounds=2
    )

    result = selector.select(
        [_candidate("a", 0.9), _candidate("b", 0.59)],
        expansion_round=2,
    )

    assert result["selected_count"] == 1
    assert result["expansion_required"] is False
    assert result["expansion_round"] == 2


def test_ties_use_retriever_rank_then_chunk_id():
    selector = AdaptiveKSelector(k_min=1, k_max=5, score_threshold=0.6)
    candidates = [
        _candidate("chunk-b", 0.8, 2),
        _candidate("chunk-c", 0.8, 1),
        _candidate("chunk-a", 0.8, 1),
    ]

    result = selector.select(candidates)

    assert [item["chunk_id"] for item in result["selected_candidates"]] == [
        "chunk-a",
        "chunk-c",
        "chunk-b",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k_min": 0}, "k_min"),
        ({"k_max": 0}, "k_max"),
        ({"k_min": 4, "k_max": 3}, "k_min"),
        ({"score_threshold": float("nan")}, "score_threshold"),
        ({"max_expansion_rounds": -1}, "max_expansion_rounds"),
    ],
)
def test_validates_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AdaptiveKSelector(**kwargs)


@pytest.mark.parametrize(
    ("candidates", "round_number", "message"),
    [
        ("invalid", 0, "sequence"),
        ([None], 0, "dictionary"),
        ([{"chunk_id": "missing-score"}], 0, "reranker_score"),
        ([_candidate("a", float("inf"))], 0, "finite"),
        ([_candidate("a", True)], 0, "numeric"),
        ([], 3, "expansion_round"),
    ],
)
def test_validates_selection_inputs(candidates, round_number, message):
    selector = AdaptiveKSelector(max_expansion_rounds=2)

    with pytest.raises(ValueError, match=message):
        selector.select(candidates, expansion_round=round_number)


def test_does_not_mutate_input_candidates():
    selector = AdaptiveKSelector(k_min=1)
    candidate = _candidate("a", 0.9)

    result = selector.select([candidate])

    assert "reranker_score" in candidate
    assert result["selected_candidates"][0] is not candidate
    assert result["selected_candidates"][0]["metadata"] is not candidate["metadata"]


def test_default_k_min_is_three():
    assert AdaptiveKSelector().k_min == 3


def test_configured_jina_threshold_keeps_relevant_hippa_chunks():
    # Scores observed in the local HiPPA retrieval diagnostic.
    selector = AdaptiveKSelector(k_min=3, k_max=10, score_threshold=0.30, max_expansion_rounds=0)
    scores = [0.52000745, 0.42786398, 0.35014987, 0.34723646, 0.33636139, 0.31369049, 0.29587377, 0.24919898]
    candidates = [_candidate(f"hippa-{index}", score, index + 1) for index, score in enumerate(scores)]

    result = selector.select(candidates)

    assert result["selected_count"] == 6
    assert result["selected_candidates"][0]["reranker_score"] == pytest.approx(0.52000745)
    assert all(item["reranker_score"] >= 0.30 for item in result["selected_candidates"])
