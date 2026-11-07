from __future__ import annotations

import math
from numbers import Real
from typing import Any, Sequence


class AdaptiveKSelector:
    """Select reranked chunks and signal when expansion is needed.

    ``score_threshold`` is model/configuration-specific, not a calibrated
    probability of relevance. Calibrate it against held-out relevance
    judgments for the configured cross-encoder before production use.
    """

    def __init__(
        self,
        k_min: int = 3,
        k_max: int = 10,
        score_threshold: float = 0.6,
        max_expansion_rounds: int = 2,
    ) -> None:
        if isinstance(k_min, bool) or not isinstance(k_min, int) or k_min < 1:
            raise ValueError("k_min must be an integer of at least 1.")
        if isinstance(k_max, bool) or not isinstance(k_max, int) or k_max < 1:
            raise ValueError("k_max must be an integer of at least 1.")
        if k_min > k_max:
            raise ValueError("k_min cannot be greater than k_max.")
        if (
            isinstance(score_threshold, bool)
            or not isinstance(score_threshold, Real)
            or not math.isfinite(float(score_threshold))
        ):
            raise ValueError("score_threshold must be a finite number.")
        if (
            isinstance(max_expansion_rounds, bool)
            or not isinstance(max_expansion_rounds, int)
            or max_expansion_rounds < 0
        ):
            raise ValueError("max_expansion_rounds must be a non-negative integer.")

        self.k_min = k_min
        self.k_max = k_max
        self.score_threshold = float(score_threshold)
        self.max_expansion_rounds = max_expansion_rounds

    def select(
        self,
        reranked_candidates: Sequence[dict[str, Any]],
        expansion_round: int = 0,
    ) -> dict[str, Any]:
        """Filter by reranker score, cap the result count, and signal expansion.

        expansion_round is the number of expansion rounds already performed.
        The selector does not retrieve candidates or increment the round itself.
        The caller passes the updated value after each expansion.
        """
        if not isinstance(reranked_candidates, Sequence) or isinstance(
            reranked_candidates, (str, bytes)
        ):
            raise ValueError("reranked_candidates must be a sequence.")
        if (
            isinstance(expansion_round, bool)
            or not isinstance(expansion_round, int)
            or expansion_round < 0
            or expansion_round > self.max_expansion_rounds
        ):
            raise ValueError(
                "expansion_round must be between 0 and max_expansion_rounds."
            )

        eligible = []
        for index, candidate in enumerate(reranked_candidates):
            if not isinstance(candidate, dict):
                raise ValueError(f"Candidate {index} must be a dictionary.")
            if "reranker_score" not in candidate:
                raise ValueError(
                    f"Candidate {index} is missing required field: reranker_score."
                )

            score = candidate["reranker_score"]
            if isinstance(score, bool) or not isinstance(score, Real):
                raise ValueError(f"Candidate {index} reranker_score must be numeric.")
            score = float(score)
            if not math.isfinite(score):
                raise ValueError(f"Candidate {index} reranker_score must be finite.")

            if score >= self.score_threshold:
                enriched = dict(candidate)
                if isinstance(candidate.get("metadata"), dict):
                    enriched["metadata"] = dict(candidate["metadata"])
                eligible.append(enriched)

        # Match the reranker's deterministic tie-breaking policy.
        eligible.sort(
            key=lambda candidate: (
                -float(candidate["reranker_score"]),
                candidate.get("rank", float("inf")),
                str(candidate.get("chunk_id", "")),
            )
        )

        selected = eligible[: self.k_max]
        expansion_required = (
            len(selected) < self.k_min
            and expansion_round < self.max_expansion_rounds
        )

        return {
            "selected_candidates": selected,
            "selected_count": len(selected),
            "eligible_count": len(eligible),
            "expansion_required": expansion_required,
            "expansion_round": expansion_round,
            "max_expansion_rounds": self.max_expansion_rounds,
            "score_threshold": self.score_threshold,
        }
