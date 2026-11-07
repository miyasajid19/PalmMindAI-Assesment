
from collections import defaultdict

from langchain_core.documents import Document


def _key(doc: Document) -> str:
    metadata = doc.metadata or {}

    # Prefer version-aware identity.
    document_version_id = metadata.get("document_version_id")
    chunk_index = metadata.get("chunk_index")

    if document_version_id is not None and chunk_index is not None:
        return f"{document_version_id}:{chunk_index}"

    # Backward compatibility for non-versioned documents.
    document_id = metadata.get("document_id")

    if document_id is not None and chunk_index is not None:
        return f"{document_id}:{chunk_index}"

    return (
        f"{metadata.get('source', '')}:"
        f"{metadata.get('page', '')}:"
        f"{doc.page_content}"
    )


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[Document, float]]:
    if k < 1:
        raise ValueError("RRF constant k must be at least 1.")

    weights = weights if weights is not None else [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists.")

    scores = defaultdict(float)
    documents = {}

    for result_list, weight in zip(ranked_lists, weights):
        for rank, (document, _raw_score) in enumerate(result_list, 1):
            key = _key(document)
            scores[key] += float(weight) / (k + rank)
            documents.setdefault(key, document)

    rank_positions = [
        {
            _key(document): rank
            for rank, (document, _) in enumerate(result_list, 1)
        }
        for result_list in ranked_lists
    ]
    
    ranked_keys = sorted(
        scores,
        key=lambda key: (
            -scores[key],
            rank_positions[1].get(key, float("inf")) if len(rank_positions) > 1 else float("inf"),
            rank_positions[0].get(key, float("inf")) if rank_positions else float("inf"),
        ),
    )

    return [
        (documents[key], scores[key])
        for key in ranked_keys
    ]
