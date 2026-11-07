"""Helpers for excluding previously retrieved chunks in Qdrant."""

from qdrant_client import models


def build_excluded_chunk_filters(
    excluded_chunk_ids: list[str] | None,
) -> list[models.Filter]:
    """Build Qdrant subfilters for stable ``document_id:chunk_index`` IDs.

    The fusion layer normally emits ``document_version_id:chunk_index``.
    The document_id alternative supports legacy chunks without a version ID.
    Each returned nested filter matches one exact chunk and can be supplied
    to a parent Filter.must_not list.
    """

    filters: list[models.Filter] = []
    seen: set[str] = set()

    for chunk_id in excluded_chunk_ids or []:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("excluded_chunk_ids must contain non-empty strings.")

        normalized_id = chunk_id.strip()
        if normalized_id in seen:
            continue
        seen.add(normalized_id)

        try:
            document_identity, raw_chunk_index = normalized_id.rsplit(":", 1)
            chunk_index = int(raw_chunk_index)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid chunk ID {normalized_id!r}; expected identity:chunk_index."
            ) from exc

        if not document_identity:
            raise ValueError(
                f"Invalid chunk ID {normalized_id!r}; document identity is empty."
            )

        index_condition = models.FieldCondition(
            key="metadata.chunk_index",
            match=models.MatchValue(value=chunk_index),
        )

        # IDs are normally version-scoped. Include the logical-document
        # alternative for legacy metadata where a version ID is unavailable.
        for identity_key in (
            "metadata.document_version_id",
            "metadata.document_id",
        ):
            filters.append(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key=identity_key,
                            match=models.MatchValue(value=document_identity),
                        ),
                        index_condition,
                    ]
                )
            )

    return filters
