
from __future__ import annotations

from qdrant_client import models


REQUIRED_PAYLOAD_INDEXES = {
    "metadata.document_version_id": models.PayloadSchemaType.KEYWORD,
    "metadata.document_id": models.PayloadSchemaType.KEYWORD,
    "metadata.chunk_index": models.PayloadSchemaType.INTEGER,
    "metadata.status": models.PayloadSchemaType.KEYWORD,
}


BM25_TEXT_FIELD = "page_content"

BM25_TEXT_INDEX = models.TextIndexParams(
    type=models.TextIndexType.TEXT,
    tokenizer=models.TokenizerType.WORD,
    lowercase=True,
    scoring=models.TextScoringParams(
        type=models.TextScoringType.BM25
    ),
)


def _validate_keyword_index(
    field_name: str,
    existing_index,
    expected_type: models.PayloadSchemaType,
) -> None:
    existing_type = getattr(
        existing_index,
        "data_type",
        existing_index,
    )

    if existing_type != expected_type:
        raise RuntimeError(
            f"Incompatible Qdrant payload index for "
            f"'{field_name}'. Expected {expected_type}, "
            f"found {existing_type}."
        )


def _validate_bm25_text_index(
    field_name: str,
    existing_index,
) -> None:
    existing_type = getattr(
        existing_index,
        "data_type",
        existing_index,
    )

    if existing_type != models.PayloadSchemaType.TEXT:
        raise RuntimeError(
            f"Incompatible Qdrant index for '{field_name}'. "
            f"Expected TEXT, found {existing_type}."
        )

    params = getattr(existing_index, "params", None)

    if params is None:
        raise RuntimeError(
            f"Qdrant text index '{field_name}' has no "
            "configuration parameters."
        )

    actual_tokenizer = getattr(params, "tokenizer", None)
    actual_lowercase = getattr(params, "lowercase", None)
    actual_scoring = getattr(params, "scoring", None)
    actual_scoring_type = getattr(actual_scoring, "type", None)

    if (
        actual_tokenizer != models.TokenizerType.WORD
        or actual_lowercase is not True
        or actual_scoring_type != models.TextScoringType.BM25
    ):
        raise RuntimeError(
            f"Incompatible BM25 text index configuration "
            f"for '{field_name}'. Expected tokenizer=WORD, "
            f"lowercase=True and scoring=BM25. "
            f"Found params={params}."
        )


def ensure_payload_indexes(
    qdrant_client,
    collection_name: str,
) -> None:
    """
    Ensure required Qdrant payload indexes exist.

    Existing compatible indexes are preserved.
    Missing indexes are created.
    Incompatible existing indexes raise RuntimeError.

    All existing indexes are validated before any missing
    index is created.
    """

    collection_info = qdrant_client.get_collection(
        collection_name=collection_name
    )

    existing_schema = collection_info.payload_schema or {}

    missing_keyword_indexes = []
    missing_text_indexes = []

    # Validate keyword indexes.
    for field_name, expected_type in REQUIRED_PAYLOAD_INDEXES.items():
        existing_index = existing_schema.get(field_name)

        if existing_index is None:
            missing_keyword_indexes.append(
                (field_name, expected_type)
            )
            continue

        _validate_keyword_index(
            field_name,
            existing_index,
            expected_type,
        )

    # Validate BM25 full-text index.
    existing_text_index = existing_schema.get(BM25_TEXT_FIELD)

    if existing_text_index is None:
        missing_text_indexes.append(
            (BM25_TEXT_FIELD, BM25_TEXT_INDEX)
        )
    else:
        _validate_bm25_text_index(
            BM25_TEXT_FIELD,
            existing_text_index,
        )

    # Create missing keyword indexes.
    for field_name, expected_type in missing_keyword_indexes:
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=expected_type,
            wait=True,
        )

    # Create missing BM25 full-text index.
    for field_name, field_schema in missing_text_indexes:
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )
