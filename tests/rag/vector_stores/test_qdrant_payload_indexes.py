from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client import models

from app.rag.vector_stores.qdrant_payload_indexes import (
    BM25_TEXT_FIELD,
    BM25_TEXT_INDEX,
    REQUIRED_PAYLOAD_INDEXES,
    ensure_payload_indexes,
)


COLLECTION_NAME = "test_collection"


def make_qdrant_client(existing_indexes=None):
    qdrant_client = MagicMock()
    qdrant_client.get_collection.return_value = SimpleNamespace(
        payload_schema=existing_indexes or {}
    )
    return qdrant_client


def make_index_info(data_type):
    return SimpleNamespace(data_type=data_type)


def make_bm25_index_info():
    return SimpleNamespace(
        data_type=models.PayloadSchemaType.TEXT,
        params=BM25_TEXT_INDEX,
    )


def test_all_existing_indexes_are_preserved():
    existing_indexes = {
        field: make_index_info(index_type)
        for field, index_type in REQUIRED_PAYLOAD_INDEXES.items()
    }
    existing_indexes[BM25_TEXT_FIELD] = make_bm25_index_info()
    qdrant_client = make_qdrant_client(existing_indexes)

    ensure_payload_indexes(qdrant_client, COLLECTION_NAME)

    qdrant_client.create_payload_index.assert_not_called()


def test_missing_indexes_are_created():
    qdrant_client = make_qdrant_client()

    ensure_payload_indexes(qdrant_client, COLLECTION_NAME)

    assert qdrant_client.create_payload_index.call_count == 5
    for field_name, expected_type in REQUIRED_PAYLOAD_INDEXES.items():
        qdrant_client.create_payload_index.assert_any_call(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=expected_type,
            wait=True,
        )
    qdrant_client.create_payload_index.assert_any_call(
        collection_name=COLLECTION_NAME,
        field_name=BM25_TEXT_FIELD,
        field_schema=BM25_TEXT_INDEX,
        wait=True,
    )


def test_existing_and_missing_indexes_are_combined():
    existing_field = "metadata.status"
    qdrant_client = make_qdrant_client(
        {
            existing_field: make_index_info(
                models.PayloadSchemaType.KEYWORD
            )
        }
    )

    ensure_payload_indexes(qdrant_client, COLLECTION_NAME)

    assert qdrant_client.create_payload_index.call_count == 4
    created_fields = {
        call.kwargs["field_name"]
        for call in qdrant_client.create_payload_index.call_args_list
    }
    assert created_fields == (
        set(REQUIRED_PAYLOAD_INDEXES) | {BM25_TEXT_FIELD}
    ) - {existing_field}


def test_incompatible_index_is_rejected_before_creation():
    incompatible_field = "metadata.status"
    qdrant_client = make_qdrant_client(
        {
            incompatible_field: make_index_info(
                models.PayloadSchemaType.INTEGER
            )
        }
    )

    with pytest.raises(RuntimeError, match="Incompatible Qdrant payload index"):
        ensure_payload_indexes(qdrant_client, COLLECTION_NAME)

    qdrant_client.create_payload_index.assert_not_called()
