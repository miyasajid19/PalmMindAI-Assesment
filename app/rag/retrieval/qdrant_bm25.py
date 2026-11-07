from __future__ import annotations

from langchain_core.documents import Document
from qdrant_client import models

from .chunk_exclusion import build_excluded_chunk_filters
from ..vector_stores.qdrant import client as qdrant_client


class QdrantBM25Retriever:
    """BM25 retrieval across the shared corpus without tenant/user filters."""

    def __init__(self, vector_store=None):
        self.vector_store = vector_store or qdrant_client

    def search(
        self,
        query: str,
        k: int = 20,
        document_id: str | None = None,
        status: str | None = "active",
        candidate_document_version_ids: list[str] | None = None,
        exclude: bool = False,
        excluded_chunk_ids: list[str] | None = None,
    ) -> list[tuple[Document, float]]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty.")
        if k < 1:
            raise ValueError("k must be at least 1.")
        if status is not None and not status.strip():
            raise ValueError("status cannot be empty.")
        if not isinstance(exclude, bool):
            raise ValueError("exclude must be a boolean.")

        if candidate_document_version_ids is not None:
            candidate_document_version_ids = list(dict.fromkeys(
                str(version_id).strip()
                for version_id in candidate_document_version_ids
                if version_id and str(version_id).strip()
            ))
            if not candidate_document_version_ids:
                return []

        return self.vector_store.bm25_search_with_score(
            query=query,
            k=k,
            query_filter=self._build_filter(
                document_id=document_id,
                status=status,
                candidate_document_version_ids=candidate_document_version_ids,
                exclude=exclude,
                excluded_chunk_ids=excluded_chunk_ids,
            ),
        )

    @staticmethod
    def _build_filter(
        *,
        document_id: str | None,
        status: str | None,
        candidate_document_version_ids: list[str] | None,
        exclude: bool = False,
        excluded_chunk_ids: list[str] | None = None,
    ):
        conditions = []
        if document_id is not None:
            conditions.append(models.FieldCondition(
                key="metadata.document_id",
                match=models.MatchValue(value=str(document_id)),
            ))
        if status is not None:
            conditions.append(models.FieldCondition(
                key="metadata.status",
                match=models.MatchValue(value=status),
            ))
        if candidate_document_version_ids is not None:
            conditions.append(models.FieldCondition(
                key="metadata.document_version_id",
                match=models.MatchAny(any=candidate_document_version_ids),
            ))

        excluded_filters = build_excluded_chunk_filters(excluded_chunk_ids) if exclude else []
        return models.Filter(must=conditions, must_not=excluded_filters) if conditions or excluded_filters else None


bm25_retriever = QdrantBM25Retriever()
