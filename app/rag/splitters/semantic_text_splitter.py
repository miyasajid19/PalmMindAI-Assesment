"""Embedding-guided semantic text splitter for LangChain Documents."""

from __future__ import annotations

import re
from math import sqrt
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SemanticTextSplitter:
    """Group neighboring sentence segments until a semantic topic shift occurs."""

    def __init__(
        self,
        embedding_client: Any | None = None,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.72,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1.")

        if embedding_client is None:
            from ..embeddings.jina_embeddings import embedding_client as default_client
            embedding_client = default_client

        self.embedding_client = embedding_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self._oversized_segment_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
        )

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding vectors must have matching dimensions.")
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

    def _segments(self, text: str) -> list[str]:
        sentence_segments = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
            if segment.strip()
        ]
        bounded_segments: list[str] = []
        for segment in sentence_segments:
            if len(segment) <= self.chunk_size:
                bounded_segments.append(segment)
            else:
                bounded_segments.extend(
                    piece.strip()
                    for piece in self._oversized_segment_splitter.split_text(segment)
                    if piece.strip()
                )
        return bounded_segments

    def split_documents(self, documents: list[Document]) -> list[Document]:
        output: list[Document] = []

        for document in documents:
            text = (document.page_content or "").strip()
            if not text:
                continue

            segments = self._segments(text)
            if not segments:
                continue

            vectors = self.embedding_client.embed_texts(segments)
            if len(vectors) != len(segments):
                raise ValueError(
                    "Embedding client returned a vector count that does not match "
                    "the semantic segments."
                )

            groups: list[list[str]] = []
            current: list[str] = []
            current_length = 0

            for index, segment in enumerate(segments):
                semantic_boundary = (
                    bool(current)
                    and self._cosine_similarity(vectors[index - 1], vectors[index])
                    < self.similarity_threshold
                )
                size_boundary = bool(current) and (
                    current_length + len(segment) + 1 > self.chunk_size
                )

                if semantic_boundary or size_boundary:
                    groups.append(current)
                    overlap_segments: list[str] = []
                    overlap_length = 0
                    for previous in reversed(current):
                        if overlap_length + len(previous) > self.chunk_overlap:
                            break
                        overlap_segments.insert(0, previous)
                        overlap_length += len(previous) + 1
                    current = overlap_segments
                    current_length = sum(len(item) + 1 for item in current)

                if current and current_length + len(segment) + 1 > self.chunk_size:
                    groups.append(current)
                    current = []
                    current_length = 0

                current.append(segment)
                current_length += len(segment) + 1

            if current:
                groups.append(current)

            for group in groups:
                chunk_text = " ".join(group).strip()
                if chunk_text:
                    output.append(
                        Document(
                            page_content=chunk_text,
                            metadata=dict(document.metadata or {}),
                        )
                    )

        return output
