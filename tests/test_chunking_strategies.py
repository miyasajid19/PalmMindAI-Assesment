from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.rag.splitters.semantic_text_splitter import SemanticTextSplitter
from app.services.document_indexing import DocumentIndexingService


class TopicEmbeddingStub:
    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "database" in lower:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors


def test_semantic_splitter_separates_topic_shift_and_keeps_metadata():
    splitter = SemanticTextSplitter(
        embedding_client=TopicEmbeddingStub(),
        chunk_size=1000,
        chunk_overlap=0,
        similarity_threshold=0.72,
    )
    document = Document(
        page_content=(
            "Artificial intelligence uses models. "
            "Machine learning trains those models. "
            "Database systems store structured records."
        ),
        metadata={"page": 4, "source": "notes.txt"},
    )

    chunks = splitter.split_documents([document])

    assert len(chunks) == 2
    assert "Artificial intelligence" in chunks[0].page_content
    assert "Machine learning" in chunks[0].page_content
    assert "Database systems" in chunks[1].page_content
    assert all(chunk.metadata == {"page": 4, "source": "notes.txt"} for chunk in chunks)


def test_semantic_splitter_skips_empty_documents():
    splitter = SemanticTextSplitter(embedding_client=TopicEmbeddingStub())
    assert splitter.split_documents([Document(page_content="  ")]) == []


def test_indexing_service_uses_requested_semantic_strategy():
    recursive_splitter = MagicMock()
    semantic_splitter = MagicMock()
    semantic_splitter.split_documents.return_value = [
        Document(page_content="Semantic chunk", metadata={"page": 1})
    ]
    qdrant = MagicMock()
    qdrant.add_documents.side_effect = lambda documents, ids: ids

    service = DocumentIndexingService(
        text_splitter=recursive_splitter,
        semantic_splitter=semantic_splitter,
        qdrant_store=qdrant,
    )

    result = service.index_document_version(
        page_documents=[Document(page_content="Source text", metadata={"page": 1})],
        document_id="document-1",
        document_version_id="version-1",
        version_number=1,
        source_filename="source.txt",
        chunking_strategy="semantic",
    )

    recursive_splitter.split_documents.assert_not_called()
    semantic_splitter.split_documents.assert_called_once()
    assert result.chunking_strategy == "semantic"
    indexed_chunk = qdrant.add_documents.call_args.args[0][0]
    assert indexed_chunk.metadata["chunking_strategy"] == "semantic"


def test_indexing_service_rejects_unknown_strategy_before_splitting():
    recursive_splitter = MagicMock()
    service = DocumentIndexingService(
        text_splitter=recursive_splitter,
        qdrant_store=MagicMock(),
    )

    with pytest.raises(ValueError, match="Choose 'recursive' or 'semantic'"):
        service.index_document_version(
            page_documents=[Document(page_content="Source text")],
            document_id="document-1",
            document_version_id="version-1",
            version_number=1,
            source_filename="source.txt",
            chunking_strategy="other",
        )

    recursive_splitter.split_documents.assert_not_called()
