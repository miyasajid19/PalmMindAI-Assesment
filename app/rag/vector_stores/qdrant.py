
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parents[3]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_core.documents import Document

from ..embeddings.jina_embeddings import embedding_client
from ...core.config import settings
from .qdrant_payload_indexes import ensure_payload_indexes


class QdrantVectorStoreClient:

    def __init__(self):
        self.embedding_model = embedding_client.get_embedding_model()

        self.collection_name = settings.QDRANT_COLLECTION_NAME

        # Connect to Qdrant.
        self.client = QdrantClient(
            url=settings.QDRANT_API_ENDPOINT,
            api_key=settings.QDRANT_API_KEY
        )

        # Create collection if it does not exist.
        self._create_collection_if_not_exists()

        # Ensure payload indexes exist before retrieval or indexing.
        ensure_payload_indexes(
            self.client,
            self.collection_name,
        )

        # Initialize LangChain vector store.
        collection_info = self.client.get_collection(self.collection_name)
        vector_config = collection_info.config.params.vectors
        if isinstance(vector_config, dict):
            raise ValueError(
                f"Collection {self.collection_name!r} uses named vectors; "
                "this integration expects an unnamed dense vector."
            )
        expected_dimension = getattr(self.embedding_model, "dimensions", None)
        if expected_dimension is not None and vector_config.size != expected_dimension:
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} has "
                f"{vector_config.size} dimensions, but configured Jina embeddings "
                f"produce {expected_dimension}."
            )
        if vector_config.distance != models.Distance.COSINE:
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} uses "
                f"{vector_config.distance}, expected COSINE."
            )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            validate_collection_config=False,
        )

    # --------------------------------------------------
    # COLLECTION MANAGEMENT
    # --------------------------------------------------

    def _create_collection_if_not_exists(self):

        if not self.client.collection_exists(self.collection_name):

            embedding_dimension = getattr(self.embedding_model, "dimensions", None)
            if embedding_dimension is None:
                embedding_dimension = len(
                    self.embedding_model.embed_query("dimension check")
                )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_dimension,
                    distance=models.Distance.COSINE
                )
            )

            print(
                f"Created Qdrant collection: {self.collection_name}"
            )

        else:

            print(
                f"Using existing Qdrant collection: {self.collection_name}"
            )

    # --------------------------------------------------
    # ADD DOCUMENTS
    # --------------------------------------------------

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:

        if not documents:
            return []

        if ids is not None:
            if len(ids) != len(documents):
                raise ValueError(
                    "The number of IDs must match the number of documents."
                )

            if len(set(ids)) != len(ids):
                raise ValueError("Document IDs must be unique.")

        inserted_ids = self.vector_store.add_documents(
            documents=documents,
            ids=ids,
        )

        # Preserve the existing root-level metadata representation.
        payload_by_id = {
            point_id: dict(document.metadata or {})
            for point_id, document in zip(inserted_ids, documents)
        }

        for point_id, payload in payload_by_id.items():
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[point_id],
                wait=True,
            )

        return inserted_ids

    # --------------------------------------------------
    # DENSE SEARCH
    # --------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ):

        return self.vector_store.similarity_search(
            query=query,
            k=k
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        query_filter: models.Filter | None = None
    ):

        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            query_filter=query_filter
        )

    # --------------------------------------------------
    # NATIVE QDRANT BM25 SEARCH
    # --------------------------------------------------

    def bm25_search_with_score(
        self,
        query: str,
        k: int = 20,
        query_filter: models.Filter | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Perform native BM25 full-text search using Qdrant.

        Searches the indexed page_content field and returns
        LangChain Documents with their Qdrant BM25 scores.

        Metadata filtering is applied inside Qdrant.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty.")

        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=models.TextQuery(
                text=query.strip()
            ),
            using="page_content",
            query_filter=query_filter,
            limit=k,
            with_payload=[
                "page_content",
                "metadata",
            ],
            with_vectors=False,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            page_content = payload.get("page_content")
            metadata = payload.get("metadata")

            if not isinstance(page_content, str):
                raise RuntimeError(
                    f"Qdrant point {point.id} is missing "
                    "a valid page_content payload."
                )

            if not isinstance(metadata, dict):
                raise RuntimeError(
                    f"Qdrant point {point.id} is missing "
                    "valid LangChain metadata."
                )

            document = Document(
                page_content=page_content,
                metadata=metadata,
            )

            score = float(point.score)

            results.append((document, score))

        return results

    # --------------------------------------------------
    # UPDATE LIFECYCLE STATUS
    # --------------------------------------------------

    def update_version_status(
        self,
        document_version_id: str,
        status: str,
    ) -> None:
        """Update lifecycle status for every chunk of a version."""

        if not document_version_id or not document_version_id.strip():
            raise ValueError("document_version_id cannot be empty.")

        if not status or not status.strip():
            raise ValueError("status cannot be empty.")

        points_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.document_version_id",
                    match=models.MatchValue(
                        value=document_version_id
                    ),
                )
            ]
        )

        # Update root-level status.
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"status": status},
            points=points_filter,
            wait=True,
        )

        # Update only metadata.status, preserving other metadata.
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"status": status},
            points=points_filter,
            key="metadata",
            wait=True,
        )

    # --------------------------------------------------
    # DELETE SPECIFIC POINTS
    # --------------------------------------------------

    def delete_points(
        self,
        point_ids: list[str | int],
    ) -> None:
        """
        Delete specific Qdrant points by their IDs.

        Used for compensating cleanup when indexing fails.
        """

        if not point_ids:
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=point_ids
            ),
            wait=True,
        )

    # --------------------------------------------------
    # DELETE COLLECTION
    # --------------------------------------------------

    def delete_collection(self):

        self.client.delete_collection(self.collection_name)

        print(
            f"Deleted collection: {self.collection_name}"
        )


# Singleton instance
client = QdrantVectorStoreClient()


if __name__ == "__main__":

    documents = [
        Document(
            page_content="Hello world!",
            metadata={"source": "example1"}
        ),
        Document(
            page_content="This is a test.",
            metadata={"source": "example2"}
        )
    ]

    point_ids = client.add_documents(documents)

    print("Inserted point IDs:", point_ids)

    results = client.similarity_search("Hello", k=2)

    for result in results:
        print(result)

    # client.delete_points(point_ids)
    # client.delete_collection()
