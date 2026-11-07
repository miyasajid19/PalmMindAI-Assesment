"""Application composition for the dependency-injected RAG workflow."""

from app.agents.Generation import GenerationAgent
from app.agents.Multi_query_generator import MultiQueryGeneratorAgent
from app.core.database import SessionLocal
from app.core.config import settings
from app.rag.retrieval.adaptive_k_selector import AdaptiveKSelector
from app.rag.retrieval.candidate_document_generator import CandidateDocumentGenerator
from app.rag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from app.rag.retrieval.pipeline import HybridRetrievalPipeline
from app.rag.vector_stores.qdrant_summary import summary_store
from app.repositories.document_version_repositories import DocumentVersionRepository
from app.services.retrieval.retrieval_service import RetrievalOrchestrator
from app.validation import static_validation_layer
from app.workflows.rag.graph import create_rag_graph


class SessionManagedDocumentVersionRepository:
    """Read the shared catalogue with a short-lived SQLAlchemy session per call."""

    @staticmethod
    def get_shared_retrieval_catalog() -> list[dict[str, object]]:
        with SessionLocal() as session:
            return DocumentVersionRepository(session).get_shared_retrieval_catalog()


def build_rag_graph():
    """Construct the production RAG graph from configured app dependencies."""
    retrieval_orchestrator = RetrievalOrchestrator(
        document_version_repository=SessionManagedDocumentVersionRepository(),
        multi_query_agent=MultiQueryGeneratorAgent(),
        candidate_document_generator=CandidateDocumentGenerator(
            summary_store=summary_store,
        ),
        hybrid_retrieval_pipeline=HybridRetrievalPipeline(),
        cross_encoder_reranker=CrossEncoderReranker(),
        adaptive_k_selector=AdaptiveKSelector(
            k_min=3,
            k_max=10,
            score_threshold=settings.RERANKER_SCORE_THRESHOLD,
            max_expansion_rounds=0,
        ),
    )
    return create_rag_graph(
        retriever=retrieval_orchestrator,
        generation_agent=GenerationAgent(),
        validator=static_validation_layer,
    )
