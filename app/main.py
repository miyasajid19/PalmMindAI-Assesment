"""SaviraVoice FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.responses import Response

from app.api.routes import router
from app.services.conversation_orchestrator import ConversationOrchestrator


def create_app(
    *,
    rag_graph=None,
    conversation_orchestrator: ConversationOrchestrator | None = None,
    initialize_rag: bool = True,
) -> FastAPI:
    """Create an application and compose the production RAG graph on startup."""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        initialized_here = False
        if application.state.rag_graph is None and initialize_rag:
            from app.services.application_graph import build_rag_graph

            application.state.rag_graph = build_rag_graph()
            initialized_here = True
        application.state.rag_initialized_here = initialized_here
        try:
            yield
        finally:
            from app.core.redis import close_redis

            await close_redis()
            if initialized_here:
                from app.core.database import close_database

                close_database()

    application = FastAPI(
        title="SaviraVoice API",
        version="0.1.0",
        description="Conversational RAG and interview scheduling API.",
        lifespan=lifespan,
    )
    application.state.rag_graph = rag_graph
    application.state.conversation_orchestrator = (
        conversation_orchestrator or ConversationOrchestrator()
    )
    application.include_router(router)

    @application.head("/", tags=["health"])
    async def root_head() -> Response:
        """Provide a lightweight root endpoint for service availability checks."""
        return Response(status_code=200)

    @application.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
