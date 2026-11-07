"""FastAPI request/response boundary for SaviraVoice workflows."""

import json
import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.services.conversation_orchestrator import ConversationOrchestrator
from app.services.document_ingestion.ingestion_service import (
    DocumentIngestionError,
    DocumentIngestionService,
)
from app.services.document_indexing.indexing_orchestrator import (
    DocumentIndexingOrchestrationError,
    DocumentIndexingOrchestrator,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["conversation"])


class ChatRequest(BaseModel):
    """A single message within a persistent conversation session."""

    session_id: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=10000)


class ChatResponse(BaseModel):
    """Normalized response returned to API clients."""

    session_id: str
    answer: str
    route: str


class DocumentIngestionResponse(BaseModel):
    """Summary of document ingestion and its resulting index lifecycle."""

    original_filename: str
    title: str
    status: str
    matching_classification: str
    page_count: int
    document_id: str | None = None
    document_version_id: str | None = None
    version_number: int | None = None
    chunk_count: int | None = None
    chunking_strategy: str | None = None
    validation_warnings: list[str] = Field(default_factory=list)


@router.post("/documents/ingest", response_model=DocumentIngestionResponse)
async def ingest_document(
    file: UploadFile = File(..., description="A PDF or TXT document"),
    chunking_strategy: str = Form("recursive", pattern="^(recursive|semantic)$"),
) -> DocumentIngestionResponse:
    """Upload, validate, register, index, and activate a PDF or TXT document."""
    original_filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    extension = Path(original_filename).suffix.lower()
    if not original_filename or extension not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=415,
            detail="Only PDF and TXT uploads with a filename are supported.",
        )

    try:
        with TemporaryDirectory(prefix="savira_voice_upload_") as temporary_directory:
            local_path = Path(temporary_directory) / original_filename
            with local_path.open("wb") as destination:
                while chunk := await file.read(1024 * 1024):
                    destination.write(chunk)

            def ingest_and_index() -> tuple[Any, Any | None]:
                ingestion_result = DocumentIngestionService().ingest_file(local_path)
                indexing_result = None
                if ingestion_result.status == "registered_pending":
                    registration = ingestion_result.registration_result
                    if registration is None:
                        raise RuntimeError(
                            "Ingestion returned pending status without registration details."
                        )
                    indexing_result = DocumentIndexingOrchestrator().index_pending_version(
                        registration.document_version_id,
                        chunking_strategy=chunking_strategy,
                    )
                return ingestion_result, indexing_result

            ingestion_result, indexing_result = await run_in_threadpool(ingest_and_index)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DocumentIndexingOrchestrationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document ingestion or indexing failed.",
        ) from exc
    finally:
        await file.close()

    registration = ingestion_result.registration_result
    return DocumentIngestionResponse(
        original_filename=ingestion_result.original_filename,
        title=ingestion_result.summary_output.title,
        status=indexing_result.status if indexing_result is not None else ingestion_result.status,
        matching_classification=ingestion_result.matching_result.classification,
        page_count=(indexing_result.page_count if indexing_result is not None else ingestion_result.page_count),
        document_id=(indexing_result.document_id if indexing_result is not None else (str(registration.document_id) if registration else None)),
        document_version_id=(indexing_result.document_version_id if indexing_result is not None else (str(registration.document_version_id) if registration else None)),
        version_number=(indexing_result.version_number if indexing_result is not None else (registration.version_number if registration else None)),
        chunk_count=indexing_result.chunk_count if indexing_result is not None else None,
        chunking_strategy=indexing_result.chunking_strategy if indexing_result is not None else None,
        validation_warnings=list(ingestion_result.validation_warnings),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """Run one user turn through booking dialogue or the RAG graph."""
    orchestrator: ConversationOrchestrator = request.app.state.conversation_orchestrator
    graph: Any | None = request.app.state.rag_graph

    history = await orchestrator.history_service.load_history(payload.session_id)
    intent = await orchestrator.detect_intent(query=payload.query, history=history)
    if intent not in {"interview_booking", "booking_followup", "greeting"} and graph is None:
        raise HTTPException(
            status_code=503,
            detail="The RAG workflow is not configured for this application.",
        )

    try:
        result = await orchestrator.run_turn(
            graph=graph,
            session_id=payload.session_id,
            query=payload.query,
            intent=intent,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    answer = ConversationOrchestrator._extract_answer(result)
    if not answer.strip():
        raise HTTPException(
            status_code=502,
            detail="The conversation workflow returned an empty answer.",
        )

    return ChatResponse(
        session_id=payload.session_id,
        answer=answer,
        route=("interview_booking" if intent in {"interview_booking", "booking_followup"} else intent if intent == "greeting" else "rag"),
    )


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """Stream a chat turn as Server-Sent Events (SSE)."""
    orchestrator: ConversationOrchestrator = request.app.state.conversation_orchestrator
    graph: Any | None = request.app.state.rag_graph
    history = await orchestrator.history_service.load_history(payload.session_id)
    intent = await orchestrator.detect_intent(query=payload.query, history=history)
    if intent not in {"interview_booking", "booking_followup", "greeting"} and graph is None:
        raise HTTPException(
            status_code=503,
            detail="The RAG workflow is not configured for this application.",
        )

    async def event_stream():
        started = time.perf_counter()
        first_delta_seconds: float | None = None
        try:
            if intent in {"interview_booking", "booking_followup", "greeting"}:
                result = await orchestrator.run_turn(
                    graph=graph,
                    session_id=payload.session_id,
                    query=payload.query,
                    intent=intent,
                )
                answer = ConversationOrchestrator._extract_answer(result)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                yield "event: completed\ndata: " + json.dumps({
                    "session_id": payload.session_id,
                    "answer": answer,
                    "route": ("interview_booking" if intent in {"interview_booking", "booking_followup"} else "greeting"),
                    "first_token_ms": None,
                    "total_latency_ms": elapsed_ms,
                    "streamed": False,
                }, ensure_ascii=False) + "\n\n"
                return

            graph_input = {"query": payload.query, "context": history, "streaming": True}
            final_state: dict[str, Any] | None = None
            async for mode, chunk in graph.astream(
                graph_input,
                stream_mode=["custom", "values"],
            ):
                if mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "delta":
                    if first_delta_seconds is None:
                        first_delta_seconds = time.perf_counter() - started
                    yield "event: delta\ndata: " + json.dumps({
                        "delta": chunk.get("delta", ""),
                        "segment_index": chunk.get("segment_index", 0),
                    }, ensure_ascii=False) + "\n\n"
                elif mode == "values" and isinstance(chunk, dict):
                    final_state = chunk

            if final_state is None:
                raise RuntimeError("RAG stream finished without a final graph state.")
            answer = ConversationOrchestrator._extract_answer(final_state)
            if not answer.strip():
                raise RuntimeError("The conversation workflow returned an empty answer.")
            await orchestrator.history_service.save_turn(payload.session_id, payload.query, answer)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            first_token_ms = round(first_delta_seconds * 1000, 2) if first_delta_seconds is not None else None
            yield "event: completed\ndata: " + json.dumps({
                "session_id": payload.session_id,
                "answer": answer,
                "route": "rag",
                "first_token_ms": first_token_ms,
                "total_latency_ms": elapsed_ms,
                "streamed": True,
            }, ensure_ascii=False) + "\n\n"
        except Exception:
            logger.exception("chat_stream_failed session_id=%s", payload.session_id)
            yield "event: error\ndata: " + json.dumps({
                "message": "Streaming chat failed due to an internal error.",
            }, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
