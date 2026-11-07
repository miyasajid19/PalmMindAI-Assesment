from collections.abc import Sequence

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.main import create_app
from app.services.conversation_orchestrator import ConversationOrchestrator


class InMemoryHistory:
    def __init__(self):
        self.sessions: dict[str, list[BaseMessage]] = {}

    async def load_history(self, session_id: str) -> list[BaseMessage]:
        return list(self.sessions.get(session_id, []))

    async def save_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        messages = self.sessions.setdefault(session_id, [])
        messages.extend(
            [
                HumanMessage(content=user_message),
                AIMessage(content=assistant_message),
            ]
        )




class FakeIntentClassifier:
    async def classify(self, *, query: str, booking_context_active: bool = False) -> str:
        lowered = query.lower()
        if booking_context_active:
            return "booking_followup"
        if any(word in lowered for word in ("book", "schedule", "appointment", "reserve")) and any(
            word in lowered for word in ("interview", "appointment", "hr", "meet")
        ):
            return "interview_booking"
        if lowered.strip() in {"hi", "hello", "hey"}:
            return "greeting"
        return "rag"


class FakeBookingConversation:
    def __init__(self):
        self.calls = []

    async def respond(self, *, query: str, history: Sequence[BaseMessage] | None = None) -> str:
        self.calls.append((query, list(history or [])))
        if not history:
            return (
                "For the interview, please provide your full name, email, exact date, "
                "and preferred time."
            )
        if len(history) == 2:
            return (
                "Please confirm your interview details: Sajid Miya, sajid@example.com, "
                "2030-06-10 at 10:30."
            )
        return "Confirmed. Your interview is booked successfully."


class FakeRAGGraph:
    def __init__(self):
        self.inputs = []

    async def ainvoke(self, payload, config=None):
        self.inputs.append((payload, config))
        return {"generation_output": {"segments": [{"text": "Answer grounded in docs."}]}}


def test_chat_api_runs_full_booking_dialogue_over_http_and_persists_each_turn():
    history = InMemoryHistory()
    booking = FakeBookingConversation()
    orchestrator = ConversationOrchestrator(
        history_service=history,
        interview_booking_conversation=booking,
        classifier=FakeIntentClassifier(),
    )
    client = TestClient(create_app(conversation_orchestrator=orchestrator, initialize_rag=False))

    first = client.post(
        "/api/v1/chat",
        json={"session_id": "api-booking-session", "query": "I'd like to schedule an interview."},
    )
    second = client.post(
        "/api/v1/chat",
        json={
            "session_id": "api-booking-session",
            "query": "Sajid Miya, sajid@example.com, 2030-06-10 at 10:30.",
        },
    )
    third = client.post(
        "/api/v1/chat",
        json={"session_id": "api-booking-session", "query": "Yes, I confirm."},
    )

    assert first.status_code == second.status_code == third.status_code == 200
    assert first.json()["route"] == "interview_booking"
    assert "please provide" in first.json()["answer"].lower()
    assert "please confirm" in second.json()["answer"].lower()
    assert third.json()["answer"] == "Confirmed. Your interview is booked successfully."
    assert len(booking.calls) == 3
    assert len(history.sessions["api-booking-session"]) == 6


def test_chat_api_routes_regular_query_through_rag_graph():
    history = InMemoryHistory()
    graph = FakeRAGGraph()
    orchestrator = ConversationOrchestrator(history_service=history, classifier=FakeIntentClassifier())
    client = TestClient(
        create_app(rag_graph=graph, conversation_orchestrator=orchestrator, initialize_rag=False)
    )

    response = client.post(
        "/api/v1/chat",
        json={"session_id": "api-rag-session", "query": "What does the document say?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "api-rag-session",
        "answer": "Answer grounded in docs.",
        "route": "rag",
    }
    assert len(graph.inputs) == 1
    assert graph.inputs[0][0]["query"] == "What does the document say?"
    assert len(history.sessions["api-rag-session"]) == 2


def test_chat_api_reports_rag_unavailable_when_graph_is_not_configured():
    client = TestClient(create_app(conversation_orchestrator=ConversationOrchestrator(
        history_service=InMemoryHistory(), classifier=FakeIntentClassifier(),
    ), initialize_rag=False))

    response = client.post(
        "/api/v1/chat",
        json={"session_id": "api-no-graph", "query": "Ask a document question"},
    )

    assert response.status_code == 503
    assert "RAG workflow is not configured" in response.json()["detail"]


def test_health_endpoint_reports_application_status():
    client = TestClient(create_app(initialize_rag=False))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_application_startup_composes_rag_graph(monkeypatch):
    import sys
    from types import SimpleNamespace

    graph = FakeRAGGraph()
    monkeypatch.setitem(
        sys.modules,
        "app.services.application_graph",
        SimpleNamespace(build_rag_graph=lambda: graph),
    )
    orchestrator = ConversationOrchestrator(history_service=InMemoryHistory(), classifier=FakeIntentClassifier())

    with TestClient(
        create_app(conversation_orchestrator=orchestrator, initialize_rag=True)
    ) as client:
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "startup-rag", "query": "Ask a document question"},
        )

    assert response.status_code == 200
    assert response.json()["route"] == "rag"
    assert graph.inputs


from datetime import date, timedelta
from unittest.mock import MagicMock
from uuid import uuid4

from langchain_core.messages import ToolMessage


class FakeActualBookingToolModel:
    def __init__(self, booking_args):
        self.booking_args = booking_args

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        if any(
            isinstance(message, ToolMessage)
            and "Interview booked successfully" in message.content
            for message in messages
        ):
            return AIMessage(content="Confirmed. Your interview is booked.")
        latest_human = next(
            message.content
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        )
        if latest_human == "I want to schedule an interview.":
            return AIMessage(
                content="For the interview, provide your full name, email, exact date, and time."
            )
        if latest_human.startswith("Sajid Miya, sajid@example.com"):
            return AIMessage(
                content="Please confirm the interview details: name Sajid Miya, "
                "email sajid@example.com, date and time listed as supplied."
            )
        if latest_human == "Yes, I confirm.":
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "book_interview",
                        "args": self.booking_args,
                        "id": "api-booking-tool-call",
                        "type": "tool_call",
                    }
                ],
            )
        raise AssertionError(f"Unexpected human message: {latest_human}")


def test_http_booking_flow_invokes_real_langchain_tool_and_persists_booking(monkeypatch):
    from app.services.interview_booking_conversation import InterviewBookingConversation

    booking_date = (date.today() + timedelta(days=3)).isoformat()
    booking_args = {
        "name": "Sajid Miya",
        "email": "sajid@example.com",
        "interview_date": booking_date,
        "interview_time": "10:30",
        "confirmed": True,
    }
    db = MagicMock()
    db.scalar.return_value = None
    db.refresh.side_effect = lambda booking: setattr(booking, "id", uuid4())
    session_context = MagicMock()
    session_context.__enter__.return_value = db
    session_context.__exit__.return_value = False
    monkeypatch.setattr(
        "app.tools.interview_booking_tool.get_db_session",
        lambda: session_context,
    )

    history = InMemoryHistory()
    booking_handler = InterviewBookingConversation(
        model=FakeActualBookingToolModel(booking_args)
    )
    orchestrator = ConversationOrchestrator(
        history_service=history,
        interview_booking_conversation=booking_handler,
        classifier=FakeIntentClassifier(),
    )
    client = TestClient(
        create_app(conversation_orchestrator=orchestrator, initialize_rag=False)
    )

    first = client.post(
        "/api/v1/chat",
        json={"session_id": "http-real-tool", "query": "I want to schedule an interview."},
    )
    second = client.post(
        "/api/v1/chat",
        json={
            "session_id": "http-real-tool",
            "query": f"Sajid Miya, sajid@example.com, {booking_date} at 10:30.",
        },
    )
    third = client.post(
        "/api/v1/chat",
        json={"session_id": "http-real-tool", "query": "Yes, I confirm."},
    )

    assert first.status_code == second.status_code == third.status_code == 200
    assert third.json()["answer"] == "Confirmed. Your interview is booked."
    assert third.json()["route"] == "interview_booking"
    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once()
    assert len(history.sessions["http-real-tool"]) == 6


from types import SimpleNamespace


def test_document_ingestion_api_uploads_and_indexes_pdf_over_http(monkeypatch):
    from app.api import routes

    document_id = uuid4()
    version_id = uuid4()
    ingestion_result = SimpleNamespace(
        status="registered_pending",
        original_filename="uploaded.pdf",
        page_count=2,
        summary_output=SimpleNamespace(title="Uploaded policy"),
        matching_result=SimpleNamespace(classification="no_match"),
        registration_result=SimpleNamespace(
            document_id=document_id,
            document_version_id=version_id,
            version_number=1,
        ),
        validation_warnings=("A warning",),
    )

    class FakeIngestionService:
        def ingest_file(self, file_path):
            assert file_path.name == "uploaded.pdf"
            assert file_path.read_bytes() == b"%PDF-fake-payload"
            return ingestion_result

    class FakeIndexingOrchestrator:
        def index_pending_version(self, received_version_id, *, chunking_strategy):
            assert received_version_id == version_id
            assert chunking_strategy == "semantic"
            return SimpleNamespace(
                status="active",
                page_count=2,
                document_id=str(document_id),
                document_version_id=str(version_id),
                version_number=1,
                chunk_count=7,
                chunking_strategy="semantic",
            )

    monkeypatch.setattr(routes, "DocumentIngestionService", FakeIngestionService)
    monkeypatch.setattr(routes, "DocumentIndexingOrchestrator", FakeIndexingOrchestrator)
    client = TestClient(create_app(initialize_rag=False))

    response = client.post(
        "/api/v1/documents/ingest",
        files={"file": ("uploaded.pdf", b"%PDF-fake-payload", "application/pdf")},
        data={"chunking_strategy": "semantic"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "original_filename": "uploaded.pdf",
        "title": "Uploaded policy",
        "status": "active",
        "matching_classification": "no_match",
        "page_count": 2,
        "document_id": str(document_id),
        "document_version_id": str(version_id),
        "version_number": 1,
        "chunk_count": 7,
        "chunking_strategy": "semantic",
        "validation_warnings": ["A warning"],
    }


def test_document_ingestion_api_returns_review_required_without_indexing(monkeypatch):
    from app.api import routes

    ingestion_result = SimpleNamespace(
        status="review_required",
        original_filename="candidate.txt",
        page_count=1,
        summary_output=SimpleNamespace(title="Possible duplicate"),
        matching_result=SimpleNamespace(classification="potential_match"),
        registration_result=None,
        validation_warnings=(),
    )

    class FakeIngestionService:
        def ingest_file(self, file_path):
            return ingestion_result

    class MustNotIndex:
        def __init__(self):
            raise AssertionError("review_required documents must not be indexed")

    monkeypatch.setattr(routes, "DocumentIngestionService", FakeIngestionService)
    monkeypatch.setattr(routes, "DocumentIndexingOrchestrator", MustNotIndex)
    client = TestClient(create_app(initialize_rag=False))

    response = client.post(
        "/api/v1/documents/ingest",
        files={"file": ("candidate.txt", b"candidate content", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "review_required"
    assert response.json()["matching_classification"] == "potential_match"
    assert response.json()["document_version_id"] is None
    assert response.json()["chunk_count"] is None


def test_document_ingestion_api_rejects_unsupported_file_type():
    client = TestClient(create_app(initialize_rag=False))

    response = client.post(
        "/api/v1/documents/ingest",
        files={"file": ("payload.docx", b"payload", "application/octet-stream")},
    )

    assert response.status_code == 415

 
class FakeStreamingRAGGraph:
    async def astream(self, payload, stream_mode):
        assert payload["streaming"] is True
        assert "custom" in stream_mode and "values" in stream_mode
        yield "custom", {"type": "delta", "delta": "Streamed answer", "segment_index": 0}
        yield "values", {
            "generation_output": {"segments": [{"text": "Streamed answer"}]},
            "validation_result": None,
        }


def test_chat_stream_api_emits_sse_deltas_completion_and_latency_metrics():
    history = InMemoryHistory()
    graph = FakeStreamingRAGGraph()
    orchestrator = ConversationOrchestrator(history_service=history, classifier=FakeIntentClassifier())
    client = TestClient(
        create_app(rag_graph=graph, conversation_orchestrator=orchestrator, initialize_rag=False)
    )

    response = client.post(
        "/api/v1/chat/stream",
        json={"session_id": "api-stream-session", "query": "Stream this answer"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: delta" in response.text
    assert '"delta": "Streamed answer"' in response.text
    assert "event: completed" in response.text
    assert '"answer": "Streamed answer"' in response.text
    assert '"route": "rag"' in response.text
    assert '"streamed": true' in response.text
    assert '"first_token_ms":' in response.text
    assert '"total_latency_ms":' in response.text
    assert len(history.sessions["api-stream-session"]) == 2

def test_chat_api_routes_greeting_to_greeting_handler():
    orchestrator = ConversationOrchestrator(
        history_service=InMemoryHistory(), classifier=FakeIntentClassifier()
    )
    client = TestClient(
        create_app(conversation_orchestrator=orchestrator, initialize_rag=False)
    )

    response = client.post(
        "/api/v1/chat",
        json={"session_id": "greeting-session", "query": "HI"},
    )

    assert response.status_code == 200
    assert response.json()["route"] == "greeting"
    assert "book an interview" in response.json()["answer"].lower()


def test_chat_api_routes_appointment_request_to_booking():
    history = InMemoryHistory()
    booking = FakeBookingConversation()
    orchestrator = ConversationOrchestrator(
        history_service=history,
        interview_booking_conversation=booking,
        classifier=FakeIntentClassifier(),
    )
    client = TestClient(
        create_app(conversation_orchestrator=orchestrator, initialize_rag=False)
    )

    response = client.post(
        "/api/v1/chat",
        json={"session_id": "appointment-session", "query": "Hi, I want to book appointment."},
    )

    assert response.status_code == 200
    assert response.json()["route"] == "interview_booking"
    assert "please provide" in response.json()["answer"].lower()


def test_intent_classifier_extracts_systemone_choice_and_sends_jev_payload(monkeypatch):
    import asyncio
    from app.services.intent_classifier import IntentClassifier

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"intent": {"value": "interview_booking"}}

    def fake_post(url, *, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "payload": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("app.services.intent_classifier.requests.post", fake_post)
    classifier = IntentClassifier(api_key="test-key")

    intent = asyncio.run(
        classifier.classify(query="I want to book an appointment", booking_context_active=False)
    )

    assert intent == "interview_booking"
    assert captured["payload"]["model"] == "jev-1.13-free"
    assert "interview_booking" in captured["payload"]["questions"]["intent"]["criteria"]
    assert captured["headers"]["Authorization"] == "Bearer test-key"

