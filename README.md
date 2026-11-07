# SaviraVoice

SaviraVoice contains the Python components for a shared-corpus document ingestion and retrieval-augmented generation (RAG) system.

## Core capabilities

- A common `DocumentLoader` dispatches `.pdf` files to the PDF/OCR handler and `.txt` files to the UTF-8 text handler; both yield normalized LangChain documents for the shared ingestion and indexing pipelines
- Document registration and version lifecycle management
- Qdrant summary candidate generation and hybrid dense/BM25 chunk retrieval
- Result fusion, reranking, adaptive retrieval expansion, and RAG workflow nodes
- Shared PostgreSQL document catalogue; retrieval and ingestion interfaces do not require organization or user IDs

## Conversational history

`ConversationOrchestrator` hydrates the RAG graph's `context` from Redis before each turn, then persists the user message and the generated assistant answer after graph completion. History is stored under `conversation:{session_id}`, trimmed to the most recent 10 turns, and uses a sliding inactivity TTL of 30 minutes by default.

Configure these optional environment values to adjust it:

- `CONVERSATION_HISTORY_TTL_SECONDS` (default: `1800`)
- `CONVERSATION_HISTORY_MAX_TURNS` (default: `10`)

At the request boundary, pass the compiled RAG graph and a stable conversation/session ID to the orchestrator:

```python
from app.services.conversation_orchestrator import ConversationOrchestrator

conversation_orchestrator = ConversationOrchestrator()
result = await conversation_orchestrator.run_turn(
    graph=rag_graph,
    session_id=session_id,
    query=user_message,
)
```

Each request should use the same session ID to continue a conversation. A new session ID starts an independent history. Redis connectivity is configured with `AIVEN_REDIS_URI`.

## Setup

1. Install dependencies with uv sync.
2. Copy .env.example to .env.
3. Configure the PostgreSQL, Qdrant, object-storage, Redis, and LLM settings required by the components you run.



RAG reranker selection uses RERANKER_SCORE_THRESHOLD (default 0.30) to filter Jina relevance scores. Tune this value against a labeled retrieval evaluation set before production rollout.

## PDF OCR

PDF extraction is native-text-first: PyMuPDF extracts text, and pages below the configured
MIN_NATIVE_CHARS threshold are sent to the configured OCR engine. The default engine is
FastOCR (OCR_ENGINE=fastocr), a hosted OCR API; EasyOCR is no longer included as a local
model dependency. Set FASTOCR_API_KEY to enable cloud OCR. The API base URL defaults to
https://api.fastocr.org and can be overridden with FASTOCR_BASE_URL; the request timeout
is configurable with FASTOCR_TIMEOUT_SECONDS (default 300). Selected page images are sent
externally to FastOCR, so account for API usage costs and document-data handling policies.
Tesseract remains available as an optional local alternate (OCR_ENGINE=tesseract) if its
system executable and language data are installed.

## Utility scripts

Run commands from the project root:

- Ingest a PDF and optionally index its pending version:
  uv run python scripts/ingestion.py path/to/file.pdf

- Generate summary-based candidates:
  uv run python scripts/candidate_documents.py --query "question text"

- Search chunks using BM25:
  uv run python scripts/qdrant_bm25.py "question text"

- Inspect the Qdrant summary collection:
  uv run python scripts/qdrant_summary_store.py list

These utilities use configured external services and may require credentials and reachable infrastructure.

## Tests

Run the test suite with uv run pytest -q.

Integration tests that require sample PDFs or external services may be skipped when their prerequisites are unavailable.

## Interview booking conversation

The conversation orchestrator routes interview scheduling requests to
`InterviewBookingConversation` rather than sending those turns through document
retrieval. The booking dialogue gathers a candidate's name, email, exact date,
and time, summarizes the details, and requires an explicit affirmative
confirmation before invoking the LangChain `book_interview` tool.

The tool validates inputs, rejects past slots, checks the requested date/time,
and persists a booking to PostgreSQL. A unique constraint on
`(interview_date, interview_time)` protects against concurrent bookings for the
same slot. Times are stored as application-local, timezone-naive values because
the current table schema does not include a timezone field.

For a database where `interview_bookings` already exists, apply the one-time
migration in `app/core/migrations/20261107_add_interview_slot_unique.sql`.
Review and resolve duplicate date/time slots before running it. `create_tables()`
creates missing tables but does not modify existing table schemas.

## FastAPI

The application exposes document ingestion, chat, and health endpoints. At startup, the
default \`app\` application composes the RAG graph using the configured
PostgreSQL, Qdrant, embedding, and LLM services. The graph composition uses a
short-lived SQLAlchemy session for each shared-catalogue read.

Run the API from the project root:

\`\`\`powershell
uv sync
uv run uvicorn app.main:app --reload
\`\`\`

Open the interactive API docs at \`http://127.0.0.1:8000/docs\`.

### Document ingestion request

`POST /api/v1/documents/ingest` accepts a multipart form upload with:

- `file`: required PDF or TXT document
- `chunking_strategy`: optional `recursive` (default) or `semantic`

For a newly registered document, the endpoint completes indexing and activation
before responding. Potential duplicates return `review_required` and are not
indexed; exact duplicates return `duplicate` without creating a new version.
The response includes the resulting status, matching classification, document
and version IDs when registered, page/chunk counts, and validation warnings.

Example using curl:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/documents/ingest `
  -F "file=@path/to/document.pdf" `
  -F "chunking_strategy=recursive"
```

### Chat request

\`POST /api/v1/chat\`

\`\`\`json
{
  "session_id": "candidate-session-001",
  "query": "I'd like to schedule an interview."
}
\`\`\`

The response includes the stable \`session_id\`, an \`answer\`, and a \`route\`
(\`interview_booking\` or \`rag\`). Reuse the same session ID to continue the
conversation; a new ID starts a new Redis-backed history.

### Health check

\`GET /health\` returns \`{"status": "ok"}\` when the application process is
running. It is a process health endpoint and does not independently ping
PostgreSQL, Redis, or Qdrant.

