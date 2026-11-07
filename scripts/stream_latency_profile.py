"""Measure end-to-end latency for the SSE chat API."""
from __future__ import annotations
import json, statistics, sys, time
from pathlib import Path
from uuid import uuid4
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from fastapi.testclient import TestClient
from app.main import create_app
QUERIES = (
    "Summarize the main findings in the retrieval evaluation execution log.",
    "What is the purpose of hybrid dense and BM25 retrieval in this system?",
    "Explain the document indexing lifecycle briefly.",
)
def main():
    client = TestClient(create_app())
    startup = time.perf_counter()
    with client:
        print(f"APP_STARTUP_SECONDS={time.perf_counter() - startup:.3f}")
        first_samples, total_samples = [], []
        for i, query in enumerate(QUERIES, 1):
            started = time.perf_counter()
            first_delta, completion = None, None
            try:
                with client.stream("POST", "/api/v1/chat/stream", json={"session_id": f"stream-latency-{uuid4()}", "query": query}) as response:
                    if response.status_code != 200:
                        print(f"SAMPLE_{i}_FAILED HTTP_STATUS={response.status_code}")
                        continue
                    event = ""
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if line.startswith("event: "):
                            event = line[7:]
                            if event == "delta" and first_delta is None:
                                first_delta = time.perf_counter() - started
                        elif line.startswith("data: ") and event == "completed":
                            completion = json.loads(line[6:])
                        elif line.startswith("data: ") and event == "error":
                            print(f"SAMPLE_{i}_STREAM_ERROR={line[6:]}")
            except Exception as exc:
                print(f"SAMPLE_{i}_FAILED ERROR_TYPE={type(exc).__name__} ERROR={exc}")
                continue
            total = time.perf_counter() - started
            if completion:
                total_samples.append(total)
                if first_delta is not None:
                    first_samples.append(first_delta)
                print(f"SAMPLE_{i}_HTTP_STATUS=200 FIRST_TOKEN_SECONDS={(first_delta if first_delta is not None else float('nan')):.3f} TOTAL_SECONDS={total:.3f} SERVER_FIRST_TOKEN_MS={completion.get('first_token_ms')} SERVER_TOTAL_LATENCY_MS={completion.get('total_latency_ms')} ANSWER_CHARS={len(completion.get('answer', ''))} STREAMED={completion.get('streamed')}")
            else:
                print(f"SAMPLE_{i}_FAILED_NO_COMPLETION_EVENT TOTAL_SECONDS={total:.3f}")
        print(f"SUCCESSFUL_STREAM_SAMPLES={len(total_samples)}")
        if first_samples:
            print(f"FIRST_TOKEN_MEDIAN_SECONDS={statistics.median(first_samples):.3f}")
            print(f"FIRST_TOKEN_MEAN_SECONDS={statistics.mean(first_samples):.3f}")
        if total_samples:
            print(f"TOTAL_MEDIAN_SECONDS={statistics.median(total_samples):.3f}")
            print(f"TOTAL_MEAN_SECONDS={statistics.mean(total_samples):.3f}")
            print(f"TOTAL_MIN_SECONDS={min(total_samples):.3f}")
            print(f"TOTAL_MAX_SECONDS={max(total_samples):.3f}")
if __name__ == "__main__":
    main()
