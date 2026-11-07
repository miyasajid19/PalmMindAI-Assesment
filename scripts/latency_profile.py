"""Read-only latency profile for the SaviraVoice chat API.

Measures application startup separately and samples end-to-end chat request
latency against configured services. Does not patch or tune application code.
Run from the repository root: uv run python scripts/latency_profile.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from uuid import uuid4

# Ensure the repository root is importable when this file is executed as a script.
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


def main() -> None:
    app = create_app()
    client = TestClient(app)
    startup_started = time.perf_counter()
    with client:
        startup_seconds = time.perf_counter() - startup_started
        print(f"APP_STARTUP_SECONDS={startup_seconds:.3f}")

        samples: list[float] = []
        failed_samples: list[float] = []
        for number, query in enumerate(QUERIES, start=1):
            started = time.perf_counter()
            try:
                response = client.post(
                    "/api/v1/chat",
                    json={"session_id": f"latency-profile-{uuid4()}", "query": query},
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                failed_samples.append(elapsed)
                print(
                    f"CHAT_SAMPLE_{number}_FAILED_AFTER_SECONDS={elapsed:.3f} "
                    f"ERROR_TYPE={type(exc).__name__} ERROR={exc}"
                )
                continue

            elapsed = time.perf_counter() - started
            if response.status_code == 200:
                samples.append(elapsed)
            answer_chars = (
                len(response.json().get("answer", ""))
                if response.status_code == 200
                else 0
            )
            print(
                f"CHAT_SAMPLE_{number}_SECONDS={elapsed:.3f} "
                f"HTTP_STATUS={response.status_code} ANSWER_CHARS={answer_chars}"
            )

        print(f"CHAT_SAMPLE_COUNT={len(samples)}")
        print(f"CHAT_FAILED_SAMPLE_COUNT={len(failed_samples)}")
        if samples:
            print(f"CHAT_MIN_SECONDS={min(samples):.3f}")
            print(f"CHAT_MEDIAN_SECONDS={statistics.median(samples):.3f}")
            print(f"CHAT_MEAN_SECONDS={statistics.mean(samples):.3f}")
            print(f"CHAT_MAX_SECONDS={max(samples):.3f}")
        else:
            print("CHAT_LATENCY_STATS=UNAVAILABLE_NO_SUCCESSFUL_REQUESTS")


if __name__ == "__main__":
    main()
