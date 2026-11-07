"""Manually exercise the Generation Agent with realistic reranker-shaped inputs.

Run from the project root:
    uv run python scripts/generator.py

This script calls the configured LLM for every scenario in both non-streaming
and streaming modes. It does not call the retrieval stack or external stores.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.Generation import (
    GenerationAgent,
    GenerationStreamEventType,
)


def make_candidate(
    *,
    chunk_id: str,
    content: str,
    title: str,
    page: int,
    section: str,
    rank: int,
    reranker_score: float,
) -> dict[str, Any]:
    """Build the exact dictionary shape returned by CrossEncoderReranker."""
    return {
        "chunk_id": chunk_id,
        "page_content": content,
        "metadata": {
            "document_title": title,
            "source_filename": title,
            "page": page,
            "section_heading": section,
        },
        "global_rank": rank,
        "global_rrf_score": 1.0 / (60 + rank),
        "provenance": [
            {
                "query": "hardcoded demonstration retrieval query",
                "branch_index": 0,
                "rank": rank,
                "score": 0.90 - (rank * 0.05),
            }
        ],
        "reranker_score": reranker_score,
    }


SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "1. PERFECT / HIGHLY RELEVANT MATCH",
        "query": "How many annual leave days do employees receive, and how many can they carry forward?",
        "history": [
            HumanMessage(content="I am checking the employee leave policy."),
            AIMessage(content="Sure. What would you like to know about it?"),
        ],
        "documents": [
            make_candidate(
                chunk_id="demo-leave-001",
                content=(
                    "Employees are entitled to 20 days of annual leave per "
                    "calendar year. Unused annual leave may be carried forward "
                    "up to a maximum of 5 days."
                ),
                title="Employee Leave Policy.pdf",
                page=4,
                section="Annual Leave Entitlement",
                rank=1,
                reranker_score=0.96,
            ),
            make_candidate(
                chunk_id="demo-leave-002",
                content=(
                    "Annual leave requests must be submitted to the employee's "
                    "manager for approval before the leave begins."
                ),
                title="Employee Leave Policy.pdf",
                page=5,
                section="Leave Approval",
                rank=2,
                reranker_score=0.88,
            ),
        ],
    },
    {
        "name": "2. PARTIAL / INCOMPLETE MATCH",
        "query": "How many annual leave days and sick leave days do employees receive?",
        "history": [],
        "documents": [
            make_candidate(
                chunk_id="demo-partial-001",
                content=(
                    "Employees are entitled to 20 days of annual leave per "
                    "calendar year. Unused annual leave may be carried forward "
                    "up to a maximum of 5 days."
                ),
                title="Employee Leave Policy.pdf",
                page=4,
                section="Annual Leave Entitlement",
                rank=1,
                reranker_score=0.84,
            ),
            make_candidate(
                chunk_id="demo-partial-002",
                content=(
                    "Annual leave should be planned in consultation with the "
                    "employee's reporting manager."
                ),
                title="Employee Leave Policy.pdf",
                page=5,
                section="Leave Planning",
                rank=2,
                reranker_score=0.68,
            ),
        ],
    },
    {
        "name": "3. IRRELEVANT / NO USEFUL MATCH",
        "query": "What is the company's parental leave policy and how long is it?",
        "history": [
            HumanMessage(content="I have a question about employee benefits."),
            AIMessage(content="Go ahead and ask your question."),
        ],
        "documents": [
            make_candidate(
                chunk_id="demo-irrelevant-001",
                content=(
                    "The office cafeteria operates from 9:00 AM to 6:00 PM "
                    "on working days. Employees can use their access cards "
                    "to make purchases."
                ),
                title="Office Facilities Guide.pdf",
                page=2,
                section="Cafeteria",
                rank=1,
                # Above the selector's default 0.60 threshold, but the
                # retrieved text is still semantically unrelated to this query.
                reranker_score=0.62,
            ),
            make_candidate(
                chunk_id="demo-irrelevant-002",
                content=(
                    "Employees should report building maintenance issues "
                    "to the facilities help desk."
                ),
                title="Office Facilities Guide.pdf",
                page=7,
                section="Building Maintenance",
                rank=2,
                reranker_score=0.61,
            ),
        ],
    },
]


def print_json(value: Any) -> None:
    """Print Pydantic models and ordinary values as readable JSON."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, indent=2, ensure_ascii=False))


async def run_scenario(
    agent: GenerationAgent,
    scenario: dict[str, Any],
) -> None:
    name = scenario["name"]
    query = scenario["query"]
    documents = scenario["documents"]
    history = scenario["history"]

    print("\n" + "#" * 100)
    print(name)
    print("#" * 100)
    print(f"Query: {query}")
    print(f"Conversation history messages: {len(history)}")
    print(f"Reranked candidate documents: {len(documents)}")

    print("\n" + "-" * 36 + " NON-STREAMING " + "-" * 36)
    try:
        result = agent.generate(
            current_query=query,
            retrieved_documents=documents,
            conversation_history=history,
        )
        print_json(result)
    except Exception as exc:
        print(f"NON-STREAMING ERROR: {type(exc).__name__}: {exc}")

    print("\n" + "-" * 36 + " STREAMING " + "-" * 39)
    print("Live answer text:")
    try:
        async for event in agent.astream_generate(
            current_query=query,
            retrieved_documents=documents,
            conversation_history=history,
        ):
            if event.event_type == GenerationStreamEventType.CONTENT_DELTA:
                print(event.delta, end="", flush=True)
            elif event.event_type == GenerationStreamEventType.COMPLETED:
                print("\n\nFinal validated structured output:")
                print_json(event.output)
    except Exception as exc:
        print(f"\nSTREAMING ERROR: {type(exc).__name__}: {exc}")
        print(
            "The stream did not produce a final validated result. "
            "Any text printed above was provisional."
        )


async def main() -> None:
    print("=" * 100)
    print("SAVIRACORTEX GENERATION PIPELINE - MANUAL SMOKE TEST")
    print("=" * 100)
    print("The configured LLM endpoint will be called 6 times (3 scenarios x 2 modes).")
    print("Inputs are hardcoded; retrieval, reranking, and databases are not invoked.")

    # One agent instance reuses the configured model client for all scenarios.
    agent = GenerationAgent()

    for scenario in SCENARIOS:
        await run_scenario(agent, scenario)

    print("\n" + "=" * 100)
    print("GENERATION SMOKE TEST FINISHED")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
