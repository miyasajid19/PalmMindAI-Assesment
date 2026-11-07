"""Generate and print shared-corpus document candidates.

Run from the project root:
    uv run python scripts/candidate_documents.py --query "What is in this document?"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.Multi_query_generator.schemas import GeneratedQuery, QueryVersionScope
from app.core.database import get_db_session
from app.rag.retrieval.candidate_document_generator import CandidateDocumentGenerator
from app.rag.vector_stores.qdrant_summary import summary_store
from app.repositories.document_version_repositories import DocumentVersionRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate shared-corpus document candidates from summaries."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--version-scope",
        choices=[scope.value for scope in QueryVersionScope],
        default=QueryVersionScope.BOTH.value,
    )
    parser.add_argument(
        "--target-year",
        action="append",
        type=int,
        default=[],
        help="Limit results to a version year; may be supplied multiple times.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.40)
    return parser


async def generate_candidates(args: argparse.Namespace) -> None:
    query = args.query.strip()
    if not query:
        raise ValueError("--query cannot be empty.")

    generated_query = GeneratedQuery(
        query=query,
        version_scope=QueryVersionScope(args.version_scope),
        target_versions=args.target_year,
    )

    with get_db_session() as db:
        catalogue = DocumentVersionRepository(db).get_shared_retrieval_catalog()

    generator = CandidateDocumentGenerator(
        summary_store=summary_store,
        similarity_threshold=args.threshold,
        search_limit=args.limit,
    )
    results = await generator.generate_candidates(
        queries=[generated_query],
        document_catalogue=catalogue,
    )

    result = results[0]
    print(f"Query: {result.query}")
    print(f"Version scope: {result.version_scope.value}")
    print(f"Target versions: {result.target_versions or 'any'}")
    print(f"Shared catalogue entries: {len(catalogue)}")

    if result.retrieval_error:
        print(f"Retrieval error: {result.retrieval_error}")
        return

    print(f"Candidates: {len(result.candidates)}")
    for index, candidate in enumerate(result.candidates, start=1):
        print("\n" + "=" * 90)
        print(f"Candidate #{index}")
        print(f"Document ID:         {candidate.document_id}")
        print(f"Document version ID: {candidate.document_version_id}")
        print(f"Title:               {candidate.document_title}")
        print(f"Version year:        {candidate.version_year}")
        print(f"Version status:      {candidate.version_status}")
        print(f"Similarity score:    {candidate.similarity_score:.6f}")
        print(f"Summary:\n{candidate.document_summary}")


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(generate_candidates(args))


if __name__ == "__main__":
    main()
