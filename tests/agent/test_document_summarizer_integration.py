
import os
from pathlib import Path

import pytest

# pytest.importorskip("pymupdf")

from app.agents.Document_summarizer import (
    DocumentSummarizerAgent,
    DocumentSummarizerConfig,
)

from app.rag.ingestion.pdf_sampler import PDFSampler
from app.rag.pdf_reader.loader import PDFLoader


def test_real_document_summarization():

    pdf_path = Path(__file__).parents[2] / "data" / "sample.pdf"

    if not pdf_path.exists():
        pytest.skip(f"Integration PDF not found: {pdf_path}")

    required_settings = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
    missing_settings = [
        name for name in required_settings if not os.getenv(name)
    ]
    if missing_settings:
        pytest.skip(
            "LLM integration settings are missing: "
            + ", ".join(missing_settings)
        )

    # Initialize PDF components
    pdf_loader = PDFLoader()
    pdf_sampler = PDFSampler(max_pages=5)

    # Initialize summarizer
    summarizer = DocumentSummarizerAgent(
        config=DocumentSummarizerConfig(
            summary_length=500,
            summary_style="descriptive and technical",
            language="English",
            temperature=0.2,
            max_output_tokens=2000,
        )
    )

    # Extract PDF pages
    extracted_pages = pdf_loader.load_pdf(str(pdf_path))

    assert extracted_pages, "No pages extracted from PDF."

    # Sample pages
    sampled_pages = pdf_sampler.sample(extracted_pages)

    assert sampled_pages, "No pages sampled from PDF."

    # Generate summary using actual LLM
    result = summarizer.summarize(sampled_pages)

    # Display results
    print("\n========== DOCUMENT SUMMARY ==========\n")

    print("Title:", result.title)
    print("Document Type:", result.document_type)
    print("Description:", result.description)

    print("\nSummary:\n")
    print(result.summary)

    # Basic output validation
    assert result.title
    assert result.document_type
    assert result.description
    assert result.summary
