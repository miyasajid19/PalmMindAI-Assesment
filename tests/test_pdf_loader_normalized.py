from pathlib import Path

import pytest

pytest.importorskip("pymupdf")

from app.rag.pdf_reader.loader import PDFLoader


@pytest.fixture
def sample_pdf() -> Path:
    pdf_path = Path(__file__).parents[1] / "data" / "HiPPA.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF not found: {pdf_path}")
    return pdf_path


def test_pdf_loader_normalizes_pages(sample_pdf):
    loader = PDFLoader(text_threshold=100, ocr_engine="fastocr")
    documents = loader.load_pdf(sample_pdf)

    assert documents

    for expected_page, document in enumerate(documents, start=1):
        assert isinstance(document.page_content, str)
        assert document.metadata.get("page") == expected_page
        assert "\x00" not in document.page_content
        assert "\r" not in document.page_content
