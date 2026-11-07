import hashlib

import pytest

from app.rag.ingestion.hash_service import PDFHashService


def test_calculate_file_hash(tmp_path):
    pdf_path = tmp_path / "document.pdf"
    content = b"PDF test content"
    pdf_path.write_bytes(content)

    assert PDFHashService.calculate_file_hash(pdf_path) == hashlib.sha256(
        content
    ).hexdigest()


def test_calculate_file_hash_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        PDFHashService.calculate_file_hash(tmp_path / "missing.pdf")


def test_calculate_file_hash_rejects_directory(tmp_path):
    with pytest.raises(ValueError):
        PDFHashService.calculate_file_hash(tmp_path)


def test_calculate_content_hash():
    text = "This is normalized PDF text."

    assert PDFHashService.calculate_content_hash(text) == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def test_calculate_content_hash_requires_text():
    with pytest.raises(TypeError):
        PDFHashService.calculate_content_hash(None)
