import pytest

pymupdf = pytest.importorskip("pymupdf")

from app.rag.pdf_reader.validator import PDFValidator


@pytest.fixture
def valid_pdf(tmp_path):
    pdf_path = tmp_path / "document.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    return pdf_path


def test_valid_pdf(valid_pdf):
    result = PDFValidator().validate(valid_pdf)

    assert result.is_valid
    assert result.file_size > 0
    assert result.page_count == 1
    assert result.errors == []


def test_missing_pdf_is_invalid(tmp_path):
    result = PDFValidator().validate(tmp_path / "missing.pdf")

    assert not result.is_valid
    assert "File does not exist." in result.errors


def test_invalid_pdf_signature(tmp_path):
    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_bytes(b"not a PDF")

    result = PDFValidator().validate(pdf_path)

    assert not result.is_valid
    assert any("PDF signature" in error for error in result.errors)


def test_validator_rejects_invalid_limits():
    with pytest.raises(ValueError):
        PDFValidator(max_file_size_mb=0)

    with pytest.raises(ValueError):
        PDFValidator(max_pages=0)
