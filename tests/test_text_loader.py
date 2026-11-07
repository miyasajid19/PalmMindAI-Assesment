

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.rag.pdf_reader.loader import DocumentLoader, TextLoader
from app.rag.pdf_reader.validator import TextValidator


def test_text_loader_normalizes_utf8_and_preserves_source_metadata(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_bytes(b"\xef\xbb\xbfHello  world.\r\n\r\nSecond\x00 line.")

    documents = TextLoader().load_txt(source)

    assert len(documents) == 1
    document = documents[0]
    assert isinstance(document, Document)
    assert document.page_content == "Hello world.\n\nSecond line."
    assert document.metadata == {
        "source": str(source.resolve()),
        "page": 1,
        "extraction_method": "plain_text",
        "encoding": "utf-8",
    }


def test_text_loader_rejects_non_txt_files(tmp_path):
    source = tmp_path / "sample.pdf"
    source.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a .txt file"):
        TextLoader().load_txt(source)


def test_text_loader_rejects_non_utf8_files(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="UTF-8"):
        TextLoader().load_txt(source)


def test_text_validator_accepts_nonempty_utf8_txt(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("Hello from text ingestion", encoding="utf-8")

    result = TextValidator().validate(source)

    assert result.is_valid is True
    assert result.page_count == 1
    assert result.errors == []


@pytest.mark.parametrize(
    ("filename", "content", "expected_error"),
    [
        ("empty.txt", b"", "empty"),
        ("wrong.pdf", b"some text", "extension"),
        ("invalid.txt", b"\xff", "UTF-8"),
    ],
)
def test_text_validator_rejects_invalid_text_files(
    tmp_path, filename, content, expected_error
):
    source = tmp_path / filename
    source.write_bytes(content)

    result = TextValidator().validate(source)

    assert result.is_valid is False
    assert expected_error.lower() in " ".join(result.errors).lower()


@pytest.mark.parametrize("suffix", [".pdf", ".PDF"])
def test_document_loader_dispatches_pdf_to_pdf_handler(tmp_path, suffix):
    source = tmp_path / f"sample{suffix}"
    source.write_bytes(b"pdf fixture")
    pdf_handler = MagicMock()
    pdf_handler.load_pdf.return_value = [Document(page_content="PDF")]
    text_handler = MagicMock()

    loader = DocumentLoader(pdf_loader=pdf_handler, text_loader=text_handler)
    documents = loader.load_document(source)

    assert documents[0].page_content == "PDF"
    pdf_handler.load_pdf.assert_called_once_with(source.resolve())
    text_handler.load_txt.assert_not_called()


def test_document_loader_dispatches_txt_to_text_handler(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("text fixture", encoding="utf-8")
    pdf_handler = MagicMock()
    text_handler = MagicMock()
    text_handler.load_txt.return_value = [Document(page_content="TXT")]

    loader = DocumentLoader(pdf_loader=pdf_handler, text_loader=text_handler)
    documents = loader.load_document(source)

    assert documents[0].page_content == "TXT"
    text_handler.load_txt.assert_called_once_with(source.resolve())
    pdf_handler.load_pdf.assert_not_called()


def test_document_loader_rejects_unsupported_extension(tmp_path):
    source = tmp_path / "sample.docx"
    source.write_bytes(b"not supported")
    pdf_handler = MagicMock()
    text_handler = MagicMock()

    loader = DocumentLoader(pdf_loader=pdf_handler, text_loader=text_handler)

    with pytest.raises(ValueError, match="Supported extensions are: .pdf, .txt"):
        loader.load_document(source)

    pdf_handler.load_pdf.assert_not_called()
    text_handler.load_txt.assert_not_called()
