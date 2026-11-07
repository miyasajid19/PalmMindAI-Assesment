"""Native-first PDF extraction with configurable page-level OCR fallback."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .native import native_client
from .ocr import ocr_client
from ..ingestion.text_normalizer import TextNormalizer
from ...core.config import settings


OCREngine = Literal["tesseract", "fastocr"]


def _language_codes(value: str | None) -> list[str]:
    return [
        language.strip()
        for language in (value or "en").split(",")
        if language.strip()
    ]


def _tesseract_language(value: str | None) -> str:
    mapping = {
        "en": "eng",
        "hi": "hin",
        "ne": "nep",
        "fr": "fra",
        "de": "deu",
        "es": "spa",
        "zh": "chi_sim",
        "ja": "jpn",
    }
    return "+".join(
        mapping.get(language, language)
        for language in _language_codes(value)
    )


class PDFLoader:
    """Extract native text first and OCR selected pages when configured."""

    SUPPORTED_OCR_ENGINES = {"tesseract", "fastocr"}

    def __init__(
        self,
        text_threshold: int | None = None,
        ocr_engine: OCREngine | None = None,
        ocr_options: dict[str, Any] | None = None,
        *,
        ocr_all_pages: bool | None = None,
        merge_with_native: bool | None = None,
    ) -> None:
        threshold = (
            settings.MIN_NATIVE_CHARS
            if text_threshold is None
            else text_threshold
        )
        engine = (
            (settings.OCR_ENGINE or "fastocr").strip().lower()
            if ocr_engine is None
            else ocr_engine
        )

        if isinstance(threshold, bool) or not isinstance(threshold, int):
            raise TypeError("text_threshold must be an integer.")
        if threshold < 0:
            raise ValueError("text_threshold cannot be negative.")
        if engine not in self.SUPPORTED_OCR_ENGINES:
            raise ValueError(
                f"Unsupported OCR engine {engine!r}. "
                f"Choose from: {', '.join(sorted(self.SUPPORTED_OCR_ENGINES))}."
            )

        configured_languages = _language_codes(settings.OCR_LANGS)
        defaults = {
            "dpi": settings.RENDER_DPI,
            "language": _tesseract_language(settings.OCR_LANGS),
            "languages": ",".join(configured_languages),
            "config": "--psm 3",
        }
        defaults.update(ocr_options or {})

        self.text_threshold = threshold
        self.ocr_engine = engine
        self.ocr_options = defaults
        self.ocr_all_pages = (
            settings.OCR_ALL_PAGES
            if ocr_all_pages is None
            else ocr_all_pages
        )
        self.merge_with_native = (
            settings.OCR_MERGE_WITH_NATIVE
            if merge_with_native is None
            else merge_with_native
        )

        if not isinstance(self.ocr_all_pages, bool):
            raise TypeError("ocr_all_pages must be a boolean.")
        if not isinstance(self.merge_with_native, bool):
            raise TypeError("merge_with_native must be a boolean.")

    @staticmethod
    def _text_length(text: str | Document) -> int:
        value = text.page_content if isinstance(text, Document) else text
        return len("".join((value or "").split()))

    @staticmethod
    def _normalize_documents(documents: list[Document]) -> list[Document]:
        return [
            Document(
                page_content=TextNormalizer.normalize(document.page_content or ""),
                metadata=dict(document.metadata or {}),
            )
            for document in documents
        ]

    @staticmethod
    def _merge_text(native_text: str, ocr_text: str) -> str:
        native = (native_text or "").strip()
        ocr = (ocr_text or "").strip()
        if not native:
            return ocr
        if not ocr:
            return native

        native_key = " ".join(native.casefold().split())
        ocr_key = " ".join(ocr.casefold().split())
        if native_key in ocr_key:
            return ocr
        if ocr_key in native_key:
            return native
        return f"{native}\n\n{ocr}"

    @staticmethod
    def _mean_ocr_confidence(metadata: dict[str, Any]) -> float | None:
        configured_mean = metadata.get("mean_confidence")
        if (
            isinstance(configured_mean, (int, float))
            and not isinstance(configured_mean, bool)
            and 0.0 <= float(configured_mean) <= 1.0
        ):
            return float(configured_mean)

        detections = metadata.get("detections")
        if not isinstance(detections, list):
            return None
        values = [
            float(item["confidence"])
            for item in detections
            if isinstance(item, dict)
            and isinstance(item.get("confidence"), (int, float))
            and not isinstance(item.get("confidence"), bool)
            and 0.0 <= float(item["confidence"]) <= 1.0
        ]
        return sum(values) / len(values) if values else None

    @classmethod
    def _ocr_is_usable(
        cls,
        native_text: str,
        ocr_text: str,
        ocr_metadata: dict[str, Any],
    ) -> bool:
        if cls._text_length(ocr_text) == 0:
            return False
        if cls._text_length(native_text) == 0:
            return True
        confidence = cls._mean_ocr_confidence(ocr_metadata)
        return confidence is None or confidence >= 0.5

    @classmethod
    def _ocr_is_better(
        cls,
        native_text: str,
        ocr_text: str,
        ocr_metadata: dict[str, Any],
    ) -> bool:
        native_length = cls._text_length(native_text)
        ocr_length = cls._text_length(ocr_text)
        return (
            ocr_length > native_length
            and cls._ocr_is_usable(native_text, ocr_text, ocr_metadata)
        )

    def _run_page_ocr(self, pdf_path: Path, page_number: int) -> Document:
        options = self.ocr_options

        if self.ocr_engine == "tesseract":
            result = ocr_client.tesseract_pdf_page(
                pdf_path,
                page_number=page_number,
                dpi=options.get("dpi"),
                language=options.get("language", "eng"),
                config=options.get("config", "--psm 3"),
            )
        else:
            result = ocr_client.fastocr_pdf_page(
                pdf_path,
                page_number=page_number,
                dpi=options.get("dpi"),
                languages=options.get("languages", "en"),
            )

        if isinstance(result, Document):
            return result

        if isinstance(result, list):
            if any(not isinstance(item, Document) for item in result):
                raise TypeError("FastOCR must return Document instances.")
            text = "\n".join(
                item.page_content.strip()
                for item in result
                if item.page_content and item.page_content.strip()
            )
            metadata = dict(result[0].metadata or {}) if result else {}
            if len(result) > 1:
                metadata["ocr_document_count"] = len(result)
            return Document(page_content=text, metadata=metadata)

        raise TypeError(
            f"{self.ocr_engine} returned {type(result).__name__}; "
            "expected Document or list[Document]."
        )

    def load_pdf(self, pdf_path: str | Path) -> list[Document]:
        path = Path(pdf_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"PDF file does not exist: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {path}")

        native_documents = native_client.load_pdf(path)
        final_documents: list[Document] = []

        for page_index, native_document in enumerate(native_documents):
            page_number = page_index + 1
            native_text = native_document.page_content or ""
            native_length = self._text_length(native_text)
            should_ocr = self.ocr_all_pages or native_length < self.text_threshold

            metadata = dict(native_document.metadata or {})
            metadata.update({
                "source": str(path),
                "page": page_number,
                "native_text_length": native_length,
                "ocr_attempted": False,
                "ocr_used": False,
            })

            final_text = native_text
            if should_ocr:
                ocr_document = self._run_page_ocr(path, page_number)
                ocr_text = ocr_document.page_content or ""
                ocr_length = self._text_length(ocr_text)
                metadata.update(dict(ocr_document.metadata or {}))
                metadata.update({
                    "source": str(path),
                    "page": page_number,
                    "native_text_length": native_length,
                    "ocr_text_length": ocr_length,
                    "ocr_attempted": True,
                })

                confidence = self._mean_ocr_confidence(
                    dict(ocr_document.metadata or {})
                )
                metadata["ocr_mean_confidence"] = confidence
                ocr_is_better = self._ocr_is_better(
                    native_text,
                    ocr_text,
                    dict(ocr_document.metadata or {}),
                )

                ocr_usable = self._ocr_is_usable(
                    native_text,
                    ocr_text,
                    dict(ocr_document.metadata or {}),
                )

                # Do not let a tiny OCR fragment contaminate useful native
                # text, and replace genuinely insufficient native text before
                # considering an explicit merge.
                if (
                    self.merge_with_native
                    and ocr_usable
                    and not (
                        ocr_is_better
                        and self._text_length(native_text) < 10
                    )
                    and (
                        ocr_is_better
                        or confidence is not None
                    )
                ):
                    final_text = self._merge_text(native_text, ocr_text)
                    ocr_contributed = final_text != native_text
                    metadata["extraction_method"] = (
                        f"pymupdf+{self.ocr_engine}"
                        if ocr_contributed
                        else "pymupdf"
                    )
                elif ocr_is_better:
                    final_text = ocr_text
                    ocr_contributed = True
                    metadata["extraction_method"] = self.ocr_engine
                else:
                    # Retain native extraction when OCR is empty, shorter,
                    # or has low confidence compared with available native text.
                    ocr_contributed = False
                    metadata["extraction_method"] = "pymupdf"

                metadata["ocr_used"] = ocr_contributed
            else:
                metadata["extraction_method"] = "pymupdf"

            final_documents.append(
                Document(page_content=final_text, metadata=metadata)
            )

        return self._normalize_documents(final_documents)

    def load(self, pdf_path: str | Path) -> list[Document]:
        return self.load_pdf(pdf_path)


pdf_loader = PDFLoader()


if __name__ == "__main__":
    from rich import print

    pdf_path = r"C:\Users\miyas\Downloads\organized.pdf"
    documents = pdf_loader.load(pdf_path)
    for document in documents:
        print(
            f"Page {document.metadata.get('page')}: "
            f"{len(document.page_content)} characters"
        )
        print(document)
        print("-" * 80)


class TextLoader:
    """Load UTF-8 plain-text files as normalized LangChain Documents."""

    @staticmethod
    def load_txt(text_path: str | Path) -> list[Document]:
        path = Path(text_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Text file does not exist: {path}")
        if path.suffix.lower() != ".txt":
            raise ValueError(f"Expected a .txt file: {path}")

        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Text file must be encoded as UTF-8: {path}"
            ) from exc

        normalized = TextNormalizer.normalize(text)

        return [
            Document(
                page_content=normalized,
                metadata={
                    "source": str(path),
                    "page": 1,
                    "extraction_method": "plain_text",
                    "encoding": "utf-8",
                },
            )
        ]

    def load(self, text_path: str | Path) -> list[Document]:
        return self.load_txt(text_path)


text_loader = TextLoader()

class DocumentLoader:
    """Dispatch documents to the format-specific extraction loader."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(
        self,
        *,
        pdf_loader: PDFLoader | Any | None = None,
        text_loader: TextLoader | Any | None = None,
    ) -> None:
        self.pdf_loader = pdf_loader or globals()["pdf_loader"]
        self.text_loader = text_loader or globals()["text_loader"]

    def load_document(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path).expanduser().resolve()
        extension = path.suffix.lower()
        if extension == ".pdf":
            return self.pdf_loader.load_pdf(path)
        if extension == ".txt":
            return self.text_loader.load_txt(path)
        raise ValueError(
            f"Unsupported document type {extension!r}. "
            "Supported extensions are: .pdf, .txt."
        )

    def load(self, file_path: str | Path) -> list[Document]:
        """Alias for load_document for conventional loader interfaces."""
        return self.load_document(file_path)


document_loader = DocumentLoader()

