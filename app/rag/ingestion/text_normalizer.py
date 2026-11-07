import re
import unicodedata


class TextNormalizer:
    """
    Normalizes extracted PDF text while preserving its
    semantic and structural information for RAG.
    """

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize extracted PDF text.

        Preserves:
        - Paragraph boundaries
        - Headings
        - Punctuation
        - Table-like spacing
        - Page markers

        Removes:
        - Null bytes
        - Unwanted control characters
        - Excessive whitespace
        - Excessive blank lines
        """

        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        if not text:
            return ""

        # 1. Unicode normalization.
        text = unicodedata.normalize("NFC", text)

        # 2. Normalize line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Remove null bytes.
        text = text.replace("\x00", "")

        # 4. Remove unwanted control characters.
        # Preserve newline and tab characters.
        text = "".join(
            char
            for char in text
            if char in ("\n", "\t", "\u200c", "\u200d")
            or not unicodedata.category(char).startswith("C")
        )

        # 5. Normalize horizontal whitespace.
        # Do not collapse tabs because they may separate table columns.
        text = re.sub(r"[^\S\n\t]+", " ", text)

        # 6. Remove spaces around line boundaries.
        lines = [
            line.strip()
            for line in text.split("\n")
        ]

        text = "\n".join(lines)

        # 7. Limit excessive blank lines to two.
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 8. Remove leading and trailing whitespace.
        return text.strip()

    @staticmethod
    def normalize_pages(pages: list[str]) -> list[str]:
        """
        Normalize extracted text page by page.

        Keeps page boundaries intact.
        """
        if not isinstance(pages, list):
            raise TypeError("pages must be a list of strings.")

        normalized_pages = []

        for page_number, page_text in enumerate(pages, start=1):
            if not isinstance(page_text, str):
                raise TypeError(
                    f"Page {page_number} must contain a string."
                )

            normalized_pages.append(
                TextNormalizer.normalize(page_text)
            )

        return normalized_pages

    @staticmethod
    def combine_pages(pages: list[str]) -> str:
        """
        Combine normalized pages while retaining explicit
        page boundaries.
        """
        normalized_pages = TextNormalizer.normalize_pages(pages)

        return "\n\n".join(
            f"[PAGE {page_number}]\n{page_text}"
            for page_number, page_text in enumerate(
                normalized_pages,
                start=1,
            )
        )