import hashlib
from pathlib import Path


class PDFHashService:
    """
    Generates SHA-256 hashes for original PDF files
    and their normalized extracted text.
    """

    @staticmethod
    def calculate_file_hash(
        file_path: str | Path,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """
        Calculate SHA-256 of the original PDF file.

        Reads the file in chunks to avoid loading large PDFs
        entirely into memory.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        sha256 = hashlib.sha256()

        with path.open("rb") as file:
            while chunk := file.read(chunk_size):
                sha256.update(chunk)

        return sha256.hexdigest()

    @staticmethod
    def calculate_content_hash(normalized_text: str) -> str:
        """
        Calculate SHA-256 of normalized extracted text.
        """
        if not isinstance(normalized_text, str):
            raise TypeError("normalized_text must be a string")

        content = normalized_text.encode("utf-8")

        return hashlib.sha256(content).hexdigest()