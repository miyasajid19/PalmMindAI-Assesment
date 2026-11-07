import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.storage import FileStorage


@pytest.fixture
def local_pdf() -> Path:
    pdf_path = Path(__file__).parents[1] / "data" / "HiPPA.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF not found: {pdf_path}")
    return pdf_path


def test_file_storage_round_trip(local_pdf, tmp_path):
    required_settings = (
        "B2_ENDPOINT_URL",
        "B2_REGION",
        "B2_KEY_ID",
        "B2_APPLICATION_KEY",
        "B2_BUCKET_NAME",
    )
    missing_settings = [
        name for name in required_settings if not os.getenv(name)
    ]
    if missing_settings:
        pytest.skip(
            "B2 integration settings are missing: "
            + ", ".join(missing_settings)
        )

    storage = FileStorage()
    document_id = uuid4()
    version_id = uuid4()
    storage_key = storage.build_document_version_key(
        document_id=document_id,
        version_id=version_id,
    )
    assert storage_key.startswith(
        f"{storage.prefix}documents/{document_id}/"
    )
    downloaded_path = tmp_path / local_pdf.name

    try:
        result = storage.upload_file(
            local_path=local_pdf,
            storage_key=storage_key,
            content_type="application/pdf",
            metadata={
                "document-id": str(document_id),
                "version-id": str(version_id),
            },
        )

        assert result
        assert storage.exists(storage_key)
        assert storage.get_metadata(storage_key) is not None
        assert storage.generate_download_url(
            storage_key,
            expiration_seconds=900,
        )

        storage.download_file(
            storage_key=storage_key,
            destination=downloaded_path,
        )

        assert downloaded_path.read_bytes() == local_pdf.read_bytes()
    finally:
        if storage.exists(storage_key):
            storage.delete_file(storage_key)
