from uuid import uuid4

import pytest

from app.core.database import get_db_session
from app.models.documents import Document
from app.models.document_versions import DocumentVersion
from app.services.document_registration import (
    DocumentRegistrationData,
    DocumentRegistrationService,
    VersionRegistrationData,
)


def make_version_data(filename="document.pdf", version=1, *, file_hash=None):
    return VersionRegistrationData(
        original_filename=filename,
        storage_key=f"assignment/test/{uuid4()}/v{version}.pdf",
        file_hash=file_hash or uuid4().hex * 2,
        content_hash=f"content-{uuid4().hex}",
        summary="Test document summary.",
        page_count=5,
    )


def test_register_new_document_stores_document_and_first_version():
    with get_db_session() as db:
        try:
            service = DocumentRegistrationService(db)
            result = service.register_new_document(
                DocumentRegistrationData(
                    title=f"Assignment document {uuid4()}",
                    document_type="PDF",
                    description="Assignment-focused document.",
                ),
                make_version_data(),
            )
            document = db.get(Document, result.document_id)
            version = db.get(DocumentVersion, result.document_version_id)

            assert document is not None
            assert version is not None
            assert version.document_id == document.id
            assert version.version_number == 1
            assert version.status == "pending"
            assert document.archived is False
        finally:
            db.rollback()


def test_register_next_version_for_document():
    with get_db_session() as db:
        try:
            service = DocumentRegistrationService(db)
            first = service.register_new_document(
                DocumentRegistrationData(
                    title=f"Versioned document {uuid4()}",
                    document_type="PDF",
                ),
                make_version_data("v1.pdf", 1),
            )
            first_version = db.get(DocumentVersion, first.document_version_id)
            first_version.status = "active"
            db.flush()

            second = service.register_new_version(
                first.document_id,
                make_version_data("v2.pdf", 2),
            )
            second_version = db.get(DocumentVersion, second.document_version_id)

            assert second.version_number == 2
            assert second_version is not None
            assert second_version.status == "pending"
            assert second_version.supersedes_version_id == first_version.id
            assert first_version.status == "active"
        finally:
            db.rollback()


def test_register_version_rejects_missing_document():
    with get_db_session() as db:
        try:
            with pytest.raises(ValueError, match="Document not found"):
                DocumentRegistrationService(db).register_new_version(
                    uuid4(),
                    make_version_data(),
                )
        finally:
            db.rollback()


def test_register_version_rejects_archived_document():
    with get_db_session() as db:
        try:
            service = DocumentRegistrationService(db)
            first = service.register_new_document(
                DocumentRegistrationData(
                    title=f"Archived document {uuid4()}",
                    document_type="PDF",
                ),
                make_version_data(),
            )
            service.document_repository.archive(first.document_id)

            with pytest.raises(ValueError, match="archived document"):
                service.register_new_version(
                    first.document_id,
                    make_version_data("v2.pdf", 2),
                )
        finally:
            db.rollback()


def test_registration_validates_required_document_fields():
    with pytest.raises(ValueError, match="title"):
        DocumentRegistrationService._validate_document_data(
            DocumentRegistrationData(title=" ", document_type="PDF")
        )


def test_registration_validates_file_hash_and_summary():
    with pytest.raises(ValueError, match="File SHA-256 hash"):
        DocumentRegistrationService._validate_version_data(
            VersionRegistrationData(
                original_filename="x.pdf",
                storage_key="x",
                summary="valid summary",
            )
        )
    with pytest.raises(ValueError, match="summary cannot be empty"):
        DocumentRegistrationService._validate_version_data(
            VersionRegistrationData(
                original_filename="x.pdf",
                storage_key="x",
                file_hash="a" * 64,
                summary=" ",
            )
        )


def test_registration_supports_preassigned_ids():
    document_id, version_id = uuid4(), uuid4()
    with get_db_session() as db:
        try:
            result = DocumentRegistrationService(db).register_new_document(
                DocumentRegistrationData(
                    title=f"Assigned ID doc {uuid4()}",
                    document_type="TXT",
                ),
                make_version_data("text.txt"),
                document_id=document_id,
                version_id=version_id,
            )
            assert result.document_id == document_id
            assert result.document_version_id == version_id
        finally:
            db.rollback()
