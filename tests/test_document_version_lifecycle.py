import uuid

import pytest

from app.core.database import get_db_session
from app.repositories import DocumentRepository, DocumentVersionRepository


def create_document_and_version(db, status="pending", file_hash=None, content_hash=None):
    document = DocumentRepository(db).create(
        title=f"Lifecycle doc {uuid.uuid4()}",
        document_type="PDF",
    )
    version = DocumentVersionRepository(db).create(
        document_id=document.id,
        version_number=1,
        original_filename="sample.pdf",
        storage_key=f"test/documents/{uuid.uuid4()}.pdf",
        file_hash=file_hash,
        content_hash=content_hash,
        summary="Test summary.",
        status=status,
        page_count=2,
    )
    return document, version


def test_document_defaults_not_archived_and_version_pending():
    with get_db_session() as db:
        try:
            document, version = create_document_and_version(db)
            assert document.archived is False
            assert version.status == "pending"
        finally:
            db.rollback()


def test_archive_unarchive_changes_document_eligibility():
    with get_db_session() as db:
        try:
            document, active = create_document_and_version(db, status="active")
            repository = DocumentRepository(db)
            versions = DocumentVersionRepository(db)

            repository.archive(document.id)
            assert document.archived is True
            assert versions.get_active_version(document.id) is None
            assert versions.get_retrieval_eligible_versions(document.id) == []

            repository.unarchive(document.id)
            assert document.archived is False
            assert versions.get_active_version(document.id).id == active.id
        finally:
            db.rollback()


def test_hash_matching_is_global_and_not_organization_scoped():
    with get_db_session() as db:
        try:
            hash_value = uuid.uuid4().hex
            _, version = create_document_and_version(
                db,
                status="active",
                file_hash=hash_value,
                content_hash=hash_value,
            )
            matches = DocumentVersionRepository(db).get_by_hash(file_hash=hash_value)
            assert [item.id for item in matches] == [version.id]
        finally:
            db.rollback()


def test_activation_supersedes_previous_active_version():
    with get_db_session() as db:
        try:
            document, active_v1 = create_document_and_version(db, status="active")
            repository = DocumentVersionRepository(db)
            pending_v2 = repository.create(
                document_id=document.id,
                version_number=2,
                original_filename="sample-v2.pdf",
                storage_key=f"test/documents/{uuid.uuid4()}.pdf",
                summary="Version two summary.",
                status="pending",
                page_count=3,
            )

            activated, previous = repository.activate_pending_version(pending_v2.id)
            assert activated.id == pending_v2.id and activated.status == "active"
            assert previous is not None and previous.id == active_v1.id
            assert previous.status == "superseded"
            assert repository.get_active_version(document.id).id == pending_v2.id
        finally:
            db.rollback()


def test_only_pending_versions_can_be_activated():
    with get_db_session() as db:
        try:
            _, active_version = create_document_and_version(db, status="active")
            with pytest.raises(ValueError, match="Only pending versions"):
                DocumentVersionRepository(db).activate_pending_version(active_version.id)
        finally:
            db.rollback()
