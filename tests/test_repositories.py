import uuid

from app.core.database import get_db_session
from app.repositories import DocumentRepository, DocumentVersionRepository


def test_document_and_version_repositories_use_document_ids_only():
    with get_db_session() as db:
        try:
            document_repo = DocumentRepository(db)
            version_repo = DocumentVersionRepository(db)
            document = document_repo.create(
                title=f"Assignment test document {uuid.uuid4()}",
                document_type="Text",
                description="Standalone document.",
            )
            version = version_repo.create(
                document_id=document.id,
                version_number=1,
                original_filename="sample.txt",
                storage_key=f"documents/{document.id}/sample.txt",
                summary="Sample summary.",
                status="active",
                page_count=1,
            )
            db.flush()

            retrieved = document_repo.get_by_id(document.id)
            versions = version_repo.get_by_document(document.id)

            assert retrieved is not None and retrieved.id == document.id
            assert len(versions) == 1
            assert versions[0].id == version.id
            assert versions[0].version_number == 1
        finally:
            db.rollback()


def test_document_repository_list_update_archive_and_delete():
    with get_db_session() as db:
        try:
            repository = DocumentRepository(db)
            document = repository.create(
                title=f"Update test {uuid.uuid4()}",
                document_type="PDF",
            )
            assert repository.get_all(document_type="PDF")
            updated = repository.update(document.id, title="Renamed document")
            assert updated is not None and updated.title == "Renamed document"

            archived = repository.archive(document.id)
            assert archived is not None and archived.archived is True
            unarchived = repository.unarchive(document.id)
            assert unarchived is not None and unarchived.archived is False
            assert repository.delete(document.id) is True
        finally:
            db.rollback()



def test_shared_retrieval_catalog_includes_global_eligible_versions_only():
    with get_db_session() as db:
        try:
            document_repo = DocumentRepository(db)
            version_repo = DocumentVersionRepository(db)
            document = document_repo.create(
                title=f"Shared catalogue test {uuid.uuid4()}",
                document_type="PDF",
                description="Global corpus catalogue.",
            )
            active = version_repo.create(
                document_id=document.id,
                version_number=1,
                original_filename="active.pdf",
                storage_key=f"documents/{document.id}/active.pdf",
                summary="Active version summary.",
                status="active",
                page_count=1,
            )
            superseded = version_repo.create(
                document_id=document.id,
                version_number=2,
                original_filename="historical.pdf",
                storage_key=f"documents/{document.id}/historical.pdf",
                summary="Historical version summary.",
                status="superseded",
                page_count=1,
            )
            version_repo.create(
                document_id=document.id,
                version_number=3,
                original_filename="pending.pdf",
                storage_key=f"documents/{document.id}/pending.pdf",
                summary="Pending version summary.",
                status="pending",
                page_count=1,
            )
            archived_document = document_repo.create(
                title=f"Archived shared catalogue test {uuid.uuid4()}",
                document_type="PDF",
            )
            archived_version = version_repo.create(
                document_id=archived_document.id,
                version_number=1,
                original_filename="archived.pdf",
                storage_key=f"documents/{archived_document.id}/archived.pdf",
                summary="Archived document summary.",
                status="active",
                page_count=1,
            )
            document_repo.archive(archived_document.id)

            catalogue = version_repo.get_shared_retrieval_catalog()
            matching = [
                entry for entry in catalogue
                if entry["document_id"] == str(document.id)
            ]

            assert {entry["document_version_id"] for entry in matching} == {
                str(active.id), str(superseded.id)
            }
            assert all(entry["version_status"] in {"active", "superseded"} for entry in matching)
            assert str(archived_version.id) not in {
                entry["document_version_id"] for entry in catalogue
            }
        finally:
            db.rollback()
