import inspect

from app.core.database import Base
from app.models import Document, DocumentVersion, InterviewBooking
from app.repositories import DocumentRepository, DocumentVersionRepository
from app.services.document_registration.registration_service import (
    DocumentRegistrationService,
)


def test_assignment_models_register_only_expected_tables():
    assert set(Base.metadata.tables) == {
        "documents",
        "document_versions",
        "interview_bookings",
    }


def test_document_schema_has_no_tenant_or_user_columns():
    columns = set(Document.__table__.columns.keys())
    assert columns == {
        "id",
        "title",
        "document_type",
        "description",
        "archived",
        "created_at",
    }


def test_document_version_is_referenced_by_document_id():
    assert "document_id" in DocumentVersion.__table__.columns
    foreign_keys = list(DocumentVersion.__table__.c.document_id.foreign_keys)
    document_foreign_keys = [
        foreign_key
        for foreign_key in foreign_keys
        if foreign_key.target_fullname == "documents.id"
    ]
    assert len(document_foreign_keys) == 1


def test_interview_booking_fields_are_present():
    assert set(InterviewBooking.__table__.columns.keys()) == {
        "id",
        "name",
        "email",
        "interview_date",
        "interview_time",
        "created_at",
    }


def test_ingestion_persistence_interfaces_have_no_tenant_or_user_arguments():
    methods = (
        DocumentRepository.create,
        DocumentRepository.get_by_id,
        DocumentRepository.get_by_title,
        DocumentRepository.get_all,
        DocumentRepository.update,
        DocumentRepository.archive,
        DocumentRepository.unarchive,
        DocumentRepository.delete,
        DocumentVersionRepository.create,
        DocumentVersionRepository.get_by_id,
        DocumentVersionRepository.get_by_document,
        DocumentVersionRepository.get_active_version,
        DocumentVersionRepository.get_by_hash,
        DocumentVersionRepository.get_retrieval_eligible_versions,
        DocumentVersionRepository.update,
        DocumentVersionRepository.change_status,
        DocumentVersionRepository.activate_pending_version,
        DocumentVersionRepository.delete,
        DocumentRegistrationService.register_new_document,
        DocumentRegistrationService.register_new_version,
    )
    forbidden_arguments = {"organization_id", "user_id", "uploaded_by_user_id"}

    for method in methods:
        assert forbidden_arguments.isdisjoint(inspect.signature(method).parameters), (
            f"{method.__qualname__} still accepts tenant/user arguments"
        )
