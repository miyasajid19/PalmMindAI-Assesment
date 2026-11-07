
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch
from uuid import uuid4

import pytest
from langchain_core.documents import Document

from app.services.document_indexing.indexing_orchestrator import (
    DocumentIndexingOrchestrator,
    DocumentIndexingOrchestrationError,
)


@pytest.fixture
def setup_orchestrator(tmp_path):
    version_id = uuid4()
    document_id = uuid4()

    version = SimpleNamespace(
        id=version_id,
        document_id=document_id,
        version_number=2,
        storage_key="voice-ai/test/document.pdf",
        original_filename="document.pdf",
        summary="Test document summary.",
        file_hash=hashlib.sha256(b"fake-pdf-content").hexdigest(),
        status="pending",
        page_count=0,
    )

    document = SimpleNamespace(
        id=document_id,
        title="Test Document",
        archived=False,
    )

    activated_version = SimpleNamespace(
        id=version_id,
        document_id=document_id,
        version_number=2,
        summary="Test document summary.",
    )

    previous_version = SimpleNamespace(id=uuid4())

    db = MagicMock()
    session = MagicMock()
    session.__enter__.return_value = db
    session.__exit__.return_value = False

    file_storage = MagicMock()

    def download_file(storage_key, destination):
        Path(destination).write_bytes(b"fake-pdf-content")

    file_storage.download_file.side_effect = download_file

    loader = MagicMock()
    loader.load_pdf.return_value = [
        Document(
            page_content="Test PDF content.",
            metadata={"page": 1, "ocr_used": False},
        )
    ]

    text_loader = MagicMock()
    text_loader.load_txt.return_value = [
        Document(page_content="Test TXT content.", metadata={"page": 1})
    ]

    indexing_result = SimpleNamespace(
        document_id=str(document_id),
        document_version_id=str(version_id),
        version_number=2,
        page_count=1,
        ocr_page_count=0,
        chunk_count=3,
        qdrant_point_ids=["qdrant-1", "qdrant-2", "qdrant-3"],
        status="completed",
    )

    indexing_service = MagicMock()
    indexing_service.index_document_version.return_value = indexing_result

    summary_storage = MagicMock()
    summary_storage.upsert_summary.side_effect = [
        str(version_id),
        str(version_id),
    ]

    orchestrator = DocumentIndexingOrchestrator(
        session_factory=MagicMock(return_value=session),
        file_storage=file_storage,
        loader=loader,
        text_loader=text_loader,
        indexing_service=indexing_service,
        summary_storage=summary_storage,
    )

    return SimpleNamespace(
        orchestrator=orchestrator,
        version_id=version_id,
        version=version,
        document=document,
        activated_version=activated_version,
        previous_version=previous_version,
        db=db,
        session=session,
        file_storage=file_storage,
        loader=loader,
        text_loader=text_loader,
        indexing_service=indexing_service,
        indexing_result=indexing_result,
        summary_storage=summary_storage,
    )


def patch_version_loading(setup):
    return patch.object(
        DocumentIndexingOrchestrator,
        "_load_version",
        return_value=(setup.version, setup.document),
    )


def test_successful_indexing_and_activation(setup_orchestrator):
    setup = setup_orchestrator

    repository = MagicMock()
    repository.activate_pending_version.return_value = (
        setup.activated_version,
        None,
    )

    with (
        patch_version_loading(setup),
        patch(
            "app.services.document_indexing.indexing_orchestrator.DocumentVersionRepository",
            return_value=repository,
        ),
    ):
        result = setup.orchestrator.index_pending_version(
            setup.version_id,
            )

    assert result.status == "active"
    assert result.document_id == str(setup.version.document_id)
    assert result.document_version_id == str(setup.version_id)
    assert result.version_number == 2
    assert result.page_count == 1
    assert result.ocr_page_count == 0
    assert result.chunk_count == 3
    assert result.qdrant_point_ids == [
        "qdrant-1", "qdrant-2", "qdrant-3"
    ]
    assert result.summary_point_id == str(setup.version_id)

    setup.db.flush.assert_called_once()
    setup.db.commit.assert_called_once()
    setup.db.rollback.assert_not_called()

    setup.file_storage.download_file.assert_called_once()
    setup.loader.load_pdf.assert_called_once()
    setup.indexing_service.index_document_version.assert_called_once()
    repository.activate_pending_version.assert_called_once_with(
        setup.version_id,
    )

    # New version's chunk lifecycle is synchronized after commit.
    setup.indexing_service.qdrant_store.update_version_status.assert_called_once_with(
        document_version_id=str(setup.version_id),
        status="active",
    )
    # Summary is first pending, then active.
    assert setup.summary_storage.upsert_summary.call_args_list == [
        call(
            document_id=setup.version.document_id,
            document_version_id=setup.version_id,
            version_number=2,
            title=setup.document.title,
            summary=setup.version.summary,
            status="pending",
            ),
        call(
            document_id=str(setup.activated_version.document_id),
            document_version_id=str(setup.activated_version.id),
            version_number=2,
            title=setup.document.title,
            summary=setup.activated_version.summary,
            status="active",
            ),
    ]

    assert result.summary_sync_pending is False
    assert result.lifecycle_sync_pending is False
    assert result.sync_errors == []


def test_previous_active_version_is_synchronized_as_superseded(
    setup_orchestrator,
):
    setup = setup_orchestrator

    repository = MagicMock()
    repository.activate_pending_version.return_value = (
        setup.activated_version,
        setup.previous_version,
    )

    with (
        patch_version_loading(setup),
        patch(
            "app.services.document_indexing.indexing_orchestrator.DocumentVersionRepository",
            return_value=repository,
        ),
    ):
        result = setup.orchestrator.index_pending_version(
            setup.version_id,
            )

    qdrant_updates = (
        setup.indexing_service.qdrant_store.update_version_status.call_args_list
    )
    assert qdrant_updates == [
        call(
            document_version_id=str(setup.version_id),
            status="active",
        ),
        call(
            document_version_id=str(setup.previous_version.id),
            status="superseded",
        ),
    ]

    assert result.status == "active"
    assert result.lifecycle_sync_pending is False


def test_summary_pending_upsert_failure_after_write_cleans_summary(
    setup_orchestrator,
):
    setup = setup_orchestrator
    persisted_summaries = []

    def upsert_then_raise(**kwargs):
        # Simulate Qdrant storing the point before the response fails.
        persisted_summaries.append(kwargs)
        raise RuntimeError("Summary response lost after Qdrant upsert")

    setup.summary_storage.upsert_summary.side_effect = upsert_then_raise

    with patch_version_loading(setup):
        with pytest.raises(
            DocumentIndexingOrchestrationError,
            match="transaction was rolled back",
        ):
            setup.orchestrator.index_pending_version(
                setup.version_id,
                    )

    assert len(persisted_summaries) == 1
    assert persisted_summaries[0]["status"] == "pending"
    setup.db.rollback.assert_called_once()
    setup.db.commit.assert_not_called()

    setup.indexing_service.qdrant_store.delete_points.assert_called_once_with(
        setup.indexing_result.qdrant_point_ids
    )
    setup.summary_storage.delete_summary.assert_called_once_with(
        setup.version_id
    )


def test_activation_failure_rolls_back_and_cleans_all_indexes(
    setup_orchestrator,
):
    setup = setup_orchestrator

    repository = MagicMock()
    repository.activate_pending_version.side_effect = RuntimeError(
        "Activation failed"
    )

    with (
        patch_version_loading(setup),
        patch(
            "app.services.document_indexing.indexing_orchestrator.DocumentVersionRepository",
            return_value=repository,
        ),
    ):
        with pytest.raises(DocumentIndexingOrchestrationError):
            setup.orchestrator.index_pending_version(
                setup.version_id,
                    )

    setup.db.rollback.assert_called_once()
    setup.db.commit.assert_not_called()

    setup.indexing_service.qdrant_store.delete_points.assert_called_once()
    setup.summary_storage.delete_summary.assert_called_once_with(
        setup.version_id
    )

    setup.file_storage.delete_file.assert_not_called()


def test_post_commit_summary_sync_failure_does_not_rollback(
    setup_orchestrator,
):
    setup = setup_orchestrator

    repository = MagicMock()
    repository.activate_pending_version.return_value = (
        setup.activated_version,
        None,
    )

    setup.summary_storage.upsert_summary.side_effect = [
        str(setup.version_id),
        RuntimeError("Summary status sync failed"),
    ]

    with (
        patch_version_loading(setup),
        patch(
            "app.services.document_indexing.indexing_orchestrator.DocumentVersionRepository",
            return_value=repository,
        ),
    ):
        result = setup.orchestrator.index_pending_version(
            setup.version_id,
            )

    assert result.status == "active"
    assert result.summary_sync_pending is True
    assert result.lifecycle_sync_pending is False
    assert any(
        error.startswith("Summary status update failed")
        for error in result.sync_errors
    )

    setup.db.commit.assert_called_once()
    setup.db.rollback.assert_not_called()

    # Chunk lifecycle synchronization still happens.
    setup.indexing_service.qdrant_store.update_version_status.assert_called_once()
@pytest.mark.parametrize(
    "store_name",
    ["qdrant_store"],
)
def test_post_commit_chunk_sync_failure_is_reported_without_rollback(
    setup_orchestrator,
    store_name,
):
    setup = setup_orchestrator

    repository = MagicMock()
    repository.activate_pending_version.return_value = (
        setup.activated_version,
        None,
    )

    store = getattr(setup.indexing_service, store_name)
    store.update_version_status.side_effect = RuntimeError(
        "Lifecycle sync failed"
    )

    with (
        patch_version_loading(setup),
        patch(
            "app.services.document_indexing.indexing_orchestrator.DocumentVersionRepository",
            return_value=repository,
        ),
    ):
        result = setup.orchestrator.index_pending_version(
            setup.version_id,
            )

    assert result.status == "active"
    assert result.lifecycle_sync_pending is True
    assert result.sync_errors

    setup.db.commit.assert_called_once()
    setup.db.rollback.assert_not_called()


def test_invalid_pending_version_is_rejected_before_download(
    setup_orchestrator,
):
    setup = setup_orchestrator

    with patch.object(
        DocumentIndexingOrchestrator,
        "_load_version",
        side_effect=ValueError("Only pending versions can be indexed."),
    ):
        with pytest.raises(DocumentIndexingOrchestrationError):
            setup.orchestrator.index_pending_version(
                setup.version_id,
                    )

    setup.file_storage.download_file.assert_not_called()
    setup.loader.load_pdf.assert_not_called()
    setup.indexing_service.index_document_version.assert_not_called()

    setup.db.rollback.assert_called_once()
    setup.db.commit.assert_not_called()




def test_b2_download_failure_rolls_back_without_indexing(
    setup_orchestrator,
):
    setup = setup_orchestrator
    setup.file_storage.download_file.side_effect = RuntimeError(
        "B2 unavailable"
    )

    with patch_version_loading(setup):
        with pytest.raises(DocumentIndexingOrchestrationError):
            setup.orchestrator.index_pending_version(
                setup.version_id,
                    )

    setup.db.rollback.assert_called_once()
    setup.db.commit.assert_not_called()

    setup.loader.load_pdf.assert_not_called()
    setup.indexing_service.index_document_version.assert_not_called()
    setup.summary_storage.upsert_summary.assert_not_called()

    setup.file_storage.delete_file.assert_not_called()


def test_empty_downloaded_pdf_is_rejected(
    setup_orchestrator,
):
    setup = setup_orchestrator

    setup.file_storage.download_file.side_effect = (
        lambda storage_key, destination: Path(destination).write_bytes(b"")
    )

    with patch_version_loading(setup):
        with pytest.raises(
            DocumentIndexingOrchestrationError,
            match="transaction was rolled back",
        ):
            setup.orchestrator.index_pending_version(
                setup.version_id,
                    )

    setup.db.rollback.assert_called_once()
    setup.db.commit.assert_not_called()

    setup.loader.load_pdf.assert_not_called()
    setup.indexing_service.index_document_version.assert_not_called()
    setup.summary_storage.upsert_summary.assert_not_called()


def test_downloaded_pdf_hash_mismatch_is_rejected_before_extraction(
    setup_orchestrator,
):
    setup = setup_orchestrator
    setup.version.file_hash = hashlib.sha256(b"different-pdf-content").hexdigest()

    with patch_version_loading(setup):
        with pytest.raises(DocumentIndexingOrchestrationError) as exc_info:
            setup.orchestrator.index_pending_version(
                setup.version_id,
                    )

    assert "SHA-256 does not match" in str(exc_info.value.__cause__)
    setup.loader.load_pdf.assert_not_called()
    setup.indexing_service.index_document_version.assert_not_called()
    setup.summary_storage.upsert_summary.assert_not_called()
    setup.db.rollback.assert_called_once()
    setup.db.commit.assert_not_called()


def test_pdf_extraction_failure_rolls_back(
    setup_orchestrator,
):
    setup = setup_orchestrator
    setup.loader.load_pdf.side_effect = RuntimeError(
        "PDF extraction failed"
    )

    with patch_version_loading(setup):
        with pytest.raises(DocumentIndexingOrchestrationError):
            setup.orchestrator.index_pending_version(
                setup.version_id,
                    )

    setup.db.rollback.assert_called_once()
    setup.db.commit.assert_not_called()

    setup.indexing_service.index_document_version.assert_not_called()
    setup.summary_storage.upsert_summary.assert_not_called()

    setup.file_storage.delete_file.assert_not_called()


def test_empty_pdf_extraction_result_is_rejected(
    setup_orchestrator,
):
    setup = setup_orchestrator
    setup.loader.load_pdf.return_value = []

    with patch_version_loading(setup):
        with pytest.raises(
            DocumentIndexingOrchestrationError,
            match="transaction was rolled back",
        ):
            setup.orchestrator.index_pending_version(
                setup.version_id,
                    )

    setup.db.rollback.assert_called_once()
    setup.db.commit.assert_not_called()

    setup.indexing_service.index_document_version.assert_not_called()
    setup.summary_storage.upsert_summary.assert_not_called()


def test_indexing_service_failure_rolls_back(
    setup_orchestrator,
):
    setup = setup_orchestrator
    setup.indexing_service.index_document_version.side_effect = (
        RuntimeError("Indexing failed")
    )

    with patch_version_loading(setup):
        with pytest.raises(DocumentIndexingOrchestrationError):
            setup.orchestrator.index_pending_version(
                setup.version_id,
                    )

    setup.db.rollback.assert_called_once()
    setup.db.commit.assert_not_called()

    setup.summary_storage.upsert_summary.assert_not_called()

    # The indexing service owns cleanup when indexing itself fails.
    # The orchestrator has not received an IndexingResult.
    setup.indexing_service.qdrant_store.delete_points.assert_not_called()


def test_activation_returning_none_rolls_back_and_cleans(
    setup_orchestrator,
):
    setup = setup_orchestrator

    repository = MagicMock()
    repository.activate_pending_version.return_value = None

    with (
        patch_version_loading(setup),
        patch(
            "app.services.document_indexing.indexing_orchestrator.DocumentVersionRepository",
            return_value=repository,
        ),
    ):
        with pytest.raises(
            DocumentIndexingOrchestrationError,
            match="transaction was rolled back",
        ):
            setup.orchestrator.index_pending_version(
                setup.version_id,
                    )

    setup.db.rollback.assert_called_once()
    setup.db.commit.assert_not_called()

    setup.indexing_service.qdrant_store.delete_points.assert_called_once()
    setup.summary_storage.delete_summary.assert_called_once_with(
        setup.version_id
    )


def test_cleanup_failures_are_reported(
    setup_orchestrator,
):
    setup = setup_orchestrator

    repository = MagicMock()
    repository.activate_pending_version.side_effect = RuntimeError(
        "Activation failed"
    )

    setup.indexing_service.qdrant_store.delete_points.side_effect = (
        RuntimeError("Qdrant cleanup unavailable")
    )
    setup.summary_storage.delete_summary.side_effect = (
        RuntimeError("Summary cleanup unavailable")
    )

    with (
        patch_version_loading(setup),
        patch(
            "app.services.document_indexing.indexing_orchestrator.DocumentVersionRepository",
            return_value=repository,
        ),
    ):
        with pytest.raises(
            DocumentIndexingOrchestrationError
        ) as exc_info:
            setup.orchestrator.index_pending_version(
                setup.version_id,
                    )

    error = exc_info.value

    assert len(error.cleanup_errors) == 2

    assert any(
        "Qdrant cleanup failed" in item
        for item in error.cleanup_errors
    )
    assert any(
        "Summary cleanup failed" in item
        for item in error.cleanup_errors
    )

    setup.db.rollback.assert_called_once()
    setup.db.commit.assert_not_called()

    # Original PDF must remain untouched.
    setup.file_storage.delete_file.assert_not_called()



def test_txt_version_downloads_and_uses_text_loader_for_indexing(
    setup_orchestrator,
):
    setup = setup_orchestrator
    setup.version.original_filename = "document.txt"
    setup.version.storage_key = "voice-ai/test/document.txt"

    repository = MagicMock()
    repository.activate_pending_version.return_value = (
        setup.activated_version,
        None,
    )

    with (
        patch_version_loading(setup),
        patch(
            "app.services.document_indexing.indexing_orchestrator.DocumentVersionRepository",
            return_value=repository,
        ),
    ):
        result = setup.orchestrator.index_pending_version(setup.version_id)

    assert result.status == "active"
    setup.file_storage.download_file.assert_called_once()
    assert Path(
        setup.file_storage.download_file.call_args.kwargs["destination"]
    ).name == "document.txt"
    setup.text_loader.load_txt.assert_called_once()
    setup.loader.load_pdf.assert_not_called()
    setup.indexing_service.index_document_version.assert_called_once()
    setup.db.commit.assert_called_once()



def test_selected_semantic_strategy_is_forwarded_to_chunk_indexer(
    setup_orchestrator,
):
    setup = setup_orchestrator
    repository = MagicMock()
    repository.activate_pending_version.return_value = (
        setup.activated_version,
        None,
    )

    with (
        patch_version_loading(setup),
        patch(
            "app.services.document_indexing.indexing_orchestrator.DocumentVersionRepository",
            return_value=repository,
        ),
    ):
        result = setup.orchestrator.index_pending_version(
            setup.version_id,
            chunking_strategy="semantic",
        )

    assert result.chunking_strategy == "semantic"
    assert (
        setup.indexing_service.index_document_version.call_args.kwargs[
            "chunking_strategy"
        ]
        == "semantic"
    )
