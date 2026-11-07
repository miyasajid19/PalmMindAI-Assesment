
from uuid import uuid4

import pytest
from langchain_core.documents import Document

from app.services.document_indexing.indexing_service import (
    DocumentIndexingService,
)


@pytest.mark.integration
def test_document_indexing_live_stores():

    service = DocumentIndexingService()

    document_id = str(uuid4())
    document_version_id = str(uuid4())

    # Synthetic page-level content; deliberately long enough
    # to produce multiple chunks with the real splitter.
    page_text = """
    Employee Leave Policy

    Employees are entitled to annual leave according to
    the organization's leave policy. Annual leave requests
    must be submitted to the employee's reporting manager.

    Sick leave may be requested when an employee is unable
    to work because of illness. Supporting medical documents
    may be required according to company policy.

    All leave requests must be reviewed and approved by
    the appropriate department manager before the leave
    period begins.

    The human resources department maintains employee
    leave records and ensures that leave balances are
    updated accurately.

    """ * 20

    page_documents = [
        Document(
            page_content=page_text,
            metadata={
                "source": "integration_test.pdf",
                "page": 1,
                "extraction_method": "pymupdf",
                "ocr_used": False,
            },
        )
    ]

    qdrant_point_ids = []

    try:
        # Run the actual indexing service.
        result = service.index_document_version(
            page_documents=page_documents,
            document_id=document_id,
            document_version_id=document_version_id,
            version_number=1,
            source_filename="integration_test.pdf",
        )

        qdrant_point_ids = result.qdrant_point_ids

        # ----------------------------------------
        # 1. Validate indexing result
        # ----------------------------------------

        assert result.status == "completed"
        assert result.page_count == 1
        assert result.ocr_page_count == 0
        assert result.chunk_count > 1

        assert len(result.qdrant_point_ids) == result.chunk_count

        # ----------------------------------------
        # 2. Verify Qdrant insertion
        # ----------------------------------------

        retrieved_points = service.qdrant_store.client.retrieve(
            collection_name=service.qdrant_store.collection_name,
            ids=result.qdrant_point_ids,
            with_payload=True,
            with_vectors=False,
        )

        assert len(retrieved_points) == result.chunk_count

        for point in retrieved_points:
            payload = point.payload

            assert payload["document_id"] == document_id
            assert payload["document_version_id"] == document_version_id
            assert payload["version_number"] == 1
            assert payload["status"] == "pending"

        print("\nINDEXING RESULT")
        print(result.to_dict())

        print("\nLIVE STORE VERIFICATION")
        print(f"Qdrant points verified: {len(retrieved_points)}")

    finally:
        # ----------------------------------------
        # 4. Clean up only this test's data
        # ----------------------------------------

        service.qdrant_store.delete_points(qdrant_point_ids)

        # Verify Qdrant cleanup.
        if qdrant_point_ids:
            remaining_points = service.qdrant_store.client.retrieve(
                collection_name=service.qdrant_store.collection_name,
                ids=qdrant_point_ids,
                with_payload=False,
                with_vectors=False,
            )

            assert len(remaining_points) == 0

        print("\nCLEANUP VERIFIED")
