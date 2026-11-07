from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.documents import Document


class DocumentRepository:
    """Repository for logical documents in the shared corpus."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        title: str,
        document_type: str,
        description: str | None = None,
        document_id: UUID | None = None,
    ) -> Document:
        document = Document(
            id=document_id,
            title=title,
            document_type=document_type,
            description=description,
        )
        self.db.add(document)
        self.db.flush()
        self.db.refresh(document)
        return document

    def get_by_id(self, document_id: UUID) -> Document | None:
        return self.db.scalar(select(Document).where(Document.id == document_id))

    def get_by_title(self, title: str) -> list[Document]:
        statement = (
            select(Document)
            .where(Document.title.ilike(title))
            .order_by(Document.created_at.desc())
        )
        return list(self.db.scalars(statement).all())

    def get_all(
        self,
        document_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Document]:
        statement = select(Document)
        if document_type:
            statement = statement.where(Document.document_type == document_type)
        statement = (
            statement.order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.scalars(statement).all())

    def update(self, document_id: UUID, **kwargs) -> Document | None:
        document = self.get_by_id(document_id)
        if document is None:
            return None
        allowed_fields = {"title", "document_type", "description"}
        invalid_fields = set(kwargs) - allowed_fields
        if invalid_fields:
            raise ValueError(f"Invalid document fields: {invalid_fields}")
        for field, value in kwargs.items():
            setattr(document, field, value)
        self.db.flush()
        self.db.refresh(document)
        return document

    def archive(self, document_id: UUID) -> Document | None:
        document = self.get_by_id(document_id)
        if document is None:
            return None
        if document.archived:
            raise ValueError("This document is already archived.")
        document.archived = True
        self.db.flush()
        self.db.refresh(document)
        return document

    def unarchive(self, document_id: UUID) -> Document | None:
        document = self.get_by_id(document_id)
        if document is None:
            return None
        if not document.archived:
            raise ValueError("This document is not archived.")
        document.archived = False
        self.db.flush()
        self.db.refresh(document)
        return document

    def delete(self, document_id: UUID) -> bool:
        document = self.get_by_id(document_id)
        if document is None:
            return False
        if document.versions:
            raise ValueError(
                "Cannot delete a document that has versions. "
                "Handle its versions explicitly first."
            )
        self.db.delete(document)
        self.db.flush()
        return True
