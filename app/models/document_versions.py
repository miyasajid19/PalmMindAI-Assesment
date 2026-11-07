from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    and_,
    String,
    Text,
    Integer,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    CheckConstraint,
    UniqueConstraint,
    Index,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, foreign, mapped_column, relationship, remote

from ..core.database import Base

if TYPE_CHECKING:
    from .documents import Document


def _supersedes_join():
    return and_(
        remote(DocumentVersion.document_id)
        == DocumentVersion.document_id,
        remote(DocumentVersion.id)
        == foreign(DocumentVersion.supersedes_version_id),
    )


class DocumentVersion(Base):

    __tablename__ = "document_versions"

    __table_args__ = (

        # A document cannot have duplicate version numbers.
        UniqueConstraint(
            "document_id",
            "version_number",
            name="uq_document_version_number",
        ),
        UniqueConstraint(
            "document_id",
            "id",
            name="uq_document_version_document_id",
        ),
        ForeignKeyConstraint(
            ["document_id", "supersedes_version_id"],
            ["document_versions.document_id", "document_versions.id"],
            ondelete="RESTRICT",
            name="fk_document_version_supersedes",
        ),

        # Version number must be positive.
        CheckConstraint(
            "version_number > 0",
            name="chk_version_number",
        ),

        # Page count cannot be negative.
        CheckConstraint(
            "page_count IS NULL OR page_count >= 0",
            name="chk_page_count",
        ),

        # Effective end date cannot precede start date.
        CheckConstraint(
            """
            effective_until IS NULL
            OR effective_from IS NULL
            OR effective_until >= effective_from
            """,
            name="chk_effective_dates",
        ),

        # Allowed lifecycle statuses.
        CheckConstraint(
            """
            status IN (
                'pending',
                'active',
                'superseded'
            )
            """,
            name="chk_document_version_status",
        ),

        # Only one active version per document.
        Index(
            "idx_one_active_version",
            "document_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),

        # Hash lookup indexes.
        Index(
            "idx_versions_content_hash",
            "content_hash",
        ),

        Index(
            "idx_versions_file_hash",
            "file_hash",
        ),

        # Status filtering index.
        Index(
            "idx_versions_status",
            "status",
        ),

    )

    # =================================================
    # Primary Key
    # =================================================

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # =================================================
    # Parent Document Reference
    # =================================================

    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "documents.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    # =================================================
    # Version Information
    # =================================================

    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    version_label: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    original_filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    # =================================================
    # Duplicate Detection
    # =================================================

    # SHA-256 of original PDF bytes.
    file_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    # SHA-256 of normalized extracted text.
    content_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    # =================================================
    # Document Summary
    # =================================================

    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # =================================================
    # Lifecycle Status
    # =================================================

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        server_default="pending",
    )

    # =================================================
    # Storage Key
    # =================================================

    storage_key: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
        unique=True,
    )

    # =================================================
    # Effective Dates
    # =================================================

    effective_from: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    effective_until: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    # =================================================
    # Version History
    # =================================================

    supersedes_version_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
    )

    # =================================================
    # PDF Information
    # =================================================

    page_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # =================================================
    # Timestamp
    # =================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # =================================================
    # Relationships
    # =================================================

    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="versions",
    )

    # Reference to the previous version.
    supersedes_version: Mapped["DocumentVersion | None"] = relationship(
        "DocumentVersion",
        primaryjoin=_supersedes_join,
        remote_side=lambda: [
            DocumentVersion.document_id,
            DocumentVersion.id,
        ],
        foreign_keys=[supersedes_version_id],
        back_populates="superseded_by",
    )

    # Versions that replace this version.
    superseded_by: Mapped[list["DocumentVersion"]] = relationship(
        "DocumentVersion",
        foreign_keys=[supersedes_version_id],
        back_populates="supersedes_version",
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentVersion("
            f"id={self.id}, "
            f"document_id={self.document_id}, "
            f"version={self.version_number}, "
            f"status={self.status!r})>"
        )