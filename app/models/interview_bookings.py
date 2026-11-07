from __future__ import annotations

from datetime import date, datetime, time
from uuid import UUID, uuid4

from sqlalchemy import Date, DateTime, String, Time, Uuid, func, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class InterviewBooking(Base):
    """Persist interview booking details collected in the conversational RAG."""

    __tablename__ = "interview_bookings"
    __table_args__ = (
        UniqueConstraint(
            "interview_date",
            "interview_time",
            name="uq_interview_bookings_date_time",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    interview_date: Mapped[date] = mapped_column(Date, nullable=False)
    interview_time: Mapped[time] = mapped_column(Time, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<InterviewBooking(id={self.id}, name={self.name!r}, "
            f"email={self.email!r}, date={self.interview_date})>"
        )
