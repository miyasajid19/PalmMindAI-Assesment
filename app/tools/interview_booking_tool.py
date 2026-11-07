"""LangChain tool for booking interview slots in the application database."""

from __future__ import annotations

import re
from datetime import date, datetime, time
from typing import Annotated

from langchain_core.tools import tool
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.database import get_db_session
from app.models.interview_bookings import InterviewBooking


_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


@tool
def book_interview(
    name: Annotated[str, "Candidate's full name."],
    email: Annotated[str, "Candidate's valid email address."],
    interview_date: Annotated[str, "Interview date in YYYY-MM-DD format."],
    interview_time: Annotated[str, "Interview time in 24-hour HH:MM format."],
    confirmed: Annotated[bool, "True only after the candidate explicitly confirms the summarized booking details."],
) -> str:
    """Book an interview only after the candidate explicitly confirms the details.

    Checks the slot before insertion. A database unique constraint also guards
    against concurrent attempts for the same date and time.
    """
    if confirmed is not True:
        return "Booking not created: explicit candidate confirmation is required."

    clean_name = name.strip() if isinstance(name, str) else ""
    clean_email = email.strip().lower() if isinstance(email, str) else ""

    if not clean_name or len(clean_name) > 255:
        return "Booking not created: name must contain 1 to 255 characters."
    if not _EMAIL_PATTERN.fullmatch(clean_email) or len(clean_email) > 320:
        return "Booking not created: please provide a valid email address."
    if not isinstance(interview_date, str) or not _DATE_PATTERN.fullmatch(interview_date):
        return "Booking not created: date must use YYYY-MM-DD format."
    if not isinstance(interview_time, str) or not _TIME_PATTERN.fullmatch(interview_time):
        return "Booking not created: time must use HH:MM (24-hour) format."

    try:
        parsed_date = date.fromisoformat(interview_date)
        parsed_time = time.fromisoformat(interview_time)
    except ValueError:
        return "Booking not created: the date or time is not a valid calendar value."

    now = datetime.now()
    if parsed_date < now.date() or (
        parsed_date == now.date()
        and parsed_time <= now.time().replace(second=0, microsecond=0)
    ):
        return "Booking not created: please choose a future interview date and time."

    with get_db_session() as db:
        existing_booking = db.scalar(
            select(InterviewBooking.id)
            .where(
                InterviewBooking.interview_date == parsed_date,
                InterviewBooking.interview_time == parsed_time,
            )
            .limit(1)
        )
        if existing_booking is not None:
            return (
                "That interview slot is already booked. Please ask the candidate "
                "to choose another date or time."
            )

        booking = InterviewBooking(
            name=clean_name,
            email=clean_email,
            interview_date=parsed_date,
            interview_time=parsed_time,
        )
        db.add(booking)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            return (
                "That interview slot was just booked by someone else. "
                "Please ask the candidate to choose another date or time."
            )
        db.refresh(booking)

    return (
        "Interview booked successfully. "
        f"Booking ID: {booking.id}. Candidate: {clean_name}. "
        f"Email: {clean_email}. Date: {parsed_date.isoformat()}. "
        f"Time: {parsed_time.strftime('%H:%M')}."
    )
