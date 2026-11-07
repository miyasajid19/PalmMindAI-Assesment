from datetime import date, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.agents.Generation.agent import GenerationAgent
from app.models.interview_bookings import InterviewBooking
from app.tools import book_interview


def test_interview_booking_is_registered_as_a_langchain_tool():
    assert book_interview.name == "book_interview"
    assert set(book_interview.args_schema.model_fields) == {
        "name",
        "email",
        "interview_date",
        "interview_time",
        "confirmed",
    }


def test_generation_agent_exposes_interview_booking_tool():
    agent = GenerationAgent(model=lambda _payload: {})
    assert book_interview in agent.available_tools


def test_booking_tool_rejects_unconfirmed_request_before_database_access():
    result = book_interview.invoke(
        {
            "name": "Candidate",
            "email": "candidate@example.com",
            "interview_date": "2030-06-10",
            "interview_time": "10:30",
            "confirmed": False,
        }
    )
    assert "explicit candidate confirmation" in result


def test_booking_tool_rejects_invalid_email_before_database_access():
    result = book_interview.invoke(
        {
            "name": "Candidate",
            "email": "not-an-email",
            "interview_date": "2030-06-10",
            "interview_time": "10:30",
            "confirmed": True,
        }
    )
    assert "valid email address" in result


def test_booking_tool_rejects_malformed_date_and_time_before_database_access():
    date_result = book_interview.invoke(
        {
            "name": "Candidate",
            "email": "candidate@example.com",
            "interview_date": "10-06-2030",
            "interview_time": "10:30",
            "confirmed": True,
        }
    )
    time_result = book_interview.invoke(
        {
            "name": "Candidate",
            "email": "candidate@example.com",
            "interview_date": "2030-06-10",
            "interview_time": "25:90",
            "confirmed": True,
        }
    )
    assert "YYYY-MM-DD" in date_result
    assert "HH:MM" in time_result


def _valid_booking_payload():
    return {
        "name": "Candidate",
        "email": "candidate@example.com",
        "interview_date": (date.today() + timedelta(days=2)).isoformat(),
        "interview_time": "11:30",
        "confirmed": True,
    }


def _mock_db_session(existing_booking_id=None):
    db = MagicMock()
    db.scalar.return_value = existing_booking_id
    session_context = MagicMock()
    session_context.__enter__.return_value = db
    session_context.__exit__.return_value = False
    return db, session_context


def test_booking_tool_returns_conflict_without_inserting_when_slot_is_taken():
    db, session_context = _mock_db_session(existing_booking_id=uuid4())

    with patch("app.tools.interview_booking_tool.get_db_session", return_value=session_context):
        result = book_interview.invoke(_valid_booking_payload())

    assert "already booked" in result
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_booking_tool_persists_and_confirms_available_slot():
    db, session_context = _mock_db_session()
    db.refresh.side_effect = lambda booking: setattr(booking, "id", uuid4())

    with patch("app.tools.interview_booking_tool.get_db_session", return_value=session_context):
        result = book_interview.invoke(_valid_booking_payload())

    assert "Interview booked successfully" in result
    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once()


def test_interview_booking_model_has_unique_date_time_constraint():
    constraints = [
        constraint
        for constraint in InterviewBooking.__table__.constraints
        if getattr(constraint, "name", None) == "uq_interview_bookings_date_time"
    ]
    assert len(constraints) == 1
    assert {column.name for column in constraints[0].columns} == {
        "interview_date",
        "interview_time",
    }


import pytest
from langchain_core.messages import AIMessage, ToolMessage


class FakeToolCallingModel:
    def __init__(self, args):
        self.args = args
        self.calls = 0

    def bind_tools(self, tools):
        assert book_interview in tools
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "book_interview",
                        "args": self.args,
                        "id": "booking-call-1",
                        "type": "tool_call",
                    }
                ],
            )
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        assert len(tool_messages) == 1
        assert "Interview booked successfully" in tool_messages[0].content
        return AIMessage(content="Your interview is booked.")


@pytest.mark.asyncio
async def test_booking_conversation_invokes_tool_and_uses_tool_result_for_confirmation():
    from app.services.interview_booking_conversation import InterviewBookingConversation

    db, session_context = _mock_db_session()
    db.refresh.side_effect = lambda booking: setattr(booking, "id", uuid4())
    model = FakeToolCallingModel(_valid_booking_payload())
    conversation = InterviewBookingConversation(model=model)

    with patch("app.tools.interview_booking_tool.get_db_session", return_value=session_context):
        response = await conversation.respond(
            query="Yes, I confirm the details.",
            history=[],
        )

    assert response == "Your interview is booked."
    assert model.calls == 2
    db.add.assert_called_once()
    db.commit.assert_called_once()
