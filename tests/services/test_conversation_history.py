import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.services.conversation_history import ConversationHistoryService
from app.services.conversation_orchestrator import ConversationOrchestrator


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.operations = []

    def rpush(self, key, *values):
        self.operations.append(("rpush", key, values))
        return self

    def ltrim(self, key, start, end):
        self.operations.append(("ltrim", key, start, end))
        return self

    def expire(self, key, ttl):
        self.operations.append(("expire", key, ttl))
        return self

    async def execute(self):
        for operation in self.operations:
            name, *args = operation
            if name == "rpush":
                key, values = args
                self.client.data.setdefault(key, []).extend(values)
            elif name == "ltrim":
                key, start, end = args
                items = self.client.data.get(key, [])
                self.client.data[key] = items[start:] if end == -1 else items[start:end + 1]
            elif name == "expire":
                key, ttl = args
                self.client.ttls[key] = ttl
        return [True] * len(self.operations)

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, exc, _tb):
        return False


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.ttls = {}

    async def lrange(self, key, start, end):
        return self.data.get(key, [])[start:] if end == -1 else self.data.get(key, [])[start:end + 1]

    async def expire(self, key, ttl):
        self.ttls[key] = ttl
        return True

    def pipeline(self, transaction=True):
        _ = transaction  # Mirrors redis-py's pipeline signature for the test fake.
        return FakePipeline(self)


@pytest.mark.asyncio
async def test_load_history_decodes_roles_and_refreshes_sliding_ttl():
    redis = FakeRedis()
    redis.data["conversation:s-1"] = [
        json.dumps({"role": "user", "content": "hello"}),
        json.dumps({"role": "assistant", "content": "hi"}),
    ]
    service = ConversationHistoryService(redis, ttl_seconds=1800, max_turns=10)

    history = await service.load_history("s-1")

    assert [type(message) for message in history] == [HumanMessage, AIMessage]
    assert [message.content for message in history] == ["hello", "hi"]
    assert redis.ttls["conversation:s-1"] == 1800


@pytest.mark.asyncio
async def test_save_turn_appends_pair_caps_history_and_sets_ttl():
    redis = FakeRedis()
    service = ConversationHistoryService(redis, ttl_seconds=60, max_turns=2)

    await service.save_turn("s-2", "q1", "a1")
    await service.save_turn("s-2", "q2", "a2")
    await service.save_turn("s-2", "q3", "a3")

    history = await service.load_history("s-2")
    assert [message.content for message in history] == ["q2", "a2", "q3", "a3"]
    assert redis.ttls["conversation:s-2"] == 60


class FakeGraph:
    def __init__(self):
        self.inputs = []

    async def ainvoke(self, inputs, config=None):
        self.inputs.append((inputs, config))
        return {
            "generation_output": {
                "segments": [{"text": "I can help with that."}],
            },
            "validation_result": None,
        }


@pytest.mark.asyncio
async def test_conversation_orchestrator_hydrates_then_persists_completed_turn():
    redis = FakeRedis()
    history_service = ConversationHistoryService(redis, ttl_seconds=1800, max_turns=10)
    await history_service.save_turn("session-a", "Tell me about ML", "ML is a field of AI.")
    graph = FakeGraph()
    orchestrator = ConversationOrchestrator(history_service)

    result = await orchestrator.run_turn(
        graph=graph,
        session_id="session-a",
        query="What are its types?",
    )

    graph_input, graph_config = graph.inputs[0]
    assert graph_config is None
    assert [message.content for message in graph_input["context"]] == [
        "Tell me about ML",
        "ML is a field of AI.",
    ]
    assert graph_input["query"] == "What are its types?"
    assert result["generation_output"]["segments"][0]["text"] == "I can help with that."
    history = await history_service.load_history("session-a")
    assert [message.content for message in history] == [
        "Tell me about ML",
        "ML is a field of AI.",
        "What are its types?",
        "I can help with that.",
    ]


@pytest.mark.asyncio
async def test_orchestrator_does_not_persist_when_graph_has_no_answer():
    redis = FakeRedis()
    service = ConversationHistoryService(redis)
    graph = FakeGraph()
    graph.ainvoke = async_no_answer
    orchestrator = ConversationOrchestrator(service)

    await orchestrator.run_turn(graph=graph, session_id="empty", query="query with no generated answer")
    assert redis.data.get("conversation:empty", []) == []


async def async_no_answer(inputs, config=None):
    return {"generation_output": None, "validation_result": None}


class FakeInterviewBookingConversation:
    def __init__(self, answer="Please provide your full name and email."):
        self.answer = answer
        self.calls = []

    async def respond(self, *, query, history=None):
        self.calls.append((query, history))
        return self.answer


@pytest.mark.asyncio
async def test_orchestrator_routes_booking_intent_and_persists_booking_response():
    redis = FakeRedis()
    history_service = ConversationHistoryService(redis, ttl_seconds=1800, max_turns=10)
    booking_flow = FakeInterviewBookingConversation(
        "Please share your full name, email, date, and preferred time."
    )
    graph = FakeGraph()
    orchestrator = ConversationOrchestrator(history_service, booking_flow)

    result = await orchestrator.run_turn(
        graph=graph,
        session_id="booking-session",
        query="I'd like to schedule an interview.",
    )

    assert result["booking_response"].startswith("Please share")
    assert graph.inputs == []
    assert len(booking_flow.calls) == 1
    history = await history_service.load_history("booking-session")
    assert [message.content for message in history] == [
        "I'd like to schedule an interview.",
        "Please share your full name, email, date, and preferred time.",
    ]


@pytest.mark.asyncio
async def test_orchestrator_keeps_booking_route_for_follow_up_without_interview_keyword():
    redis = FakeRedis()
    history_service = ConversationHistoryService(redis, ttl_seconds=1800, max_turns=10)
    await history_service.save_turn(
        "booking-follow-up",
        "I'd like to book an interview.",
        "For the interview, please provide your full name, email, exact date, and time.",
    )
    booking_flow = FakeInterviewBookingConversation("Thanks. What time works best?")
    graph = FakeGraph()
    orchestrator = ConversationOrchestrator(history_service, booking_flow)

    result = await orchestrator.run_turn(
        graph=graph,
        session_id="booking-follow-up",
        query="Sajid Miya",
    )

    assert result["booking_response"] == "Thanks. What time works best?"
    assert graph.inputs == []
    assert booking_flow.calls[0][0] == "Sajid Miya"


from unittest.mock import MagicMock, patch
from uuid import uuid4

from langchain_core.messages import ToolMessage


class BookingLifecycleModel:
    def __init__(self, booking_args):
        self.booking_args = booking_args
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            assert "Interview booked successfully" in last_message.content
            return AIMessage(content="Confirmed—your interview is booked.")
        query = last_message.content
        if query == "I want to schedule an interview.":
            return AIMessage(
                content="For the interview, please share your full name, email, exact date, and time."
            )
        if query == "Sajid Miya, sajid@example.com, 2030-06-10 at 10:30.":
            return AIMessage(
                content="Please confirm: Sajid Miya, sajid@example.com, 2030-06-10 at 10:30."
            )
        if query == "Yes, I confirm.":
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "book_interview",
                        "args": self.booking_args,
                        "id": "booking-lifecycle-call",
                        "type": "tool_call",
                    }
                ],
            )
        raise AssertionError(f"Unexpected conversational query: {query}")


@pytest.mark.asyncio
async def test_interview_booking_lifecycle_routes_collects_confirms_persists_and_saves_history():
    from app.services.interview_booking_conversation import InterviewBookingConversation

    redis = FakeRedis()
    history_service = ConversationHistoryService(redis, ttl_seconds=1800, max_turns=10)
    booking_args = {
        "name": "Sajid Miya",
        "email": "sajid@example.com",
        "interview_date": "2030-06-10",
        "interview_time": "10:30",
        "confirmed": True,
    }
    model = BookingLifecycleModel(booking_args)
    booking_flow = InterviewBookingConversation(model=model)
    orchestrator = ConversationOrchestrator(history_service, booking_flow)
    graph = FakeGraph()

    db = MagicMock()
    db.scalar.return_value = None
    db.refresh.side_effect = lambda booking: setattr(booking, "id", uuid4())
    session_context = MagicMock()
    session_context.__enter__.return_value = db
    session_context.__exit__.return_value = False

    with patch(
        "app.tools.interview_booking_tool.get_db_session",
        return_value=session_context,
    ):
        first = await orchestrator.run_turn(
            graph=graph,
            session_id="booking-lifecycle",
            query="I want to schedule an interview.",
        )
        second = await orchestrator.run_turn(
            graph=graph,
            session_id="booking-lifecycle",
            query="Sajid Miya, sajid@example.com, 2030-06-10 at 10:30.",
        )
        third = await orchestrator.run_turn(
            graph=graph,
            session_id="booking-lifecycle",
            query="Yes, I confirm.",
        )

    assert "please share" in first["booking_response"].lower()
    assert "please confirm" in second["booking_response"].lower()
    assert third["booking_response"] == "Confirmed—your interview is booked."
    assert graph.inputs == []
    db.add.assert_called_once()
    db.commit.assert_called_once()
    history = await history_service.load_history("booking-lifecycle")
    assert len(history) == 6
    assert history[-1].content == "Confirmed—your interview is booked."


@pytest.mark.asyncio
async def test_booking_follow_up_routes_when_initial_name_prompt_omits_interview_keyword():
    redis = FakeRedis()
    history_service = ConversationHistoryService(redis, ttl_seconds=1800, max_turns=10)
    await history_service.save_turn(
        "booking-name-follow-up",
        "I would like to schedule an interview.",
        "Could you please provide your full name?",
    )
    booking_flow = FakeInterviewBookingConversation("Thanks. What is your email address?")
    graph = FakeGraph()
    orchestrator = ConversationOrchestrator(history_service, booking_flow)

    result = await orchestrator.run_turn(
        graph=graph,
        session_id="booking-name-follow-up",
        query="Sajid Miya",
    )

    assert result["booking_response"] == "Thanks. What is your email address?"
    assert graph.inputs == []
    assert booking_flow.calls[0][0] == "Sajid Miya"


@pytest.mark.asyncio
async def test_name_prompt_without_recent_booking_intent_stays_on_rag_route():
    redis = FakeRedis()
    history_service = ConversationHistoryService(redis, ttl_seconds=1800, max_turns=10)
    await history_service.save_turn(
        "ordinary-rag-session",
        "Can you explain how names are stored in a database?",
        "Could you provide the full name of the field?",
    )
    graph = FakeGraph()
    orchestrator = ConversationOrchestrator(history_service)

    result = await orchestrator.run_turn(
        graph=graph,
        session_id="ordinary-rag-session",
        query="Sajid Miya",
    )

    assert "booking_response" not in result
    assert len(graph.inputs) == 1
