"""Redis-backed conversation history for the conversational RAG workflow."""

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from redis.asyncio import Redis

from app.core.config import settings
from app.core.redis import get_redis


class ConversationHistoryService:
    """Persist a bounded chat history with a sliding inactivity TTL."""

    def __init__(self, redis_client: Redis | None = None, *, ttl_seconds: int | None = None, max_turns: int | None = None) -> None:
        self._redis_client = redis_client
        self.ttl_seconds = ttl_seconds or settings.CONVERSATION_HISTORY_TTL_SECONDS
        self.max_turns = max_turns or settings.CONVERSATION_HISTORY_MAX_TURNS
        if self.ttl_seconds < 1:
            raise ValueError("ttl_seconds must be a positive integer.")
        if self.max_turns < 1:
            raise ValueError("max_turns must be a positive integer.")

    async def _redis(self) -> Redis:
        return self._redis_client or await get_redis()

    @staticmethod
    def _key(session_id: str) -> str:
        normalized = session_id.strip()
        if not normalized:
            raise ValueError("session_id must not be empty.")
        return f"conversation:{normalized}"

    async def load_history(self, session_id: str) -> list[BaseMessage]:
        """Load recent messages in order and refresh the inactivity TTL."""
        client = await self._redis()
        key = self._key(session_id)
        encoded_messages = await client.lrange(key, 0, -1)
        if not encoded_messages:
            return []
        history = [self._decode_message(item) for item in encoded_messages]
        await client.expire(key, self.ttl_seconds)
        return history[-(self.max_turns * 2):]

    async def save_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """Append a completed turn, bound history, and refresh its TTL."""
        if not user_message.strip() or not assistant_message.strip():
            raise ValueError("Both user and assistant messages must be non-empty.")
        client = await self._redis()
        key = self._key(session_id)
        entries = [
            self._encode_message(HumanMessage(content=user_message)),
            self._encode_message(AIMessage(content=assistant_message)),
        ]
        max_messages = self.max_turns * 2
        async with client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *entries)
            pipe.ltrim(key, -max_messages, -1)
            pipe.expire(key, self.ttl_seconds)
            await pipe.execute()

    @staticmethod
    def _encode_message(message: BaseMessage) -> str:
        role = "user" if isinstance(message, HumanMessage) else "system" if isinstance(message, SystemMessage) else "assistant"
        content = message.content
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        return json.dumps({"role": role, "content": content}, ensure_ascii=False)

    @staticmethod
    def _decode_message(encoded: str | bytes) -> BaseMessage:
        if isinstance(encoded, bytes):
            encoded = encoded.decode("utf-8")
        data: dict[str, Any] = json.loads(encoded)
        role, content = data.get("role"), data.get("content", "")
        if role == "user":
            return HumanMessage(content=content)
        if role == "system":
            return SystemMessage(content=content)
        return AIMessage(content=content)
