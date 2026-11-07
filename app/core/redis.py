import asyncio

from redis.asyncio import Redis

from .config import settings


_redis_client: Redis | None = None
_redis_loop: asyncio.AbstractEventLoop | None = None


async def get_redis() -> Redis:
    """
    Return the shared asynchronous Valkey client.

    The connection is created lazily and reused across requests.
    Actual connectivity is established when a Redis command executes.
    """
    global _redis_client, _redis_loop

    if not settings.AIVEN_REDIS_URI:
        raise RuntimeError(
            "AIVEN_REDIS_URI is missing from environment variables."
        )

    current_loop = asyncio.get_running_loop()

    if _redis_client is not None and _redis_loop is not current_loop:
        if _redis_loop is not None and not _redis_loop.is_closed():
            await _redis_client.aclose()
        _redis_client = None
        _redis_loop = None

    if _redis_client is None:
        _redis_client = Redis.from_url(
            settings.AIVEN_REDIS_URI,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30,
        )
        _redis_loop = current_loop

    return _redis_client


async def close_redis() -> None:
    """Close the shared Valkey connection pool."""
    global _redis_client, _redis_loop

    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        _redis_loop = None