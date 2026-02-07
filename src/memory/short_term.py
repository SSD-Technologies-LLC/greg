import json
import logging
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

logger = logging.getLogger(__name__)

_Redis = Redis  # alias to avoid mypy issues with redis.asyncio stubs


class ShortTermMemory:
    def __init__(self, redis: Any, buffer_size: int = 200) -> None:
        self._redis: Any = redis
        self._buffer_size = buffer_size

    def _chat_key(self, chat_id: int) -> str:
        return f"chat:{chat_id}:messages"

    def _state_key(self, chat_id: int) -> str:
        return f"chat:{chat_id}:state"

    async def store_message(self, chat_id: int, user_id: int, username: str, text: str) -> int:
        msg = json.dumps(
            {
                "user_id": user_id,
                "username": username,
                "text": text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chat_id": chat_id,
            },
            ensure_ascii=False,
        )
        key = self._chat_key(chat_id)
        await self._redis.rpush(key, msg)
        await self._redis.expire(key, 48 * 3600)  # 48h TTL safety net
        length = await self._redis.llen(key)
        return length

    async def get_recent_messages(self, chat_id: int, count: int = 50) -> list[dict]:
        key = self._chat_key(chat_id)
        raw = await self._redis.lrange(key, -count, -1)
        return [json.loads(m) for m in raw]

    async def get_overflow_messages(self, chat_id: int) -> list[dict] | None:
        key = self._chat_key(chat_id)
        length = await self._redis.llen(key)
        if length <= self._buffer_size:
            return None
        overflow_count = length - self._buffer_size
        raw = await self._redis.lrange(key, 0, overflow_count - 1)
        return [json.loads(m) for m in raw]

    async def trim_overflow(self, chat_id: int) -> None:
        key = self._chat_key(chat_id)
        length = await self._redis.llen(key)
        if length > self._buffer_size:
            await self._redis.ltrim(key, length - self._buffer_size, -1)

    async def get_buffer_length(self, chat_id: int) -> int:
        return await self._redis.llen(self._chat_key(chat_id))

    async def set_chat_state(self, chat_id: int, state: dict) -> None:
        key = self._state_key(chat_id)
        await self._redis.set(key, json.dumps(state, ensure_ascii=False))
        await self._redis.expire(key, 48 * 3600)

    async def get_chat_state(self, chat_id: int) -> dict | None:
        raw = await self._redis.get(self._state_key(chat_id))
        return json.loads(raw) if raw else None
