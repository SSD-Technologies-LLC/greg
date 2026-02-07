import json
from unittest.mock import AsyncMock

import pytest

from src.memory.short_term import ShortTermMemory


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.llen = AsyncMock(return_value=0)
    r.rpush = AsyncMock()
    r.lrange = AsyncMock(return_value=[])
    r.ltrim = AsyncMock()
    r.set = AsyncMock()
    r.get = AsyncMock(return_value=None)
    r.expire = AsyncMock()
    return r


@pytest.fixture
def stm(mock_redis):
    return ShortTermMemory(mock_redis, buffer_size=200)


@pytest.mark.asyncio
async def test_store_message(stm, mock_redis):
    await stm.store_message(chat_id=123, user_id=456, username="alice", text="hello")
    mock_redis.rpush.assert_called_once()
    key = mock_redis.rpush.call_args[0][0]
    assert key == "chat:123:messages"
    data = json.loads(mock_redis.rpush.call_args[0][1])
    assert data["user_id"] == 456
    assert data["text"] == "hello"


@pytest.mark.asyncio
async def test_get_recent_messages(stm, mock_redis):
    msgs = [json.dumps({"user_id": 1, "username": "a", "text": "hi", "timestamp": "t", "chat_id": 1}).encode()]
    mock_redis.lrange = AsyncMock(return_value=msgs)
    result = await stm.get_recent_messages(chat_id=1, count=50)
    assert len(result) == 1
    assert result[0]["text"] == "hi"


@pytest.mark.asyncio
async def test_get_overflow_messages(stm, mock_redis):
    mock_redis.llen = AsyncMock(return_value=210)
    msgs = [
        json.dumps({"user_id": 1, "username": "a", "text": f"msg{i}", "timestamp": "t", "chat_id": 1}).encode()
        for i in range(50)
    ]
    mock_redis.lrange = AsyncMock(return_value=msgs)
    result = await stm.get_overflow_messages(chat_id=1)
    assert len(result) == 50
