from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.long_term import LongTermMemory

DEFAULT_EMOTIONS = {
    "warmth": 0.0,
    "trust": 0.0,
    "respect": 0.0,
    "annoyance": 0.0,
    "interest": 0.0,
    "loyalty": 0.0,
}


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    conn.transaction.return_value.__aenter__ = AsyncMock()
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool, conn


@pytest.fixture
def ltm(mock_pool):
    pool, _ = mock_pool
    return LongTermMemory(pool)


@pytest.mark.asyncio
async def test_get_or_create_profile_existing(ltm, mock_pool):
    _, conn = mock_pool
    row = {
        "user_id": 1,
        "chat_id": 10,
        "display_name": "Alice",
        "facts": {},
        "personality_traits": {},
        "emotional_state": DEFAULT_EMOTIONS,
    }
    conn.fetchrow = AsyncMock(return_value=row)
    result = await ltm.get_or_create_profile(user_id=1, chat_id=10, display_name="Alice")
    assert result["display_name"] == "Alice"


@pytest.mark.asyncio
async def test_update_facts(ltm, mock_pool):
    _, conn = mock_pool
    conn.execute = AsyncMock()
    await ltm.update_facts(user_id=1, chat_id=10, facts={"job": "dev"})
    conn.execute.assert_called_once()
    query = conn.execute.call_args[0][0]
    assert "facts" in query


@pytest.mark.asyncio
async def test_append_memory_log(ltm, mock_pool):
    _, conn = mock_pool
    conn.execute = AsyncMock()
    await ltm.append_memory_log(chat_id=10, user_id=1, memory_type="fact", content="works at Google")
    conn.execute.assert_called_once()
