"""Comprehensive tests for LongTermMemory — PostgreSQL profile and memory operations."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.long_term import DEFAULT_EMOTIONS, LongTermMemory


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


@pytest.fixture
def ltm(mock_pool):
    pool, _ = mock_pool
    return LongTermMemory(pool)


# ---------------------------------------------------------------------------
# Profile management
# ---------------------------------------------------------------------------


class TestProfileManagement:
    @pytest.mark.asyncio
    async def test_get_existing_profile(self, ltm, mock_pool):
        _, conn = mock_pool
        row = {
            "user_id": 1,
            "chat_id": 10,
            "display_name": "Alice",
            "facts": {"job": "dev"},
            "personality_traits": {},
            "emotional_state": DEFAULT_EMOTIONS.copy(),
        }
        conn.fetchrow = AsyncMock(return_value=row)
        result = await ltm.get_or_create_profile(1, 10, "Alice")
        assert result["display_name"] == "Alice"
        assert result["facts"]["job"] == "dev"

    @pytest.mark.asyncio
    async def test_create_new_profile(self, ltm, mock_pool):
        _, conn = mock_pool
        new_row = {
            "user_id": 1,
            "chat_id": 10,
            "display_name": "Alice",
            "facts": {},
            "personality_traits": {},
            "emotional_state": DEFAULT_EMOTIONS.copy(),
        }
        # First fetchrow returns None (not found), then returns the created row
        conn.fetchrow = AsyncMock(side_effect=[None, new_row])
        conn.execute = AsyncMock()
        result = await ltm.get_or_create_profile(1, 10, "Alice")
        assert result["user_id"] == 1
        conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_profile_existing(self, ltm, mock_pool):
        _, conn = mock_pool
        row = {"user_id": 1, "chat_id": 10, "display_name": "Bob"}
        conn.fetchrow = AsyncMock(return_value=row)
        result = await ltm.get_profile(1, 10)
        assert result["display_name"] == "Bob"

    @pytest.mark.asyncio
    async def test_get_profile_missing(self, ltm, mock_pool):
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)
        result = await ltm.get_profile(1, 10)
        assert result is None


# ---------------------------------------------------------------------------
# Facts and traits
# ---------------------------------------------------------------------------


class TestFactsAndTraits:
    @pytest.mark.asyncio
    async def test_update_facts(self, ltm, mock_pool):
        _, conn = mock_pool
        conn.execute = AsyncMock()
        await ltm.update_facts(1, 10, {"job": "engineer"})
        conn.execute.assert_called_once()
        query = conn.execute.call_args[0][0]
        assert "facts" in query
        assert "||" in query  # JSONB merge

    @pytest.mark.asyncio
    async def test_update_personality_traits(self, ltm, mock_pool):
        _, conn = mock_pool
        conn.execute = AsyncMock()
        await ltm.update_personality_traits(1, 10, {"humor": "dry"})
        conn.execute.assert_called_once()
        query = conn.execute.call_args[0][0]
        assert "personality_traits" in query


# ---------------------------------------------------------------------------
# Emotional state
# ---------------------------------------------------------------------------


class TestEmotionalState:
    @pytest.mark.asyncio
    async def test_update_emotional_state(self, ltm, mock_pool):
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"emotional_state": DEFAULT_EMOTIONS.copy()})
        conn.execute = AsyncMock()

        result = await ltm.update_emotional_state(1, 10, {"warmth": 0.1, "trust": 0.05})
        assert result["warmth"] == 0.1
        assert result["trust"] == 0.05
        conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_emotional_state_clamped(self, ltm, mock_pool):
        _, conn = mock_pool
        current = DEFAULT_EMOTIONS.copy()
        current["warmth"] = 0.95
        conn.fetchrow = AsyncMock(return_value={"emotional_state": current})
        conn.execute = AsyncMock()

        result = await ltm.update_emotional_state(1, 10, {"warmth": 0.2})
        assert result["warmth"] == 1.0  # Clamped at max

    @pytest.mark.asyncio
    async def test_emotional_state_clamped_negative(self, ltm, mock_pool):
        _, conn = mock_pool
        current = DEFAULT_EMOTIONS.copy()
        current["annoyance"] = -0.9
        conn.fetchrow = AsyncMock(return_value={"emotional_state": current})
        conn.execute = AsyncMock()

        result = await ltm.update_emotional_state(1, 10, {"annoyance": -0.2})
        assert result["annoyance"] == -1.0  # Clamped at min

    @pytest.mark.asyncio
    async def test_emotional_state_missing_profile(self, ltm, mock_pool):
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)
        result = await ltm.update_emotional_state(1, 10, {"warmth": 0.1})
        assert result == DEFAULT_EMOTIONS

    @pytest.mark.asyncio
    async def test_get_emotional_state(self, ltm, mock_pool):
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(
            return_value={
                "emotional_state": {
                    "warmth": 0.5,
                    "trust": 0.1,
                    "respect": 0.0,
                    "annoyance": 0.0,
                    "interest": 0.0,
                    "loyalty": 0.0,
                }
            }
        )
        result = await ltm.get_emotional_state(1, 10)
        assert result["warmth"] == 0.5

    @pytest.mark.asyncio
    async def test_get_emotional_state_missing(self, ltm, mock_pool):
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)
        result = await ltm.get_emotional_state(1, 10)
        assert result == DEFAULT_EMOTIONS

    @pytest.mark.asyncio
    async def test_emotional_state_json_string(self, ltm, mock_pool):
        """Emotional state stored as JSON string should be parsed."""
        _, conn = mock_pool
        state = json.dumps(DEFAULT_EMOTIONS)
        conn.fetchrow = AsyncMock(return_value={"emotional_state": state})
        result = await ltm.get_emotional_state(1, 10)
        assert isinstance(result, dict)
        assert "warmth" in result

    @pytest.mark.asyncio
    async def test_unknown_delta_key_ignored(self, ltm, mock_pool):
        _, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"emotional_state": DEFAULT_EMOTIONS.copy()})
        conn.execute = AsyncMock()
        result = await ltm.update_emotional_state(1, 10, {"unknown_key": 0.5})
        # unknown_key should not appear, only known emotions remain
        assert "unknown_key" not in result


# ---------------------------------------------------------------------------
# Group context
# ---------------------------------------------------------------------------


class TestGroupContext:
    @pytest.mark.asyncio
    async def test_get_existing_group(self, ltm, mock_pool):
        _, conn = mock_pool
        row = {"chat_id": 10, "inside_jokes": ["joke1"], "recurring_topics": []}
        conn.fetchrow = AsyncMock(return_value=row)
        result = await ltm.get_group_context(10)
        assert result["inside_jokes"] == ["joke1"]

    @pytest.mark.asyncio
    async def test_create_new_group(self, ltm, mock_pool):
        _, conn = mock_pool
        new_row = {"chat_id": 10, "inside_jokes": [], "recurring_topics": []}
        conn.fetchrow = AsyncMock(side_effect=[None, new_row])
        conn.execute = AsyncMock()
        result = await ltm.get_group_context(10)
        assert result["chat_id"] == 10
        conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_group_context(self, ltm, mock_pool):
        _, conn = mock_pool
        conn.execute = AsyncMock()
        await ltm.update_group_context(10, inside_jokes=["new joke"])
        conn.execute.assert_called_once()


# ---------------------------------------------------------------------------
# Memory log
# ---------------------------------------------------------------------------


class TestMemoryLog:
    @pytest.mark.asyncio
    async def test_append_memory_log(self, ltm, mock_pool):
        _, conn = mock_pool
        conn.execute = AsyncMock()
        await ltm.append_memory_log(10, 1, "fact", "works at Google")
        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert "memory_log" in args[0]
        assert args[3] == "fact"
        assert args[4] == "works at Google"

    @pytest.mark.asyncio
    async def test_append_memory_log_null_user(self, ltm, mock_pool):
        _, conn = mock_pool
        conn.execute = AsyncMock()
        await ltm.append_memory_log(10, None, "insight", "group dynamic")
        args = conn.execute.call_args[0]
        assert args[2] is None

    @pytest.mark.asyncio
    async def test_get_recent_memories_with_user(self, ltm, mock_pool):
        _, conn = mock_pool
        rows = [{"content": "fact1"}, {"content": "fact2"}]
        conn.fetch = AsyncMock(return_value=rows)
        result = await ltm.get_recent_memories(10, user_id=1, limit=10)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_recent_memories_without_user(self, ltm, mock_pool):
        _, conn = mock_pool
        rows = [{"content": "event1"}]
        conn.fetch = AsyncMock(return_value=rows)
        result = await ltm.get_recent_memories(10, user_id=None, limit=20)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Annoyance decay
# ---------------------------------------------------------------------------


class TestAnnoyanceDecay:
    @pytest.mark.asyncio
    async def test_decay_annoyance(self, ltm, mock_pool):
        _, conn = mock_pool
        conn.execute = AsyncMock(return_value="UPDATE 3")
        result = await ltm.decay_annoyance(0.9)
        assert result == 3

    @pytest.mark.asyncio
    async def test_decay_no_profiles(self, ltm, mock_pool):
        _, conn = mock_pool
        conn.execute = AsyncMock(return_value="UPDATE 0")
        result = await ltm.decay_annoyance(0.9)
        assert result == 0
