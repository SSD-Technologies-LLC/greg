"""Tests for ContextBuilder — assembling context from STM + LTM."""

from unittest.mock import AsyncMock

import pytest

from src.memory.context_builder import ContextBuilder


@pytest.fixture
def mock_deps():
    stm = AsyncMock()
    ltm = AsyncMock()
    return stm, ltm


@pytest.fixture
def builder(mock_deps):
    stm, ltm = mock_deps
    return ContextBuilder(stm=stm, ltm=ltm)


class TestBuildContext:

    @pytest.mark.asyncio
    async def test_returns_all_required_keys(self, builder, mock_deps):
        stm, ltm = mock_deps
        stm.get_recent_messages = AsyncMock(return_value=[])
        ltm.get_or_create_profile = AsyncMock(return_value={"user_id": 1, "display_name": "Alice"})
        ltm.get_group_context = AsyncMock(return_value={"chat_id": 10})
        ltm.get_recent_memories = AsyncMock(return_value=[])

        result = await builder.build_context(chat_id=10, user_id=1, display_name="Alice")

        assert "recent_messages" in result
        assert "user_profile" in result
        assert "group_context" in result
        assert "recent_memories" in result
        assert "other_profiles" in result

    @pytest.mark.asyncio
    async def test_collects_other_profiles(self, builder, mock_deps):
        stm, ltm = mock_deps
        stm.get_recent_messages = AsyncMock(return_value=[
            {"user_id": 1, "username": "alice", "text": "hi"},
            {"user_id": 2, "username": "bob", "text": "hello"},
            {"user_id": 3, "username": "carol", "text": "hey"},
        ])
        ltm.get_or_create_profile = AsyncMock(return_value={"user_id": 1})
        ltm.get_group_context = AsyncMock(return_value={})
        ltm.get_recent_memories = AsyncMock(return_value=[])
        ltm.get_profile = AsyncMock(return_value={
            "display_name": "Other",
            "facts": {},
            "emotional_state": {"warmth": 0.1},
        })

        result = await builder.build_context(chat_id=10, user_id=1, display_name="Alice")

        # Should fetch profiles for user 2 and 3 (not 1, that's the current user)
        assert len(result["other_profiles"]) == 2
        assert 2 in result["other_profiles"]
        assert 3 in result["other_profiles"]
        assert 1 not in result["other_profiles"]

    @pytest.mark.asyncio
    async def test_missing_other_profile_excluded(self, builder, mock_deps):
        stm, ltm = mock_deps
        stm.get_recent_messages = AsyncMock(return_value=[
            {"user_id": 1, "username": "alice", "text": "hi"},
            {"user_id": 2, "username": "bob", "text": "hello"},
        ])
        ltm.get_or_create_profile = AsyncMock(return_value={"user_id": 1})
        ltm.get_group_context = AsyncMock(return_value={})
        ltm.get_recent_memories = AsyncMock(return_value=[])
        ltm.get_profile = AsyncMock(return_value=None)  # user not found

        result = await builder.build_context(chat_id=10, user_id=1, display_name="Alice")
        assert len(result["other_profiles"]) == 0

    @pytest.mark.asyncio
    async def test_limits_other_profiles_to_10(self, builder, mock_deps):
        stm, ltm = mock_deps
        messages = [{"user_id": i, "username": f"u{i}", "text": "hi"} for i in range(20)]
        messages.append({"user_id": 999, "username": "me", "text": "current"})
        stm.get_recent_messages = AsyncMock(return_value=messages)
        ltm.get_or_create_profile = AsyncMock(return_value={"user_id": 999})
        ltm.get_group_context = AsyncMock(return_value={})
        ltm.get_recent_memories = AsyncMock(return_value=[])
        ltm.get_profile = AsyncMock(return_value={
            "display_name": "Other",
            "facts": {},
            "emotional_state": {},
        })

        result = await builder.build_context(chat_id=10, user_id=999, display_name="me")
        assert len(result["other_profiles"]) <= 10
