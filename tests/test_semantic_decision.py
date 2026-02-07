"""Tests for the semantic decision engine using Haiku evaluation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.brain.decision import DecisionEngine


@pytest.fixture
def mock_client():
    client = AsyncMock()
    response = AsyncMock()
    response.content = [
        AsyncMock(
            text=json.dumps(
                {
                    "respond": True,
                    "reason": "interesting conversation",
                    "search_needed": False,
                    "search_query": None,
                }
            )
        )
    ]
    client.messages.create = AsyncMock(return_value=response)
    return client


@pytest.fixture
def engine(mock_client):
    return DecisionEngine(
        bot_username="greg_bot",
        anthropic_client=mock_client,
        max_unprompted_per_hour=5,
        night_start=1,
        night_end=8,
        timezone="Europe/Moscow",
    )


class TestDirectTriggers:
    @pytest.mark.asyncio
    async def test_reply_to_bot_returns_direct(self, engine, mock_client):
        result = await engine.evaluate(
            chat_id=1,
            text="I disagree",
            is_reply_to_bot=True,
            recent_messages=[],
        )
        assert result.should_respond is True
        assert result.is_direct is True
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_at_mention_returns_direct(self, engine, mock_client):
        result = await engine.evaluate(
            chat_id=1,
            text="@greg_bot what do you think?",
            is_reply_to_bot=False,
            recent_messages=[],
        )
        assert result.should_respond is True
        assert result.is_direct is True
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_russian_name_returns_direct(self, engine, mock_client):
        result = await engine.evaluate(
            chat_id=1,
            text="Гриша, ты чё думаешь?",
            is_reply_to_bot=False,
            recent_messages=[],
        )
        assert result.should_respond is True
        assert result.is_direct is True
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_username_in_text_returns_direct(self, engine, mock_client):
        result = await engine.evaluate(
            chat_id=1,
            text="greg_bot is cool",
            is_reply_to_bot=False,
            recent_messages=[],
        )
        assert result.should_respond is True
        assert result.is_direct is True


class TestSemanticEvaluation:
    """Semantic evaluation tests — patched to daytime to avoid night gate."""

    @pytest.mark.asyncio
    async def test_haiku_says_respond(self, engine, mock_client):
        with patch("src.brain.decision.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=12)
            result = await engine.evaluate(
                chat_id=1,
                text="what do you guys think about the new iPhone?",
                is_reply_to_bot=False,
                recent_messages=[{"username": "alice", "text": "hi", "user_id": 1}],
            )
        assert result.should_respond is True
        assert result.is_direct is False
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_haiku_says_dont_respond(self, engine, mock_client):
        response = AsyncMock()
        response.content = [
            AsyncMock(
                text=json.dumps(
                    {
                        "respond": False,
                        "reason": "boring small talk",
                        "search_needed": False,
                        "search_query": None,
                    }
                )
            )
        ]
        mock_client.messages.create = AsyncMock(return_value=response)

        with patch("src.brain.decision.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=12)
            result = await engine.evaluate(
                chat_id=1,
                text="ok",
                is_reply_to_bot=False,
                recent_messages=[],
            )
        assert result.should_respond is False

    @pytest.mark.asyncio
    async def test_search_needed_flag(self, engine, mock_client):
        response = AsyncMock()
        response.content = [
            AsyncMock(
                text=json.dumps(
                    {
                        "respond": True,
                        "reason": "factual question",
                        "search_needed": True,
                        "search_query": "Bitcoin price today",
                    }
                )
            )
        ]
        mock_client.messages.create = AsyncMock(return_value=response)

        with patch("src.brain.decision.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=12)
            result = await engine.evaluate(
                chat_id=1,
                text="what's Bitcoin at right now?",
                is_reply_to_bot=False,
                recent_messages=[],
            )
        assert result.search_needed is True
        assert result.search_query == "Bitcoin price today"

    @pytest.mark.asyncio
    async def test_api_failure_defaults_to_no_response(self, engine, mock_client):
        mock_client.messages.create = AsyncMock(side_effect=Exception("API down"))
        with patch("src.brain.decision.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=12)
            result = await engine.evaluate(
                chat_id=1,
                text="hello",
                is_reply_to_bot=False,
                recent_messages=[],
            )
        assert result.should_respond is False

    @pytest.mark.asyncio
    async def test_invalid_json_defaults_to_no_response(self, engine, mock_client):
        response = AsyncMock()
        response.content = [AsyncMock(text="not json at all")]
        mock_client.messages.create = AsyncMock(return_value=response)
        with patch("src.brain.decision.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=12)
            result = await engine.evaluate(
                chat_id=1,
                text="hello",
                is_reply_to_bot=False,
                recent_messages=[],
            )
        assert result.should_respond is False


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limit_blocks_unprompted(self, engine, mock_client):
        for _ in range(5):
            engine.record_response(chat_id=1, is_direct=False)
        result = await engine.evaluate(
            chat_id=1,
            text="interesting topic",
            is_reply_to_bot=False,
            recent_messages=[],
        )
        assert result.should_respond is False
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_allows_direct(self, engine, mock_client):
        for _ in range(5):
            engine.record_response(chat_id=1, is_direct=False)
        result = await engine.evaluate(
            chat_id=1,
            text="@greg_bot hello",
            is_reply_to_bot=False,
            recent_messages=[],
        )
        assert result.should_respond is True


class TestNightMode:
    @pytest.mark.asyncio
    async def test_night_mode_skips_api_call(self, engine, mock_client):
        with patch("src.brain.decision.datetime") as mock_dt:
            mock_now = MagicMock()
            mock_now.hour = 3
            mock_dt.now.return_value = mock_now
            result = await engine.evaluate(
                chat_id=1,
                text="hello",
                is_reply_to_bot=False,
                recent_messages=[],
            )
            assert result.should_respond is False
            mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_direct_bypasses_night_mode(self, engine, mock_client):
        with patch("src.brain.decision.datetime") as mock_dt:
            mock_now = MagicMock()
            mock_now.hour = 3
            mock_dt.now.return_value = mock_now
            result = await engine.evaluate(
                chat_id=1,
                text="Гриша!",
                is_reply_to_bot=False,
                recent_messages=[],
            )
            assert result.should_respond is True


class TestRecordResponse:
    def test_records_last_response_time(self, engine):
        engine.record_response(chat_id=1, is_direct=False)
        assert 1 in engine._last_response

    def test_direct_not_logged_as_unprompted(self, engine):
        engine.record_response(chat_id=1, is_direct=True)
        assert len(engine._unprompted_log[1]) == 0

    def test_unprompted_logged(self, engine):
        engine.record_response(chat_id=1, is_direct=False)
        assert len(engine._unprompted_log[1]) == 1
