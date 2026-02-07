"""Comprehensive tests for DecisionEngine — scoring, rate limiting, and response decisions."""

import time
from unittest.mock import patch

import pytest

from src.brain.decision import DecisionEngine


@pytest.fixture
def engine():
    return DecisionEngine(
        bot_username="greg_bot",
        response_threshold=0.4,
        random_factor=0.0,  # Disable randomness for deterministic tests
        cooldown_messages=3,
        max_unprompted_per_hour=5,
        night_start=1,
        night_end=8,
        timezone="Europe/Moscow",
    )


# ---------------------------------------------------------------------------
# Direct mentions & names
# ---------------------------------------------------------------------------

class TestDirectMentions:

    @pytest.mark.asyncio
    async def test_at_mention(self, engine):
        score = await engine.calculate_score(1, "@greg_bot hey", False, [])
        assert score >= 1.0

    @pytest.mark.asyncio
    async def test_username_in_text(self, engine):
        score = await engine.calculate_score(1, "greg_bot what do you think?", False, [])
        assert score >= 1.0

    @pytest.mark.asyncio
    async def test_reply_to_bot(self, engine):
        score = await engine.calculate_score(1, "whatever", True, [])
        assert score >= 1.0

    @pytest.mark.asyncio
    async def test_russian_name_грег(self, engine):
        score = await engine.calculate_score(1, "грег, ты тут?", False, [])
        assert score >= 1.0

    @pytest.mark.asyncio
    async def test_russian_name_гриша(self, engine):
        score = await engine.calculate_score(1, "гриша, привет!", False, [])
        assert score >= 1.0

    @pytest.mark.asyncio
    async def test_russian_name_григорий(self, engine):
        score = await engine.calculate_score(1, "григорий, как дела?", False, [])
        assert score >= 1.0

    @pytest.mark.asyncio
    async def test_case_insensitive_mention(self, engine):
        score = await engine.calculate_score(1, "@GREG_BOT hello", False, [])
        assert score >= 1.0


# ---------------------------------------------------------------------------
# Topic scoring
# ---------------------------------------------------------------------------

class TestTopicScoring:

    @pytest.mark.asyncio
    async def test_hot_take_topic(self, engine):
        score = await engine.calculate_score(1, "ai is taking over the world", False, [])
        assert score >= 0.5

    @pytest.mark.asyncio
    async def test_interest_topic(self, engine):
        score = await engine.calculate_score(1, "let's talk about philosophy", False, [])
        assert score >= 0.3

    @pytest.mark.asyncio
    async def test_bland_message_low_score(self, engine):
        # No topics, no sentiment, lots of messages (no newcomer boost)
        recent = [{"user_id": i, "username": f"u{i}", "text": "x"} for i in range(25)]
        score = await engine.calculate_score(1, "ok", False, recent)
        assert score < 0.4


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------

class TestSentimentScoring:

    @pytest.mark.asyncio
    async def test_vulnerable_keyword(self, engine):
        score = await engine.calculate_score(1, "мне так грустно сегодня", False, [])
        assert score >= 0.6

    @pytest.mark.asyncio
    async def test_positive_strong_keyword(self, engine):
        score = await engine.calculate_score(1, "меня повысили на работе!", False, [])
        assert score >= 0.6

    @pytest.mark.asyncio
    async def test_negative_strong_keyword(self, engine):
        score = await engine.calculate_score(1, "ненавижу понедельники", False, [])
        assert score >= 0.5

    @pytest.mark.asyncio
    async def test_conflict_keyword(self, engine):
        score = await engine.calculate_score(1, "это полный бред", False, [])
        assert score >= 0.5


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

class TestMomentum:

    @pytest.mark.asyncio
    async def test_momentum_when_bot_absent(self, engine):
        recent = [{"user_id": i, "username": f"user{i}", "text": "msg"} for i in range(15)]
        score = engine._score_momentum(recent)
        assert score == 0.2

    @pytest.mark.asyncio
    async def test_no_momentum_when_bot_present(self, engine):
        recent = [{"user_id": i, "username": f"user{i}", "text": "msg"} for i in range(9)]
        recent.append({"user_id": 0, "username": "greg_bot", "text": "hey"})
        score = engine._score_momentum(recent)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_no_momentum_few_messages(self, engine):
        recent = [{"user_id": 1, "username": "a", "text": "hi"}]
        score = engine._score_momentum(recent)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Newcomer boost
# ---------------------------------------------------------------------------

class TestNewcomerBoost:

    def test_few_messages_gets_boost(self, engine):
        recent = [{"user_id": 1, "username": "a", "text": "hi"}] * 10
        assert engine._newcomer_boost(recent) == 0.5

    def test_many_messages_no_boost(self, engine):
        recent = [{"user_id": 1, "username": "a", "text": "hi"}] * 25
        assert engine._newcomer_boost(recent) == 0.0

    def test_exactly_20_no_boost(self, engine):
        recent = [{"user_id": 1, "username": "a", "text": "hi"}] * 20
        assert engine._newcomer_boost(recent) == 0.0


# ---------------------------------------------------------------------------
# Cooldown penalty
# ---------------------------------------------------------------------------

class TestCooldownPenalty:

    def test_bot_spoke_recently(self, engine):
        recent = [
            {"user_id": 0, "username": "greg_bot", "text": "x"},
            {"user_id": 1, "username": "alice", "text": "y"},
            {"user_id": 2, "username": "bob", "text": "z"},
        ]
        assert engine._cooldown_penalty(recent) == 0.3

    def test_bot_not_in_last_n(self, engine):
        recent = [
            {"user_id": 0, "username": "greg_bot", "text": "old"},
            {"user_id": 1, "username": "a", "text": "1"},
            {"user_id": 2, "username": "b", "text": "2"},
            {"user_id": 3, "username": "c", "text": "3"},
            {"user_id": 4, "username": "d", "text": "4"},
        ]
        # Only checks last 3 messages — bot is 5 back
        assert engine._cooldown_penalty(recent) == 0.0

    def test_empty_messages(self, engine):
        assert engine._cooldown_penalty([]) == 0.0


# ---------------------------------------------------------------------------
# Night penalty
# ---------------------------------------------------------------------------

class TestNightPenalty:

    def test_night_time(self, engine):
        with patch("src.brain.decision.datetime") as mock_dt:
            mock_now = mock_dt.now.return_value
            mock_now.hour = 3  # 3am — in night window
            assert engine._night_penalty() == 0.2

    def test_day_time(self, engine):
        with patch("src.brain.decision.datetime") as mock_dt:
            mock_now = mock_dt.now.return_value
            mock_now.hour = 14  # 2pm — daytime
            assert engine._night_penalty() == 0.0


# ---------------------------------------------------------------------------
# Active conversation boost
# ---------------------------------------------------------------------------

class TestActiveConversationBoost:

    def test_recent_response_gives_boost(self, engine):
        engine._active_convos[1] = time.time()  # Just now
        assert engine._active_conversation_boost(1) == 0.4

    def test_old_response_no_boost(self, engine):
        engine._active_convos[1] = time.time() - 300  # 5 minutes ago
        assert engine._active_conversation_boost(1) == 0.0

    def test_no_prior_response_no_boost(self, engine):
        assert engine._active_conversation_boost(999) == 0.0


# ---------------------------------------------------------------------------
# should_respond
# ---------------------------------------------------------------------------

class TestShouldRespond:

    def test_direct_always_responds(self, engine):
        assert engine.should_respond(1.0, chat_id=1, is_direct=True) is True

    def test_below_threshold_no_response(self, engine):
        assert engine.should_respond(0.1, chat_id=1, is_direct=False) is False

    def test_above_threshold_responds(self, engine):
        assert engine.should_respond(0.5, chat_id=1, is_direct=False) is True

    def test_rate_limit_blocks(self, engine):
        # Fill up rate limit
        for _ in range(10):
            engine.record_response(chat_id=1, is_direct=False)
        assert engine.should_respond(0.5, chat_id=1, is_direct=False) is False

    def test_direct_bypasses_rate_limit(self, engine):
        for _ in range(10):
            engine.record_response(chat_id=1, is_direct=False)
        assert engine.should_respond(1.0, chat_id=1, is_direct=True) is True


# ---------------------------------------------------------------------------
# record_response
# ---------------------------------------------------------------------------

class TestRecordResponse:

    def test_records_active_convo(self, engine):
        engine.record_response(chat_id=1, is_direct=False)
        assert 1 in engine._active_convos

    def test_unprompted_logged(self, engine):
        engine.record_response(chat_id=1, is_direct=False)
        assert len(engine._unprompted_log[1]) == 1

    def test_direct_not_logged_as_unprompted(self, engine):
        engine.record_response(chat_id=1, is_direct=True)
        assert len(engine._unprompted_log[1]) == 0

    def test_direct_still_records_active_convo(self, engine):
        engine.record_response(chat_id=1, is_direct=True)
        assert 1 in engine._active_convos


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimit:

    def test_within_limit(self, engine):
        assert engine._check_rate_limit(1) is True

    def test_exceeds_limit(self, engine):
        for _ in range(6):
            engine._unprompted_log[1].append(time.time())
        assert engine._check_rate_limit(1) is False

    def test_old_entries_cleaned(self, engine):
        # Add old entries (2 hours ago)
        engine._unprompted_log[1] = [time.time() - 7200 for _ in range(10)]
        assert engine._check_rate_limit(1) is True
