import time
from unittest.mock import AsyncMock, patch

import pytest

from src.brain.decision import DecisionEngine


@pytest.fixture
def engine():
    return DecisionEngine(
        bot_username="greg_bot",
        response_threshold=0.4,
        random_factor=0.07,
        cooldown_messages=3,
        max_unprompted_per_hour=5,
        night_start=1,
        night_end=8,
        timezone="Europe/Moscow",
    )


@pytest.mark.asyncio
async def test_direct_mention_always_responds(engine):
    score = await engine.calculate_score(
        chat_id=1,
        text="@greg_bot what do you think?",
        is_reply_to_bot=False,
        recent_messages=[],
    )
    assert score >= 1.0


@pytest.mark.asyncio
async def test_reply_to_bot_always_responds(engine):
    score = await engine.calculate_score(
        chat_id=1,
        text="I disagree",
        is_reply_to_bot=True,
        recent_messages=[],
    )
    assert score >= 1.0


@pytest.mark.asyncio
async def test_cooldown_penalty(engine):
    recent = [
        {"user_id": 0, "username": "greg_bot", "text": "hey", "timestamp": "t", "chat_id": 1},
        {"user_id": 1, "username": "alice", "text": "ok", "timestamp": "t", "chat_id": 1},
    ]
    score = await engine.calculate_score(
        chat_id=1,
        text="random message about weather",
        is_reply_to_bot=False,
        recent_messages=recent,
    )
    # With cooldown active, score should be low for a bland message
    assert score < 0.5
