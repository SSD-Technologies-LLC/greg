import json
from unittest.mock import AsyncMock

import pytest

from src.brain.emotions import EmotionTracker


@pytest.fixture
def mock_deps():
    ltm = AsyncMock()
    client = AsyncMock()
    return ltm, client


@pytest.fixture
def tracker(mock_deps):
    ltm, client = mock_deps
    return EmotionTracker(ltm=ltm, anthropic_client=client)


@pytest.mark.asyncio
async def test_evaluate_interaction(tracker, mock_deps):
    ltm, client = mock_deps
    ltm.get_emotional_state = AsyncMock(return_value={
        "warmth": 0.0, "trust": 0.0, "respect": 0.0,
        "annoyance": 0.0, "interest": 0.0, "loyalty": 0.0,
    })
    ltm.update_emotional_state = AsyncMock(return_value={
        "warmth": 0.05, "trust": 0.0, "respect": 0.0,
        "annoyance": 0.0, "interest": 0.02, "loyalty": 0.0,
    })
    ltm.append_memory_log = AsyncMock()

    api_response = AsyncMock()
    api_response.content = [AsyncMock(text=json.dumps({
        "deltas": {"warmth": 0.05, "interest": 0.02},
        "reasoning": "Friendly greeting, showing interest",
    }))]
    client.messages.create = AsyncMock(return_value=api_response)

    result = await tracker.evaluate_interaction(
        chat_id=1, user_id=2, username="alice",
        message_text="Hey Greg! How's it going?",
        greg_response="All good, you?",
    )
    assert result["warmth"] == 0.05
    ltm.update_emotional_state.assert_called_once()
    ltm.append_memory_log.assert_called_once()
