import json
from unittest.mock import AsyncMock

import pytest

from src.memory.distiller import Distiller


@pytest.fixture
def mock_deps():
    stm = AsyncMock()
    ltm = AsyncMock()
    client = AsyncMock()
    return stm, ltm, client


@pytest.fixture
def distiller(mock_deps):
    stm, ltm, client = mock_deps
    return Distiller(stm=stm, ltm=ltm, anthropic_client=client)


@pytest.mark.asyncio
async def test_distill_skips_when_no_overflow(distiller, mock_deps):
    stm, _, _ = mock_deps
    stm.get_overflow_messages = AsyncMock(return_value=None)
    result = await distiller.distill(chat_id=123)
    assert result is False


@pytest.mark.asyncio
async def test_distill_extracts_and_stores(distiller, mock_deps):
    stm, ltm, client = mock_deps
    messages = [
        {"user_id": 1, "username": "alice", "text": "I got a new job at Google", "timestamp": "t", "chat_id": 123},
        {"user_id": 2, "username": "bob", "text": "Nice! I'm jealous", "timestamp": "t", "chat_id": 123},
    ]
    stm.get_overflow_messages = AsyncMock(return_value=messages)
    stm.trim_overflow = AsyncMock()

    api_response = AsyncMock()
    api_response.content = [AsyncMock(text=json.dumps({
        "users": {
            "1": {"facts": {"job": "Google"}, "personality_insights": {"ambitious": 0.8}},
            "2": {"facts": {}, "personality_insights": {"humor_style": "self-deprecating"}},
        },
        "group": {"inside_jokes": [], "recurring_topics": ["work"]},
    }))]
    client.messages.create = AsyncMock(return_value=api_response)

    result = await distiller.distill(chat_id=123)
    assert result is True
    ltm.update_facts.assert_called()
    stm.trim_overflow.assert_called_once_with(chat_id=123)
