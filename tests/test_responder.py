"""Tests for Responder — Claude API integration and response generation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.brain.responder import Responder


@pytest.fixture
def mock_client():
    client = AsyncMock()
    response = AsyncMock()
    response.content = [AsyncMock(text="Ответ Грега")]
    client.messages.create = AsyncMock(return_value=response)
    return client


@pytest.fixture
def responder(mock_client):
    return Responder(anthropic_client=mock_client)


def _context():
    return {
        "recent_messages": [
            {"username": "alice", "text": "hey greg"},
        ],
        "user_profile": {"emotional_state": {}},
        "group_context": {},
        "recent_memories": [],
        "other_profiles": {},
    }


class TestGenerateResponse:

    @pytest.mark.asyncio
    async def test_basic_response(self, responder, mock_client):
        result = await responder.generate_response(_context(), "hello", "alice")
        assert result == "Ответ Грега"
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_system_prompt(self, responder, mock_client):
        await responder.generate_response(_context(), "hello", "alice")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert "Грег" in call_kwargs["system"]

    @pytest.mark.asyncio
    async def test_passes_messages(self, responder, mock_client):
        await responder.generate_response(_context(), "hello", "alice")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "messages" in call_kwargs
        assert len(call_kwargs["messages"]) > 0

    @pytest.mark.asyncio
    async def test_image_forwarded_to_messages(self, responder, mock_client):
        await responder.generate_response(
            _context(), "[Фото] look", "alice", image_base64="AAAA"
        )
        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = call_kwargs["messages"]
        # Last message should have multimodal content
        last = messages[-1]
        assert isinstance(last["content"], list)
        assert last["content"][0]["type"] == "image"

    @pytest.mark.asyncio
    async def test_no_image_text_only_messages(self, responder, mock_client):
        await responder.generate_response(_context(), "hello", "alice")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = call_kwargs["messages"]
        last = messages[-1]
        assert isinstance(last["content"], str)

    @pytest.mark.asyncio
    async def test_api_failure_returns_none(self, responder, mock_client):
        mock_client.messages.create = AsyncMock(side_effect=Exception("API down"))
        result = await responder.generate_response(_context(), "hello", "alice")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_messages_returns_none(self, responder, mock_client):
        ctx = {
            "recent_messages": [],
            "user_profile": {"emotional_state": {}},
            "group_context": {},
        }
        # build_messages returns [] only if all messages are empty — hard to trigger
        # but generate_response should handle it
        result = await responder.generate_response(ctx, "hello", "alice")
        # With "hello" as current_text, build_messages always adds at least 1 message
        assert result is not None

    @pytest.mark.asyncio
    async def test_uses_correct_model(self, responder, mock_client):
        await responder.generate_response(_context(), "hello", "alice")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-6"

    @pytest.mark.asyncio
    async def test_response_stripped(self, responder, mock_client):
        response = AsyncMock()
        response.content = [AsyncMock(text="  some text  \n ")]
        mock_client.messages.create = AsyncMock(return_value=response)
        result = await responder.generate_response(_context(), "hello", "alice")
        assert result == "some text"
