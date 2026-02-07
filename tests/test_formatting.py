"""Tests for output formatting — sanitization, splitting, truncation handling."""

from unittest.mock import AsyncMock, patch

import pytest

from src.brain.responder import sanitize_response
from src.bot.sender import MessageSender


class TestSanitizeResponse:

    def test_strips_literal_separator(self):
        text = "Part 1\\n---\\nPart 2"
        result = sanitize_response(text)
        assert "\\n---\\n" not in result

    def test_strips_dashes_separator(self):
        text = "Part 1\n---\nPart 2"
        result = sanitize_response(text)
        assert "\n---\n" not in result

    def test_strips_leaked_username_format(self):
        text = "Нормальный текст\n[alice]: leaked message\nещё текст"
        result = sanitize_response(text)
        assert "[alice]:" not in result

    def test_strips_bot_username_format(self):
        text = "[greg_ssd_bot]: Спасибо"
        result = sanitize_response(text)
        assert "[greg_ssd_bot]:" not in result

    def test_trims_truncated_response(self):
        text = "Полное предложение. Неполное предлож"
        result = sanitize_response(text)
        assert result == "Полное предложение."

    def test_preserves_complete_response(self):
        text = "Полное предложение. Второе тоже!"
        result = sanitize_response(text)
        assert result == text

    def test_single_complete_sentence(self):
        text = "Просто предложение."
        result = sanitize_response(text)
        assert result == text

    def test_empty_string(self):
        result = sanitize_response("")
        assert result == ""

    def test_ellipsis_is_valid_ending(self):
        text = "Ну такое..."
        result = sanitize_response(text)
        assert result == text

    def test_question_mark_is_valid_ending(self):
        text = "А ты как думаешь?"
        result = sanitize_response(text)
        assert result == text


class TestSenderRobustSplit:

    @pytest.mark.asyncio
    async def test_split_on_dashes_with_spaces(self):
        bot = AsyncMock()
        sender = MessageSender(bot=bot)
        with patch("src.bot.sender.asyncio.sleep", new_callable=AsyncMock):
            await sender.send_response(chat_id=1, text="Part 1\n ---  \nPart 2")
        assert bot.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_split_on_long_dashes(self):
        bot = AsyncMock()
        sender = MessageSender(bot=bot)
        with patch("src.bot.sender.asyncio.sleep", new_callable=AsyncMock):
            await sender.send_response(chat_id=1, text="Part 1\n-----\nPart 2")
        assert bot.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_no_separator_single_message(self):
        bot = AsyncMock()
        sender = MessageSender(bot=bot)
        with patch("src.bot.sender.asyncio.sleep", new_callable=AsyncMock):
            await sender.send_response(chat_id=1, text="Just one message")
        assert bot.send_message.call_count == 1
