"""Tests for MessageSender — typing animation and message splitting."""

from unittest.mock import AsyncMock, patch

import pytest

from src.bot.sender import MessageSender


@pytest.fixture
def mock_bot():
    bot = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()
    return bot


@pytest.fixture
def sender(mock_bot):
    return MessageSender(bot=mock_bot)


class TestSendResponse:
    @pytest.mark.asyncio
    async def test_single_message(self, sender, mock_bot):
        with patch("src.bot.sender.asyncio.sleep", new_callable=AsyncMock):
            await sender.send_response(chat_id=1, text="Hello!")
        mock_bot.send_message.assert_called_once_with(chat_id=1, text="Hello!", reply_to_message_id=None)

    @pytest.mark.asyncio
    async def test_reply_to(self, sender, mock_bot):
        with patch("src.bot.sender.asyncio.sleep", new_callable=AsyncMock):
            await sender.send_response(chat_id=1, text="Reply", reply_to=42)
        call_kwargs = mock_bot.send_message.call_args.kwargs
        assert call_kwargs["reply_to_message_id"] == 42

    @pytest.mark.asyncio
    async def test_multi_part_message(self, sender, mock_bot):
        with patch("src.bot.sender.asyncio.sleep", new_callable=AsyncMock):
            await sender.send_response(chat_id=1, text="Part 1\n---\nPart 2", reply_to=10)
        assert mock_bot.send_message.call_count == 2
        # First part gets reply_to, second doesn't
        first_call = mock_bot.send_message.call_args_list[0]
        assert first_call.kwargs["reply_to_message_id"] == 10
        second_call = mock_bot.send_message.call_args_list[1]
        assert second_call.kwargs["reply_to_message_id"] is None

    @pytest.mark.asyncio
    async def test_empty_text_no_send(self, sender, mock_bot):
        with patch("src.bot.sender.asyncio.sleep", new_callable=AsyncMock):
            await sender.send_response(chat_id=1, text="")
        mock_bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_typing_action_sent(self, sender, mock_bot):
        with patch("src.bot.sender.asyncio.sleep", new_callable=AsyncMock):
            await sender.send_response(chat_id=1, text="Hello")
        mock_bot.send_chat_action.assert_called_with(1, "typing")

    @pytest.mark.asyncio
    async def test_whitespace_only_parts_stripped(self, sender, mock_bot):
        with patch("src.bot.sender.asyncio.sleep", new_callable=AsyncMock):
            await sender.send_response(chat_id=1, text="Real\n---\n   \n---\nAlso real")
        # Middle part is whitespace-only, should be stripped out
        assert mock_bot.send_message.call_count == 2
