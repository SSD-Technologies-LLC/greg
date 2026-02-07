import asyncio
import logging
import random
import re

from aiogram import Bot

logger = logging.getLogger(__name__)

TYPING_SPEED = 12  # chars per second
_SEPARATOR_RE = re.compile(r"\n\s*-{3,}\s*\n")


class MessageSender:
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_response(self, chat_id: int, text: str, reply_to: int | None = None) -> None:
        parts = [p.strip() for p in _SEPARATOR_RE.split(text) if p.strip()]
        if not parts:
            return

        for i, part in enumerate(parts):
            delay = len(part) / TYPING_SPEED + random.uniform(0.5, 1.5)
            delay = min(delay, 5.0)

            await self._bot.send_chat_action(chat_id, "typing")
            await asyncio.sleep(delay)

            await self._bot.send_message(
                chat_id=chat_id,
                text=part,
                reply_to_message_id=reply_to if i == 0 else None,
            )

            if i < len(parts) - 1:
                await asyncio.sleep(random.uniform(0.3, 1.0))
