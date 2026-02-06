import asyncio
import logging

from aiogram import Router
from aiogram.types import Message

from config.settings import settings
from src.bot.sender import MessageSender
from src.brain.decision import DecisionEngine
from src.brain.emotions import EmotionTracker
from src.brain.responder import Responder
from src.memory.context_builder import ContextBuilder
from src.memory.distiller import Distiller
from src.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)

router = Router()


class MessageHandler:
    def __init__(
        self,
        sender: MessageSender,
        decision_engine: DecisionEngine,
        responder: Responder,
        emotion_tracker: EmotionTracker,
        context_builder: ContextBuilder,
        stm: ShortTermMemory,
        distiller: Distiller,
    ) -> None:
        self._sender = sender
        self._decision = decision_engine
        self._responder = responder
        self._emotions = emotion_tracker
        self._context = context_builder
        self._stm = stm
        self._distiller = distiller

    async def handle_message(self, message: Message) -> None:
        if not message.text or not message.from_user:
            return

        chat_id = message.chat.id
        user_id = message.from_user.id
        username = message.from_user.username or message.from_user.first_name or str(user_id)
        text = message.text

        # Store in short-term memory
        buffer_len = await self._stm.store_message(chat_id, user_id, username, text)

        # Trigger distillation if buffer overflowing
        if buffer_len > settings.greg_redis_buffer_size:
            asyncio.create_task(self._safe_distill(chat_id))

        # Check if Greg should respond
        is_reply_to_bot = (
            message.reply_to_message is not None
            and message.reply_to_message.from_user is not None
            and message.reply_to_message.from_user.username is not None
            and message.reply_to_message.from_user.username.lower() == settings.greg_bot_username.lower()
        )

        recent = await self._stm.get_recent_messages(chat_id, count=20)

        score = await self._decision.calculate_score(
            chat_id=chat_id,
            text=text,
            is_reply_to_bot=is_reply_to_bot,
            recent_messages=recent,
        )

        is_direct = score >= 1.0
        if not self._decision.should_respond(score, chat_id, is_direct):
            return

        # Build context and generate response
        context = await self._context.build_context(chat_id, user_id, username)
        response = await self._responder.generate_response(context, text, username)

        if not response:
            return

        # Send response
        reply_to = message.message_id if is_direct else None
        await self._sender.send_response(chat_id, response, reply_to=reply_to)

        # Store Greg's response in short-term memory
        await self._stm.store_message(
            chat_id, 0, settings.greg_bot_username, response.replace("\n---\n", " ")
        )

        # Record for rate limiting
        self._decision.record_response(chat_id, is_direct)

        # Evaluate emotional impact in background
        asyncio.create_task(
            self._safe_emotion_update(chat_id, user_id, username, text, response)
        )

    async def _safe_distill(self, chat_id: int) -> None:
        try:
            await self._distiller.distill(chat_id)
        except Exception:
            logger.exception("Background distillation failed for chat %d", chat_id)

    async def _safe_emotion_update(
        self, chat_id: int, user_id: int, username: str, text: str, response: str
    ) -> None:
        try:
            await self._emotions.evaluate_interaction(
                chat_id, user_id, username, text, response
            )
        except Exception:
            logger.exception("Background emotion update failed for user %d", user_id)
