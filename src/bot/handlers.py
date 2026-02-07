import asyncio
import base64
import logging
from io import BytesIO

from aiogram import Router
from aiogram.types import Message

from config.settings import settings
from src.bot.sender import MessageSender
from src.brain.decision import DecisionEngine
from src.brain.emotions import EmotionTracker
from src.brain.responder import Responder
from src.brain.searcher import WebSearcher
from src.brain.transcriber import VoiceTranscriber
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
        searcher: WebSearcher | None = None,
        transcriber: VoiceTranscriber | None = None,
    ) -> None:
        self._sender = sender
        self._decision = decision_engine
        self._responder = responder
        self._emotions = emotion_tracker
        self._context = context_builder
        self._stm = stm
        self._distiller = distiller
        self._searcher = searcher
        self._transcriber = transcriber

    async def handle_message(self, message: Message) -> None:
        if not message.from_user:
            return
        if not (
            message.text or message.caption or message.photo or message.video or message.voice or message.video_note
        ):
            return

        chat_id = message.chat.id
        user_id = message.from_user.id
        username = message.from_user.username or message.from_user.first_name or str(user_id)

        display_text, image_base64 = await self._extract_media(message)

        logger.info("Message from %s in chat %d: %s", username, chat_id, display_text[:80])

        buffer_len = await self._stm.store_message(chat_id, user_id, username, display_text)

        if buffer_len > settings.greg_redis_buffer_size:
            asyncio.create_task(self._safe_distill(chat_id))

        is_private = message.chat.type == "private"

        is_reply_to_bot = (
            message.reply_to_message is not None
            and message.reply_to_message.from_user is not None
            and message.reply_to_message.from_user.username is not None
            and message.reply_to_message.from_user.username.lower() == settings.greg_bot_username.lower()
        )

        recent = await self._stm.get_recent_messages(chat_id, count=20)

        if is_private:
            should_respond = True
            is_direct = True
            search_needed = False
            search_query = None
        else:
            result = await self._decision.evaluate(
                chat_id=chat_id,
                text=display_text,
                is_reply_to_bot=is_reply_to_bot,
                recent_messages=recent,
            )
            should_respond = result.should_respond
            is_direct = result.is_direct
            search_needed = result.search_needed
            search_query = result.search_query

        if not should_respond:
            return

        # Web search if needed (run in thread to avoid blocking event loop)
        search_context = None
        if search_needed and search_query and self._searcher:
            search_context = await asyncio.to_thread(self._searcher.search, search_query, settings.tavily_max_results)

        context = await self._context.build_context(chat_id, user_id, username)
        response = await self._responder.generate_response(
            context,
            display_text,
            username,
            image_base64=image_base64,
            search_context=search_context,
        )

        if not response:
            return

        reply_to = message.message_id if is_direct else None
        await self._sender.send_response(chat_id, response, reply_to=reply_to)

        await self._stm.store_message(chat_id, 0, settings.greg_bot_username, response.replace("\n---\n", " "))

        self._decision.record_response(chat_id, is_direct)

        asyncio.create_task(self._safe_emotion_update(chat_id, user_id, username, display_text, response))

    async def _extract_media(self, message: Message) -> tuple[str, str | None]:
        caption = message.text or message.caption or ""
        image_base64 = None

        if message.photo:
            label = "[Фото]"
            image_base64 = await self._download_image(message, message.photo[-1].file_id)
        elif message.video:
            label = "[Видео]"
            if message.video.thumbnail:
                image_base64 = await self._download_image(message, message.video.thumbnail.file_id)
        elif message.video_note:
            label = "[Видеосообщение]"
            if message.video_note.thumbnail:
                image_base64 = await self._download_image(message, message.video_note.thumbnail.file_id)
        elif message.voice:
            label = "[Голосовое сообщение]"
            transcription = await self._transcribe_voice(message, message.voice.file_id)
            if transcription:
                display_text = f"{label} {transcription}"
                if caption:
                    display_text = f"{label} {caption} — {transcription}"
                return display_text, None
        else:
            return caption, None

        display_text = f"{label} {caption}".strip() if caption else label
        return display_text, image_base64

    async def _transcribe_voice(self, message: Message, file_id: str) -> str | None:
        if not self._transcriber:
            return None
        try:
            buf = BytesIO()
            assert message.bot is not None
            await message.bot.download(file_id, destination=buf)
            audio_data = buf.getvalue()
            if not audio_data:
                return None
            return await self._transcriber.transcribe(audio_data)
        except Exception:
            logger.exception("Failed to download voice %s", file_id)
            return None

    async def _download_image(self, message: Message, file_id: str) -> str | None:
        try:
            buf = BytesIO()
            assert message.bot is not None
            await message.bot.download(file_id, destination=buf)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            logger.exception("Failed to download file %s", file_id)
            return None

    async def _safe_distill(self, chat_id: int) -> None:
        try:
            await self._distiller.distill(chat_id)
        except Exception:
            logger.exception("Background distillation failed for chat %d", chat_id)

    async def _safe_emotion_update(self, chat_id: int, user_id: int, username: str, text: str, response: str) -> None:
        try:
            await self._emotions.evaluate_interaction(chat_id, user_id, username, text, response)
        except Exception:
            logger.exception("Background emotion update failed for user %d", user_id)
