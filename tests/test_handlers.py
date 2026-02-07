"""End-to-end tests for MessageHandler — the main message processing pipeline."""

import asyncio
import base64
import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.handlers import MessageHandler
from src.brain.decision import DecisionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_message(
    text=None,
    caption=None,
    chat_id=1,
    chat_type="group",
    user_id=42,
    username="alice",
    photo=None,
    video=None,
    voice=None,
    video_note=None,
    reply_to_bot=False,
    bot_username="greg_bot",
):
    """Build a mock aiogram Message with the required nested attributes."""
    msg = MagicMock()
    msg.text = text
    msg.caption = caption
    msg.chat.id = chat_id
    msg.chat.type = chat_type
    msg.from_user.id = user_id
    msg.from_user.username = username
    msg.from_user.first_name = username
    msg.message_id = 100

    msg.photo = photo
    msg.video = video
    msg.voice = voice
    msg.video_note = video_note

    if reply_to_bot:
        msg.reply_to_message.from_user.username = bot_username
    else:
        msg.reply_to_message = None

    # bot.download writes fake JPEG bytes
    async def fake_download(file_id, destination):
        destination.write(b"\xff\xd8fake-jpeg-bytes")

    msg.bot.download = AsyncMock(side_effect=fake_download)
    return msg


def _make_photo():
    """Return a list of mock PhotoSize objects (smallest -> largest)."""
    small = MagicMock()
    small.file_id = "photo_small"
    large = MagicMock()
    large.file_id = "photo_large"
    return [small, large]


def _make_video(with_thumbnail=True):
    vid = MagicMock()
    if with_thumbnail:
        vid.thumbnail.file_id = "video_thumb"
    else:
        vid.thumbnail = None
    return vid


def _make_video_note(with_thumbnail=True):
    vn = MagicMock()
    if with_thumbnail:
        vn.thumbnail.file_id = "vnote_thumb"
    else:
        vn.thumbnail = None
    return vn


def _make_voice():
    return MagicMock()


@pytest.fixture
def deps():
    """Shared mock dependencies for MessageHandler."""
    sender = AsyncMock()
    decision = MagicMock()
    decision.evaluate = AsyncMock(return_value=DecisionResult(
        should_respond=True, is_direct=True, search_needed=False, search_query=None,
    ))
    decision.record_response = MagicMock()

    responder = AsyncMock()
    responder.generate_response = AsyncMock(return_value="Привет!")

    emotions = AsyncMock()
    emotions.evaluate_interaction = AsyncMock(return_value={})

    ctx_builder = AsyncMock()
    ctx_builder.build_context = AsyncMock(return_value={
        "recent_messages": [],
        "user_profile": {},
        "group_context": {},
        "recent_memories": [],
        "other_profiles": {},
    })

    stm = AsyncMock()
    stm.store_message = AsyncMock(return_value=10)
    stm.get_recent_messages = AsyncMock(return_value=[])

    distiller = AsyncMock()
    distiller.distill = AsyncMock(return_value=True)

    searcher = MagicMock()
    searcher.search = MagicMock(return_value=None)

    return sender, decision, responder, emotions, ctx_builder, stm, distiller, searcher


@pytest.fixture
def handler(deps):
    sender, decision, responder, emotions, ctx_builder, stm, distiller, searcher = deps
    return MessageHandler(
        sender=sender,
        decision_engine=decision,
        responder=responder,
        emotion_tracker=emotions,
        context_builder=ctx_builder,
        stm=stm,
        distiller=distiller,
        searcher=searcher,
    )


# ---------------------------------------------------------------------------
# Guard clauses
# ---------------------------------------------------------------------------

class TestGuardClauses:

    @pytest.mark.asyncio
    async def test_no_from_user(self, handler, deps):
        msg = MagicMock()
        msg.from_user = None
        await handler.handle_message(msg)
        deps[5].store_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_content(self, handler, deps):
        msg = _make_message()
        msg.text = None
        msg.caption = None
        msg.photo = None
        msg.video = None
        msg.voice = None
        msg.video_note = None
        await handler.handle_message(msg)
        deps[5].store_message.assert_not_called()


# ---------------------------------------------------------------------------
# Text messages (existing behaviour preserved)
# ---------------------------------------------------------------------------

class TestTextMessages:

    @pytest.mark.asyncio
    async def test_plain_text_stores_and_responds(self, handler, deps):
        sender, decision, responder, emotions, ctx_builder, stm, distiller, searcher = deps
        msg = _make_message(text="Привет всем")
        await handler.handle_message(msg)

        stm.store_message.assert_any_call(1, 42, "alice", "Привет всем")
        responder.generate_response.assert_called_once()
        # image_base64 should be None for plain text
        call_kw = responder.generate_response.call_args
        assert call_kw.kwargs.get("image_base64") is None
        sender.send_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_decision_engine_no_respond_skips_response(self, handler, deps):
        sender, decision, responder, *_ = deps
        decision.evaluate = AsyncMock(return_value=DecisionResult(
            should_respond=False, is_direct=False,
        ))
        msg = _make_message(text="random stuff")
        await handler.handle_message(msg)
        responder.generate_response.assert_not_called()
        sender.send_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_private_chat_always_responds(self, handler, deps):
        sender, decision, responder, *_ = deps
        msg = _make_message(text="hey", chat_type="private")
        await handler.handle_message(msg)
        # In private chat, evaluate should NOT be called
        decision.evaluate.assert_not_called()
        responder.generate_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_response_not_sent(self, handler, deps):
        sender, _, responder, *_ = deps
        responder.generate_response = AsyncMock(return_value=None)
        msg = _make_message(text="hello")
        await handler.handle_message(msg)
        sender.send_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_reply_to_bot_is_direct(self, handler, deps):
        sender, decision, responder, *_ = deps
        with patch("src.bot.handlers.settings") as mock_settings:
            mock_settings.greg_bot_username = "greg_bot"
            mock_settings.greg_redis_buffer_size = 200
            mock_settings.tavily_max_results = 3
            msg = _make_message(text="I disagree", reply_to_bot=True)
            await handler.handle_message(msg)
        # reply_to should be set (is_direct = True)
        call_args = sender.send_response.call_args
        assert call_args.kwargs.get("reply_to") == 100 or call_args[0][2] == 100 or \
            (len(call_args.args) > 2 and call_args.args[2] == 100) or \
            call_args.kwargs.get("reply_to") is not None


# ---------------------------------------------------------------------------
# Media messages: Photo
# ---------------------------------------------------------------------------

class TestPhotoMessages:

    @pytest.mark.asyncio
    async def test_photo_with_caption(self, handler, deps):
        sender, decision, responder, emotions, ctx_builder, stm, distiller, searcher = deps
        msg = _make_message(caption="Смотрите что нашёл", photo=_make_photo())
        await handler.handle_message(msg)

        # STM should get display_text with [Фото] label
        store_call = stm.store_message.call_args_list[0]
        assert store_call[0][3] == "[Фото] Смотрите что нашёл"

        # Responder should receive image_base64
        resp_call = responder.generate_response.call_args
        assert resp_call.kwargs["image_base64"] is not None
        # Verify it's valid base64
        decoded = base64.b64decode(resp_call.kwargs["image_base64"])
        assert decoded == b"\xff\xd8fake-jpeg-bytes"

    @pytest.mark.asyncio
    async def test_photo_without_caption(self, handler, deps):
        _, _, _, _, _, stm, _, _ = deps
        msg = _make_message(photo=_make_photo())
        await handler.handle_message(msg)

        store_call = stm.store_message.call_args_list[0]
        assert store_call[0][3] == "[Фото]"

    @pytest.mark.asyncio
    async def test_photo_downloads_largest_size(self, handler, deps):
        msg = _make_message(photo=_make_photo())
        await handler.handle_message(msg)
        # Should download photo[-1] (largest)
        msg.bot.download.assert_called_once()
        call_args = msg.bot.download.call_args
        assert call_args[0][0] == "photo_large"


# ---------------------------------------------------------------------------
# Media messages: Video
# ---------------------------------------------------------------------------

class TestVideoMessages:

    @pytest.mark.asyncio
    async def test_video_with_thumbnail(self, handler, deps):
        _, _, responder, _, _, stm, _, _ = deps
        msg = _make_message(caption="Классное видео", video=_make_video())
        await handler.handle_message(msg)

        store_call = stm.store_message.call_args_list[0]
        assert store_call[0][3] == "[Видео] Классное видео"

        resp_call = responder.generate_response.call_args
        assert resp_call.kwargs["image_base64"] is not None

    @pytest.mark.asyncio
    async def test_video_without_thumbnail(self, handler, deps):
        _, _, responder, _, _, stm, _, _ = deps
        msg = _make_message(video=_make_video(with_thumbnail=False))
        await handler.handle_message(msg)

        store_call = stm.store_message.call_args_list[0]
        assert store_call[0][3] == "[Видео]"

        resp_call = responder.generate_response.call_args
        assert resp_call.kwargs["image_base64"] is None


# ---------------------------------------------------------------------------
# Media messages: Video note (circles)
# ---------------------------------------------------------------------------

class TestVideoNoteMessages:

    @pytest.mark.asyncio
    async def test_video_note_with_thumbnail(self, handler, deps):
        _, _, responder, _, _, stm, _, _ = deps
        msg = _make_message(video_note=_make_video_note())
        await handler.handle_message(msg)

        store_call = stm.store_message.call_args_list[0]
        assert store_call[0][3] == "[Видеосообщение]"

        resp_call = responder.generate_response.call_args
        assert resp_call.kwargs["image_base64"] is not None

    @pytest.mark.asyncio
    async def test_video_note_without_thumbnail(self, handler, deps):
        _, _, responder, _, _, stm, _, _ = deps
        msg = _make_message(video_note=_make_video_note(with_thumbnail=False))
        await handler.handle_message(msg)

        store_call = stm.store_message.call_args_list[0]
        assert store_call[0][3] == "[Видеосообщение]"

        resp_call = responder.generate_response.call_args
        assert resp_call.kwargs["image_base64"] is None


# ---------------------------------------------------------------------------
# Media messages: Voice
# ---------------------------------------------------------------------------

class TestVoiceMessages:

    @pytest.mark.asyncio
    async def test_voice_message(self, handler, deps):
        _, _, responder, _, _, stm, _, _ = deps
        msg = _make_message(voice=_make_voice())
        await handler.handle_message(msg)

        store_call = stm.store_message.call_args_list[0]
        assert store_call[0][3] == "[Голосовое сообщение]"

        # No image for voice
        resp_call = responder.generate_response.call_args
        assert resp_call.kwargs["image_base64"] is None


# ---------------------------------------------------------------------------
# Download failure handling
# ---------------------------------------------------------------------------

class TestDownloadFailure:

    @pytest.mark.asyncio
    async def test_photo_download_failure_still_processes(self, handler, deps):
        _, _, responder, _, _, stm, _, _ = deps
        msg = _make_message(caption="Look!", photo=_make_photo())
        msg.bot.download = AsyncMock(side_effect=Exception("Network error"))
        await handler.handle_message(msg)

        # Should still store display_text
        store_call = stm.store_message.call_args_list[0]
        assert store_call[0][3] == "[Фото] Look!"

        # image_base64 should be None after failure
        resp_call = responder.generate_response.call_args
        assert resp_call.kwargs["image_base64"] is None


# ---------------------------------------------------------------------------
# Distillation trigger
# ---------------------------------------------------------------------------

class TestDistillation:

    @pytest.mark.asyncio
    async def test_distillation_triggered_on_overflow(self, handler, deps):
        _, _, _, _, _, stm, distiller, _ = deps
        stm.store_message = AsyncMock(return_value=250)  # > buffer_size (200)
        msg = _make_message(text="one more message")

        with patch("src.bot.handlers.settings") as mock_settings:
            mock_settings.greg_bot_username = "greg_bot"
            mock_settings.greg_redis_buffer_size = 200
            mock_settings.greg_max_response_tokens = 300
            mock_settings.tavily_max_results = 3
            await handler.handle_message(msg)

        # Give asyncio.create_task a chance to run
        await asyncio.sleep(0.05)
        distiller.distill.assert_called_once_with(1)


# ---------------------------------------------------------------------------
# Greg's own response storage
# ---------------------------------------------------------------------------

class TestResponseStorage:

    @pytest.mark.asyncio
    async def test_gregs_response_stored_in_stm(self, handler, deps):
        sender, _, responder, _, _, stm, _, _ = deps
        responder.generate_response = AsyncMock(return_value="Ответ Грега")
        msg = _make_message(text="hey")

        with patch("src.bot.handlers.settings") as mock_settings:
            mock_settings.greg_bot_username = "greg_bot"
            mock_settings.greg_redis_buffer_size = 200
            mock_settings.greg_max_response_tokens = 300
            mock_settings.tavily_max_results = 3
            await handler.handle_message(msg)

        # Second call to store_message should be Greg's response
        calls = stm.store_message.call_args_list
        assert len(calls) == 2
        greg_call = calls[1]
        assert greg_call[0][1] == 0  # user_id=0 for Greg
        assert greg_call[0][2] == "greg_bot"
        assert greg_call[0][3] == "Ответ Грега"

    @pytest.mark.asyncio
    async def test_response_separator_stripped(self, handler, deps):
        _, _, responder, _, _, stm, _, _ = deps
        responder.generate_response = AsyncMock(return_value="Часть 1\n---\nЧасть 2")
        msg = _make_message(text="hey")

        with patch("src.bot.handlers.settings") as mock_settings:
            mock_settings.greg_bot_username = "greg_bot"
            mock_settings.greg_redis_buffer_size = 200
            mock_settings.greg_max_response_tokens = 300
            mock_settings.tavily_max_results = 3
            await handler.handle_message(msg)

        greg_call = stm.store_message.call_args_list[1]
        stored_text = greg_call[0][3]
        assert "\n---\n" not in stored_text
        assert "Часть 1" in stored_text
        assert "Часть 2" in stored_text


# ---------------------------------------------------------------------------
# Search integration
# ---------------------------------------------------------------------------

class TestSearchIntegration:

    @pytest.mark.asyncio
    async def test_search_triggered_when_needed(self, handler, deps):
        sender, decision, responder, _, _, _, _, searcher = deps
        decision.evaluate = AsyncMock(return_value=DecisionResult(
            should_respond=True, is_direct=False,
            search_needed=True, search_query="weather moscow",
        ))
        searcher.search = MagicMock(return_value="Moscow: 5°C, cloudy")

        msg = _make_message(text="what's the weather in Moscow?")
        with patch("src.bot.handlers.settings") as mock_settings:
            mock_settings.greg_bot_username = "greg_bot"
            mock_settings.greg_redis_buffer_size = 200
            mock_settings.tavily_max_results = 3
            await handler.handle_message(msg)

        searcher.search.assert_called_once_with("weather moscow", max_results=3)
        call_kwargs = responder.generate_response.call_args.kwargs
        assert call_kwargs["search_context"] == "Moscow: 5°C, cloudy"

    @pytest.mark.asyncio
    async def test_no_search_when_not_needed(self, handler, deps):
        _, _, _, _, _, _, _, searcher = deps
        msg = _make_message(text="hey")
        await handler.handle_message(msg)
        searcher.search.assert_not_called()
