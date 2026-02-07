"""Tests for VoiceTranscriber — OpenAI Whisper integration."""

from unittest.mock import AsyncMock

import pytest

from src.brain.transcriber import VoiceTranscriber


class TestTranscribe:
    @pytest.mark.asyncio
    async def test_successful_transcription(self):
        client = AsyncMock()
        client.audio.transcriptions.create = AsyncMock(return_value=AsyncMock(text="Привет, как дела?"))
        transcriber = VoiceTranscriber(client=client)
        result = await transcriber.transcribe(b"fake-audio-data")
        assert result == "Привет, как дела?"
        client.audio.transcriptions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_whisper_params(self):
        client = AsyncMock()
        client.audio.transcriptions.create = AsyncMock(return_value=AsyncMock(text="text"))
        transcriber = VoiceTranscriber(client=client)
        await transcriber.transcribe(b"audio")
        call_kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "whisper-1"
        assert call_kwargs["language"] == "ru"

    @pytest.mark.asyncio
    async def test_api_error_returns_none(self):
        client = AsyncMock()
        client.audio.transcriptions.create = AsyncMock(side_effect=Exception("API down"))
        transcriber = VoiceTranscriber(client=client)
        result = await transcriber.transcribe(b"audio")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_audio_returns_none(self):
        client = AsyncMock()
        transcriber = VoiceTranscriber(client=client)
        result = await transcriber.transcribe(b"")
        assert result is None
        client.audio.transcriptions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_result_returns_none(self):
        client = AsyncMock()
        client.audio.transcriptions.create = AsyncMock(return_value=AsyncMock(text="   "))
        transcriber = VoiceTranscriber(client=client)
        result = await transcriber.transcribe(b"audio")
        assert result is None


class TestDisabled:
    @pytest.mark.asyncio
    async def test_no_client_returns_none(self):
        transcriber = VoiceTranscriber(client=None)
        result = await transcriber.transcribe(b"audio")
        assert result is None
