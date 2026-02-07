import logging
from io import BytesIO
from typing import Any

logger = logging.getLogger(__name__)


class VoiceTranscriber:
    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    async def transcribe(self, audio_data: bytes) -> str | None:
        if self._client is None:
            return None
        if not audio_data:
            return None

        try:
            buf = BytesIO(audio_data)
            buf.name = "voice.ogg"
            transcript = await self._client.audio.transcriptions.create(
                model="whisper-1",
                file=buf,
                language="ru",
            )
            text = transcript.text.strip()
            if not text:
                return None
            logger.info("Transcribed voice message (%d chars)", len(text))
            return text
        except Exception:
            logger.exception("Voice transcription failed")
            return None
