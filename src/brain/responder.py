import logging

from anthropic import AsyncAnthropic

from config.settings import settings
from src.brain.personality import PersonalityEngine

logger = logging.getLogger(__name__)


class Responder:
    def __init__(self, anthropic_client: AsyncAnthropic) -> None:
        self._client = anthropic_client
        self._personality = PersonalityEngine()

    async def generate_response(
        self, context: dict, current_text: str, current_username: str,
        *, image_base64: str | None = None,
    ) -> str | None:
        system_prompt = self._personality.build_system_prompt(context)
        messages = self._personality.build_messages(
            context, current_text, current_username, image_base64=image_base64
        )

        if not messages:
            return None

        try:
            response = await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=settings.greg_max_response_tokens,
                system=system_prompt,
                messages=messages,
            )
            text = response.content[0].text.strip()
            logger.info("Generated response (%d chars) for %s", len(text), current_username)
            return text
        except Exception:
            logger.exception("Failed to generate response")
            return None
