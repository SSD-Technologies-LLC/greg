import logging
import re

from anthropic import AsyncAnthropic

from config.settings import settings
from src.brain.personality import PersonalityEngine

logger = logging.getLogger(__name__)

# Pattern to match leaked [username]: format
_LEAKED_FORMAT_RE = re.compile(r"^\[[\w]+\]:.*$", re.MULTILINE)
# Pattern to match literal \n---\n (escaped)
_LITERAL_SEP_RE = re.compile(r"\\n---\\n")
# Terminal punctuation (including Russian)
_TERMINAL_PUNCT = re.compile(r"[.!?…»)\"']$")


def sanitize_response(text: str) -> str:
    if not text:
        return text

    # Strip literal escaped separators
    text = _LITERAL_SEP_RE.sub(" ", text)

    # Strip actual --- separators (will be handled by sender, but clean up remnants)
    text = re.sub(r"\n\s*-{3,}\s*\n", " ", text)

    # Strip leaked [username]: lines
    text = _LEAKED_FORMAT_RE.sub("", text)

    # Clean up extra whitespace from removals
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Handle truncated responses — trim to last complete sentence
    if text and not _TERMINAL_PUNCT.search(text):
        last_punct = -1
        for m in re.finditer(r"[.!?…]", text):
            last_punct = m.end()
        if last_punct > 0:
            text = text[:last_punct].strip()

    return text


class Responder:
    def __init__(self, anthropic_client: AsyncAnthropic) -> None:
        self._client = anthropic_client
        self._personality = PersonalityEngine()

    async def generate_response(
        self, context: dict, current_text: str, current_username: str,
        *, image_base64: str | None = None,
        search_context: str | None = None,
    ) -> str | None:
        system_prompt = self._personality.build_system_prompt(context)

        if search_context:
            system_prompt += f"\n\n[Результаты поиска — используй если релевантно, не цитируй дословно]:\n{search_context}"

        messages = self._personality.build_messages(
            context, current_text, current_username, image_base64=image_base64
        )

        if not messages:
            return None

        try:
            response = await self._client.messages.create(
                model=settings.greg_response_model,
                max_tokens=settings.greg_max_response_tokens,
                system=system_prompt,
                messages=messages,
            )
            text = response.content[0].text.strip()
            text = sanitize_response(text)
            logger.info("Generated response (%d chars) for %s", len(text), current_username)
            return text if text else None
        except Exception:
            logger.exception("Failed to generate response")
            return None
