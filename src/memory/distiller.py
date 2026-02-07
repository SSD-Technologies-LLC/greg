import json
import logging

from anthropic import AsyncAnthropic

from config.settings import settings
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory
from src.utils.json_parser import safe_parse_json

logger = logging.getLogger(__name__)

DISTILL_SYSTEM = "You are a memory extraction module. Output only valid JSON, nothing else."

DISTILL_PROMPT = """Проанализируй эти сообщения из группового чата и извлеки:

1. Новые факты о каждом участнике (работа, отношения, интересы, события жизни)
2. Инсайты о личности каждого участника (стиль общения, юмор, ценности)
3. Групповой контекст: внутренние шутки, повторяющиеся темы

Верни ТОЛЬКО валидный JSON в формате:
{{
    "users": {{
        "<user_id>": {{
            "facts": {{"key": "value"}},
            "personality_insights": {{"trait": 0.0-1.0}}
        }}
    }},
    "group": {{
        "inside_jokes": ["joke"],
        "recurring_topics": ["topic"]
    }}
}}

Если ничего нового — верни пустые объекты. Не выдумывай то, чего нет в сообщениях.

Сообщения:
{messages}"""

MAX_RETRIES = 2


class Distiller:
    def __init__(
        self,
        stm: ShortTermMemory,
        ltm: LongTermMemory,
        anthropic_client: AsyncAnthropic,
    ) -> None:
        self._stm = stm
        self._ltm = ltm
        self._client = anthropic_client

    async def _call_with_retry(self, messages_text: str, chat_id: int) -> dict | None:
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._client.messages.create(
                    model=settings.greg_decision_model,
                    max_tokens=1024,
                    system=DISTILL_SYSTEM,
                    messages=[{"role": "user", "content": DISTILL_PROMPT.format(messages=messages_text)}],
                )
                raw = response.content[0].text  # type: ignore[union-attr]
                data = safe_parse_json(raw)
                if data is not None:
                    return data
                logger.warning(
                    "Distillation attempt %d/%d returned unparseable JSON for chat %d",
                    attempt + 1,
                    MAX_RETRIES,
                    chat_id,
                )
            except Exception:
                logger.exception(
                    "Distillation attempt %d/%d failed for chat %d",
                    attempt + 1,
                    MAX_RETRIES,
                    chat_id,
                )
        return None

    async def distill(self, chat_id: int) -> bool:
        overflow = await self._stm.get_overflow_messages(chat_id)
        if not overflow:
            return False

        logger.info("Distilling %d messages for chat %d", len(overflow), chat_id)

        messages_text = "\n".join(f"[{m['username']}({m['user_id']})]: {m['text']}" for m in overflow)

        data = await self._call_with_retry(messages_text, chat_id)
        if data is None:
            logger.error("Distillation failed after %d retries for chat %d", MAX_RETRIES, chat_id)
            return False

        users = data.get("users", {})
        for uid_str, info in users.items():
            uid = int(uid_str)
            facts = info.get("facts", {})
            traits = info.get("personality_insights", {})

            if facts:
                await self._ltm.update_facts(uid, chat_id, facts)
                await self._ltm.append_memory_log(chat_id, uid, "fact", json.dumps(facts, ensure_ascii=False))
            if traits:
                await self._ltm.update_personality_traits(uid, chat_id, traits)
                await self._ltm.append_memory_log(chat_id, uid, "insight", json.dumps(traits, ensure_ascii=False))

        group = data.get("group", {})
        if group.get("inside_jokes") or group.get("recurring_topics"):
            await self._ltm.update_group_context(chat_id, **group)

        await self._stm.trim_overflow(chat_id=chat_id)
        logger.info("Distillation complete for chat %d", chat_id)
        return True
