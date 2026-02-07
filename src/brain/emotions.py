import json
import logging

from anthropic import AsyncAnthropic

from config.settings import settings
from src.memory.long_term import LongTermMemory
from src.utils.json_parser import safe_parse_json

logger = logging.getLogger(__name__)

EMOTION_SYSTEM = "You are an emotional analysis module. Output only valid JSON, nothing else."

EMOTION_PROMPT = """Ты оцениваешь эмоциональное взаимодействие. Грег — участник чата с характером.

Текущее эмоциональное состояние Грега к этому человеку:
{current_state}

Сообщение от {username}: "{message}"
Ответ Грега: "{response}"

Оцени, как это взаимодействие должно изменить эмоции Грега к этому человеку.
Верни ТОЛЬКО валидный JSON:
{{
    "deltas": {{
        "warmth": float (-0.1 to 0.1),
        "trust": float (-0.1 to 0.1),
        "respect": float (-0.1 to 0.1),
        "annoyance": float (-0.1 to 0.1),
        "interest": float (-0.1 to 0.1),
        "loyalty": float (-0.05 to 0.05)
    }},
    "reasoning": "brief explanation in Russian"
}}

Дельты должны быть маленькими. Большие сдвиги только при значимых событиях.
Не включай дельту если изменений нет (опусти ключ или поставь 0)."""


class EmotionTracker:
    def __init__(self, ltm: LongTermMemory, anthropic_client: AsyncAnthropic) -> None:
        self._ltm = ltm
        self._client = anthropic_client

    async def evaluate_interaction(
        self,
        chat_id: int,
        user_id: int,
        username: str,
        message_text: str,
        greg_response: str,
    ) -> dict:
        current = await self._ltm.get_emotional_state(user_id, chat_id)

        try:
            response = await self._client.messages.create(
                model=settings.greg_decision_model,
                max_tokens=256,
                system=EMOTION_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": EMOTION_PROMPT.format(
                            current_state=json.dumps(current),
                            username=username,
                            message=message_text,
                            response=greg_response,
                        ),
                    }
                ],
            )
            raw = response.content[0].text.strip()  # type: ignore[union-attr]
            data = safe_parse_json(raw)
            if data is None:
                logger.warning("No valid JSON in emotion response for user %d", user_id)
                return current
        except Exception:
            logger.exception("Emotion evaluation failed for user %d in chat %d", user_id, chat_id)
            return current

        deltas = data.get("deltas", {})
        reasoning = data.get("reasoning", "")

        new_state = await self._ltm.update_emotional_state(user_id, chat_id, deltas)

        await self._ltm.append_memory_log(
            chat_id,
            user_id,
            "emotion_change",
            json.dumps({"deltas": deltas, "reasoning": reasoning, "new_state": new_state}, ensure_ascii=False),
        )

        logger.info(
            "Emotion update for user %d in chat %d: %s (%s)",
            user_id,
            chat_id,
            deltas,
            reasoning,
        )
        return new_state

    async def get_state(self, user_id: int, chat_id: int) -> dict:
        return await self._ltm.get_emotional_state(user_id, chat_id)
