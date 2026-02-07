import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from anthropic import AsyncAnthropic

from config.settings import settings
from config.topics import GREG_NAMES
from src.utils.json_parser import safe_parse_json

logger = logging.getLogger(__name__)

DECISION_PROMPT = """Ты — фильтр внимания Грега. Грег — острый на язык, с мнением по любому поводу друг в групповом чате.
Он вступает в разговор когда: тема интересная, кто-то сказал чушь, есть шутка которую можно вставить, кто-то задал вопрос на который он знает ответ, или настроение разговора требует его присутствия.
Он молчит когда: скучный смолтолк, люди общаются между собой и ему нечего добавить, или он только что говорил и нового сказать нечего.

Последние сообщения:
{recent_messages}

Грег последний раз говорил: {seconds_ago}с назад

Ответь ТОЛЬКО валидным JSON: {{"respond": bool, "reason": "одна строка", "search_needed": bool, "search_query": "запрос или null"}}"""


@dataclass
class DecisionResult:
    should_respond: bool
    is_direct: bool
    search_needed: bool = False
    search_query: str | None = None


class DecisionEngine:
    def __init__(
        self,
        bot_username: str,
        anthropic_client: AsyncAnthropic,
        max_unprompted_per_hour: int = 5,
        night_start: int = 1,
        night_end: int = 8,
        timezone: str = "Europe/Moscow",
    ) -> None:
        self._bot_username = bot_username.lower()
        self._client = anthropic_client
        self._max_unprompted = max_unprompted_per_hour
        self._night_start = night_start
        self._night_end = night_end
        self._tz = ZoneInfo(timezone)
        self._unprompted_log: dict[int, list[float]] = defaultdict(list)
        self._last_response: dict[int, float] = {}

    async def evaluate(
        self,
        chat_id: int,
        text: str,
        is_reply_to_bot: bool,
        recent_messages: list[dict],
    ) -> DecisionResult:
        text_lower = text.lower()

        # Direct triggers — no API call needed
        if is_reply_to_bot or f"@{self._bot_username}" in text_lower or self._bot_username in text_lower:
            return DecisionResult(should_respond=True, is_direct=True)
        if any(name in text_lower for name in GREG_NAMES):
            return DecisionResult(should_respond=True, is_direct=True)

        # Night mode gate — skip API call
        if self._is_night():
            return DecisionResult(should_respond=False, is_direct=False)

        # Rate limit gate — skip API call
        if not self._check_rate_limit(chat_id):
            logger.info("Rate limit hit for chat %d", chat_id)
            return DecisionResult(should_respond=False, is_direct=False)

        # Semantic evaluation via Haiku
        return await self._semantic_evaluate(chat_id, recent_messages)

    def record_response(self, chat_id: int, is_direct: bool) -> None:
        self._last_response[chat_id] = time.time()
        if not is_direct:
            self._unprompted_log[chat_id].append(time.time())

    async def _semantic_evaluate(self, chat_id: int, recent_messages: list[dict]) -> DecisionResult:
        last_spoke = self._last_response.get(chat_id, 0)
        seconds_ago = int(time.time() - last_spoke) if last_spoke else 9999

        formatted = "\n".join(f"[{m.get('username', '?')}]: {m.get('text', '')}" for m in recent_messages[-10:])

        try:
            response = await self._client.messages.create(
                model=settings.greg_decision_model,
                max_tokens=128,
                system="Output only valid JSON.",
                messages=[
                    {
                        "role": "user",
                        "content": DECISION_PROMPT.format(
                            recent_messages=formatted or "(пусто)",
                            seconds_ago=seconds_ago,
                        ),
                    }
                ],
            )
            raw = response.content[0].text
            data = safe_parse_json(raw)
            if data is None:
                logger.warning("Invalid JSON from decision model for chat %d", chat_id)
                return DecisionResult(should_respond=False, is_direct=False)

            return DecisionResult(
                should_respond=data.get("respond", False),
                is_direct=False,
                search_needed=data.get("search_needed", False),
                search_query=data.get("search_query"),
            )
        except Exception:
            logger.exception("Semantic decision failed for chat %d", chat_id)
            return DecisionResult(should_respond=False, is_direct=False)

    def _is_night(self) -> bool:
        now = datetime.now(self._tz)
        return self._night_start <= now.hour < self._night_end

    def _check_rate_limit(self, chat_id: int) -> bool:
        now = time.time()
        hour_ago = now - 3600
        self._unprompted_log[chat_id] = [t for t in self._unprompted_log[chat_id] if t > hour_ago]
        return len(self._unprompted_log[chat_id]) < self._max_unprompted
