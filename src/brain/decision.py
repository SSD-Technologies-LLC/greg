import logging
import random
import time
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from config.topics import HOT_TAKES, INTERESTS, SENTIMENT_KEYWORDS

logger = logging.getLogger(__name__)


class DecisionEngine:
    def __init__(
        self,
        bot_username: str,
        response_threshold: float = 0.4,
        random_factor: float = 0.07,
        cooldown_messages: int = 3,
        max_unprompted_per_hour: int = 5,
        night_start: int = 1,
        night_end: int = 8,
        timezone: str = "Europe/Moscow",
    ) -> None:
        self._bot_username = bot_username.lower()
        self._threshold = response_threshold
        self._random_factor = random_factor
        self._cooldown_messages = cooldown_messages
        self._max_unprompted = max_unprompted_per_hour
        self._night_start = night_start
        self._night_end = night_end
        self._tz = ZoneInfo(timezone)
        self._unprompted_log: dict[int, list[float]] = defaultdict(list)

    async def calculate_score(
        self,
        chat_id: int,
        text: str,
        is_reply_to_bot: bool,
        recent_messages: list[dict],
    ) -> float:
        text_lower = text.lower()

        if is_reply_to_bot or f"@{self._bot_username}" in text_lower or self._bot_username in text_lower:
            return 1.0

        score = 0.0
        score += self._score_topics(text_lower)
        score += self._score_sentiment(text_lower)
        score += self._score_momentum(recent_messages)
        score += random.uniform(0, self._random_factor)
        score -= self._cooldown_penalty(recent_messages)
        score -= self._night_penalty()

        return max(0.0, score)

    def should_respond(self, score: float, chat_id: int, is_direct: bool) -> bool:
        if is_direct:
            return True
        if score < self._threshold:
            return False
        if not self._check_rate_limit(chat_id):
            logger.info("Rate limit hit for chat %d", chat_id)
            return False
        return True

    def record_response(self, chat_id: int, is_direct: bool) -> None:
        if not is_direct:
            self._unprompted_log[chat_id].append(time.time())

    def _score_topics(self, text: str) -> float:
        for topic in HOT_TAKES:
            if topic in text:
                return 0.5
        for topic in INTERESTS:
            if topic in text:
                return 0.3
        return 0.0

    def _score_sentiment(self, text: str) -> float:
        for category, keywords in SENTIMENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    if category == "vulnerable":
                        return 0.4
                    if category in ("negative_strong", "conflict"):
                        return 0.3
                    if category == "positive_strong":
                        return 0.2
                    return 0.1
        return 0.0

    def _score_momentum(self, recent_messages: list[dict]) -> float:
        if len(recent_messages) < 10:
            return 0.0
        bot_in_recent = any(
            m.get("username", "").lower() == self._bot_username
            for m in recent_messages[-10:]
        )
        if not bot_in_recent:
            return 0.2
        return 0.0

    def _cooldown_penalty(self, recent_messages: list[dict]) -> float:
        if not recent_messages:
            return 0.0
        last_n = recent_messages[-self._cooldown_messages:]
        bot_spoke = any(
            m.get("username", "").lower() == self._bot_username for m in last_n
        )
        return 0.3 if bot_spoke else 0.0

    def _night_penalty(self) -> float:
        now = datetime.now(self._tz)
        if self._night_start <= now.hour < self._night_end:
            return 0.2
        return 0.0

    def _check_rate_limit(self, chat_id: int) -> bool:
        now = time.time()
        hour_ago = now - 3600
        self._unprompted_log[chat_id] = [
            t for t in self._unprompted_log[chat_id] if t > hour_ago
        ]
        return len(self._unprompted_log[chat_id]) < self._max_unprompted
