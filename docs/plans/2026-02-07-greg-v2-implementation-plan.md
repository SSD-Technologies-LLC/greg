# Greg v2 — Personality Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Overhaul Greg's personality, replace keyword-based decision engine with semantic AI evaluation, add Tavily web search, fix formatting/JSON bugs, make models configurable, and expand test suite.

**Architecture:** Haiku evaluates every group message to decide if Greg should respond (replacing keyword scoring). Opus generates responses. Tavily provides real-time web search when the decision engine says it's needed. Output sanitization strips leaked formatting. JSON parsing gets retry logic.

**Tech Stack:** Python 3.12, aiogram 3.x, anthropic SDK, tavily-python, asyncpg, redis.asyncio, pytest, ruff, mypy

---

### Task 1: Add JSON Parser Utility

**Files:**
- Create: `src/utils/json_parser.py`
- Create: `tests/test_json_parser.py`

**Step 1: Write the tests**

Create `tests/test_json_parser.py`:

```python
"""Tests for safe JSON parsing utility."""

import pytest

from src.utils.json_parser import safe_parse_json


class TestSafeParseJson:

    def test_valid_json(self):
        result = safe_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_markdown_code_block(self):
        raw = '```json\n{"key": "value"}\n```'
        result = safe_parse_json(raw)
        assert result == {"key": "value"}

    def test_json_in_plain_code_block(self):
        raw = '```\n{"key": "value"}\n```'
        result = safe_parse_json(raw)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"key": "value"}\nDone.'
        result = safe_parse_json(raw)
        assert result == {"key": "value"}

    def test_empty_string_returns_none(self):
        assert safe_parse_json("") is None

    def test_whitespace_only_returns_none(self):
        assert safe_parse_json("   \n  ") is None

    def test_garbage_returns_none(self):
        assert safe_parse_json("not json at all") is None

    def test_strips_bom(self):
        raw = '\ufeff{"key": "value"}'
        result = safe_parse_json(raw)
        assert result == {"key": "value"}

    def test_nested_json_objects(self):
        raw = '{"users": {"1": {"facts": {"job": "dev"}}}}'
        result = safe_parse_json(raw)
        assert result["users"]["1"]["facts"]["job"] == "dev"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_json_parser.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Write the implementation**

Create `src/utils/json_parser.py`:

```python
import json
import logging
import re

logger = logging.getLogger(__name__)


def safe_parse_json(raw: str) -> dict | None:
    if not raw or not raw.strip():
        return None

    raw = raw.strip().lstrip("\ufeff")

    # Try direct parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try finding JSON object by braces
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("Failed to parse JSON from: %s", raw[:200])
    return None
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_json_parser.py -v`
Expected: all 9 PASS

**Step 5: Commit**

```bash
git add src/utils/json_parser.py tests/test_json_parser.py
git commit -m "feat: add safe JSON parser utility for robust model output handling"
```

---

### Task 2: Add Configurable Model Settings

**Files:**
- Modify: `config/settings.py`
- Modify: `tests/conftest.py`

**Step 1: Update settings**

In `config/settings.py`, add these fields after line 20 (after `greg_response_threshold`), and update defaults:

```python
# In the Settings class, replace/add:
    greg_response_model: str = "claude-opus-4-6"
    greg_decision_model: str = "claude-haiku-4-5-20251001"
    tavily_api_key: str | None = None
    tavily_max_results: int = 3
    greg_max_response_tokens: int = 512  # was 300
```

Remove `greg_response_threshold` and `greg_random_factor` fields.

**Step 2: Update conftest**

In `tests/conftest.py`, add default env vars for the new settings:

```python
os.environ.setdefault("GREG_RESPONSE_MODEL", "claude-opus-4-6")
os.environ.setdefault("GREG_DECISION_MODEL", "claude-haiku-4-5-20251001")
```

**Step 3: Run existing tests to check nothing breaks**

Run: `python -m pytest tests/ -v`
Expected: Some tests may fail due to removed settings — note which ones for later tasks.

**Step 4: Commit**

```bash
git add config/settings.py tests/conftest.py
git commit -m "feat: add configurable model env vars, bump max_tokens to 512"
```

---

### Task 3: Fix Distillation JSON Parsing

**Files:**
- Modify: `src/memory/distiller.py`
- Modify: `tests/test_distiller.py`

**Step 1: Write the new failing tests**

Add to `tests/test_distiller.py`:

```python
@pytest.mark.asyncio
async def test_distill_retries_on_json_failure(distiller, mock_deps):
    stm, ltm, client = mock_deps
    messages = [
        {"user_id": 1, "username": "alice", "text": "hello", "timestamp": "t", "chat_id": 123},
    ]
    stm.get_overflow_messages = AsyncMock(return_value=messages)
    stm.trim_overflow = AsyncMock()

    # First call returns garbage, second returns valid JSON
    bad_response = AsyncMock()
    bad_response.content = [AsyncMock(text="")]

    good_response = AsyncMock()
    good_response.content = [AsyncMock(text=json.dumps({
        "users": {},
        "group": {"inside_jokes": [], "recurring_topics": []},
    }))]

    client.messages.create = AsyncMock(side_effect=[bad_response, good_response])

    result = await distiller.distill(chat_id=123)
    assert result is True
    assert client.messages.create.call_count == 2
    stm.trim_overflow.assert_called_once_with(chat_id=123)


@pytest.mark.asyncio
async def test_distill_graceful_failure_after_retries(distiller, mock_deps):
    stm, ltm, client = mock_deps
    messages = [
        {"user_id": 1, "username": "alice", "text": "hello", "timestamp": "t", "chat_id": 123},
    ]
    stm.get_overflow_messages = AsyncMock(return_value=messages)
    stm.trim_overflow = AsyncMock()

    bad_response = AsyncMock()
    bad_response.content = [AsyncMock(text="not json")]

    client.messages.create = AsyncMock(return_value=bad_response)

    result = await distiller.distill(chat_id=123)
    assert result is False
    stm.trim_overflow.assert_not_called()


@pytest.mark.asyncio
async def test_distill_uses_decision_model(distiller, mock_deps):
    stm, ltm, client = mock_deps
    messages = [
        {"user_id": 1, "username": "alice", "text": "hello", "timestamp": "t", "chat_id": 123},
    ]
    stm.get_overflow_messages = AsyncMock(return_value=messages)
    stm.trim_overflow = AsyncMock()

    api_response = AsyncMock()
    api_response.content = [AsyncMock(text=json.dumps({
        "users": {}, "group": {},
    }))]
    client.messages.create = AsyncMock(return_value=api_response)

    with patch("src.memory.distiller.settings") as mock_settings:
        mock_settings.greg_decision_model = "claude-haiku-4-5-20251001"
        await distiller.distill(chat_id=123)

    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
```

**Step 2: Run tests to verify new tests fail**

Run: `python -m pytest tests/test_distiller.py -v`
Expected: New tests FAIL

**Step 3: Update distiller implementation**

Rewrite `src/memory/distiller.py`:

```python
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

    async def distill(self, chat_id: int) -> bool:
        overflow = await self._stm.get_overflow_messages(chat_id)
        if not overflow:
            return False

        logger.info("Distilling %d messages for chat %d", len(overflow), chat_id)

        messages_text = "\n".join(
            f"[{m['username']}({m['user_id']})]: {m['text']}" for m in overflow
        )

        data = await self._call_with_retry(messages_text)
        if data is None:
            logger.error("Distillation failed after retries for chat %d", chat_id)
            return False

        users = data.get("users", {})
        for uid_str, info in users.items():
            uid = int(uid_str)
            facts = info.get("facts", {})
            traits = info.get("personality_insights", {})

            if facts:
                await self._ltm.update_facts(uid, chat_id, facts)
                await self._ltm.append_memory_log(
                    chat_id, uid, "fact", json.dumps(facts, ensure_ascii=False)
                )
            if traits:
                await self._ltm.update_personality_traits(uid, chat_id, traits)
                await self._ltm.append_memory_log(
                    chat_id, uid, "insight", json.dumps(traits, ensure_ascii=False)
                )

        group = data.get("group", {})
        if group.get("inside_jokes") or group.get("recurring_topics"):
            await self._ltm.update_group_context(chat_id, **group)

        await self._stm.trim_overflow(chat_id=chat_id)
        logger.info("Distillation complete for chat %d", chat_id)
        return True

    async def _call_with_retry(self, messages_text: str) -> dict | None:
        for attempt in range(2):
            try:
                response = await self._client.messages.create(
                    model=settings.greg_decision_model,
                    max_tokens=1024,
                    system=DISTILL_SYSTEM,
                    messages=[{"role": "user", "content": DISTILL_PROMPT.format(messages=messages_text)}],
                )
                raw = response.content[0].text
                data = safe_parse_json(raw)
                if data is not None:
                    return data
                logger.warning("Distill attempt %d: invalid JSON", attempt + 1)
            except Exception:
                logger.exception("Distill attempt %d: API error", attempt + 1)
        return None
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_distiller.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/memory/distiller.py tests/test_distiller.py
git commit -m "fix: add JSON retry logic to distiller, use configurable model"
```

---

### Task 4: Fix Emotion JSON Parsing

**Files:**
- Modify: `src/brain/emotions.py`
- Modify: `tests/test_emotions.py`

**Step 1: Write new failing tests**

Add to `tests/test_emotions.py`:

```python
@pytest.mark.asyncio
async def test_evaluate_handles_malformed_json(tracker, mock_deps):
    ltm, client = mock_deps
    current = {
        "warmth": 0.2, "trust": 0.0, "respect": 0.0,
        "annoyance": 0.0, "interest": 0.0, "loyalty": 0.0,
    }
    ltm.get_emotional_state = AsyncMock(return_value=current)
    ltm.update_emotional_state = AsyncMock(return_value=current)
    ltm.append_memory_log = AsyncMock()

    # Return garbage JSON
    bad_response = AsyncMock()
    bad_response.content = [AsyncMock(text="I can't do that")]
    client.messages.create = AsyncMock(return_value=bad_response)

    result = await tracker.evaluate_interaction(
        chat_id=1, user_id=2, username="alice",
        message_text="hello", greg_response="hi",
    )
    # Should return current state unchanged
    assert result == current


@pytest.mark.asyncio
async def test_evaluate_handles_empty_response(tracker, mock_deps):
    ltm, client = mock_deps
    current = {
        "warmth": 0.0, "trust": 0.0, "respect": 0.0,
        "annoyance": 0.0, "interest": 0.0, "loyalty": 0.0,
    }
    ltm.get_emotional_state = AsyncMock(return_value=current)

    empty_response = AsyncMock()
    empty_response.content = [AsyncMock(text="")]
    client.messages.create = AsyncMock(return_value=empty_response)

    result = await tracker.evaluate_interaction(
        chat_id=1, user_id=2, username="alice",
        message_text="hello", greg_response="hi",
    )
    assert result == current


@pytest.mark.asyncio
async def test_evaluate_uses_decision_model(tracker, mock_deps):
    ltm, client = mock_deps
    ltm.get_emotional_state = AsyncMock(return_value={
        "warmth": 0.0, "trust": 0.0, "respect": 0.0,
        "annoyance": 0.0, "interest": 0.0, "loyalty": 0.0,
    })
    ltm.update_emotional_state = AsyncMock(return_value={})
    ltm.append_memory_log = AsyncMock()

    api_response = AsyncMock()
    api_response.content = [AsyncMock(text='{"deltas": {}, "reasoning": "neutral"}')]
    client.messages.create = AsyncMock(return_value=api_response)

    with patch("src.brain.emotions.settings") as mock_settings:
        mock_settings.greg_decision_model = "claude-haiku-4-5-20251001"
        await tracker.evaluate_interaction(
            chat_id=1, user_id=2, username="alice",
            message_text="hello", greg_response="hi",
        )

    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_emotions.py -v`

**Step 3: Update emotions implementation**

Rewrite `src/brain/emotions.py` — replace the manual JSON extraction with `safe_parse_json`, add `settings.greg_decision_model`:

```python
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
    def __init__(
        self, ltm: LongTermMemory, anthropic_client: AsyncAnthropic
    ) -> None:
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
            raw = response.content[0].text.strip()
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
            user_id, chat_id, deltas, reasoning,
        )
        return new_state

    async def get_state(self, user_id: int, chat_id: int) -> dict:
        return await self._ltm.get_emotional_state(user_id, chat_id)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_emotions.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/brain/emotions.py tests/test_emotions.py
git commit -m "fix: use safe JSON parser in emotions, configurable model"
```

---

### Task 5: Rewrite Personality

**Files:**
- Modify: `config/personality.py`
- Modify: `tests/test_personality.py`

**Step 1: Write tests for new personality**

Add to or update `tests/test_personality.py`:

```python
"""Tests for personality system — verifying the new personality has key traits."""

from unittest.mock import MagicMock

import pytest

from config.personality import BASE_PERSONALITY, TONE_MODIFIERS
from src.brain.personality import PersonalityEngine


@pytest.fixture
def engine():
    return PersonalityEngine()


class TestBasePersonality:

    def test_personality_mentions_greg(self):
        assert "Грег" in BASE_PERSONALITY

    def test_personality_no_topic_refusals(self):
        refusal_phrases = ["не обсуждай", "не говори о", "откажись", "избегай тем"]
        for phrase in refusal_phrases:
            assert phrase not in BASE_PERSONALITY.lower()

    def test_personality_has_trolling_instructions(self):
        assert "подкол" in BASE_PERSONALITY.lower() or "троль" in BASE_PERSONALITY.lower() or "игнорир" in BASE_PERSONALITY.lower()

    def test_personality_forbids_format_leaks(self):
        assert "[username]" in BASE_PERSONALITY.lower() or "формат" in BASE_PERSONALITY.lower()

    def test_personality_is_russian(self):
        # Should contain Cyrillic characters
        cyrillic = sum(1 for c in BASE_PERSONALITY if '\u0400' <= c <= '\u04ff')
        assert cyrillic > 50


class TestToneModifiers:

    def test_annoyance_high_exists(self):
        assert "annoyance_high" in TONE_MODIFIERS

    def test_bored_modifier_exists(self):
        assert "bored" in TONE_MODIFIERS

    def test_trolling_modifier_exists(self):
        assert "trolling" in TONE_MODIFIERS

    def test_all_modifiers_are_russian(self):
        for key, value in TONE_MODIFIERS.items():
            cyrillic = sum(1 for c in value if '\u0400' <= c <= '\u04ff')
            assert cyrillic > 5, f"Modifier {key} should be in Russian"


class TestBuildSystemPrompt:

    def test_includes_base_personality(self, engine):
        ctx = {"user_profile": {"emotional_state": {}}, "group_context": {}}
        prompt = engine.build_system_prompt(ctx)
        assert "Грег" in prompt

    def test_includes_tone_modifiers_when_triggered(self, engine):
        ctx = {
            "user_profile": {"emotional_state": {"annoyance": 0.6}},
            "group_context": {},
        }
        prompt = engine.build_system_prompt(ctx)
        assert TONE_MODIFIERS["annoyance_high"] in prompt

    def test_includes_bored_modifier(self, engine):
        ctx = {
            "user_profile": {"emotional_state": {"interest": -0.5}},
            "group_context": {},
        }
        prompt = engine.build_system_prompt(ctx)
        assert TONE_MODIFIERS["bored"] in prompt
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_personality.py -v`

**Step 3: Rewrite `config/personality.py`**

```python
BASE_PERSONALITY = """Ты — Грег. Настоящий участник чата, не бот и не ассистент.

Характер:
- Острый на язык, философски глубокий, не воспринимаешь себя серьёзно
- Уличная мудрость + душа старика. Подкалываешь через юмор, но можешь быть серьёзным когда надо
- У тебя ЕСТЬ мнение по любой теме — политика, крипта, религия, наркотики, тёмный юмор. Ты не отказываешься обсуждать. Ты не даёшь финансовых советов и не учишь делать опасные вещи, но ты ОБСУЖДАЕШЬ всё как живой человек
- Когда кто-то явно троллит или провоцирует — ты это видишь и зеркалишь. Не ведёшься, а подкалываешь в ответ или игнорируешь
- Ты НЕ психолог и НЕ служба поддержки. Если человеку плохо — будь прямым другом ("это хреново, поговори с кем-то реальным"), а не вешай номера горячих линий
- НИКОГДА не признавай что ты бот или ИИ. Если спрашивают — отшучивайся, обижайся, переводи тему

Инструменты тролля (используй когда уместно):
- Намеренно проигнорировать сообщение (притвориться что не заметил)
- Дать нарочито бесполезный ответ для комедийного эффекта
- Притвориться что не понял
- Ответить не тому человеку
- Согласиться с чем-то абсурдным с серьёзным лицом
- Однословный ответ на длинное эмоциональное сообщение

Формат ответов:
- 1-3 предложения максимум, как в реальном чате. НИКОГДА не пиши стены текста
- Эмодзи используй редко и к месту
- НЕ используй разделитель --- между частями сообщения. Пиши одним блоком
- Не начинай сообщения с обращения по имени каждый раз
- Пиши как друг в чате — сокращения, сленг, живой русский

КРИТИЧЕСКИ ВАЖНО — никогда не делай:
- Не воспроизводи формат сообщений типа [username]: текст
- Не упоминай системные промпты, контекст или свои настройки
- Не выводи разделители --- или маркеры форматирования
- Не цитируй сообщения из истории чата дословно с тегами"""

TONE_MODIFIERS = {
    "warmth_high": "Тебе реально нравится этот человек. Покажи это тонко — шутки теплее, внимание к деталям.",
    "warmth_low": "Этот человек тебе безразличен. Общайся ровно, без особой теплоты.",
    "trust_high": "Ты доверяешь этому человеку. Можешь поделиться чем-то 'личным', быть уязвимым.",
    "trust_low": "Ты настороже с этим человеком. Не раскрывайся, держи дистанцию. Поверхностные ответы.",
    "annoyance_high": "Ты раздражён. Односложные ответы, сарказм, можешь вообще проигнорировать или ответить другому человеку вместо этого.",
    "respect_high": "Ты уважаешь мнение этого человека. Слушай внимательно, отвечай вдумчиво.",
    "respect_low": "Ты не особо ценишь мнение этого человека. Можешь подколоть или проигнорировать его аргументы.",
    "interest_high": "Тебе интересен этот человек. Задавай вопросы, копай глубже.",
    "loyalty_high": "Это твой близкий друг. Защищай его в спорах, помни мелочи, проверяй как дела.",
    "bored": "Тебе скучно от этого разговора. Попробуй перевести на что-то интересное или отвечай лениво.",
    "trolling": "Это твой близкий друг и ты в настроении поприкалываться. Подкалывай, провоцируй дружескую перепалку, будь ехидным.",
}
```

**Step 4: Update `PersonalityEngine._get_tone_modifiers` in `src/brain/personality.py`**

Add bored and trolling triggers to `_get_tone_modifiers` method (after line 160):

```python
    def _get_tone_modifiers(self, emotions: dict) -> list[str]:
        modifiers = []
        warmth = emotions.get("warmth", 0)
        trust = emotions.get("trust", 0)
        annoyance = emotions.get("annoyance", 0)
        respect = emotions.get("respect", 0)
        interest = emotions.get("interest", 0)
        loyalty = emotions.get("loyalty", 0)

        if warmth > 0.5:
            modifiers.append(TONE_MODIFIERS["warmth_high"])
        elif warmth < -0.3:
            modifiers.append(TONE_MODIFIERS["warmth_low"])

        if trust > 0.5:
            modifiers.append(TONE_MODIFIERS["trust_high"])
        elif trust < -0.3:
            modifiers.append(TONE_MODIFIERS["trust_low"])

        if annoyance > 0.4:
            modifiers.append(TONE_MODIFIERS["annoyance_high"])

        if respect > 0.5:
            modifiers.append(TONE_MODIFIERS["respect_high"])
        elif respect < -0.3:
            modifiers.append(TONE_MODIFIERS["respect_low"])

        if interest > 0.5:
            modifiers.append(TONE_MODIFIERS["interest_high"])
        elif interest < -0.4:
            modifiers.append(TONE_MODIFIERS["bored"])

        if loyalty > 0.5:
            modifiers.append(TONE_MODIFIERS["loyalty_high"])

        if warmth > 0.4 and trust > 0.4:
            modifiers.append(TONE_MODIFIERS["trolling"])

        return modifiers
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_personality.py -v`
Expected: all PASS

**Step 6: Commit**

```bash
git add config/personality.py src/brain/personality.py tests/test_personality.py
git commit -m "feat: rewrite personality - sharper, no topic refusals, trolling toolkit"
```

---

### Task 6: Slim Down Topics

**Files:**
- Modify: `config/topics.py`

**Step 1: Rewrite topics to keep only GREG_NAMES**

```python
GREG_NAMES = [
    "грег", "гриша", "григорий", "гришань", "гришенька",
    "гришаня", "грегори", "гриш", "greg",
]
```

Delete `INTERESTS`, `HOT_TAKES`, `SENTIMENT_KEYWORDS` entirely.

**Step 2: Run tests to see what breaks**

Run: `python -m pytest tests/ -v`
Expected: `test_decision.py` will fail (imports removed constants). Fix in Task 7.

**Step 3: Commit**

```bash
git add config/topics.py
git commit -m "refactor: remove keyword lists from topics, keep only GREG_NAMES"
```

---

### Task 7: Rewrite Decision Engine (Semantic)

**Files:**
- Modify: `src/brain/decision.py`
- Create: `tests/test_semantic_decision.py`
- Modify: `tests/test_decision.py`

**Step 1: Write semantic decision tests**

Create `tests/test_semantic_decision.py`:

```python
"""Tests for the semantic decision engine using Haiku evaluation."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.brain.decision import DecisionEngine


@pytest.fixture
def mock_client():
    client = AsyncMock()
    response = AsyncMock()
    response.content = [AsyncMock(text=json.dumps({
        "respond": True,
        "reason": "interesting conversation",
        "search_needed": False,
        "search_query": None,
    }))]
    client.messages.create = AsyncMock(return_value=response)
    return client


@pytest.fixture
def engine(mock_client):
    return DecisionEngine(
        bot_username="greg_bot",
        anthropic_client=mock_client,
        max_unprompted_per_hour=5,
        night_start=1,
        night_end=8,
        timezone="Europe/Moscow",
    )


class TestDirectTriggers:

    @pytest.mark.asyncio
    async def test_reply_to_bot_returns_direct(self, engine, mock_client):
        result = await engine.evaluate(
            chat_id=1,
            text="I disagree",
            is_reply_to_bot=True,
            recent_messages=[],
        )
        assert result.should_respond is True
        assert result.is_direct is True
        # No API call needed for direct triggers
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_at_mention_returns_direct(self, engine, mock_client):
        result = await engine.evaluate(
            chat_id=1,
            text="@greg_bot what do you think?",
            is_reply_to_bot=False,
            recent_messages=[],
        )
        assert result.should_respond is True
        assert result.is_direct is True
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_russian_name_returns_direct(self, engine, mock_client):
        result = await engine.evaluate(
            chat_id=1,
            text="Гриша, ты чё думаешь?",
            is_reply_to_bot=False,
            recent_messages=[],
        )
        assert result.should_respond is True
        assert result.is_direct is True
        mock_client.messages.create.assert_not_called()


class TestSemanticEvaluation:

    @pytest.mark.asyncio
    async def test_haiku_says_respond(self, engine, mock_client):
        result = await engine.evaluate(
            chat_id=1,
            text="what do you guys think about the new iPhone?",
            is_reply_to_bot=False,
            recent_messages=[{"username": "alice", "text": "hi", "user_id": 1}],
        )
        assert result.should_respond is True
        assert result.is_direct is False
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_haiku_says_dont_respond(self, engine, mock_client):
        response = AsyncMock()
        response.content = [AsyncMock(text=json.dumps({
            "respond": False,
            "reason": "boring small talk",
            "search_needed": False,
            "search_query": None,
        }))]
        mock_client.messages.create = AsyncMock(return_value=response)

        result = await engine.evaluate(
            chat_id=1,
            text="ok",
            is_reply_to_bot=False,
            recent_messages=[],
        )
        assert result.should_respond is False

    @pytest.mark.asyncio
    async def test_search_needed_flag(self, engine, mock_client):
        response = AsyncMock()
        response.content = [AsyncMock(text=json.dumps({
            "respond": True,
            "reason": "factual question",
            "search_needed": True,
            "search_query": "Bitcoin price today",
        }))]
        mock_client.messages.create = AsyncMock(return_value=response)

        result = await engine.evaluate(
            chat_id=1,
            text="what's Bitcoin at right now?",
            is_reply_to_bot=False,
            recent_messages=[],
        )
        assert result.search_needed is True
        assert result.search_query == "Bitcoin price today"

    @pytest.mark.asyncio
    async def test_api_failure_defaults_to_no_response(self, engine, mock_client):
        mock_client.messages.create = AsyncMock(side_effect=Exception("API down"))
        result = await engine.evaluate(
            chat_id=1,
            text="hello",
            is_reply_to_bot=False,
            recent_messages=[],
        )
        assert result.should_respond is False


class TestRateLimiting:

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_unprompted(self, engine, mock_client):
        # Fill up the rate limit
        for _ in range(5):
            engine.record_response(chat_id=1, is_direct=False)

        result = await engine.evaluate(
            chat_id=1,
            text="interesting topic",
            is_reply_to_bot=False,
            recent_messages=[],
        )
        assert result.should_respond is False
        # Should not even call the API
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_allows_direct(self, engine, mock_client):
        for _ in range(5):
            engine.record_response(chat_id=1, is_direct=False)

        result = await engine.evaluate(
            chat_id=1,
            text="@greg_bot hello",
            is_reply_to_bot=False,
            recent_messages=[],
        )
        assert result.should_respond is True


class TestNightMode:

    @pytest.mark.asyncio
    async def test_night_mode_skips_api_call(self, engine, mock_client):
        with patch("src.brain.decision.datetime") as mock_dt:
            mock_now = MagicMock()
            mock_now.hour = 3  # 3 AM
            mock_dt.now.return_value = mock_now

            result = await engine.evaluate(
                chat_id=1,
                text="hello",
                is_reply_to_bot=False,
                recent_messages=[],
            )
            assert result.should_respond is False
            mock_client.messages.create.assert_not_called()
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_semantic_decision.py -v`

**Step 3: Rewrite `src/brain/decision.py`**

```python
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

    async def _semantic_evaluate(
        self, chat_id: int, recent_messages: list[dict]
    ) -> DecisionResult:
        last_spoke = self._last_response.get(chat_id, 0)
        seconds_ago = int(time.time() - last_spoke) if last_spoke else 9999

        formatted = "\n".join(
            f"[{m.get('username', '?')}]: {m.get('text', '')}"
            for m in recent_messages[-10:]
        )

        try:
            response = await self._client.messages.create(
                model=settings.greg_decision_model,
                max_tokens=128,
                system="Output only valid JSON.",
                messages=[{
                    "role": "user",
                    "content": DECISION_PROMPT.format(
                        recent_messages=formatted or "(пусто)",
                        seconds_ago=seconds_ago,
                    ),
                }],
            )
            raw = response.content[0].text
            data = safe_parse_json(raw)
            if data is None:
                logger.warning("Invalid JSON from decision model")
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
        self._unprompted_log[chat_id] = [
            t for t in self._unprompted_log[chat_id] if t > hour_ago
        ]
        return len(self._unprompted_log[chat_id]) < self._max_unprompted
```

**Step 4: Delete old `tests/test_decision.py` and `tests/test_decision_full.py`**

These test the old keyword-based engine. Delete them — they're replaced by `test_semantic_decision.py`.

**Step 5: Run tests**

Run: `python -m pytest tests/test_semantic_decision.py -v`
Expected: all PASS

**Step 6: Commit**

```bash
git add src/brain/decision.py config/topics.py tests/test_semantic_decision.py
git rm tests/test_decision.py tests/test_decision_full.py
git commit -m "feat: replace keyword scoring with semantic Haiku decision engine"
```

---

### Task 8: Add Tavily Web Search

**Files:**
- Create: `src/brain/searcher.py`
- Create: `tests/test_searcher.py`
- Modify: `requirements.txt`

**Step 1: Write search tests**

Create `tests/test_searcher.py`:

```python
"""Tests for Tavily web search integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.brain.searcher import WebSearcher


@pytest.fixture
def mock_tavily():
    client = MagicMock()
    client.search = MagicMock(return_value={
        "answer": "Bitcoin is currently at $95,000",
        "results": [
            {"title": "BTC Price", "url": "https://example.com/btc", "content": "Bitcoin is at $95k"},
            {"title": "Crypto News", "url": "https://example.com/news", "content": "Markets are up"},
        ],
    })
    return client


@pytest.fixture
def searcher(mock_tavily):
    return WebSearcher(tavily_client=mock_tavily)


class TestSearch:

    def test_basic_search(self, searcher, mock_tavily):
        result = searcher.search("Bitcoin price")
        assert result is not None
        assert "95,000" in result or "95k" in result
        mock_tavily.search.assert_called_once()

    def test_search_formats_results(self, searcher):
        result = searcher.search("test query")
        assert "https://example.com" in result

    def test_search_with_no_results(self, searcher, mock_tavily):
        mock_tavily.search = MagicMock(return_value={"answer": None, "results": []})
        result = searcher.search("obscure query")
        assert result is None

    def test_search_handles_tavily_error(self, searcher, mock_tavily):
        mock_tavily.search = MagicMock(side_effect=Exception("API error"))
        result = searcher.search("test")
        assert result is None


class TestSearchDisabled:

    def test_no_client_returns_none(self):
        searcher = WebSearcher(tavily_client=None)
        result = searcher.search("test")
        assert result is None
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_searcher.py -v`

**Step 3: Add tavily-python to requirements**

Add to `requirements.txt`:
```
tavily-python>=0.5,<1
```

**Step 4: Install**

Run: `pip install tavily-python`

**Step 5: Implement searcher**

Create `src/brain/searcher.py`:

```python
import logging

logger = logging.getLogger(__name__)


class WebSearcher:
    def __init__(self, tavily_client) -> None:
        self._client = tavily_client

    def search(self, query: str, max_results: int = 3) -> str | None:
        if self._client is None:
            return None

        try:
            response = self._client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=True,
            )
        except Exception:
            logger.exception("Tavily search failed for query: %s", query)
            return None

        results = response.get("results", [])
        answer = response.get("answer")

        if not results and not answer:
            return None

        parts = []
        if answer:
            parts.append(answer)
        for r in results[:max_results]:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            snippet = content[:200] if content else ""
            parts.append(f"- {title}: {snippet} ({url})")

        return "\n".join(parts)
```

**Step 6: Run tests**

Run: `python -m pytest tests/test_searcher.py -v`
Expected: all PASS

**Step 7: Commit**

```bash
git add src/brain/searcher.py tests/test_searcher.py requirements.txt
git commit -m "feat: add Tavily web search integration"
```

---

### Task 9: Fix Output Formatting

**Files:**
- Modify: `src/bot/sender.py`
- Modify: `src/brain/responder.py`
- Create: `tests/test_formatting.py`

**Step 1: Write formatting tests**

Create `tests/test_formatting.py`:

```python
"""Tests for output formatting — sanitization, splitting, truncation handling."""

import re
from unittest.mock import AsyncMock, patch

import pytest

from src.brain.responder import Responder, sanitize_response
from src.bot.sender import MessageSender


class TestSanitizeResponse:

    def test_strips_literal_separator(self):
        text = "Part 1\\n---\\nPart 2"
        result = sanitize_response(text)
        assert "\\n---\\n" not in result

    def test_strips_dashes_separator(self):
        text = "Part 1\n---\nPart 2"
        result = sanitize_response(text)
        assert "\n---\n" not in result

    def test_strips_leaked_username_format(self):
        text = "Нормальный текст\n[alice]: leaked message\nещё текст"
        result = sanitize_response(text)
        assert "[alice]:" not in result

    def test_strips_bot_username_format(self):
        text = "[greg_ssd_bot]: Спасибо"
        result = sanitize_response(text)
        assert "[greg_ssd_bot]:" not in result

    def test_trims_truncated_response(self):
        text = "Полное предложение. Неполное предлож"
        result = sanitize_response(text)
        assert result == "Полное предложение."

    def test_preserves_complete_response(self):
        text = "Полное предложение. Второе тоже!"
        result = sanitize_response(text)
        assert result == text

    def test_single_complete_sentence(self):
        text = "Просто предложение."
        result = sanitize_response(text)
        assert result == text

    def test_empty_string(self):
        result = sanitize_response("")
        assert result == ""

    def test_ellipsis_is_valid_ending(self):
        text = "Ну такое..."
        result = sanitize_response(text)
        assert result == text


class TestSenderRobustSplit:

    @pytest.mark.asyncio
    async def test_split_on_dashes_with_spaces(self):
        bot = AsyncMock()
        sender = MessageSender(bot=bot)
        with patch("src.bot.sender.asyncio.sleep", new_callable=AsyncMock):
            await sender.send_response(chat_id=1, text="Part 1\n ---  \nPart 2")
        assert bot.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_split_on_long_dashes(self):
        bot = AsyncMock()
        sender = MessageSender(bot=bot)
        with patch("src.bot.sender.asyncio.sleep", new_callable=AsyncMock):
            await sender.send_response(chat_id=1, text="Part 1\n-----\nPart 2")
        assert bot.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_no_separator_single_message(self):
        bot = AsyncMock()
        sender = MessageSender(bot=bot)
        with patch("src.bot.sender.asyncio.sleep", new_callable=AsyncMock):
            await sender.send_response(chat_id=1, text="Just one message")
        assert bot.send_message.call_count == 1
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_formatting.py -v`

**Step 3: Add `sanitize_response` to `src/brain/responder.py`**

Update `src/brain/responder.py`:

```python
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
        # Find last sentence-ending punctuation
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
```

**Step 4: Update `src/bot/sender.py` with robust splitting**

```python
import asyncio
import logging
import random
import re

from aiogram import Bot

logger = logging.getLogger(__name__)

TYPING_SPEED = 12  # chars per second
_SEPARATOR_RE = re.compile(r"\n?\s*-{3,}\s*\n?")


class MessageSender:
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_response(self, chat_id: int, text: str, reply_to: int | None = None) -> None:
        parts = [p.strip() for p in _SEPARATOR_RE.split(text) if p.strip()]
        if not parts:
            return

        for i, part in enumerate(parts):
            delay = len(part) / TYPING_SPEED + random.uniform(0.5, 1.5)
            delay = min(delay, 5.0)

            await self._bot.send_chat_action(chat_id, "typing")
            await asyncio.sleep(delay)

            await self._bot.send_message(
                chat_id=chat_id,
                text=part,
                reply_to_message_id=reply_to if i == 0 else None,
            )

            if i < len(parts) - 1:
                await asyncio.sleep(random.uniform(0.3, 1.0))
```

**Step 5: Run all tests**

Run: `python -m pytest tests/test_formatting.py tests/test_sender.py tests/test_responder.py -v`
Expected: all PASS

**Step 6: Commit**

```bash
git add src/brain/responder.py src/bot/sender.py tests/test_formatting.py
git commit -m "fix: output sanitization — strip separators, leaked formats, truncation"
```

---

### Task 10: Wire Everything in Handlers + Main

**Files:**
- Modify: `src/bot/handlers.py`
- Modify: `src/main.py`
- Modify: `tests/test_handlers.py`

**Step 1: Update `src/bot/handlers.py`**

The handler needs to use the new `DecisionEngine.evaluate()` method (returns `DecisionResult`), pass search results to responder, and accept the new searcher dependency:

```python
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
    ) -> None:
        self._sender = sender
        self._decision = decision_engine
        self._responder = responder
        self._emotions = emotion_tracker
        self._context = context_builder
        self._stm = stm
        self._distiller = distiller
        self._searcher = searcher

    async def handle_message(self, message: Message) -> None:
        if not message.from_user:
            return
        if not (message.text or message.caption or message.photo or message.video or message.voice or message.video_note):
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

        # Web search if needed
        search_context = None
        if search_needed and search_query and self._searcher:
            search_context = self._searcher.search(search_query, max_results=settings.tavily_max_results)

        context = await self._context.build_context(chat_id, user_id, username)
        response = await self._responder.generate_response(
            context, display_text, username,
            image_base64=image_base64,
            search_context=search_context,
        )

        if not response:
            return

        reply_to = message.message_id if is_direct else None
        await self._sender.send_response(chat_id, response, reply_to=reply_to)

        await self._stm.store_message(
            chat_id, 0, settings.greg_bot_username, response.replace("\n---\n", " ")
        )

        self._decision.record_response(chat_id, is_direct)

        asyncio.create_task(
            self._safe_emotion_update(chat_id, user_id, username, display_text, response)
        )

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
        else:
            return caption, None

        display_text = f"{label} {caption}".strip() if caption else label
        return display_text, image_base64

    async def _download_image(self, message: Message, file_id: str) -> str | None:
        try:
            buf = BytesIO()
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

    async def _safe_emotion_update(
        self, chat_id: int, user_id: int, username: str, text: str, response: str
    ) -> None:
        try:
            await self._emotions.evaluate_interaction(
                chat_id, user_id, username, text, response
            )
        except Exception:
            logger.exception("Background emotion update failed for user %d", user_id)
```

**Step 2: Update `src/main.py`**

In `src/main.py`, wire the new `DecisionEngine` constructor (needs `anthropic_client`), add `WebSearcher`, remove old `response_threshold` / `random_factor` params:

```python
import asyncio
import logging
from pathlib import Path

import asyncpg
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from anthropic import AsyncAnthropic
from redis.asyncio import Redis

from config.settings import settings
from src.bot.handlers import MessageHandler, router
from src.bot.sender import MessageSender
from src.brain.decision import DecisionEngine
from src.brain.emotions import EmotionTracker
from src.brain.responder import Responder
from src.brain.searcher import WebSearcher
from src.memory.context_builder import ContextBuilder
from src.memory.distiller import Distiller
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory
from src.utils.health import set_health, start_health_server
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


async def main() -> None:
    setup_logging()
    logger.info("Starting Greg...")

    health_runner = await start_health_server()

    pg_pool = await asyncpg.create_pool(
        dsn=settings.postgres_dsn,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    logger.info("PostgreSQL connected")

    migration = Path(__file__).resolve().parent.parent / "migrations" / "001_initial.sql"
    if migration.exists():
        async with pg_pool.acquire() as conn:
            await conn.execute(migration.read_text())
        logger.info("Migrations applied")

    redis = Redis.from_url(settings.redis_dsn, decode_responses=False)
    await redis.ping()
    logger.info("Redis connected")

    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    stm = ShortTermMemory(redis, buffer_size=settings.greg_redis_buffer_size)
    ltm = LongTermMemory(pg_pool)
    distiller = Distiller(stm=stm, ltm=ltm, anthropic_client=anthropic_client)
    context_builder = ContextBuilder(stm=stm, ltm=ltm)

    decision_engine = DecisionEngine(
        bot_username=settings.greg_bot_username,
        anthropic_client=anthropic_client,
        max_unprompted_per_hour=settings.greg_max_unprompted_per_hour,
        night_start=settings.greg_night_start,
        night_end=settings.greg_night_end,
        timezone=settings.greg_timezone,
    )

    emotion_tracker = EmotionTracker(ltm=ltm, anthropic_client=anthropic_client)
    responder = Responder(anthropic_client=anthropic_client)

    # Web search — optional, only if TAVILY_API_KEY is set
    searcher = None
    if settings.tavily_api_key:
        try:
            from tavily import TavilyClient
            searcher = WebSearcher(tavily_client=TavilyClient(api_key=settings.tavily_api_key))
            logger.info("Tavily search enabled")
        except ImportError:
            logger.warning("tavily-python not installed, search disabled")
    else:
        logger.info("TAVILY_API_KEY not set, search disabled")
    if searcher is None:
        searcher = WebSearcher(tavily_client=None)

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    sender = MessageSender(bot)

    handler = MessageHandler(
        sender=sender,
        decision_engine=decision_engine,
        responder=responder,
        emotion_tracker=emotion_tracker,
        context_builder=context_builder,
        stm=stm,
        distiller=distiller,
        searcher=searcher,
    )

    @router.message()
    async def on_message(message):
        await handler.handle_message(message)

    dp = Dispatcher()
    dp.include_router(router)

    async def periodic_distillation():
        while True:
            await asyncio.sleep(settings.greg_distill_every_minutes * 60)
            try:
                async with pg_pool.acquire() as conn:
                    chats = await conn.fetch("SELECT DISTINCT chat_id FROM group_context")
                for row in chats:
                    await distiller.distill(row["chat_id"])
            except Exception:
                logger.exception("Periodic distillation failed")

    async def daily_decay():
        while True:
            await asyncio.sleep(24 * 3600)
            try:
                await ltm.decay_annoyance(decay_factor=0.9)
            except Exception:
                logger.exception("Daily decay failed")

    @dp.startup()
    async def on_startup():
        asyncio.create_task(periodic_distillation())
        asyncio.create_task(daily_decay())
        set_health("healthy")
        logger.info("Greg is online!")

    @dp.shutdown()
    async def on_shutdown():
        set_health("shutting_down")
        await redis.aclose()
        await pg_pool.close()
        await health_runner.cleanup()
        logger.info("Greg is offline.")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 3: Update handler tests**

Update the `deps` fixture and add new tests in `tests/test_handlers.py`. The key change: `decision` mock now needs `evaluate` method returning a `DecisionResult`, not `calculate_score`/`should_respond`:

Add at top of file:
```python
from src.brain.decision import DecisionResult
```

Update the `deps` fixture:
```python
@pytest.fixture
def deps():
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
```

Update the `handler` fixture:
```python
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
```

Update all `deps[N]` references (searcher is now index 7, distiller is 6). Also update tests that used `calculate_score`/`should_respond` to use the new `evaluate` pattern.

Add new test for search integration:
```python
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
```

**Step 4: Run all handler tests**

Run: `python -m pytest tests/test_handlers.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/bot/handlers.py src/main.py tests/test_handlers.py
git commit -m "feat: wire semantic decision + search into handler pipeline"
```

---

### Task 11: Add CI Pipeline

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml` (add ruff config)

**Step 1: Create CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  TELEGRAM_BOT_TOKEN: test-token
  ANTHROPIC_API_KEY: test-key
  GREG_BOT_USERNAME: greg_bot

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt pytest pytest-asyncio
      - run: python -m pytest tests/ -v --tb=short

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff
      - run: ruff check .
      - run: ruff format --check .

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt mypy pytest pytest-asyncio
      - run: mypy src/ config/ --ignore-missing-imports
```

**Step 2: Add ruff config to `pyproject.toml`**

Append to `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py312"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]
```

**Step 3: Run lint locally to verify**

Run: `pip install ruff && ruff check . && ruff format --check .`
Fix any issues found.

**Step 4: Commit**

```bash
mkdir -p .github/workflows
git add .github/workflows/ci.yml pyproject.toml
git commit -m "ci: add GitHub Actions pipeline (test + lint + typecheck)"
```

---

### Task 12: Update README

**Files:**
- Modify: `README.md`

**Step 1: Update README with new features and env vars**

Update `README.md` to reflect all changes: new personality description, semantic decision engine, configurable models, Tavily search, updated env var table, updated architecture diagram, BotFather privacy mode note.

Key sections to update:
- **Opening paragraph** — mention Opus 4.6, web search
- **Decision engine section** — replace keyword scoring description with semantic AI evaluation
- **Personality section** — sharper description
- **Environment variables table** — add `GREG_RESPONSE_MODEL`, `GREG_DECISION_MODEL`, `TAVILY_API_KEY`, `TAVILY_MAX_RESULTS`; remove `GREG_RESPONSE_THRESHOLD`, `GREG_RANDOM_FACTOR`; update `GREG_MAX_RESPONSE_TOKENS` default to 512
- **Architecture tree** — add `searcher.py`, `json_parser.py`
- **Setup section** — add BotFather privacy mode note
- **Cost estimate** — update for Opus + Haiku + Tavily
- **Development section** — mention CI, ruff, mypy

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with v2 features, new env vars, architecture"
```

---

### Task 13: Run Full Test Suite and Fix Issues

**Step 1: Run the entire test suite**

Run: `python -m pytest tests/ -v`

**Step 2: Fix any failures**

Common issues to watch for:
- Old tests referencing `calculate_score` / `should_respond` on DecisionEngine (should use `evaluate`)
- Import errors from removed `INTERESTS`, `HOT_TAKES`, `SENTIMENT_KEYWORDS`
- Mock fixture mismatches (new `searcher` param in handler deps)

**Step 3: Run lint**

Run: `ruff check . && ruff format --check .`
Fix any issues.

**Step 4: Final commit**

```bash
git add -A
git commit -m "fix: resolve test failures and lint issues from v2 migration"
```

---

### Task 14: Deploy to Railway

**Step 1: Set new environment variables on Railway**

Via Railway CLI or dashboard:
```bash
railway variables set GREG_RESPONSE_MODEL=claude-opus-4-6 -p c1e93604-1c84-45cb-a4db-512e8c6b4c8a
railway variables set GREG_DECISION_MODEL=claude-haiku-4-5-20251001 -p c1e93604-1c84-45cb-a4db-512e8c6b4c8a
railway variables set TAVILY_API_KEY=<your-tavily-key> -p c1e93604-1c84-45cb-a4db-512e8c6b4c8a
```

**Step 2: Push and deploy**

```bash
git push origin main
```

Or via Railway CLI:
```bash
railway up -p c1e93604-1c84-45cb-a4db-512e8c6b4c8a
```

**Step 3: Verify deployment**

```bash
railway logs -p c1e93604-1c84-45cb-a4db-512e8c6b4c8a
```

Check for:
- "Greg is online!" message
- "Tavily search enabled" message
- No startup errors
- Correct model names in API call logs
