# CLAUDE.md — Instructions for AI Assistants

## Project
Greg (Гриша) — Telegram group chat bot. See `context.md` for full architecture.

## Commands
- **Run tests:** `python -m pytest tests/ -v`
- **Lint:** `ruff check .`
- **Format check:** `ruff format --check .`
- **Auto-format:** `ruff format .`
- **Type check:** `mypy src/ config/ --ignore-missing-imports`
- **Run bot:** `docker compose up -d` (requires `.env`)
- **Logs:** `docker compose logs -f greg`

## Code Style
- Python 3.12, type hints everywhere (`str | None` not `Optional[str]`)
- Async throughout — `async def`, `await`, `AsyncMock` in tests
- Imports: stdlib → third-party → `config.*` → `src.*` (enforced by ruff `I` rules)
- Line length: 120 (configured in `pyproject.toml`)
- No docstrings on simple methods; docstrings only where behavior is non-obvious
- Russian in user-facing strings, personality prompts, and display labels
- Keep responses short — Greg speaks in 1-3 sentences

## Rules

### Models
- **Never hardcode model strings.** Use `settings.greg_response_model` (Opus 4.6, for responses) and `settings.greg_decision_model` (Haiku 4.5, for decisions/distillation/emotions).
- When adding a new Claude API call, pick the appropriate model setting based on the call's purpose.

### JSON Parsing
- **Never use raw `json.loads()` on model output.** Always use `safe_parse_json()` from `src/utils/json_parser.py` — it handles markdown code blocks, BOM, and embedded JSON.
- Add retry logic when parsing model output in critical paths (see `distiller.py` `_call_with_retry()` as pattern).

### Decision Engine
- `DecisionEngine.evaluate()` returns a `DecisionResult` dataclass — never use the old `calculate_score()`/`should_respond()` API (removed).
- Direct triggers (name mentions, replies, @mentions) bypass the Haiku API call entirely.
- Private chats bypass the decision engine entirely — always respond.

### Output Safety
- All Claude responses must pass through `sanitize_response()` before sending.
- `sender.py` splits on regex `_SEPARATOR_RE` — never use exact string split on `\n---\n`.
- Greg's personality prompt explicitly forbids reproducing `[username]:` format markers and `---` separators. If leaks appear, fix the anti-leak instructions in `config/personality.py`.

### Search
- Tavily search is synchronous — always call via `asyncio.to_thread()` to avoid blocking the event loop.
- Search is optional — code must work when `TAVILY_API_KEY` is unset. `WebSearcher(tavily_client=None)` returns `None` for all queries.

### Voice Transcription
- Voice transcription uses OpenAI Whisper API (`whisper-1`, `language="ru"`) via `VoiceTranscriber`.
- Transcription is optional — code must work when `OPENAI_API_KEY` is unset. `VoiceTranscriber(client=None)` returns `None`.
- Transcription is internal only — users see Greg's natural response, not the transcript.
- Follows the same optional-dependency pattern as Tavily search.

### Personality
- Greg has no topic refusals. He discusses everything. Don't add content filters.
- Tone modifiers are driven by emotional state, not hardcoded triggers.
- `TONE_MODIFIERS` keys: `annoyance_high`, `bored`, `trolling`, `warmth_high`, `trust_high`, `respect_high`, `interest_high`, `loyalty_high`.

### Error Handling
- Claude API failures must never crash the bot. Return `None` from responder, `False` from distiller, current state from emotion tracker.
- Log exceptions with `logger.exception()` — don't silently swallow errors.
- Background tasks (distillation, emotion updates) run in `asyncio.create_task()` with try/except wrappers.

## Testing Patterns
- **asyncio_mode = "auto"** in pyproject.toml — existing tests use `@pytest.mark.asyncio` for consistency
- **Mock Redis:** `AsyncMock()` with explicit method mocks (`llen`, `rpush`, `lrange`, etc.)
- **Mock PostgreSQL:** `MagicMock()` pool with `AsyncMock` conn inside sync context manager:
  ```python
  pool = MagicMock()
  conn = AsyncMock()
  ctx = MagicMock()
  ctx.__aenter__ = AsyncMock(return_value=conn)
  ctx.__aexit__ = AsyncMock(return_value=False)
  pool.acquire.return_value = ctx
  ```
- **Mock Anthropic:** `AsyncMock()` client, mock `response.content[0].text`
- **Mock decision engine:** `decision.evaluate = AsyncMock(return_value=DecisionResult(...))` — not the old `calculate_score`
- **Handler deps fixture:** 9-tuple: `(sender, decision, responder, emotions, ctx_builder, stm, distiller, searcher, transcriber)`
- **Environment:** `tests/conftest.py` sets `TELEGRAM_BOT_TOKEN`, `ANTHROPIC_API_KEY`, `GREG_BOT_USERNAME`, `GREG_RESPONSE_MODEL`, `GREG_DECISION_MODEL` before any source imports
- **No real external services** — all tests are fully mocked
- Use class-based test grouping (`class TestFeatureName:`)

## Architecture Gotchas
- `config/settings.py` has `settings = Settings()` at module level — importing anything that touches `config.settings` will fail without env vars. Tests handle this via conftest.
- `asyncpg pool.acquire()` returns a **sync** context manager wrapping an async connection — mock with `MagicMock` not `AsyncMock`
- `Distiller.distill()` must be called with `chat_id=chat_id` (keyword argument) — tests assert this
- Emotional state in DB may be a `dict` or a JSON `str` — always handle both when reading
- `PersonalityEngine.build_messages()` deduplicates if last recent message is from the current user
- Media messages: only the **current** message gets image content blocks; history messages in Redis are text-only with labels like `[Фото]`, `[Видео]`
- `config/topics.py` only contains `GREG_NAMES` — keyword lists (`INTERESTS`, `HOT_TAKES`, `SENTIMENT_KEYWORDS`) were removed in v2

## File Map (what to edit for common tasks)
- **Add new message type:** `src/bot/handlers.py` (`_extract_media`)
- **Change when Greg responds:** `src/brain/decision.py` (`DECISION_PROMPT`, `evaluate`)
- **Change Greg's personality:** `config/personality.py` (`BASE_PERSONALITY`, `TONE_MODIFIERS`)
- **Change Greg's name aliases:** `config/topics.py` (`GREG_NAMES`)
- **Change emotion evaluation:** `src/brain/emotions.py` (`EMOTION_PROMPT`)
- **Change response/decision model:** `config/settings.py` (`greg_response_model`, `greg_decision_model`) or env vars
- **Change search behavior:** `src/brain/searcher.py`, `src/bot/handlers.py` (search wiring)
- **Change voice transcription:** `src/brain/transcriber.py`, `src/bot/handlers.py` (`_transcribe_voice`)
- **Fix output formatting:** `src/brain/responder.py` (`sanitize_response`), `src/bot/sender.py` (`_SEPARATOR_RE`)
- **Add DB fields:** `migrations/` + `src/memory/long_term.py`
- **Add tests:** `tests/` — follow existing mock patterns, use class-based grouping
