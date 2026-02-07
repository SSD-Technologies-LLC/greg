# CLAUDE.md — Instructions for AI Assistants

## Project
Greg (Гриша) — Telegram group chat bot. See `context.md` for full architecture.

## Commands
- **Run tests:** `python -m pytest tests/ -v`
- **Run bot:** `docker compose up -d` (requires `.env` with `TELEGRAM_BOT_TOKEN`, `ANTHROPIC_API_KEY`, `GREG_BOT_USERNAME`)
- **Logs:** `docker compose logs -f greg`

## Code Style
- Python 3.12, type hints everywhere (`str | None` not `Optional[str]`)
- Async throughout — `async def`, `await`, `AsyncMock` in tests
- Imports: stdlib → third-party → `config.*` → `src.*`
- No docstrings on simple methods; docstrings only where behavior is non-obvious
- Russian in user-facing strings, personality prompts, and display labels
- Keep responses short — Greg speaks in 1-3 sentences

## Testing Patterns
- **asyncio_mode = "auto"** in pyproject.toml — no need for `@pytest.mark.asyncio` on test functions (but existing tests use it, so keep consistent)
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
- **Environment:** `tests/conftest.py` sets `TELEGRAM_BOT_TOKEN`, `ANTHROPIC_API_KEY`, `GREG_BOT_USERNAME` as env vars before any source imports
- **No real external services** — all tests are fully mocked

## Architecture Gotchas
- `config/settings.py` has `settings = Settings()` at module level — importing anything that touches `config.settings` will fail without env vars. Tests handle this via conftest.
- `asyncpg pool.acquire()` returns a **sync** context manager wrapping an async connection — mock with `MagicMock` not `AsyncMock`
- `Distiller.distill()` must be called with `chat_id=chat_id` (keyword argument) — tests assert this
- Emotional state in DB may be a `dict` or a JSON `str` — always handle both when reading
- `PersonalityEngine.build_messages()` deduplicates if last recent message is from the current user
- Media messages: only the **current** message gets image content blocks; history messages in Redis are text-only with labels like `[Фото]`, `[Видео]`

## File Map (what to edit for common tasks)
- **Add new message type:** `src/bot/handlers.py` (`_extract_media`), guard clause on line 45
- **Change response behavior:** `src/brain/decision.py` (scoring), `config/topics.py` (keywords)
- **Change Greg's personality:** `config/personality.py` (`BASE_PERSONALITY`, `TONE_MODIFIERS`)
- **Change emotion evaluation:** `src/brain/emotions.py` (`EMOTION_PROMPT`)
- **Change Claude model:** grep for `claude-haiku-4-5-20251001` in `responder.py`, `distiller.py`, `emotions.py`
- **Add DB fields:** `migrations/` + `src/memory/long_term.py`
- **Add tests:** `tests/` — follow existing mock patterns, use class-based grouping
