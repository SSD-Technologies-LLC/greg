# Greg Telegram Bot — Project Context

## What is Greg?
Greg (Гриша/Григорий) is a Telegram group chat bot powered by Claude Haiku 4.5. He acts as a real chat participant — a "wise trickster" who speaks native Russian, never breaks character, and has genuine emotional relationships with users.

## Tech Stack
- **Runtime:** Python 3.12, Docker Compose (3 containers: greg, postgres, redis)
- **Bot framework:** aiogram 3.x
- **AI:** Anthropic SDK (AsyncAnthropic), model `claude-haiku-4-5-20251001`
- **Storage:** asyncpg (PostgreSQL) + redis.asyncio (Redis)
- **Config:** pydantic-settings (BaseSettings with `.env` file)

## Architecture

### Dual Memory System
- **Short-term (Redis):** 200-message circular buffer per chat. Stores JSON-encoded messages with `{user_id, username, text, timestamp, chat_id}`. TTL: 48 hours.
- **Long-term (PostgreSQL):** User profiles (per chat) with facts, personality traits, emotional state. Group context with inside jokes and recurring topics. Memory log audit trail.

### Distillation Pipeline
Every 50 messages or 30 minutes, overflow messages beyond the 200-buffer are sent to Claude Haiku which extracts facts, personality insights, inside jokes, and recurring topics. Results are merged into PostgreSQL via JSONB `||` operator.

### Decision Engine (Score-based)
Calculates a response probability score (0.0–1.0+) from:
- **Direct triggers (=1.0):** @mention, reply to bot, Russian name variants (грег, гриша, григорий, etc.)
- **Positive signals:** Hot topics (+0.5), interests (+0.3), sentiment keywords (+0.2–0.6), momentum (+0.2), newcomer boost (+0.5), active conversation (+0.4), random factor
- **Negative signals:** Cooldown penalty (-0.3), night penalty (-0.2)
- **Rate limiting:** Max 5 unprompted responses per hour per chat

### Emotion Tracker (6 dimensions)
Per-user per-chat emotional state: warmth, trust, respect, annoyance, interest, loyalty. Each bounded [-1.0, 1.0]. After every response, Claude evaluates small deltas (-0.1 to 0.1). Annoyance decays nightly (×0.9). Emotions influence Greg's tone via personality modifiers.

### Media Support
Greg processes photos (full image via vision), videos (thumbnail), video notes/circles (thumbnail), and voice messages (text label only). Images are base64-encoded and sent as multimodal content blocks to Claude. Redis history stores text descriptions with Russian labels: `[Фото]`, `[Видео]`, `[Видеосообщение]`, `[Голосовое сообщение]`.

### Message Sending
Responses simulate natural typing speed (12 chars/sec, max 5s delay). Multi-part responses split on `\n---\n` are sent as separate messages with pauses between them.

## File Structure

```
config/
  settings.py          # Settings(BaseSettings) — module-level singleton, fails without env vars
  personality.py       # BASE_PERSONALITY (Russian prompt), TONE_MODIFIERS dict
  topics.py            # GREG_NAMES, INTERESTS, HOT_TAKES, SENTIMENT_KEYWORDS

src/
  main.py              # Entry point: creates pool, redis, wires everything, starts polling
  bot/
    handlers.py        # MessageHandler — main pipeline: extract media → store → score → respond
    sender.py          # MessageSender — typing animation, multi-part splitting
  brain/
    decision.py        # DecisionEngine — score calculation, rate limiting, should_respond
    emotions.py        # EmotionTracker — Claude-evaluated emotional deltas
    responder.py       # Responder — builds prompt via PersonalityEngine, calls Claude
    personality.py     # PersonalityEngine — system prompt + message list construction
  memory/
    short_term.py      # ShortTermMemory — Redis buffer operations
    long_term.py       # LongTermMemory — PostgreSQL CRUD for profiles, groups, memory log
    distiller.py       # Distiller — overflow extraction via Claude
    context_builder.py # ContextBuilder — assembles full context dict from STM + LTM
  utils/
    health.py          # /health HTTP endpoint
    logging.py         # JSON structured logging

tests/                 # 136 tests, mock-based (no real DB/Redis/API needed)
  conftest.py          # Sets test env vars, anyio_backend fixture
  test_handlers.py     # 18 tests: guards, text, photo, video, voice, video_note, distill, storage
  test_personality.py  # 18 tests: messages (text + multimodal), system prompt, tone modifiers
  test_responder.py    # 9 tests: API calls, image forwarding, failures, model verification
  test_sender.py       # 6 tests: splitting, typing, reply_to, empty handling
  test_context_builder.py  # 4 tests: context assembly, other profiles, limits
  test_decision.py     # 3 tests (original)
  test_decision_full.py    # 32 tests: all scoring, names, rate limiting, night, active convo
  test_emotions.py     # 1 test (original)
  test_memory.py       # 3 tests (original): store, get_recent, overflow
  test_distiller.py    # 2 tests (original): skip + extract
  test_long_term.py    # 3 tests (original)
  test_long_term_full.py   # 19 tests: profiles, facts, emotions (clamping, JSON), groups, decay

migrations/
  001_initial.sql      # user_profiles, group_context, memory_log tables
```

## Data Flow (Message → Response)

1. **Telegram message arrives** → `MessageHandler.handle_message()`
2. **Extract media** → `_extract_media()` builds `display_text` + optional `image_base64`
3. **Store in Redis** → `ShortTermMemory.store_message(display_text)`
4. **Check overflow** → If buffer > 200, trigger `Distiller.distill()` in background
5. **Score decision** → `DecisionEngine.calculate_score()` (or 1.0 for private chats)
6. **Check threshold** → `DecisionEngine.should_respond()` with rate limiting
7. **Build context** → `ContextBuilder.build_context()` — recent msgs, profile, group, memories
8. **Generate response** → `Responder.generate_response(context, text, username, image_base64)`
   - `PersonalityEngine.build_system_prompt()` — personality + emotions + facts + group
   - `PersonalityEngine.build_messages()` — history (text) + current (multimodal if image)
   - Claude Haiku API call
9. **Send response** → `MessageSender.send_response()` with typing simulation
10. **Post-response** → Store Greg's reply in Redis, record for rate limiting, evaluate emotions in background

## Key Design Decisions
- `config/settings.py` has module-level `settings = Settings()` — intentionally fails without env vars (runtime only). Tests set env vars in `conftest.py`.
- asyncpg `pool.acquire()` returns sync context manager — use `MagicMock` (not `AsyncMock`) in tests.
- Distiller uses `chat_id=chat_id` (keyword arg) — tests assert keyword form.
- All Claude calls use `claude-haiku-4-5-20251001` for cost/speed.
- Emotions stored as JSONB, may be dict or JSON string — code handles both.

## Deployment
```bash
cp .env.example .env  # Fill in TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY, GREG_BOT_USERNAME
docker compose up -d  # Starts greg, postgres, redis containers
```
Migrations auto-run on first PostgreSQL start via `docker-entrypoint-initdb.d`.

## Testing
```bash
python -m pytest tests/ -v  # 136 tests, ~3 seconds, no external deps needed
```
`asyncio_mode = "auto"` in `pyproject.toml`. All tests are mock-based.
