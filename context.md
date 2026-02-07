# Greg Telegram Bot — Project Context

## What is Greg?
Greg (Гриша/Григорий) is a Telegram group chat bot. He acts as a real chat participant — sharp, opinionated, with a trolling streak. He speaks native Russian, never breaks character, has genuine emotional relationships with users, and can search the web for real-time information.

## Tech Stack
- **Runtime:** Python 3.12, Docker Compose (3 containers: greg, postgres, redis)
- **Bot framework:** aiogram 3.x
- **AI:** Anthropic SDK (AsyncAnthropic) — Opus 4.6 for responses, Haiku 4.5 for decisions/distillation/emotions
- **Storage:** asyncpg (PostgreSQL) + redis.asyncio (Redis)
- **Search:** Tavily API (optional, for real-time web search)
- **Config:** pydantic-settings (BaseSettings with `.env` file)
- **CI:** GitHub Actions (pytest + ruff + mypy)

## Architecture

### Dual Memory System
- **Short-term (Redis):** 200-message circular buffer per chat. Stores JSON-encoded messages with `{user_id, username, text, timestamp, chat_id}`. TTL: 48 hours.
- **Long-term (PostgreSQL):** User profiles (per chat) with facts, personality traits, emotional state. Group context with inside jokes and recurring topics. Memory log audit trail.

### Distillation Pipeline
Every 50 messages or 30 minutes, overflow messages beyond the 200-buffer are sent to Haiku 4.5 which extracts facts, personality insights, inside jokes, and recurring topics. Results are merged into PostgreSQL via JSONB `||` operator. Uses `safe_parse_json()` with retry logic for robustness.

### Semantic Decision Engine
Every message is evaluated for whether Greg should respond. Flow:

1. **Direct triggers** (no API call): @mention, reply to bot, name variants (грег, гриша, григорий, etc.) → always respond
2. **Night gate** (1–8 AM Moscow, no API call): skip unless direct
3. **Rate limit gate** (no API call): max 5 unprompted responses/hour/chat
4. **Haiku semantic evaluation** (API call): last 10 messages + time since Greg last spoke → returns `{respond, reason, search_needed, search_query}`

Returns a `DecisionResult` dataclass with `should_respond`, `is_direct`, `search_needed`, `search_query`.

### Web Search (Tavily)
When the decision engine flags `search_needed=true`, Greg searches the web via Tavily before responding. Results (answer + top 3 snippets with URLs) are injected into the response context. Search runs in a thread (`asyncio.to_thread`) to avoid blocking the event loop. Gracefully degrades if `TAVILY_API_KEY` is unset or Tavily is down.

### Emotion Tracker (6 dimensions)
Per-user per-chat emotional state: warmth, trust, respect, annoyance, interest, loyalty. Each bounded [-1.0, 1.0]. After every response, Haiku evaluates small deltas (-0.1 to 0.1). Annoyance decays nightly (×0.9). Emotions influence Greg's tone via personality modifiers (annoyance_high, bored, trolling).

### Personality System
Greg is a sharp, opinionated friend. No topic refusals — he discusses everything (politics, crypto, dark humor). Trolling toolkit: deliberately ignore messages, give purposefully unhelpful answers, pretend to misunderstand, agree with absurd takes. Anti-leak instructions prevent reproducing `[username]:` format markers or `---` separators.

Tone modifiers activate based on emotional state:
- `annoyance_high` (annoyance > 0.5) — meaner, one-word answers
- `bored` (interest < -0.4) — redirects conversation
- `trolling` (warmth > 0.4 AND trust > 0.4) — roasts friends

### Output Sanitization
`sanitize_response()` in responder.py cleans up model output:
- Strips literal `\n---\n` and actual separator patterns
- Removes leaked `[username]:` format lines
- Trims truncated responses to last complete sentence

### Media Support
Processes photos (full image via vision), videos (thumbnail), video notes/circles (thumbnail), and voice messages (text label only). Images are base64-encoded and sent as multimodal content blocks. Redis history stores text descriptions with Russian labels: `[Фото]`, `[Видео]`, `[Видеосообщение]`, `[Голосовое сообщение]`.

### Message Sending
Responses simulate natural typing speed (12 chars/sec, max 5s delay). Multi-part responses split on `---` separator patterns (regex-based, handles whitespace/length variants) are sent as separate messages with pauses.

## File Structure

```
config/
  settings.py          # Settings(BaseSettings) — module-level singleton, fails without env vars
  personality.py       # BASE_PERSONALITY (Russian prompt), TONE_MODIFIERS dict
  topics.py            # GREG_NAMES — name/alias variants for direct triggers

src/
  main.py              # Entry point: creates pool, redis, wires everything, starts polling
  bot/
    handlers.py        # MessageHandler — main pipeline: extract → store → evaluate → search → respond
    sender.py          # MessageSender — typing animation, regex-based multi-part splitting
  brain/
    decision.py        # DecisionEngine — semantic Haiku evaluation, rate limiting, DecisionResult
    emotions.py        # EmotionTracker — Haiku-evaluated emotional deltas with JSON retry
    responder.py       # Responder — builds prompt, calls Opus API, sanitizes output
    searcher.py        # WebSearcher — Tavily search wrapper with graceful degradation
    personality.py     # PersonalityEngine — system prompt + message list construction
  memory/
    short_term.py      # ShortTermMemory — Redis buffer operations
    long_term.py       # LongTermMemory — PostgreSQL CRUD for profiles, groups, memory log
    distiller.py       # Distiller — overflow extraction via Haiku with JSON retry
    context_builder.py # ContextBuilder — assembles full context dict from STM + LTM
  utils/
    json_parser.py     # safe_parse_json() — robust JSON extraction from model output
    health.py          # /health HTTP endpoint
    logging.py         # JSON structured logging

tests/                 # 146 tests, mock-based (no real DB/Redis/API needed)
  conftest.py          # Sets test env vars before source imports
  test_handlers.py     # 20 tests: guards, text, media, distill, storage, search integration
  test_personality.py  # 18 tests: messages, system prompt, tone modifiers, trolling logic
  test_responder.py    # 9 tests: API calls, image forwarding, failures, model, sanitization
  test_sender.py       # 6 tests: splitting, typing, reply_to, empty handling
  test_formatting.py   # 13 tests: sanitize_response, robust sender splitting
  test_semantic_decision.py  # 16 tests: direct triggers, semantic eval, rate limiting, night mode
  test_searcher.py     # 7 tests: search, formatting, errors, disabled client
  test_json_parser.py  # 9 tests: valid JSON, markdown blocks, BOM, garbage, nested
  test_distiller.py    # 5 tests: skip, extract, retry, graceful failure, model
  test_emotions.py     # 4 tests: interaction, malformed JSON, empty response, model
  test_context_builder.py  # 4 tests: context assembly, other profiles, limits
  test_long_term.py    # 3 tests: profile, facts, memory log
  test_long_term_full.py   # 19 tests: profiles, facts, emotions, groups, decay
  test_memory.py       # 3 tests: store, get_recent, overflow

migrations/
  001_initial.sql      # user_profiles, group_context, memory_log tables

.github/workflows/
  ci.yml               # pytest + ruff lint/format + mypy typecheck
```

## Data Flow (Message → Response)

1. **Telegram message arrives** → `MessageHandler.handle_message()`
2. **Extract media** → `_extract_media()` builds `display_text` + optional `image_base64`
3. **Store in Redis** → `ShortTermMemory.store_message(display_text)`
4. **Check overflow** → If buffer > 200, trigger `Distiller.distill()` in background
5. **Evaluate decision** → Private chat: always respond. Group: `DecisionEngine.evaluate()` → `DecisionResult`
6. **Web search** → If `search_needed`, call `WebSearcher.search()` via `asyncio.to_thread()`
7. **Build context** → `ContextBuilder.build_context()` — recent msgs, profile, group, memories
8. **Generate response** → `Responder.generate_response(context, text, username, image_base64, search_context)`
   - `PersonalityEngine.build_system_prompt()` — personality + emotions + facts + group + search results
   - `PersonalityEngine.build_messages()` — history (text) + current (multimodal if image)
   - Opus 4.6 API call → `sanitize_response()` cleanup
9. **Send response** → `MessageSender.send_response()` with typing simulation
10. **Post-response** → Store Greg's reply in Redis, record for rate limiting, evaluate emotions in background

## Key Design Decisions
- `config/settings.py` has module-level `settings = Settings()` — intentionally fails without env vars (runtime only). Tests set env vars in `conftest.py`.
- asyncpg `pool.acquire()` returns sync context manager — use `MagicMock` (not `AsyncMock`) in tests.
- Distiller uses `chat_id=chat_id` (keyword arg) — tests assert keyword form.
- Models are configurable: `GREG_RESPONSE_MODEL` (Opus 4.6) for responses, `GREG_DECISION_MODEL` (Haiku 4.5) for decisions/distillation/emotions. No hardcoded model strings.
- Emotions stored as JSONB, may be dict or JSON string — code handles both.
- All JSON parsing from model output uses `safe_parse_json()` with retry on failure.
- Tavily search is synchronous — wrapped in `asyncio.to_thread()` in handlers to avoid blocking.

## Deployment
```bash
cp .env.example .env  # Fill in TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY, GREG_BOT_USERNAME
docker compose up -d  # Starts greg, postgres, redis containers
```
Migrations auto-run on first PostgreSQL start via `docker-entrypoint-initdb.d`.
On Railway: migrations auto-run in `src/main.py` on startup.

## Testing
```bash
python -m pytest tests/ -v  # 146 tests, ~3 seconds, no external deps needed
ruff check .                # Lint
ruff format --check .       # Format check
```
`asyncio_mode = "auto"` in `pyproject.toml`. All tests are mock-based.
