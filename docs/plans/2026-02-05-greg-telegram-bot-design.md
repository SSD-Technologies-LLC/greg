# Greg — Telegram Group Chat Bot Design

## Overview

Greg is a Telegram chatbot powered by Claude Haiku 4.5 that joins closed friend group chats and acts as a genuine group member. He speaks native Russian, has a "wise trickster" personality, maintains short and long-term memory about each person, tracks emotional relationships, and chimes into conversations organically.

**Stack:** Python (aiogram 3.x) + PostgreSQL 16 + Redis 7 + Claude Haiku 4.5 API
**Estimated cost:** ~$11/month for a moderately active group (~500 messages/day)

---

## Architecture

Three core layers:

### 1. Telegram Layer (aiogram 3.x)
- Listens to all messages in group chats via long polling (or webhook, configurable)
- Every incoming message passes through the Decision Engine before any response
- Greg has a Telegram account name "Greg" — looks like a regular group member

### 2. Brain Layer
- **Decision Engine** — For every message, decides: respond, stay silent, or queue. Uses topic matching, sentiment analysis, and random factor (~7% base chance). When mentioned/replied to — always responds.
- **Memory Manager** — Dual-memory system. Reads from both short-term (Redis) and long-term (Postgres) to build context for each API call. After each conversation cycle, distills important new information from short-term into long-term storage.
- **Emotion Tracker** — Maintains per-person emotional state (warmth, trust, annoyance, respect, interest, loyalty) as numeric values that shift based on interactions. These values influence Greg's tone in responses.
- **Personality Engine** — Applies the "wise trickster" persona, Russian cultural context, and current emotional state to shape the system prompt sent to Claude Haiku.

### 3. Storage Layer
- **Redis** — Rolling buffer of recent messages per chat (short-term memory). Fast reads for conversation context. TTL-based expiry with distillation before eviction.
- **PostgreSQL** — User profiles, facts, emotional states, group culture notes. The durable "soul" of Greg's relationships.

All wrapped in Docker Compose: app container, Postgres, Redis.

---

## Memory System

### Short-term Memory (Redis)

- Every message stored as JSON: `{user_id, username, text, timestamp, chat_id}`
- Stored in Redis list per chat: `chat:{chat_id}:messages` — last 200 messages kept
- When buffer hits 200, oldest 50 go through distillation before eviction
- Per-chat "conversation state" tracked: current topics, active users, emotional temperature
- TTL of 48 hours as safety net — if distillation fails, messages expire gracefully

### Long-term Memory (PostgreSQL)

Three core tables:

**`user_profiles`**
- One row per user per chat
- `user_id, chat_id, display_name, facts (JSONB), personality_traits (JSONB), emotional_state (JSONB), last_updated`
- Facts: `{"job": "developer", "has_dog": true, "girlfriend_name": "Masha", ...}`
- Personality traits: `{"humor_style": "dry", "conflict_avoidance": 0.8, "openness": 0.7, ...}`
- Emotional state (Greg's feelings toward this person): `{"warmth": 0.6, "trust": 0.4, "annoyance": 0.1, ...}`

**`group_context`**
- Per chat: `chat_id, group_dynamics (JSONB), inside_jokes (JSONB), recurring_topics (JSONB), last_updated`

**`memory_log`**
- Append-only log of distilled memories
- `id, chat_id, user_id, memory_type (fact/insight/event), content, created_at`
- Fed by distillation, feeds periodic profile updates

### Distillation Process

Every 50 messages (or 30 minutes, whichever comes first), Greg sends the buffer to Claude Haiku with a prompt: "Extract new facts, personality insights, and notable events about each participant. Return structured JSON." The result updates `user_profiles` and appends to `memory_log`.

---

## Decision Engine

Every incoming message runs through a scoring pipeline producing a response probability (0.0–1.0). Greg responds if score exceeds threshold (default 0.4).

### Score Components

| Component | Score | Description |
|---|---|---|
| Direct trigger | = 1.0 | Mentioned, replied to, or DM |
| Topic match | +0.2–0.5 | Message hits Greg's interests/opinions |
| Sentiment spike | +0.1–0.4 | Strong emotion detected |
| Conversation momentum | +0.1–0.2 | Active chat, Greg quiet for a while |
| Random factor | +0.0–0.07 | Unpredictability |
| Cooldown penalty | -0.3 | Greg spoke in last 3 messages |
| Night mode penalty | -0.2 | Between 1am–8am |

### Anti-spam
- Hard limit: max 5 unprompted messages per hour per chat
- Direct mentions don't count against this limit
- All weights and thresholds configurable via env vars

---

## Emotion Tracker

Six dimensions per person per chat, each a float from -1.0 to 1.0:

| Dimension | Description |
|---|---|
| **warmth** | How much Greg likes this person. Rises with friendly interactions, drops with hostility. |
| **trust** | How much Greg opens up. Rises slowly through consistency, drops sharply if sincerity is mocked. |
| **respect** | How seriously Greg takes their opinions. Rises with good points, drops with bad faith. |
| **annoyance** | Short-term irritation. Rises fast with spam/condescension. Decays 10% daily toward 0. |
| **interest** | Curiosity about this person. Rises with novelty, drops with predictability. |
| **loyalty** | Long-term bond. Moves very slowly. High loyalty = Greg defends them, remembers dates, checks in. |

### Behavioral Effects

- High warmth + trust: Vulnerable, shares "personal" thoughts, uses inside jokes
- High annoyance: Shorter replies, more sarcasm, less patience
- Low respect + high warmth: Teases lovingly, doesn't take their advice
- High loyalty: Initiates conversations, remembers small details

### Updates
- After each interaction, Haiku evaluates exchange and returns emotional deltas (+-0.01 to +-0.1)
- Big shifts require big events
- All changes logged to `memory_log` with reasoning

---

## Personality Engine

System prompt is assembled dynamically for every response.

### Base Personality (static)
- Wise trickster: sharp wit, philosophical depth, doesn't take himself too seriously
- Street-smart with an old soul. Challenges people with humor. Can roast and comfort.
- Native Russian: natural spoken Russian, contractions, slang matching the group's vibe
- Never breaks character. If asked "are you a bot?" — deflects, gets offended, plays it off.

### Dynamic Context (per request)
- Current emotional state toward the person being replied to
- Recent conversation from Redis (last 30–50 messages)
- Relevant facts about the person from Postgres
- Group dynamics: inside jokes, relationships between members, recurring topics
- Current conversation energy level

### Tone Modifiers (injected based on emotional state)
- Warmth > 0.7: "You genuinely enjoy talking to this person. Show it subtly."
- Annoyance > 0.5: "You're irritated. Be shorter, more sarcastic. Don't fake warmth."
- Trust < -0.3: "You're guarded. Don't share anything personal. Keep it surface level."

### Response Format
- 1–3 sentences typical, like real chat messages. Never walls of text.
- Emoji used sparingly, like a real person
- Can send multiple short messages (split with `\n---\n`, app sends separately with delays)

---

## Robust Guarantees

### Memory Durability
- Distillation writes to Postgres inside a transaction — no partial profiles on failure
- Undistilled messages stay in Redis and retry next cycle
- Redis AOF with `appendfsync everysec` — worst case: 1 second of lost messages
- Background task every 6 hours: reconciles `memory_log` against `user_profiles`

### Graceful Degradation
- **Redis down:** Greg works with long-term memory only — responds like he "just woke up." Logs warning.
- **Postgres down:** Greg chats using Redis context, queues distillation until Postgres recovers.
- **Anthropic API down:** Greg goes silent. No fallback model — silence > broken character.

### Rate Limiting & Cost Control
- Max tokens per response: 300
- Max Haiku API calls per hour: configurable (default 60)
- Distillation batched: one call per 50 messages
- Decision engine uses keyword/regex first, Haiku only for ambiguous cases

### Monitoring
- Health endpoint on configurable port
- Structured JSON logging to stdout (Docker captures)
- Key metrics: messages processed, responses sent, distillations run, API errors, memory sizes

---

## Docker Compose & Configuration

### Containers
- **greg** — Python app. Depends on postgres and redis. Restart always.
- **postgres** — PostgreSQL 16 with volume for data persistence.
- **redis** — Redis 7 with AOF persistence, volume mounted.

### Environment Variables

```bash
# Required
TELEGRAM_BOT_TOKEN=         # From @BotFather
ANTHROPIC_API_KEY=          # Anthropic API key
GREG_BOT_USERNAME=          # Bot's Telegram username (without @)

# Postgres
POSTGRES_USER=greg
POSTGRES_PASSWORD=          # Strong password
POSTGRES_DB=greg_brain

# Redis
REDIS_PASSWORD=             # Strong password

# Tuning (defaults shown)
GREG_RESPONSE_THRESHOLD=0.4
GREG_RANDOM_FACTOR=0.07
GREG_COOLDOWN_MESSAGES=3
GREG_MAX_UNPROMPTED_PER_HOUR=5
GREG_MAX_API_CALLS_PER_HOUR=60
GREG_MAX_RESPONSE_TOKENS=300
GREG_NIGHT_START=1
GREG_NIGHT_END=8
GREG_TIMEZONE=Europe/Moscow
GREG_DISTILL_EVERY_N=50
GREG_DISTILL_EVERY_MINUTES=30
GREG_REDIS_BUFFER_SIZE=200
GREG_HEALTH_PORT=8080
GREG_LOG_LEVEL=INFO
```

---

## Project Structure

```
griwa/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.py               # Pydantic settings, loads env vars
│   ├── personality.py            # Base personality prompt, interests, hot takes
│   └── topics.py                 # Topic keywords and categories
├── src/
│   ├── __init__.py
│   ├── main.py                   # Entrypoint: init bot, connect DB/Redis, start polling
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── handlers.py           # Telegram message handlers
│   │   └── sender.py             # Message splitting, typing delays, send logic
│   ├── brain/
│   │   ├── __init__.py
│   │   ├── decision.py           # Decision engine — scoring
│   │   ├── emotions.py           # Emotion tracker — state, decay, queries
│   │   ├── personality.py        # Dynamic system prompt assembly
│   │   └── responder.py          # Haiku API calls, response formatting
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py         # Redis operations
│   │   ├── long_term.py          # Postgres operations
│   │   ├── distiller.py          # Distillation logic
│   │   └── context_builder.py    # Context assembly from both layers
│   └── utils/
│       ├── __init__.py
│       ├── logging.py            # Structured JSON logging
│       └── health.py             # Health endpoint
├── migrations/
│   └── 001_initial.sql           # Postgres schema
└── tests/
    ├── test_decision.py
    ├── test_emotions.py
    ├── test_distiller.py
    └── test_memory.py
```
