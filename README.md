# Greg

A Telegram group chat bot that acts as a genuine friend, not an assistant. Powered by Claude Haiku 4.5.

Greg joins your closed friend group chats, learns about each person over time, builds real emotional relationships, and chimes into conversations organically — like a friend who's always in the chat but doesn't need to respond to everything.

## How it works

**Dual memory** — like a human brain:
- **Short-term** (Redis): Rolling buffer of last 200 messages per chat. This is Greg's "working memory" — what's happening right now.
- **Long-term** (PostgreSQL): Facts about each person, personality profiles, emotional state, group dynamics, inside jokes. This is Greg's "deep memory" — who you are to him.

**Memory distillation** — when short-term memory overflows, Greg extracts important facts and insights before discarding old messages. Nothing meaningful gets lost.

**Emotional relationships** — Greg tracks 6 emotional dimensions per person:

| Dimension | What it means |
|---|---|
| Warmth | How much Greg likes you |
| Trust | How much he opens up to you |
| Respect | How seriously he takes your opinions |
| Annoyance | Short-term irritation (decays daily) |
| Interest | How curious he is about you |
| Loyalty | Long-term bond (moves slowly) |

These aren't just numbers — they change Greg's behavior. High warmth means warmer jokes. High annoyance means shorter, more sarcastic replies. High loyalty means he'll defend you in arguments.

**Decision engine** — Greg doesn't respond to every message. He scores each message based on:
- Topic relevance (does he care about this?)
- Emotional intensity (is someone venting, celebrating, arguing?)
- Conversation momentum (has the chat been active?)
- A small random factor (keeps him unpredictable)
- Cooldown (he won't dominate the chat)
- Night mode (even friends sleep)

When someone mentions him or replies to him — he always responds.

## Personality

Greg is a **wise trickster**: sharp wit, philosophical depth, doesn't take himself too seriously. Street-smart but with an old soul underneath. He speaks native Russian. He never admits to being a bot.

## Setup

### Prerequisites

- Docker and Docker Compose
- [Telegram Bot Token](https://core.telegram.org/bots#botfather) (from @BotFather)
- [Anthropic API Key](https://console.anthropic.com/)

### Deploy

```bash
git clone https://github.com/YOUR_USERNAME/griwa.git
cd griwa

cp .env.example .env
# Edit .env and fill in required values (see below)

docker compose up -d
```

Then add the bot to your Telegram group chat and start talking.

### Environment variables

**Required** — the bot won't start without these:

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From @BotFather |
| `ANTHROPIC_API_KEY` | Your Anthropic API key |
| `GREG_BOT_USERNAME` | Bot's Telegram username (without @) |
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `REDIS_PASSWORD` | Redis password |

**Optional** — tune Greg's behavior:

| Variable | Default | Description |
|---|---|---|
| `GREG_RESPONSE_THRESHOLD` | `0.4` | Score threshold to respond (0.0-1.0). Lower = chattier |
| `GREG_RANDOM_FACTOR` | `0.07` | Random chime-in chance per message |
| `GREG_COOLDOWN_MESSAGES` | `3` | Stay quiet for N messages after speaking |
| `GREG_MAX_UNPROMPTED_PER_HOUR` | `5` | Anti-spam: max unprompted messages per hour |
| `GREG_MAX_API_CALLS_PER_HOUR` | `60` | Cost control |
| `GREG_MAX_RESPONSE_TOKENS` | `300` | Keep responses short |
| `GREG_NIGHT_START` | `1` | Hour (24h) when Greg sleeps |
| `GREG_NIGHT_END` | `8` | Hour when Greg wakes up |
| `GREG_TIMEZONE` | `Europe/Moscow` | Timezone for night mode |
| `GREG_DISTILL_EVERY_N` | `50` | Distill after N messages |
| `GREG_DISTILL_EVERY_MINUTES` | `30` | Or after N minutes |
| `GREG_REDIS_BUFFER_SIZE` | `200` | Messages kept in short-term memory |
| `GREG_HEALTH_PORT` | `8080` | Health check endpoint port |
| `GREG_LOG_LEVEL` | `INFO` | Logging level |

## Architecture

```
src/
  bot/
    handlers.py      # Message handler — the main pipeline
    sender.py        # Splits messages, adds typing delays
  brain/
    decision.py      # Decides when to respond
    emotions.py      # Tracks emotional state per person
    personality.py   # Assembles dynamic system prompts
    responder.py     # Calls Claude Haiku API
  memory/
    short_term.py    # Redis — recent messages buffer
    long_term.py     # PostgreSQL — profiles, facts, emotions
    distiller.py     # Extracts facts from message overflow
    context_builder.py  # Combines both memory layers
  utils/
    logging.py       # Structured JSON logging
    health.py        # Health endpoint
  main.py            # Wires everything together
config/
  settings.py        # All configuration (from env vars)
  personality.py     # Greg's base personality prompt
  topics.py          # Topic keywords, sentiment triggers
```

## Estimated cost

~$11/month for a moderately active group (~500 messages/day) using Claude Haiku 4.5.

## Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v
```

## License

MIT
