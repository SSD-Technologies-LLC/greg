# Greg

A Telegram group chat bot that acts as a genuine friend, not an assistant. Powered by Claude Opus 4.6 (responses) and Claude Haiku 4.5 (decisions), with web search via Tavily.

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

**Decision engine** — Greg doesn't respond to every message. He uses Haiku to semantically evaluate every message in context, deciding whether it's worth responding to based on conversation flow, emotional dynamics, and relevance — not keyword matching. This gives him a natural sense of when to speak up and when to stay quiet.

When someone mentions him or replies to him — he always responds.

**Web search** — Greg can search the web via Tavily when he needs current information to give a meaningful answer.

## Personality

Greg is a **wise trickster**: sharp wit, philosophical depth, strong opinions on everything. He trolls his friends, picks sides in arguments, and never refuses a topic. Street-smart with an old soul underneath. He speaks native Russian. He never admits to being a bot. He is not a school counselor — he's the smartest guy in the group chat.

## Setup

### Prerequisites

- Docker and Docker Compose
- [Telegram Bot Token](https://core.telegram.org/bots#botfather) (from @BotFather)
- [Anthropic API Key](https://console.anthropic.com/)
- (Optional) [Tavily API Key](https://tavily.com/) for web search

**Important:** In BotFather, disable privacy mode for your bot (`/setprivacy` -> Disable) so Greg can receive all group messages, not just commands and mentions.

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
| `GREG_RESPONSE_MODEL` | `claude-opus-4-6` | Model for generating responses |
| `GREG_DECISION_MODEL` | `claude-haiku-4-5-20251001` | Model for decision-making |
| `TAVILY_API_KEY` | — | Tavily API key for web search (optional) |
| `TAVILY_MAX_RESULTS` | `3` | Max search results per query |
| `GREG_COOLDOWN_MESSAGES` | `3` | Stay quiet for N messages after speaking |
| `GREG_MAX_UNPROMPTED_PER_HOUR` | `5` | Anti-spam: max unprompted messages per hour |
| `GREG_MAX_API_CALLS_PER_HOUR` | `60` | Cost control |
| `GREG_MAX_RESPONSE_TOKENS` | `512` | Keep responses short |
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
    responder.py     # Calls Claude API
    searcher.py      # Web search via Tavily
  memory/
    short_term.py    # Redis — recent messages buffer
    long_term.py     # PostgreSQL — profiles, facts, emotions
    distiller.py     # Extracts facts from message overflow
    context_builder.py  # Combines both memory layers
  utils/
    logging.py       # Structured JSON logging
    health.py        # Health endpoint
    json_parser.py   # JSON response parsing
  main.py            # Wires everything together
config/
  settings.py        # All configuration (from env vars)
  personality.py     # Greg's base personality prompt
  topics.py          # Greg's names/aliases
```

## Estimated cost

~$50-60/month for a moderately active group (~1000 messages/day) using Claude Opus 4.6 for responses, Haiku 4.5 for decisions, and Tavily for web searches.

## Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio ruff

# Run tests (146 tests)
python -m pytest tests/ -v

# Lint
ruff check .
ruff format --check .
```

CI runs automatically on push/PR to `main` via GitHub Actions (test, lint, typecheck).

## License

MIT
