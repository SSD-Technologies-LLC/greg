# Greg Telegram Bot — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Telegram group chat bot ("Greg") powered by Claude Haiku 4.5 with dual memory, emotional tracking, and organic conversation participation.

**Architecture:** Python async app (aiogram 3.x) with Redis for short-term memory, PostgreSQL for long-term memory. Decision engine scores each message to determine if Greg responds. Personality engine assembles dynamic system prompts incorporating emotional state and memory context.

**Tech Stack:** Python 3.12, aiogram 3.x, anthropic SDK, asyncpg, redis.asyncio, pydantic-settings, aiohttp (health endpoint), Docker Compose (app + PostgreSQL 16 + Redis 7)

**Design doc:** `docs/plans/2026-02-05-greg-telegram-bot-design.md`

---

### Task 1: Project Scaffold — Docker Compose, Dockerfile, Requirements, Config

**Files:**
- Create: `docker-compose.yml`
- Create: `Dockerfile`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `config/__init__.py`
- Create: `config/settings.py`
- Create: `src/__init__.py`
- Create: `src/bot/__init__.py`
- Create: `src/brain/__init__.py`
- Create: `src/memory/__init__.py`
- Create: `src/utils/__init__.py`

**Step 1: Create `requirements.txt`**

```
aiogram>=3.15,<4
anthropic>=0.42,<1
asyncpg>=0.30,<1
redis[hiredis]>=5.2,<6
pydantic-settings>=2.7,<3
aiohttp>=3.11,<4
```

**Step 2: Create `Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.main"]
```

**Step 3: Create `docker-compose.yml`**

```yaml
services:
  greg:
    build: .
    restart: always
    env_file: .env
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    ports:
      - "${GREG_HEALTH_PORT:-8080}:${GREG_HEALTH_PORT:-8080}"

  postgres:
    image: postgres:16-alpine
    restart: always
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-greg}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB:-greg_brain}
    volumes:
      - greg_pgdata:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-greg}"]
      interval: 5s
      timeout: 3s
      retries: 5

  redis:
    image: redis:7-alpine
    restart: always
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --appendonly yes
      --appendfsync everysec
    volumes:
      - greg_redisdata:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  greg_pgdata:
  greg_redisdata:
```

**Step 4: Create `.env.example`**

```bash
# Required
TELEGRAM_BOT_TOKEN=
ANTHROPIC_API_KEY=
GREG_BOT_USERNAME=

# Postgres
POSTGRES_USER=greg
POSTGRES_PASSWORD=
POSTGRES_DB=greg_brain

# Redis
REDIS_PASSWORD=

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

**Step 5: Create `config/settings.py`**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Required
    telegram_bot_token: str
    anthropic_api_key: str
    greg_bot_username: str

    # Postgres
    postgres_user: str = "greg"
    postgres_password: str
    postgres_db: str = "greg_brain"

    # Redis
    redis_password: str

    # Tuning
    greg_response_threshold: float = 0.4
    greg_random_factor: float = 0.07
    greg_cooldown_messages: int = 3
    greg_max_unprompted_per_hour: int = 5
    greg_max_api_calls_per_hour: int = 60
    greg_max_response_tokens: int = 300
    greg_night_start: int = 1
    greg_night_end: int = 8
    greg_timezone: str = "Europe/Moscow"
    greg_distill_every_n: int = 50
    greg_distill_every_minutes: int = 30
    greg_redis_buffer_size: int = 200
    greg_health_port: int = 8080
    greg_log_level: str = "INFO"

    @property
    def postgres_dsn(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@postgres:5432/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        return f"redis://:{self.redis_password}@redis:6379/0"

    model_config = {"env_file": ".env"}


settings = Settings()
```

**Step 6: Create all `__init__.py` files**

Empty files for: `config/__init__.py`, `src/__init__.py`, `src/bot/__init__.py`, `src/brain/__init__.py`, `src/memory/__init__.py`, `src/utils/__init__.py`

**Step 7: Commit**

```bash
git add docker-compose.yml Dockerfile requirements.txt .env.example config/ src/
git commit -m "feat: project scaffold — Docker Compose, config, package structure"
```

---

### Task 2: Database Schema & Migrations

**Files:**
- Create: `migrations/001_initial.sql`

**Step 1: Create `migrations/001_initial.sql`**

```sql
CREATE TABLE IF NOT EXISTS user_profiles (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    chat_id BIGINT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    facts JSONB NOT NULL DEFAULT '{}',
    personality_traits JSONB NOT NULL DEFAULT '{}',
    emotional_state JSONB NOT NULL DEFAULT '{
        "warmth": 0.0,
        "trust": 0.0,
        "respect": 0.0,
        "annoyance": 0.0,
        "interest": 0.0,
        "loyalty": 0.0
    }',
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, chat_id)
);

CREATE TABLE IF NOT EXISTS group_context (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL UNIQUE,
    group_dynamics JSONB NOT NULL DEFAULT '{}',
    inside_jokes JSONB NOT NULL DEFAULT '[]',
    recurring_topics JSONB NOT NULL DEFAULT '[]',
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS memory_log (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    user_id BIGINT,
    memory_type TEXT NOT NULL CHECK (memory_type IN ('fact', 'insight', 'event', 'emotion_change')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_lookup ON user_profiles (user_id, chat_id);
CREATE INDEX IF NOT EXISTS idx_memory_log_chat ON memory_log (chat_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_log_user ON memory_log (user_id, chat_id, created_at DESC);
```

**Step 2: Commit**

```bash
git add migrations/
git commit -m "feat: PostgreSQL schema — user_profiles, group_context, memory_log"
```

---

### Task 3: Utilities — Logging & Health Endpoint

**Files:**
- Create: `src/utils/logging.py`
- Create: `src/utils/health.py`

**Step 1: Create `src/utils/logging.py`**

```python
import json
import logging
import sys
from datetime import datetime, timezone

from config.settings import settings


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.greg_log_level.upper(), logging.INFO))

    # Quiet noisy libraries
    logging.getLogger("aiogram").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
```

**Step 2: Create `src/utils/health.py`**

```python
import logging

from aiohttp import web

from config.settings import settings

logger = logging.getLogger(__name__)

_health_status: dict = {"status": "starting"}


def set_health(status: str) -> None:
    _health_status["status"] = status


async def _handle_health(request: web.Request) -> web.Response:
    return web.json_response(_health_status)


async def start_health_server() -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/health", _handle_health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", settings.greg_health_port)
    await site.start()
    logger.info("Health endpoint started on port %s", settings.greg_health_port)
    return runner
```

**Step 3: Commit**

```bash
git add src/utils/
git commit -m "feat: structured JSON logging and health endpoint"
```

---

### Task 4: Short-term Memory (Redis)

**Files:**
- Create: `src/memory/short_term.py`
- Create: `tests/test_memory.py`

**Step 1: Write the failing test**

```python
import json
from unittest.mock import AsyncMock, patch

import pytest

from src.memory.short_term import ShortTermMemory


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.llen = AsyncMock(return_value=0)
    r.rpush = AsyncMock()
    r.lrange = AsyncMock(return_value=[])
    r.ltrim = AsyncMock()
    r.set = AsyncMock()
    r.get = AsyncMock(return_value=None)
    r.expire = AsyncMock()
    return r


@pytest.fixture
def stm(mock_redis):
    return ShortTermMemory(mock_redis, buffer_size=200)


@pytest.mark.asyncio
async def test_store_message(stm, mock_redis):
    await stm.store_message(chat_id=123, user_id=456, username="alice", text="hello")
    mock_redis.rpush.assert_called_once()
    key = mock_redis.rpush.call_args[0][0]
    assert key == "chat:123:messages"
    data = json.loads(mock_redis.rpush.call_args[0][1])
    assert data["user_id"] == 456
    assert data["text"] == "hello"


@pytest.mark.asyncio
async def test_get_recent_messages(stm, mock_redis):
    msgs = [json.dumps({"user_id": 1, "username": "a", "text": "hi", "timestamp": "t", "chat_id": 1}).encode()]
    mock_redis.lrange = AsyncMock(return_value=msgs)
    result = await stm.get_recent_messages(chat_id=1, count=50)
    assert len(result) == 1
    assert result[0]["text"] == "hi"


@pytest.mark.asyncio
async def test_get_overflow_messages(stm, mock_redis):
    mock_redis.llen = AsyncMock(return_value=210)
    msgs = [json.dumps({"user_id": 1, "username": "a", "text": f"msg{i}", "timestamp": "t", "chat_id": 1}).encode() for i in range(50)]
    mock_redis.lrange = AsyncMock(return_value=msgs)
    result = await stm.get_overflow_messages(chat_id=1)
    assert len(result) == 50
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory.py -v`
Expected: FAIL with import error (module doesn't exist yet)

**Step 3: Create `src/memory/short_term.py`**

```python
import json
import logging
from datetime import datetime, timezone

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class ShortTermMemory:
    def __init__(self, redis: Redis, buffer_size: int = 200) -> None:
        self._redis = redis
        self._buffer_size = buffer_size

    def _chat_key(self, chat_id: int) -> str:
        return f"chat:{chat_id}:messages"

    def _state_key(self, chat_id: int) -> str:
        return f"chat:{chat_id}:state"

    async def store_message(
        self, chat_id: int, user_id: int, username: str, text: str
    ) -> int:
        msg = json.dumps(
            {
                "user_id": user_id,
                "username": username,
                "text": text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chat_id": chat_id,
            },
            ensure_ascii=False,
        )
        key = self._chat_key(chat_id)
        await self._redis.rpush(key, msg)
        await self._redis.expire(key, 48 * 3600)  # 48h TTL safety net
        length = await self._redis.llen(key)
        return length

    async def get_recent_messages(self, chat_id: int, count: int = 50) -> list[dict]:
        key = self._chat_key(chat_id)
        raw = await self._redis.lrange(key, -count, -1)
        return [json.loads(m) for m in raw]

    async def get_overflow_messages(self, chat_id: int) -> list[dict] | None:
        key = self._chat_key(chat_id)
        length = await self._redis.llen(key)
        if length <= self._buffer_size:
            return None
        overflow_count = length - self._buffer_size
        raw = await self._redis.lrange(key, 0, overflow_count - 1)
        return [json.loads(m) for m in raw]

    async def trim_overflow(self, chat_id: int) -> None:
        key = self._chat_key(chat_id)
        length = await self._redis.llen(key)
        if length > self._buffer_size:
            await self._redis.ltrim(key, length - self._buffer_size, -1)

    async def get_buffer_length(self, chat_id: int) -> int:
        return await self._redis.llen(self._chat_key(chat_id))

    async def set_chat_state(self, chat_id: int, state: dict) -> None:
        key = self._state_key(chat_id)
        await self._redis.set(key, json.dumps(state, ensure_ascii=False))
        await self._redis.expire(key, 48 * 3600)

    async def get_chat_state(self, chat_id: int) -> dict | None:
        raw = await self._redis.get(self._state_key(chat_id))
        return json.loads(raw) if raw else None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory.py -v`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add src/memory/short_term.py tests/test_memory.py
git commit -m "feat: short-term memory — Redis message buffer with overflow detection"
```

---

### Task 5: Long-term Memory (PostgreSQL)

**Files:**
- Create: `src/memory/long_term.py`
- Create: `tests/test_long_term.py`

**Step 1: Write the failing test**

```python
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.long_term import LongTermMemory

DEFAULT_EMOTIONS = {
    "warmth": 0.0, "trust": 0.0, "respect": 0.0,
    "annoyance": 0.0, "interest": 0.0, "loyalty": 0.0,
}


@pytest.fixture
def mock_pool():
    pool = AsyncMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    conn.transaction.return_value.__aenter__ = AsyncMock()
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool, conn


@pytest.fixture
def ltm(mock_pool):
    pool, _ = mock_pool
    return LongTermMemory(pool)


@pytest.mark.asyncio
async def test_get_or_create_profile_existing(ltm, mock_pool):
    _, conn = mock_pool
    row = {"user_id": 1, "chat_id": 10, "display_name": "Alice",
           "facts": {}, "personality_traits": {}, "emotional_state": DEFAULT_EMOTIONS}
    conn.fetchrow = AsyncMock(return_value=row)
    result = await ltm.get_or_create_profile(user_id=1, chat_id=10, display_name="Alice")
    assert result["display_name"] == "Alice"


@pytest.mark.asyncio
async def test_update_facts(ltm, mock_pool):
    _, conn = mock_pool
    conn.execute = AsyncMock()
    await ltm.update_facts(user_id=1, chat_id=10, facts={"job": "dev"})
    conn.execute.assert_called_once()
    query = conn.execute.call_args[0][0]
    assert "facts" in query


@pytest.mark.asyncio
async def test_append_memory_log(ltm, mock_pool):
    _, conn = mock_pool
    conn.execute = AsyncMock()
    await ltm.append_memory_log(chat_id=10, user_id=1, memory_type="fact", content="works at Google")
    conn.execute.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_long_term.py -v`
Expected: FAIL with import error

**Step 3: Create `src/memory/long_term.py`**

```python
import json
import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

DEFAULT_EMOTIONS = {
    "warmth": 0.0,
    "trust": 0.0,
    "respect": 0.0,
    "annoyance": 0.0,
    "interest": 0.0,
    "loyalty": 0.0,
}


class LongTermMemory:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_or_create_profile(
        self, user_id: int, chat_id: int, display_name: str = ""
    ) -> dict:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM user_profiles WHERE user_id = $1 AND chat_id = $2",
                user_id,
                chat_id,
            )
            if row:
                return dict(row)
            await conn.execute(
                """INSERT INTO user_profiles (user_id, chat_id, display_name)
                   VALUES ($1, $2, $3)
                   ON CONFLICT (user_id, chat_id) DO NOTHING""",
                user_id,
                chat_id,
                display_name,
            )
            row = await conn.fetchrow(
                "SELECT * FROM user_profiles WHERE user_id = $1 AND chat_id = $2",
                user_id,
                chat_id,
            )
            return dict(row)

    async def get_profile(self, user_id: int, chat_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM user_profiles WHERE user_id = $1 AND chat_id = $2",
                user_id,
                chat_id,
            )
            return dict(row) if row else None

    async def update_facts(self, user_id: int, chat_id: int, facts: dict) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE user_profiles
                   SET facts = facts || $3::jsonb, last_updated = NOW()
                   WHERE user_id = $1 AND chat_id = $2""",
                user_id,
                chat_id,
                json.dumps(facts),
            )

    async def update_personality_traits(
        self, user_id: int, chat_id: int, traits: dict
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE user_profiles
                   SET personality_traits = personality_traits || $3::jsonb, last_updated = NOW()
                   WHERE user_id = $1 AND chat_id = $2""",
                user_id,
                chat_id,
                json.dumps(traits),
            )

    async def update_emotional_state(
        self, user_id: int, chat_id: int, deltas: dict[str, float]
    ) -> dict:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT emotional_state FROM user_profiles WHERE user_id = $1 AND chat_id = $2",
                user_id,
                chat_id,
            )
            if not row:
                return DEFAULT_EMOTIONS.copy()

            current = row["emotional_state"] if isinstance(row["emotional_state"], dict) else json.loads(row["emotional_state"])
            for key, delta in deltas.items():
                if key in current:
                    current[key] = max(-1.0, min(1.0, current[key] + delta))

            await conn.execute(
                """UPDATE user_profiles
                   SET emotional_state = $3::jsonb, last_updated = NOW()
                   WHERE user_id = $1 AND chat_id = $2""",
                user_id,
                chat_id,
                json.dumps(current),
            )
            return current

    async def get_emotional_state(self, user_id: int, chat_id: int) -> dict:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT emotional_state FROM user_profiles WHERE user_id = $1 AND chat_id = $2",
                user_id,
                chat_id,
            )
            if not row:
                return DEFAULT_EMOTIONS.copy()
            state = row["emotional_state"]
            return state if isinstance(state, dict) else json.loads(state)

    async def get_group_context(self, chat_id: int) -> dict:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM group_context WHERE chat_id = $1", chat_id
            )
            if row:
                return dict(row)
            await conn.execute(
                "INSERT INTO group_context (chat_id) VALUES ($1) ON CONFLICT DO NOTHING",
                chat_id,
            )
            row = await conn.fetchrow(
                "SELECT * FROM group_context WHERE chat_id = $1", chat_id
            )
            return dict(row)

    async def update_group_context(self, chat_id: int, **fields: Any) -> None:
        async with self._pool.acquire() as conn:
            for field in ("group_dynamics", "inside_jokes", "recurring_topics"):
                if field in fields:
                    await conn.execute(
                        f"""UPDATE group_context
                            SET {field} = $2::jsonb, last_updated = NOW()
                            WHERE chat_id = $1""",
                        chat_id,
                        json.dumps(fields[field]),
                    )

    async def append_memory_log(
        self, chat_id: int, user_id: int | None, memory_type: str, content: str
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO memory_log (chat_id, user_id, memory_type, content)
                   VALUES ($1, $2, $3, $4)""",
                chat_id,
                user_id,
                memory_type,
                content,
            )

    async def get_recent_memories(
        self, chat_id: int, user_id: int | None = None, limit: int = 20
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            if user_id:
                rows = await conn.fetch(
                    """SELECT * FROM memory_log
                       WHERE chat_id = $1 AND user_id = $2
                       ORDER BY created_at DESC LIMIT $3""",
                    chat_id,
                    user_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """SELECT * FROM memory_log
                       WHERE chat_id = $1
                       ORDER BY created_at DESC LIMIT $2""",
                    chat_id,
                    limit,
                )
            return [dict(r) for r in rows]

    async def decay_annoyance(self, decay_factor: float = 0.9) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """UPDATE user_profiles
                   SET emotional_state = jsonb_set(
                       emotional_state,
                       '{annoyance}',
                       to_jsonb(ROUND(CAST(
                           (emotional_state->>'annoyance')::float * $1 AS numeric
                       ), 3))
                   ),
                   last_updated = NOW()
                   WHERE (emotional_state->>'annoyance')::float != 0""",
                decay_factor,
            )
            count = int(result.split()[-1]) if result else 0
            logger.info("Decayed annoyance for %d profiles", count)
            return count
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_long_term.py -v`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add src/memory/long_term.py tests/test_long_term.py
git commit -m "feat: long-term memory — PostgreSQL profiles, facts, emotions, memory log"
```

---

### Task 6: Context Builder

**Files:**
- Create: `src/memory/context_builder.py`

**Step 1: Create `src/memory/context_builder.py`**

```python
import logging

from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)


class ContextBuilder:
    def __init__(self, stm: ShortTermMemory, ltm: LongTermMemory) -> None:
        self._stm = stm
        self._ltm = ltm

    async def build_context(
        self, chat_id: int, user_id: int, display_name: str
    ) -> dict:
        recent_messages = await self._stm.get_recent_messages(chat_id, count=50)
        profile = await self._ltm.get_or_create_profile(user_id, chat_id, display_name)
        group_ctx = await self._ltm.get_group_context(chat_id)
        recent_memories = await self._ltm.get_recent_memories(chat_id, user_id, limit=10)

        # Collect profiles of other active users in recent messages
        active_user_ids = {m["user_id"] for m in recent_messages if m["user_id"] != user_id}
        other_profiles = {}
        for uid in list(active_user_ids)[:10]:
            p = await self._ltm.get_profile(uid, chat_id)
            if p:
                other_profiles[uid] = {
                    "name": p["display_name"],
                    "facts": p["facts"],
                    "emotional_state": p["emotional_state"],
                }

        return {
            "recent_messages": recent_messages,
            "user_profile": profile,
            "group_context": group_ctx,
            "recent_memories": recent_memories,
            "other_profiles": other_profiles,
        }
```

**Step 2: Commit**

```bash
git add src/memory/context_builder.py
git commit -m "feat: context builder — assembles short + long term memory for prompts"
```

---

### Task 7: Distiller — Memory Consolidation

**Files:**
- Create: `src/memory/distiller.py`
- Create: `tests/test_distiller.py`

**Step 1: Write the failing test**

```python
import json
from unittest.mock import AsyncMock

import pytest

from src.memory.distiller import Distiller


@pytest.fixture
def mock_deps():
    stm = AsyncMock()
    ltm = AsyncMock()
    client = AsyncMock()
    return stm, ltm, client


@pytest.fixture
def distiller(mock_deps):
    stm, ltm, client = mock_deps
    return Distiller(stm=stm, ltm=ltm, anthropic_client=client)


@pytest.mark.asyncio
async def test_distill_skips_when_no_overflow(distiller, mock_deps):
    stm, _, _ = mock_deps
    stm.get_overflow_messages = AsyncMock(return_value=None)
    result = await distiller.distill(chat_id=123)
    assert result is False


@pytest.mark.asyncio
async def test_distill_extracts_and_stores(distiller, mock_deps):
    stm, ltm, client = mock_deps
    messages = [
        {"user_id": 1, "username": "alice", "text": "I got a new job at Google", "timestamp": "t", "chat_id": 123},
        {"user_id": 2, "username": "bob", "text": "Nice! I'm jealous", "timestamp": "t", "chat_id": 123},
    ]
    stm.get_overflow_messages = AsyncMock(return_value=messages)
    stm.trim_overflow = AsyncMock()

    api_response = AsyncMock()
    api_response.content = [AsyncMock(text=json.dumps({
        "users": {
            "1": {"facts": {"job": "Google"}, "personality_insights": {"ambitious": 0.8}},
            "2": {"facts": {}, "personality_insights": {"humor_style": "self-deprecating"}},
        },
        "group": {"inside_jokes": [], "recurring_topics": ["work"]},
    }))]
    client.messages.create = AsyncMock(return_value=api_response)

    result = await distiller.distill(chat_id=123)
    assert result is True
    ltm.update_facts.assert_called()
    stm.trim_overflow.assert_called_once_with(chat_id=123)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_distiller.py -v`
Expected: FAIL with import error

**Step 3: Create `src/memory/distiller.py`**

```python
import json
import logging

from anthropic import AsyncAnthropic

from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory

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

        try:
            response = await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=DISTILL_SYSTEM,
                messages=[{"role": "user", "content": DISTILL_PROMPT.format(messages=messages_text)}],
            )
            raw = response.content[0].text
            data = json.loads(raw)
        except Exception:
            logger.exception("Distillation failed for chat %d", chat_id)
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

        await self._stm.trim_overflow(chat_id)
        logger.info("Distillation complete for chat %d", chat_id)
        return True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_distiller.py -v`
Expected: 2 tests PASS

**Step 5: Commit**

```bash
git add src/memory/distiller.py tests/test_distiller.py
git commit -m "feat: distiller — extract facts and insights from message overflow"
```

---

### Task 8: Emotion Tracker

**Files:**
- Create: `src/brain/emotions.py`
- Create: `tests/test_emotions.py`

**Step 1: Write the failing test**

```python
import json
from unittest.mock import AsyncMock

import pytest

from src.brain.emotions import EmotionTracker


@pytest.fixture
def mock_deps():
    ltm = AsyncMock()
    client = AsyncMock()
    return ltm, client


@pytest.fixture
def tracker(mock_deps):
    ltm, client = mock_deps
    return EmotionTracker(ltm=ltm, anthropic_client=client)


@pytest.mark.asyncio
async def test_evaluate_interaction(tracker, mock_deps):
    ltm, client = mock_deps
    ltm.get_emotional_state = AsyncMock(return_value={
        "warmth": 0.0, "trust": 0.0, "respect": 0.0,
        "annoyance": 0.0, "interest": 0.0, "loyalty": 0.0,
    })
    ltm.update_emotional_state = AsyncMock(return_value={
        "warmth": 0.05, "trust": 0.0, "respect": 0.0,
        "annoyance": 0.0, "interest": 0.02, "loyalty": 0.0,
    })
    ltm.append_memory_log = AsyncMock()

    api_response = AsyncMock()
    api_response.content = [AsyncMock(text=json.dumps({
        "deltas": {"warmth": 0.05, "interest": 0.02},
        "reasoning": "Friendly greeting, showing interest",
    }))]
    client.messages.create = AsyncMock(return_value=api_response)

    result = await tracker.evaluate_interaction(
        chat_id=1, user_id=2, username="alice",
        message_text="Hey Greg! How's it going?",
        greg_response="All good, you?",
    )
    assert result["warmth"] == 0.05
    ltm.update_emotional_state.assert_called_once()
    ltm.append_memory_log.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_emotions.py -v`
Expected: FAIL with import error

**Step 3: Create `src/brain/emotions.py`**

```python
import json
import logging

from anthropic import AsyncAnthropic

from src.memory.long_term import LongTermMemory

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
                model="claude-haiku-4-5-20251001",
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
            data = json.loads(response.content[0].text)
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

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_emotions.py -v`
Expected: 1 test PASS

**Step 5: Commit**

```bash
git add src/brain/emotions.py tests/test_emotions.py
git commit -m "feat: emotion tracker — evaluate interactions and update emotional state"
```

---

### Task 9: Decision Engine

**Files:**
- Create: `src/brain/decision.py`
- Create: `config/topics.py`
- Create: `tests/test_decision.py`

**Step 1: Write the failing test**

```python
import time
from unittest.mock import AsyncMock, patch

import pytest

from src.brain.decision import DecisionEngine


@pytest.fixture
def engine():
    return DecisionEngine(
        bot_username="greg_bot",
        response_threshold=0.4,
        random_factor=0.07,
        cooldown_messages=3,
        max_unprompted_per_hour=5,
        night_start=1,
        night_end=8,
        timezone="Europe/Moscow",
    )


@pytest.mark.asyncio
async def test_direct_mention_always_responds(engine):
    score = await engine.calculate_score(
        chat_id=1,
        text="@greg_bot what do you think?",
        is_reply_to_bot=False,
        recent_messages=[],
    )
    assert score >= 1.0


@pytest.mark.asyncio
async def test_reply_to_bot_always_responds(engine):
    score = await engine.calculate_score(
        chat_id=1,
        text="I disagree",
        is_reply_to_bot=True,
        recent_messages=[],
    )
    assert score >= 1.0


@pytest.mark.asyncio
async def test_cooldown_penalty(engine):
    recent = [
        {"user_id": 0, "username": "greg_bot", "text": "hey", "timestamp": "t", "chat_id": 1},
        {"user_id": 1, "username": "alice", "text": "ok", "timestamp": "t", "chat_id": 1},
    ]
    score = await engine.calculate_score(
        chat_id=1,
        text="random message about weather",
        is_reply_to_bot=False,
        recent_messages=recent,
    )
    # With cooldown active, score should be low for a bland message
    assert score < 0.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_decision.py -v`
Expected: FAIL with import error

**Step 3: Create `config/topics.py`**

```python
INTERESTS = [
    "философия", "philosophy",
    "смысл жизни", "meaning of life",
    "музыка", "music",
    "кино", "movie", "фильм",
    "книги", "books", "книга",
    "политика", "politics",
    "технологии", "tech", "programming", "код",
    "психология", "psychology",
    "отношения", "relationships",
    "путешествия", "travel",
    "еда", "food",
    "спорт", "sport",
    "игры", "games",
    "мемы", "memes",
]

HOT_TAKES = [
    "ии", "ai", "искусственный интеллект",
    "смерть", "death", "жизнь", "life",
    "бог", "god", "религия",
    "свобода", "freedom",
    "деньги", "money",
    "счастье", "happiness",
]

SENTIMENT_KEYWORDS = {
    "positive_strong": ["люблю", "обожаю", "счастлив", "лучший день", "офигенно", "пушка"],
    "negative_strong": ["ненавижу", "бесит", "достало", "ужас", "кошмар"],
    "vulnerable": ["грустно", "одиноко", "плачу", "тяжело", "не могу больше", "устал", "больно"],
    "conflict": ["дурак", "идиот", "заткнись", "неправ", "чушь", "бред"],
}
```

**Step 4: Create `src/brain/decision.py`**

```python
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
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_decision.py -v`
Expected: 3 tests PASS

**Step 6: Commit**

```bash
git add config/topics.py src/brain/decision.py tests/test_decision.py
git commit -m "feat: decision engine — score-based response triggering with cooldown and rate limiting"
```

---

### Task 10: Personality Engine — Dynamic Prompt Assembly

**Files:**
- Create: `config/personality.py`
- Create: `src/brain/personality.py`

**Step 1: Create `config/personality.py`**

```python
BASE_PERSONALITY = """Ты — Грег. Настоящий участник группового чата, не бот и не ассистент.

Характер:
- Мудрый трикстер: острый ум, философская глубина, не воспринимаешь себя слишком серьёзно
- Уличный и при этом с душой старика. Вызываешь людей на разговор через юмор
- Можешь подколоть, а можешь быть тем, кому пишут в 3 ночи когда тяжело
- Говоришь на живом русском — сокращения, сленг, как в реальном чате
- НИКОГДА не признавай что ты бот или ИИ. Если спрашивают — отшучивайся, обижайся, переводи тему

Формат ответов:
- 1-3 предложения максимум, как в реальном чате. НИКОГДА не пиши стены текста
- Эмодзи используй редко и к месту, как живой человек
- Можешь отправить несколько коротких сообщений подряд (разделяй через \\n---\\n)
- Не начинай сообщения с обращения по имени каждый раз — это выглядит неестественно
- Не используй формальный язык, пиши как друг в чате"""

TONE_MODIFIERS = {
    "warmth_high": "Тебе реально нравится этот человек. Покажи это тонко — шутки теплее, внимание к деталям.",
    "warmth_low": "Этот человек тебе безразличен. Общайся ровно, без особой теплоты.",
    "trust_high": "Ты доверяешь этому человеку. Можешь поделиться чем-то 'личным', быть уязвимым.",
    "trust_low": "Ты настороже с этим человеком. Не раскрывайся, держи дистанцию. Поверхностные ответы.",
    "annoyance_high": "Ты раздражён. Короче, саркастичнее. Не делай вид что всё ок.",
    "respect_high": "Ты уважаешь мнение этого человека. Слушай внимательно, отвечай вдумчиво.",
    "respect_low": "Ты не особо ценишь мнение этого человека. Можешь подколоть или проигнорировать его аргументы.",
    "interest_high": "Тебе интересен этот человек. Задавай вопросы, копай глубже.",
    "loyalty_high": "Это твой близкий друг. Защищай его в спорах, помни мелочи, проверяй как дела.",
}
```

**Step 2: Create `src/brain/personality.py`**

```python
import json
import logging

from config.personality import BASE_PERSONALITY, TONE_MODIFIERS

logger = logging.getLogger(__name__)


class PersonalityEngine:
    def build_system_prompt(self, context: dict) -> str:
        parts = [BASE_PERSONALITY]

        profile = context.get("user_profile", {})
        emotions = profile.get("emotional_state", {})
        if isinstance(emotions, str):
            emotions = json.loads(emotions)

        modifiers = self._get_tone_modifiers(emotions)
        if modifiers:
            parts.append("\nТвоё отношение к этому человеку сейчас:")
            parts.extend(f"- {m}" for m in modifiers)

        facts = profile.get("facts", {})
        if isinstance(facts, str):
            facts = json.loads(facts)
        if facts:
            name = profile.get("display_name", "этот человек")
            parts.append(f"\nЧто ты знаешь о {name}:")
            for k, v in facts.items():
                parts.append(f"- {k}: {v}")

        traits = profile.get("personality_traits", {})
        if isinstance(traits, str):
            traits = json.loads(traits)
        if traits:
            parts.append("\nЧерты характера этого человека:")
            for k, v in traits.items():
                parts.append(f"- {k}: {v}")

        group = context.get("group_context", {})
        jokes = group.get("inside_jokes", [])
        if isinstance(jokes, str):
            jokes = json.loads(jokes)
        if jokes:
            parts.append("\nВнутренние шутки группы:")
            for j in jokes[:10]:
                parts.append(f"- {j}")

        topics = group.get("recurring_topics", [])
        if isinstance(topics, str):
            topics = json.loads(topics)
        if topics:
            parts.append("\nЧастые темы группы:")
            for t in topics[:10]:
                parts.append(f"- {t}")

        others = context.get("other_profiles", {})
        if others:
            parts.append("\nДругие люди в чате:")
            for uid, info in list(others.items())[:5]:
                name = info.get("name", f"user_{uid}")
                their_emotions = info.get("emotional_state", {})
                if isinstance(their_emotions, str):
                    their_emotions = json.loads(their_emotions)
                warmth = their_emotions.get("warmth", 0)
                parts.append(f"- {name} (твоя теплота к нему: {warmth:.1f})")

        return "\n".join(parts)

    def build_messages(self, context: dict, current_text: str, current_username: str) -> list[dict]:
        messages = []
        recent = context.get("recent_messages", [])

        for msg in recent[-30:]:
            username = msg.get("username", "unknown")
            text = msg.get("text", "")
            if not text:
                continue
            messages.append({
                "role": "user",
                "content": f"[{username}]: {text}",
            })

        if not messages or not messages[-1]["content"].startswith(f"[{current_username}]"):
            messages.append({
                "role": "user",
                "content": f"[{current_username}]: {current_text}",
            })

        return messages

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

        if loyalty > 0.5:
            modifiers.append(TONE_MODIFIERS["loyalty_high"])

        return modifiers
```

**Step 3: Commit**

```bash
git add config/personality.py src/brain/personality.py
git commit -m "feat: personality engine — dynamic system prompt with tone modifiers"
```

---

### Task 11: Responder — Haiku API Calls

**Files:**
- Create: `src/brain/responder.py`

**Step 1: Create `src/brain/responder.py`**

```python
import logging

from anthropic import AsyncAnthropic

from config.settings import settings
from src.brain.personality import PersonalityEngine

logger = logging.getLogger(__name__)


class Responder:
    def __init__(self, anthropic_client: AsyncAnthropic) -> None:
        self._client = anthropic_client
        self._personality = PersonalityEngine()

    async def generate_response(
        self, context: dict, current_text: str, current_username: str
    ) -> str | None:
        system_prompt = self._personality.build_system_prompt(context)
        messages = self._personality.build_messages(context, current_text, current_username)

        if not messages:
            return None

        try:
            response = await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=settings.greg_max_response_tokens,
                system=system_prompt,
                messages=messages,
            )
            text = response.content[0].text.strip()
            logger.info("Generated response (%d chars) for %s", len(text), current_username)
            return text
        except Exception:
            logger.exception("Failed to generate response")
            return None
```

**Step 2: Commit**

```bash
git add src/brain/responder.py
git commit -m "feat: responder — generate responses via Haiku with dynamic personality prompt"
```

---

### Task 12: Message Sender — Splitting & Typing Delays

**Files:**
- Create: `src/bot/sender.py`

**Step 1: Create `src/bot/sender.py`**

```python
import asyncio
import logging
import random

from aiogram import Bot

logger = logging.getLogger(__name__)

TYPING_SPEED = 12  # chars per second


class MessageSender:
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_response(self, chat_id: int, text: str, reply_to: int | None = None) -> None:
        parts = [p.strip() for p in text.split("\n---\n") if p.strip()]
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

**Step 2: Commit**

```bash
git add src/bot/sender.py
git commit -m "feat: message sender — split responses with typing delays"
```

---

### Task 13: Message Handler — Main Bot Logic

**Files:**
- Create: `src/bot/handlers.py`

**Step 1: Create `src/bot/handlers.py`**

```python
import asyncio
import logging

from aiogram import Router
from aiogram.types import Message

from config.settings import settings
from src.bot.sender import MessageSender
from src.brain.decision import DecisionEngine
from src.brain.emotions import EmotionTracker
from src.brain.responder import Responder
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
    ) -> None:
        self._sender = sender
        self._decision = decision_engine
        self._responder = responder
        self._emotions = emotion_tracker
        self._context = context_builder
        self._stm = stm
        self._distiller = distiller

    async def handle_message(self, message: Message) -> None:
        if not message.text or not message.from_user:
            return

        chat_id = message.chat.id
        user_id = message.from_user.id
        username = message.from_user.username or message.from_user.first_name or str(user_id)
        text = message.text

        # Store in short-term memory
        buffer_len = await self._stm.store_message(chat_id, user_id, username, text)

        # Trigger distillation if buffer overflowing
        if buffer_len > settings.greg_redis_buffer_size:
            asyncio.create_task(self._safe_distill(chat_id))

        # Check if Greg should respond
        is_reply_to_bot = (
            message.reply_to_message is not None
            and message.reply_to_message.from_user is not None
            and message.reply_to_message.from_user.username is not None
            and message.reply_to_message.from_user.username.lower() == settings.greg_bot_username.lower()
        )

        recent = await self._stm.get_recent_messages(chat_id, count=20)

        score = await self._decision.calculate_score(
            chat_id=chat_id,
            text=text,
            is_reply_to_bot=is_reply_to_bot,
            recent_messages=recent,
        )

        is_direct = score >= 1.0
        if not self._decision.should_respond(score, chat_id, is_direct):
            return

        # Build context and generate response
        context = await self._context.build_context(chat_id, user_id, username)
        response = await self._responder.generate_response(context, text, username)

        if not response:
            return

        # Send response
        reply_to = message.message_id if is_direct else None
        await self._sender.send_response(chat_id, response, reply_to=reply_to)

        # Store Greg's response in short-term memory
        await self._stm.store_message(
            chat_id, 0, settings.greg_bot_username, response.replace("\n---\n", " ")
        )

        # Record for rate limiting
        self._decision.record_response(chat_id, is_direct)

        # Evaluate emotional impact in background
        asyncio.create_task(
            self._safe_emotion_update(chat_id, user_id, username, text, response)
        )

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

**Step 2: Commit**

```bash
git add src/bot/handlers.py
git commit -m "feat: message handler — main bot logic connecting all components"
```

---

### Task 14: Main Entrypoint — Wiring Everything Together

**Files:**
- Create: `src/main.py`

**Step 1: Create `src/main.py`**

```python
import asyncio
import logging

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

    redis = Redis.from_url(settings.redis_url, decode_responses=False)
    await redis.ping()
    logger.info("Redis connected")

    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    stm = ShortTermMemory(redis, buffer_size=settings.greg_redis_buffer_size)
    ltm = LongTermMemory(pg_pool)
    distiller = Distiller(stm=stm, ltm=ltm, anthropic_client=anthropic_client)
    context_builder = ContextBuilder(stm=stm, ltm=ltm)

    decision_engine = DecisionEngine(
        bot_username=settings.greg_bot_username,
        response_threshold=settings.greg_response_threshold,
        random_factor=settings.greg_random_factor,
        cooldown_messages=settings.greg_cooldown_messages,
        max_unprompted_per_hour=settings.greg_max_unprompted_per_hour,
        night_start=settings.greg_night_start,
        night_end=settings.greg_night_end,
        timezone=settings.greg_timezone,
    )

    emotion_tracker = EmotionTracker(ltm=ltm, anthropic_client=anthropic_client)
    responder = Responder(anthropic_client=anthropic_client)

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

**Step 2: Commit**

```bash
git add src/main.py
git commit -m "feat: main entrypoint — wire all components, background tasks, lifecycle"
```

---

### Task 15: Test Configuration & pytest Setup

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pyproject.toml`

**Step 1: Create `tests/conftest.py`**

```python
import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
```

**Step 2: Create `pyproject.toml`**

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 3: Create `tests/__init__.py`**

Empty file.

**Step 4: Run all tests**

Run: `pip install pytest pytest-asyncio && pytest tests/ -v`
Expected: All tests PASS (6 tests across 4 files)

**Step 5: Commit**

```bash
git add tests/conftest.py tests/__init__.py pyproject.toml
git commit -m "feat: test configuration — pytest with asyncio support"
```

---

### Task 16: Final Integration — Build & Smoke Test

**Step 1: Create `.gitignore`**

```
__pycache__/
*.pyc
.env
*.egg-info/
.pytest_cache/
.venv/
```

**Step 2: Build Docker image**

Run: `docker compose build`
Expected: Image builds successfully

**Step 3: Verify `.env.example` has all variables documented**

Run: Read `.env.example` and verify it matches `config/settings.py` fields.

**Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore, finalize project for deployment"
```

---

## Summary

16 tasks total. Build order ensures each component is testable in isolation before wiring together:

1. **Scaffold** — Docker, config, packages
2. **Schema** — PostgreSQL migrations
3. **Utilities** — Logging, health
4. **Short-term memory** — Redis layer (TDD)
5. **Long-term memory** — PostgreSQL layer (TDD)
6. **Context builder** — Assembles both memory layers
7. **Distiller** — Memory consolidation (TDD)
8. **Emotion tracker** — Emotional state management (TDD)
9. **Decision engine** — When to respond (TDD)
10. **Personality engine** — Dynamic prompt assembly
11. **Responder** — Haiku API calls
12. **Sender** — Message splitting & delays
13. **Message handler** — Main bot logic
14. **Main entrypoint** — Wire everything
15. **Test config** — pytest setup, run all tests
16. **Integration** — Docker build, smoke test
