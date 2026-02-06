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
