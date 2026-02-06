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
