import os

# Set required env vars before any source imports trigger Settings()
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("GREG_BOT_USERNAME", "greg_bot")
os.environ.setdefault("GREG_RESPONSE_MODEL", "claude-opus-4-6")
os.environ.setdefault("GREG_DECISION_MODEL", "claude-haiku-4-5-20251001")

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
