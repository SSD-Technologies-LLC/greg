import os

# Set required env vars before any source imports trigger Settings()
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("GREG_BOT_USERNAME", "greg_bot")

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
