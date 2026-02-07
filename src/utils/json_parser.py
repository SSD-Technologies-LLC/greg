import json
import logging
import re

logger = logging.getLogger(__name__)


def safe_parse_json(raw: str) -> dict | None:
    if not raw or not raw.strip():
        return None

    raw = raw.strip().lstrip("\ufeff")

    # Try direct parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try finding JSON object by braces
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("Failed to parse JSON from: %s", raw[:200])
    return None
