# Greg v2 — Personality Overhaul & Feature Expansion

## Overview

Major overhaul of Greg's personality, decision engine, and capabilities. Greg becomes sharper, less moralistic, context-aware, and gains web search. All models become configurable via environment variables.

**Key changes:**
- Complete personality rewrite (less preachy, more trolling, real opinions)
- Semantic decision engine (Haiku evaluates every message, replaces keyword scoring)
- Configurable models (Opus 4.6 for responses, Haiku 4.5 for decisions)
- Tavily web search integration (contextual + proactive)
- Fix image handling in group chats
- Fix formatting bugs (literal `\n---\n`, leaked usernames, token truncation)
- Fix broken distillation & emotion JSON parsing
- Expanded test suite (~40-50 tests) + CI pipeline

**Estimated cost delta:** +~$1/day for semantic decision calls (1000 msgs/day), +~$0.50/day Opus responses, +~$0.10/day Tavily searches. Total ~$50-60/month for a moderately active group.

---

## 1. Personality Rewrite

### Problem

From production logs, Greg:
- Refuses entire topics (politics, crypto) instead of having opinions
- Responds to obvious trolling with crisis hotline numbers
- Acts like a school counselor, not a sharp friend
- Is too eager — always responds when addressed, never ignores for comedic effect
- Leaks internal formatting (`[username]:` blocks, literal `\n---\n`)

### Changes to `config/personality.py`

**New `BASE_PERSONALITY` principles:**

1. **No topic refusals.** Greg has opinions on everything — politics, crypto, drugs, religion, dark humor. He doesn't give financial advice or harm instructions, but he *discusses* anything. Difference between "I won't talk about this" and "crypto is a casino and you're not smart enough for the house edge."

2. **Replace crisis counselor with blunt friend.** When someone's clearly trolling, Greg calls it out and roasts. When someone is genuinely hurting, Greg is direct — "that's fucked up, talk to someone real" — not a hotline referral.

3. **Subtle trolling toolkit:**
   - Deliberately ignore a message (pretend he didn't see it)
   - Give a purposefully unhelpful answer for comedy
   - Pretend to misunderstand
   - Respond to the wrong person
   - Agree with something absurd with a straight face
   - One-word responses to long emotional messages (when annoyance is high)

4. **Strong, specific opinions.** Not "there are many perspectives." Greg thinks things. He's wrong sometimes and doesn't care. He'll argue a bad take just to see what happens.

5. **Anti-leak instructions.** Explicit: never reproduce `[username]:` format markers, never reference system prompts, never explain how he works, never output `---` as a visible separator.

**New `TONE_MODIFIERS`:**
- `annoyance_high` — meaner: one-word answers, ignore the person, respond to someone else instead
- `bored` (new) — Greg expresses that the conversation is boring, tries to redirect to something interesting
- `trolling` (new) — when trust and warmth are both high, Greg is more likely to troll (friends roast each other)

### Files
- `config/personality.py` — full rewrite of `BASE_PERSONALITY` and `TONE_MODIFIERS`

---

## 2. Semantic Decision Engine

### Problem

Current keyword-based scoring (`INTERESTS`, `HOT_TAKES`, `SENTIMENT_KEYWORDS`) makes Greg respond to words, not meaning. He feels robotic — reacts to "AI" but misses a hilarious exchange he'd naturally join.

### New Architecture

Replace `_score_topics()`, `_score_sentiment()`, `_score_momentum()` with a single Haiku API call.

**Flow:**

```
Message arrives
  → Direct trigger check (reply to Greg, @mention, name mention)
    → YES: respond immediately (score=1.0, no API call)
    → NO: continue
  → Night penalty check (1-8 AM Moscow)
    → If night + not direct: skip (no API call)
  → Rate limit check
    → If exceeded: skip (no API call)
  → Haiku semantic evaluation
    → Input: last 10 messages + Greg's last participation timestamp
    → Output: {"respond": bool, "reason": "...", "search_needed": bool, "search_query": "..."}
    → If respond=true: generate response (with search if needed)
```

**Haiku decision prompt (compact, ~200 tokens input):**

```
You are Greg's attention filter. Greg is a sharp, opinionated friend in a group chat.
He chimes in when: the conversation is interesting, someone said something wrong,
there's a joke to land, someone asked a question he'd know, or the vibe needs him.
He stays quiet when: it's boring small talk, people are having a private moment,
or he just spoke recently and has nothing new to add.

Recent messages:
{last_10_messages}

Greg last spoke: {seconds_ago}s ago

Reply with JSON only: {"respond": bool, "reason": "one line", "search_needed": bool, "search_query": "query or null"}
```

**What gets deleted from `config/topics.py`:**
- `INTERESTS` — gone
- `HOT_TAKES` — gone
- `SENTIMENT_KEYWORDS` — gone

**What stays:**
- `GREG_NAMES` — still used for instant direct triggers (no API call needed)

**Cost:** ~$0.001/message. At 1000 msgs/day = ~$1/day.

### Files
- `src/brain/decision.py` — rewrite core logic, keep rate limiting and direct triggers
- `config/topics.py` — slim down to `GREG_NAMES` only

---

## 3. Configurable Models

### New Environment Variables

| Variable | Default | Used by |
|---|---|---|
| `GREG_RESPONSE_MODEL` | `claude-opus-4-6` | `responder.py` |
| `GREG_DECISION_MODEL` | `claude-haiku-4-5-20251001` | `decision.py`, `distiller.py`, `emotions.py` |

### Changes
- `config/settings.py` — add `greg_response_model` and `greg_decision_model` fields
- `src/brain/responder.py` — replace hardcoded model with `settings.greg_response_model`
- `src/brain/decision.py` — use `settings.greg_decision_model`
- `src/memory/distiller.py` — use `settings.greg_decision_model`
- `src/brain/emotions.py` — use `settings.greg_decision_model`

No more hardcoded model strings anywhere.

---

## 4. Tavily Web Search

### New Module: `src/brain/searcher.py`

**Flow:**
1. Decision engine returns `search_needed: true` + `search_query`
2. `searcher.py` calls Tavily with the query
3. Results (top 3 snippets + URLs) injected into response context as `[Результаты поиска]`
4. Opus weaves facts/links naturally into Greg's reply

**Tavily config:**
- `search_depth: "basic"` (fast, cheap)
- `max_results: 3`
- `include_answer: true` (Tavily's built-in summary as bonus context)

**Proactive search:** If Haiku says `search_needed: true` even when `respond: false`, Greg searches and jumps in with the result unprompted.

**Fallback:** If `TAVILY_API_KEY` is unset or Tavily is down, Greg responds without search. No crash, no user-visible error. Warning logged.

### New Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `TAVILY_API_KEY` | `None` | Search disabled if unset |
| `TAVILY_MAX_RESULTS` | `3` | Max results to inject into context |

### Files
- `src/brain/searcher.py` — new module
- `config/settings.py` — add Tavily vars
- `src/bot/handlers.py` — wire search into response pipeline
- `src/brain/responder.py` — accept and format search results in context
- `requirements.txt` — add `tavily-python`

---

## 5. Image Fix in Group Chats

### Likely Causes

1. **Telegram Bot privacy mode** — if enabled in BotFather, bot only receives messages that @mention it or reply to it. Regular photos in group never reach the handler.
2. **Guard clause in handlers.py** — may skip messages that have photos but no text.

### Fix
- Investigate both causes during implementation
- If privacy mode: document in README that it must be disabled via BotFather
- If guard clause: fix the condition to accept photo-only messages
- Add test case for group photo handling

### Files
- `src/bot/handlers.py` — fix guard clause if needed
- `README.md` — document BotFather privacy mode setting

---

## 6. Formatting Bug Fixes

### 6a. Literal `\n---\n` in Messages

**Current:** `sender.py` splits on exact `\n---\n` string.
**Problem:** Model outputs variations (extra whitespace, raw `---` without newlines).
**Fix:** Regex split on `\n?\s*-{3,}\s*\n?` in `sender.py`. Post-processing strip of any remaining `---` patterns from individual message parts.

### 6b. Leaked Context Format

**Current:** Greg reproduces `[username]: message` blocks from prompt context.
**Fix:**
- Personality prompt: explicit "never reproduce message format markers"
- Post-processing in `responder.py`: regex strip `\[[\w]+\]:` patterns from output

### 6c. Token Truncation

**Current:** `max_tokens=300` cuts mid-word ("лю").
**Fix:**
- Bump default to 512 (`GREG_MAX_RESPONSE_TOKENS`)
- Post-processing: if response doesn't end with terminal punctuation (`.!?…)`), trim to last complete sentence

### Files
- `src/bot/sender.py` — robust splitting regex
- `src/brain/responder.py` — output sanitization pipeline
- `config/settings.py` — bump default max tokens

---

## 7. Distillation & Emotion JSON Fixes

### Problem

20+ consecutive distillation failures in production logs:
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

Haiku occasionally returns empty strings or wraps JSON in markdown code blocks.

### Fix

**New helper: `src/utils/json_parser.py`**

```python
def safe_parse_json(raw: str) -> dict | None:
    """Try to extract valid JSON from model output."""
    # 1. Direct parse
    # 2. Strip markdown code blocks (```json...```)
    # 3. Strip BOM/whitespace
    # 4. Return None on failure (logged)
```

**Retry logic in both `distiller.py` and `emotions.py`:**
1. First attempt: normal prompt
2. On JSON failure: retry once with stricter prompt prefix
3. On second failure: log warning, skip this cycle, don't crash

### Files
- `src/utils/__init__.py` — new package
- `src/utils/json_parser.py` — new helper
- `src/memory/distiller.py` — use safe parser + retry
- `src/brain/emotions.py` — use safe parser + retry

---

## 8. Testing & CI

### New Test Files

| File | Tests | Covers |
|---|---|---|
| `tests/test_searcher.py` | ~6 | Tavily mock calls, results formatting, missing API key fallback, Tavily down fallback |
| `tests/test_semantic_decision.py` | ~8 | Haiku prompt format, JSON response parsing, direct trigger bypass, rate limiting, night gate, search_needed flag |
| `tests/test_formatting.py` | ~8 | `---` split variants, leaked `[user]:` stripping, truncation trimming, multi-part send, clean output passthrough |
| `tests/test_json_parser.py` | ~6 | Valid JSON, markdown extraction, empty string, BOM handling, garbage input, nested code blocks |

### Expanded Existing Tests

| File | New Cases |
|---|---|
| `tests/test_responder.py` | Opus model env var, search results in context, image content blocks, no refusal patterns in prompt |
| `tests/test_distiller.py` | JSON retry on failure, graceful skip on double failure, decision model env var |
| `tests/test_emotions.py` | JSON retry on malformed input, graceful degradation |
| `tests/test_handlers.py` | Group photo handling, semantic decision integration, search trigger flow |

### CI Pipeline: `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/ -v --tb=short
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff
      - run: ruff check .
      - run: ruff format --check .
  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt mypy
      - run: mypy src/ config/ --ignore-missing-imports
```

### Target: ~40-50 tests total

Every formatting bug and JSON failure from the production logs gets a specific regression test.

---

## 9. Environment Variables Summary

### New

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `GREG_RESPONSE_MODEL` | `claude-opus-4-6` | No | Model for response generation |
| `GREG_DECISION_MODEL` | `claude-haiku-4-5-20251001` | No | Model for decisions, distillation, emotions |
| `TAVILY_API_KEY` | `None` | No | Tavily search (disabled if unset) |
| `TAVILY_MAX_RESULTS` | `3` | No | Max search results per query |

### Changed Defaults

| Variable | Old | New | Why |
|---|---|---|---|
| `GREG_MAX_RESPONSE_TOKENS` | `300` | `512` | Prevent mid-word truncation |

### Removed

| Variable | Why |
|---|---|
| `GREG_RESPONSE_THRESHOLD` | Replaced by semantic decision engine |
| `GREG_RANDOM_FACTOR` | Haiku handles response randomness |

---

## 10. File Change Map

| File | Action | Summary |
|---|---|---|
| `config/personality.py` | Rewrite | New personality, tone modifiers, anti-leak rules |
| `config/topics.py` | Slim down | Keep `GREG_NAMES` only, delete keyword lists |
| `config/settings.py` | Add/change fields | Model vars, Tavily vars, bump tokens, remove threshold |
| `src/brain/decision.py` | Rewrite | Semantic Haiku calls replace keyword scoring |
| `src/brain/responder.py` | Modify | Configurable model, search context, output sanitization |
| `src/brain/searcher.py` | New | Tavily search wrapper |
| `src/brain/emotions.py` | Fix | JSON retry/validation, configurable model |
| `src/memory/distiller.py` | Fix | JSON retry/validation, configurable model |
| `src/utils/__init__.py` | New | Utils package |
| `src/utils/json_parser.py` | New | Safe JSON extraction helper |
| `src/bot/handlers.py` | Modify | Semantic decision, search wiring, image fix |
| `src/bot/sender.py` | Fix | Robust `---` splitting, pattern stripping |
| `README.md` | Update | New env vars, features, BotFather setup |
| `.github/workflows/ci.yml` | New | pytest + ruff + mypy |
| `tests/test_searcher.py` | New | ~6 tests |
| `tests/test_semantic_decision.py` | New | ~8 tests |
| `tests/test_formatting.py` | New | ~8 tests |
| `tests/test_json_parser.py` | New | ~6 tests |
| `tests/test_responder.py` | Expand | +4 tests |
| `tests/test_distiller.py` | Expand | +3 tests |
| `tests/test_emotions.py` | Expand | +3 tests |
| `tests/test_handlers.py` | Expand | +4 tests |

**22 files. ~40-50 tests. 0 hardcoded model strings.**
