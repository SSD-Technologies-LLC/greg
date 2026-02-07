# Voice Message Transcription — Design

## Summary

Add voice message transcription via OpenAI Whisper API so Greg can "hear" and respond to voice messages naturally. Transcription is internal only — users see Greg's reply, not the transcribed text. Optional dependency: works without `OPENAI_API_KEY`, degrades to current `[Голосовое сообщение]` label.

## Component: VoiceTranscriber

New file: `src/brain/transcriber.py`

```python
class VoiceTranscriber:
    def __init__(self, client: AsyncOpenAI | None) -> None: ...
    async def transcribe(self, audio_data: bytes) -> str | None: ...
```

- Constructor takes optional `openai.AsyncOpenAI` client. If `None`, all calls return `None`.
- `transcribe()` sends audio as in-memory BytesIO file to Whisper API (`model="whisper-1"`, `language="ru"`).
- Returns transcribed text on success, `None` on any error.
- Errors logged with `logger.exception()`, never raised.
- No retry logic — single voice message isn't worth retrying.
- No file length cap — Telegram's 20MB limit is the natural constraint.

## Configuration

New setting in `config/settings.py`:

```python
openai_api_key: str | None = None
```

## Wiring (src/main.py)

- If `openai_api_key` is set: create `openai.AsyncOpenAI(api_key=...)`, pass to `VoiceTranscriber`.
- If unset: `VoiceTranscriber(client=None)` — transcription disabled.
- Inject `VoiceTranscriber` into `MessageHandler`.

## Handler Integration

In `src/bot/handlers.py`, `_extract_media` voice case changes:

1. Download voice file bytes via `message.bot.download()`.
2. Call `self._transcriber.transcribe(audio_bytes)` if transcriber is available.
3. Success: `display_text = f"[Голосовое сообщение] {transcription}"`.
4. Failure or no transcriber: `display_text = "[Голосовое сообщение]"` (current behavior).

`MessageHandler.__init__` gets new optional param: `transcriber: VoiceTranscriber | None = None`.

Transcription runs inline (not background) because Greg needs the text before deciding whether to respond. Whisper API latency (~1-2s) is acceptable.

## Data Flow

No downstream changes needed. Transcribed text is embedded in `display_text`, which flows through:
- Redis storage (STM)
- Decision engine (recent messages for semantic evaluation)
- Responder (context for response generation)

## Error Handling

| Scenario | Behavior |
|---|---|
| No `OPENAI_API_KEY` | `VoiceTranscriber(client=None)`, returns `None` always |
| Whisper API error | Log exception, return `None`, use label fallback |
| Download fails | Skip transcription, use label fallback |
| Empty audio bytes | Skip transcription, use label fallback |
| Whisper returns empty string | Treat as failure, use label fallback |

## Testing

New file: `tests/test_transcriber.py`

- `TestTranscribe`: success, API error, empty audio, empty result
- `TestDisabled`: `VoiceTranscriber(client=None)` returns `None`

Updates to `tests/test_handlers.py`:

- Handler deps fixture becomes 9-tuple (add `transcriber`)
- `TestVoiceMessages`: voice with transcription, voice with failed transcription, voice with no transcriber

## Files Changed

| File | Change |
|---|---|
| `src/brain/transcriber.py` | New — `VoiceTranscriber` class |
| `config/settings.py` | Add `openai_api_key` |
| `src/main.py` | Wire OpenAI client + transcriber |
| `src/bot/handlers.py` | Download voice, call transcriber |
| `tests/test_transcriber.py` | New — transcriber unit tests |
| `tests/test_handlers.py` | Update deps tuple, add voice transcription tests |

## Dependencies

New pip dependency: `openai` (async client).
