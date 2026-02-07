"""Tests for personality system — verifying new personality traits and modifiers."""

import pytest

from config.personality import BASE_PERSONALITY, TONE_MODIFIERS
from src.brain.personality import PersonalityEngine


@pytest.fixture
def engine():
    return PersonalityEngine()


class TestBasePersonality:
    def test_personality_mentions_greg(self):
        assert "Грег" in BASE_PERSONALITY

    def test_personality_no_topic_refusals(self):
        refusal_phrases = ["не обсуждай", "не говори о", "откажись", "избегай тем"]
        for phrase in refusal_phrases:
            assert phrase not in BASE_PERSONALITY.lower(), f"Found refusal phrase: {phrase}"

    def test_personality_has_trolling_instructions(self):
        lower = BASE_PERSONALITY.lower()
        assert any(word in lower for word in ["подкол", "тролл", "игнорир", "проигнорир"])

    def test_personality_forbids_format_leaks(self):
        lower = BASE_PERSONALITY.lower()
        assert "username" in lower or "[username]" in lower

    def test_personality_forbids_separator(self):
        assert "---" in BASE_PERSONALITY

    def test_personality_is_russian(self):
        cyrillic = sum(1 for c in BASE_PERSONALITY if "\u0400" <= c <= "\u04ff")
        assert cyrillic > 50


class TestToneModifiers:
    def test_annoyance_high_exists(self):
        assert "annoyance_high" in TONE_MODIFIERS

    def test_bored_modifier_exists(self):
        assert "bored" in TONE_MODIFIERS

    def test_trolling_modifier_exists(self):
        assert "trolling" in TONE_MODIFIERS

    def test_all_modifiers_are_russian(self):
        for key, value in TONE_MODIFIERS.items():
            cyrillic = sum(1 for c in value if "\u0400" <= c <= "\u04ff")
            assert cyrillic > 5, f"Modifier {key} should be in Russian"

    def test_original_modifiers_preserved(self):
        expected = [
            "warmth_high",
            "warmth_low",
            "trust_high",
            "trust_low",
            "annoyance_high",
            "respect_high",
            "respect_low",
            "interest_high",
            "loyalty_high",
        ]
        for key in expected:
            assert key in TONE_MODIFIERS


class TestBuildSystemPrompt:
    def test_includes_base_personality(self, engine):
        ctx = {"user_profile": {"emotional_state": {}}, "group_context": {}}
        prompt = engine.build_system_prompt(ctx)
        assert "Грег" in prompt

    def test_includes_tone_modifiers_when_triggered(self, engine):
        ctx = {
            "user_profile": {"emotional_state": {"annoyance": 0.6}},
            "group_context": {},
        }
        prompt = engine.build_system_prompt(ctx)
        assert TONE_MODIFIERS["annoyance_high"] in prompt

    def test_includes_bored_modifier(self, engine):
        ctx = {
            "user_profile": {"emotional_state": {"interest": -0.5}},
            "group_context": {},
        }
        prompt = engine.build_system_prompt(ctx)
        assert TONE_MODIFIERS["bored"] in prompt

    def test_includes_trolling_modifier(self, engine):
        ctx = {
            "user_profile": {"emotional_state": {"warmth": 0.6, "trust": 0.6}},
            "group_context": {},
        }
        prompt = engine.build_system_prompt(ctx)
        assert TONE_MODIFIERS["trolling"] in prompt

    def test_no_trolling_without_both_warmth_and_trust(self, engine):
        ctx = {
            "user_profile": {"emotional_state": {"warmth": 0.6, "trust": 0.1}},
            "group_context": {},
        }
        prompt = engine.build_system_prompt(ctx)
        assert TONE_MODIFIERS["trolling"] not in prompt


class TestBuildMessages:
    def test_empty_context_adds_current(self, engine):
        ctx = {"recent_messages": []}
        msgs = engine.build_messages(ctx, "hello", "alice")
        assert len(msgs) == 1
        assert "[alice]: hello" in msgs[0]["content"]

    def test_recent_messages_included(self, engine):
        ctx = {"recent_messages": [{"username": "bob", "text": "hey"}]}
        msgs = engine.build_messages(ctx, "hello", "alice")
        assert len(msgs) == 2

    def test_image_creates_content_blocks(self, engine):
        ctx = {"recent_messages": []}
        msgs = engine.build_messages(ctx, "[Фото] look", "alice", image_base64="AAAA")
        last = msgs[-1]
        assert isinstance(last["content"], list)
        assert last["content"][0]["type"] == "image"


class TestToneModifierLogic:
    def test_neutral_returns_empty(self, engine):
        modifiers = engine._get_tone_modifiers({})
        assert modifiers == []

    def test_all_high(self, engine):
        emotions = {
            "warmth": 0.8,
            "trust": 0.8,
            "annoyance": 0.6,
            "respect": 0.8,
            "interest": 0.8,
            "loyalty": 0.8,
        }
        modifiers = engine._get_tone_modifiers(emotions)
        assert TONE_MODIFIERS["warmth_high"] in modifiers
        assert TONE_MODIFIERS["trust_high"] in modifiers
        assert TONE_MODIFIERS["annoyance_high"] in modifiers
        assert TONE_MODIFIERS["respect_high"] in modifiers
        assert TONE_MODIFIERS["interest_high"] in modifiers
        assert TONE_MODIFIERS["loyalty_high"] in modifiers
        assert TONE_MODIFIERS["trolling"] in modifiers

    def test_bored_on_low_interest(self, engine):
        modifiers = engine._get_tone_modifiers({"interest": -0.5})
        assert TONE_MODIFIERS["bored"] in modifiers

    def test_bored_not_on_neutral_interest(self, engine):
        modifiers = engine._get_tone_modifiers({"interest": -0.2})
        assert TONE_MODIFIERS["bored"] not in modifiers
