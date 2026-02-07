"""Tests for PersonalityEngine — system prompt building and multimodal messages."""

import json

import pytest

from src.brain.personality import PersonalityEngine


@pytest.fixture
def engine():
    return PersonalityEngine()


# ---------------------------------------------------------------------------
# build_messages — text only
# ---------------------------------------------------------------------------

class TestBuildMessagesText:

    def test_empty_context_adds_current(self, engine):
        msgs = engine.build_messages(
            context={"recent_messages": []},
            current_text="hello",
            current_username="alice",
        )
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "[alice]: hello"

    def test_recent_messages_included(self, engine):
        recent = [
            {"username": "bob", "text": "hi"},
            {"username": "alice", "text": "hey"},
        ]
        msgs = engine.build_messages(
            context={"recent_messages": recent},
            current_text="what's up",
            current_username="carol",
        )
        assert len(msgs) == 3
        assert msgs[0]["content"] == "[bob]: hi"
        assert msgs[1]["content"] == "[alice]: hey"
        assert msgs[2]["content"] == "[carol]: what's up"

    def test_dedup_current_user_last_message(self, engine):
        """If the last recent message is from the current user, don't duplicate."""
        recent = [
            {"username": "bob", "text": "hi"},
            {"username": "alice", "text": "hello"},
        ]
        msgs = engine.build_messages(
            context={"recent_messages": recent},
            current_text="hello",
            current_username="alice",
        )
        # Should NOT add a duplicate — last message already starts with [alice]
        assert len(msgs) == 2

    def test_skips_empty_text(self, engine):
        recent = [
            {"username": "bob", "text": ""},
            {"username": "alice", "text": "hi"},
        ]
        msgs = engine.build_messages(
            context={"recent_messages": recent},
            current_text="yo",
            current_username="carol",
        )
        # bob's empty message should be skipped
        assert len(msgs) == 2
        assert msgs[0]["content"] == "[alice]: hi"

    def test_limits_to_30_messages(self, engine):
        recent = [{"username": f"u{i}", "text": f"msg{i}"} for i in range(50)]
        msgs = engine.build_messages(
            context={"recent_messages": recent},
            current_text="latest",
            current_username="me",
        )
        # 30 from recent + 1 current
        assert len(msgs) == 31


# ---------------------------------------------------------------------------
# build_messages — multimodal (image)
# ---------------------------------------------------------------------------

class TestBuildMessagesMultimodal:

    def test_image_creates_content_blocks(self, engine):
        msgs = engine.build_messages(
            context={"recent_messages": []},
            current_text="[Фото] look at this",
            current_username="alice",
            image_base64="AAAA",
        )
        assert len(msgs) == 1
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "image"
        assert content[0]["source"]["type"] == "base64"
        assert content[0]["source"]["media_type"] == "image/jpeg"
        assert content[0]["source"]["data"] == "AAAA"
        assert content[1]["type"] == "text"
        assert "[alice]:" in content[1]["text"]

    def test_image_replaces_existing_last_message(self, engine):
        """When last recent msg is from current user, replace it with multimodal."""
        recent = [
            {"username": "alice", "text": "[Фото] my cat"},
        ]
        msgs = engine.build_messages(
            context={"recent_messages": recent},
            current_text="[Фото] my cat",
            current_username="alice",
            image_base64="BBBB",
        )
        # Should NOT duplicate — replaces in-place
        assert len(msgs) == 1
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert content[0]["source"]["data"] == "BBBB"

    def test_no_image_keeps_text(self, engine):
        msgs = engine.build_messages(
            context={"recent_messages": []},
            current_text="just text",
            current_username="bob",
            image_base64=None,
        )
        assert isinstance(msgs[0]["content"], str)

    def test_history_messages_stay_text_only(self, engine):
        recent = [
            {"username": "bob", "text": "[Фото] old photo"},
            {"username": "carol", "text": "nice"},
        ]
        msgs = engine.build_messages(
            context={"recent_messages": recent},
            current_text="[Фото] new photo",
            current_username="dave",
            image_base64="CCCC",
        )
        # History messages should be plain strings
        assert isinstance(msgs[0]["content"], str)
        assert isinstance(msgs[1]["content"], str)
        # Only the last (current) should be multimodal
        assert isinstance(msgs[2]["content"], list)


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:

    def test_base_personality_always_present(self, engine):
        prompt = engine.build_system_prompt({"user_profile": {}, "group_context": {}})
        assert "Грег" in prompt

    def test_emotion_modifiers_applied(self, engine):
        ctx = {
            "user_profile": {
                "emotional_state": {"warmth": 0.8, "trust": 0.0, "respect": 0.0,
                                    "annoyance": 0.0, "interest": 0.0, "loyalty": 0.0},
            },
        }
        prompt = engine.build_system_prompt(ctx)
        assert "нравится" in prompt  # warmth_high modifier

    def test_emotion_modifiers_negative(self, engine):
        ctx = {
            "user_profile": {
                "emotional_state": {"warmth": -0.5, "trust": -0.5, "respect": -0.5,
                                    "annoyance": 0.6, "interest": 0.0, "loyalty": 0.0},
            },
        }
        prompt = engine.build_system_prompt(ctx)
        assert "безразличен" in prompt or "раздражён" in prompt

    def test_facts_included(self, engine):
        ctx = {
            "user_profile": {
                "display_name": "Алиса",
                "facts": {"работа": "программист"},
                "emotional_state": {},
            },
        }
        prompt = engine.build_system_prompt(ctx)
        assert "программист" in prompt
        assert "Алиса" in prompt

    def test_personality_traits_included(self, engine):
        ctx = {
            "user_profile": {
                "personality_traits": {"humor_style": "сарказм"},
                "emotional_state": {},
            },
        }
        prompt = engine.build_system_prompt(ctx)
        assert "сарказм" in prompt

    def test_inside_jokes(self, engine):
        ctx = {
            "user_profile": {"emotional_state": {}},
            "group_context": {"inside_jokes": ["кот Барсик"]},
        }
        prompt = engine.build_system_prompt(ctx)
        assert "кот Барсик" in prompt

    def test_recurring_topics(self, engine):
        ctx = {
            "user_profile": {"emotional_state": {}},
            "group_context": {"recurring_topics": ["Тема работы"]},
        }
        prompt = engine.build_system_prompt(ctx)
        assert "Тема работы" in prompt

    def test_other_profiles(self, engine):
        ctx = {
            "user_profile": {"emotional_state": {}},
            "other_profiles": {
                123: {"name": "Боб", "emotional_state": {"warmth": 0.3}},
            },
        }
        prompt = engine.build_system_prompt(ctx)
        assert "Боб" in prompt

    def test_json_string_emotions_parsed(self, engine):
        """Emotions stored as JSON string in DB should be parsed correctly."""
        ctx = {
            "user_profile": {
                "emotional_state": json.dumps({
                    "warmth": 0.8, "trust": 0.0, "respect": 0.0,
                    "annoyance": 0.0, "interest": 0.0, "loyalty": 0.0,
                }),
            },
        }
        prompt = engine.build_system_prompt(ctx)
        assert "нравится" in prompt

    def test_json_string_facts_parsed(self, engine):
        ctx = {
            "user_profile": {
                "facts": json.dumps({"job": "dev"}),
                "emotional_state": {},
            },
        }
        prompt = engine.build_system_prompt(ctx)
        assert "dev" in prompt


# ---------------------------------------------------------------------------
# _get_tone_modifiers
# ---------------------------------------------------------------------------

class TestToneModifiers:

    def test_all_high(self, engine):
        emotions = {
            "warmth": 0.8, "trust": 0.8, "respect": 0.8,
            "annoyance": 0.6, "interest": 0.8, "loyalty": 0.8,
        }
        mods = engine._get_tone_modifiers(emotions)
        assert len(mods) == 6  # all 6 high modifiers

    def test_all_low(self, engine):
        emotions = {
            "warmth": -0.5, "trust": -0.5, "respect": -0.5,
            "annoyance": 0.0, "interest": 0.0, "loyalty": 0.0,
        }
        mods = engine._get_tone_modifiers(emotions)
        assert len(mods) == 3  # warmth_low, trust_low, respect_low

    def test_neutral_returns_empty(self, engine):
        emotions = {
            "warmth": 0.0, "trust": 0.0, "respect": 0.0,
            "annoyance": 0.0, "interest": 0.0, "loyalty": 0.0,
        }
        mods = engine._get_tone_modifiers(emotions)
        assert len(mods) == 0

    def test_missing_keys_default_zero(self, engine):
        mods = engine._get_tone_modifiers({})
        assert len(mods) == 0
