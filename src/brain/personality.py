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

    def build_messages(
        self,
        context: dict,
        current_text: str,
        current_username: str,
        *,
        image_base64: str | None = None,
    ) -> list[dict]:
        messages = []
        recent = context.get("recent_messages", [])

        for msg in recent[-30:]:
            username = msg.get("username", "unknown")
            text = msg.get("text", "")
            if not text:
                continue
            messages.append(
                {
                    "role": "user",
                    "content": f"[{username}]: {text}",
                }
            )

        text_content = f"[{current_username}]: {current_text}"

        if not messages or not messages[-1]["content"].startswith(f"[{current_username}]"):
            if image_base64:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": text_content},
                        ],
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": text_content,
                    }
                )
        elif image_base64:
            # Last message already matches current user — replace with multimodal
            messages[-1] = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64,
                        },
                    },
                    {"type": "text", "text": messages[-1]["content"]},
                ],
            }

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
        elif interest < -0.4:
            modifiers.append(TONE_MODIFIERS["bored"])

        if loyalty > 0.5:
            modifiers.append(TONE_MODIFIERS["loyalty_high"])

        if warmth > 0.4 and trust > 0.4:
            modifiers.append(TONE_MODIFIERS["trolling"])

        return modifiers
