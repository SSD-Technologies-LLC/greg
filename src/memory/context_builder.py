import logging

from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)


class ContextBuilder:
    def __init__(self, stm: ShortTermMemory, ltm: LongTermMemory) -> None:
        self._stm = stm
        self._ltm = ltm

    async def build_context(self, chat_id: int, user_id: int, display_name: str) -> dict:
        recent_messages = await self._stm.get_recent_messages(chat_id, count=50)
        profile = await self._ltm.get_or_create_profile(user_id, chat_id, display_name)
        group_ctx = await self._ltm.get_group_context(chat_id)
        recent_memories = await self._ltm.get_recent_memories(chat_id, user_id, limit=10)

        # Collect profiles of other active users in recent messages
        active_user_ids = {m["user_id"] for m in recent_messages if m["user_id"] != user_id}
        other_profiles = {}
        for uid in list(active_user_ids)[:10]:
            p = await self._ltm.get_profile(uid, chat_id)
            if p:
                other_profiles[uid] = {
                    "name": p["display_name"],
                    "facts": p["facts"],
                    "emotional_state": p["emotional_state"],
                }

        return {
            "recent_messages": recent_messages,
            "user_profile": profile,
            "group_context": group_ctx,
            "recent_memories": recent_memories,
            "other_profiles": other_profiles,
        }
