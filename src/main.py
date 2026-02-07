import asyncio
import logging
from pathlib import Path

import asyncpg
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from anthropic import AsyncAnthropic
from redis.asyncio import Redis

from config.settings import settings
from src.bot.handlers import MessageHandler, router
from src.bot.sender import MessageSender
from src.brain.decision import DecisionEngine
from src.brain.emotions import EmotionTracker
from src.brain.responder import Responder
from src.memory.context_builder import ContextBuilder
from src.memory.distiller import Distiller
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory
from src.utils.health import set_health, start_health_server
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


async def main() -> None:
    setup_logging()
    logger.info("Starting Greg...")

    health_runner = await start_health_server()

    pg_pool = await asyncpg.create_pool(
        dsn=settings.postgres_dsn,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    logger.info("PostgreSQL connected")

    # Auto-run migrations (Railway doesn't use docker-entrypoint-initdb.d)
    migration = Path(__file__).resolve().parent.parent / "migrations" / "001_initial.sql"
    if migration.exists():
        async with pg_pool.acquire() as conn:
            await conn.execute(migration.read_text())
        logger.info("Migrations applied")

    redis = Redis.from_url(settings.redis_dsn, decode_responses=False)
    await redis.ping()
    logger.info("Redis connected")

    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    stm = ShortTermMemory(redis, buffer_size=settings.greg_redis_buffer_size)
    ltm = LongTermMemory(pg_pool)
    distiller = Distiller(stm=stm, ltm=ltm, anthropic_client=anthropic_client)
    context_builder = ContextBuilder(stm=stm, ltm=ltm)

    decision_engine = DecisionEngine(
        bot_username=settings.greg_bot_username,
        response_threshold=settings.greg_response_threshold,
        random_factor=settings.greg_random_factor,
        cooldown_messages=settings.greg_cooldown_messages,
        max_unprompted_per_hour=settings.greg_max_unprompted_per_hour,
        night_start=settings.greg_night_start,
        night_end=settings.greg_night_end,
        timezone=settings.greg_timezone,
    )

    emotion_tracker = EmotionTracker(ltm=ltm, anthropic_client=anthropic_client)
    responder = Responder(anthropic_client=anthropic_client)

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    sender = MessageSender(bot)

    handler = MessageHandler(
        sender=sender,
        decision_engine=decision_engine,
        responder=responder,
        emotion_tracker=emotion_tracker,
        context_builder=context_builder,
        stm=stm,
        distiller=distiller,
    )

    @router.message()
    async def on_message(message):
        await handler.handle_message(message)

    dp = Dispatcher()
    dp.include_router(router)

    async def periodic_distillation():
        while True:
            await asyncio.sleep(settings.greg_distill_every_minutes * 60)
            try:
                async with pg_pool.acquire() as conn:
                    chats = await conn.fetch("SELECT DISTINCT chat_id FROM group_context")
                for row in chats:
                    await distiller.distill(row["chat_id"])
            except Exception:
                logger.exception("Periodic distillation failed")

    async def daily_decay():
        while True:
            await asyncio.sleep(24 * 3600)
            try:
                await ltm.decay_annoyance(decay_factor=0.9)
            except Exception:
                logger.exception("Daily decay failed")

    @dp.startup()
    async def on_startup():
        asyncio.create_task(periodic_distillation())
        asyncio.create_task(daily_decay())
        set_health("healthy")
        logger.info("Greg is online!")

    @dp.shutdown()
    async def on_shutdown():
        set_health("shutting_down")
        await redis.aclose()
        await pg_pool.close()
        await health_runner.cleanup()
        logger.info("Greg is offline.")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
