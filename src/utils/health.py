import logging

from aiohttp import web

from config.settings import settings

logger = logging.getLogger(__name__)

_health_status: dict = {"status": "starting"}


def set_health(status: str) -> None:
    _health_status["status"] = status


async def _handle_health(request: web.Request) -> web.Response:
    return web.json_response(_health_status)


async def start_health_server() -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/health", _handle_health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", settings.greg_health_port)
    await site.start()
    logger.info("Health endpoint started on port %s", settings.greg_health_port)
    return runner
