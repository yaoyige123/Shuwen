"""ARQ worker（P2 queue 模式）。

启动：
    python -m arq app.worker.WorkerSettings

它和 API 进程共享同一个 Redis 任务存储：API 只负责落任务 + 投递，
worker 负责真正调用 LangGraph。这样推理负载可以独立于 HTTP 层扩容。
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis
from arq.connections import RedisSettings

from app.config import settings
from app.runner import run_turn
from app.store_redis import RedisTaskStore

logger = logging.getLogger(__name__)


async def startup(ctx: dict) -> None:
    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    ctx["store"] = RedisTaskStore(client)
    logger.info("worker started (redis=%s)", settings.redis_url)


async def shutdown(ctx: dict) -> None:
    store: RedisTaskStore | None = ctx.get("store")
    if store is not None:
        await store.close()
    logger.info("worker stopped")


async def run_turn_job(
    ctx: dict, task_id: str, session_id: str, message: str
) -> str:
    store: RedisTaskStore = ctx["store"]
    logger.info("job start: task=%s session=%s", task_id, session_id)
    record = await run_turn(store, task_id, session_id, message)
    status = record.status.value if record is not None else "unknown"
    logger.info("job done: task=%s status=%s", task_id, status)
    return status


class WorkerSettings:
    functions = [run_turn_job]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)

    # 并发度：LangGraph 是同步阻塞调用，靠线程池并行，这里控制同时执行的任务数
    max_jobs = 4
    job_timeout = settings.turn_timeout_seconds + 60
    # 重试次数：任务状态机本身保证幂等，默认不自动重试，避免重复扣费
    max_tries = 1
    keep_result = 3600
