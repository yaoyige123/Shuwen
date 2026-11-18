"""FastAPI 应用工厂与入口。

启动：
    python -m app.main            # 默认 0.0.0.0:8000
    uvicorn app.main:app --reload

工厂模式的两个好处：
    1. 测试可以 ``create_app(store=FakeStore(), turn_runner=fake)`` 注入替身；
    2. 真正的共享资源（连接池、Redis、MQ）后续统一挂在 lifespan 上，
       而不是散落在模块级全局变量里。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.deps import TurnRunner
from app.api.routes import VERSION, router
from app.config import settings
from app.dispatch import ArqDispatcher, Dispatcher, InlineDispatcher
from app.metrics import metrics_response
from app.observability import ObservabilityMiddleware, configure_logging
from app.ratelimit import MemoryRateLimiter, RateLimiter, RedisRateLimiter
from app.runner import run_turn
from app.tasks import InMemoryTaskStore, InvalidTransition, TaskStore

logger = logging.getLogger(__name__)

# 自带的极简聊天页（纯静态，无构建步骤）
STATIC_DIR = Path(__file__).resolve().parent / "static"


async def _reaper_loop(store: TaskStore) -> None:
    """后台回收租约过期的任务（worker 崩溃兜底）。"""
    interval = max(1, settings.reaper_interval_seconds)
    while True:
        try:
            await asyncio.sleep(interval)
            reaped = await store.reap_expired()
            if reaped:
                logger.warning(
                    "reaped %d expired task(s): %s", len(reaped), reaped[:5]
                )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - 回收失败不应拖垮 API
            logger.exception("reaper tick failed")


def _build_store() -> TaskStore:
    """按 TASK_BACKEND 选择存储实现。"""
    if settings.task_backend == "redis":
        import redis.asyncio as aioredis

        from app.store_redis import RedisTaskStore

        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        return RedisTaskStore(client)
    return InMemoryTaskStore()


def _build_rate_limiter() -> RateLimiter:
    """按 RATE_LIMIT_BACKEND 选择限流实现。"""
    if settings.rate_limit_backend == "redis":
        import redis.asyncio as aioredis

        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        return RedisRateLimiter(
            client,
            limit=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
    return MemoryRateLimiter(
        limit=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )


def create_app(
    task_store: TaskStore | None = None,
    turn_runner: TurnRunner | None = None,
    dispatcher: Dispatcher | None = None,
    rate_limiter: RateLimiter | None = None,
) -> FastAPI:
    configure_logging()
    store: TaskStore = task_store if task_store is not None else _build_store()
    runner: TurnRunner = turn_runner if turn_runner is not None else run_turn
    disp: Dispatcher = (
        dispatcher if dispatcher is not None else InlineDispatcher(store, runner)
    )
    limiter: RateLimiter = (
        rate_limiter if rate_limiter is not None else _build_rate_limiter()
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.task_store = store
        app.state.turn_runner = runner
        app.state.dispatcher = disp
        app.state.rate_limiter = limiter
        if not settings.api_key:
            logger.warning("未配置 API_KEY，接口鉴权已关闭（仅建议开发环境）")
        logger.info(
            "CMeKG Agent API %s starting (model=%s, backend=%s, exec=%s, timeout=%ss)",
            VERSION,
            settings.chat_model,
            settings.task_backend,
            settings.execution_mode,
            settings.turn_timeout_seconds,
        )

        pool = None
        reaper = None
        if settings.reaper_interval_seconds > 0:
            reaper = asyncio.create_task(_reaper_loop(store), name="task-reaper")
            logger.info("task reaper started (every %ss)", settings.reaper_interval_seconds)

        if settings.execution_mode == "queue" and not isinstance(disp, ArqDispatcher):
            if settings.task_backend != "redis":
                logger.error(
                    "EXECUTION_MODE=queue 需要 TASK_BACKEND=redis，已降级为 inline"
                )
            else:
                try:
                    from arq import create_pool
                    from arq.connections import RedisSettings

                    pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
                    app.state.dispatcher = ArqDispatcher(pool)
                    logger.info("ARQ dispatcher ready (%s)", settings.redis_url)
                except Exception:  # noqa: BLE001 - 队列不可用时降级而非崩溃
                    logger.exception("创建 ARQ 连接池失败，降级为 inline 执行")

        try:
            yield
        finally:
            # 先把在途任务排空，再关存储：否则后台协程会在连接关闭后继续写入
            active = app.state.dispatcher
            if isinstance(active, InlineDispatcher):
                leftover = await active.drain(timeout=5.0)
                if leftover:
                    logger.warning(
                        "shutdown: %d task(s) still running after drain timeout", leftover
                    )
            if reaper is not None:
                reaper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reaper
            if pool is not None:
                await pool.aclose()
            with contextlib.suppress(Exception):
                await limiter.close()
            closer = getattr(store, "close", None)
            if closer is not None:
                await closer()
            logger.info("CMeKG Agent API stopping")

    app = FastAPI(
        title="CMeKG Agent API",
        version=VERSION,
        description="麻醉术前访视 Graph RAG 助手的异步任务接口（P1/P2）",
        lifespan=lifespan,
    )

    # 依赖注入的取值来源（lifespan 里可能被替换，例如 queue 模式换成 ARQ 调度器）
    app.state.task_store = store
    app.state.turn_runner = runner
    app.state.dispatcher = disp
    app.state.rate_limiter = limiter

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 后添加 = 更外层：trace_id 与 HTTP 指标需要包住所有路由
    app.add_middleware(ObservabilityMiddleware)

    @app.get("/metrics", include_in_schema=False)
    async def _metrics():  # pragma: no cover - 只有一行转发
        return metrics_response()

    @app.get("/", include_in_schema=False)
    async def _root() -> RedirectResponse:  # pragma: no cover - 一行跳转
        """根路径直接进聊天页。"""
        return RedirectResponse(url="/chat")

    @app.get("/chat", include_in_schema=False)
    async def _chat_page() -> FileResponse:  # pragma: no cover - 静态文件
        """极简聊天页：提交消息 + SSE 订阅进度。"""
        return FileResponse(STATIC_DIR / "chat.html", media_type="text/html")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.exception_handler(InvalidTransition)
    async def _handle_invalid_transition(  # pragma: no cover - 防御性分支
        request: Request, exc: InvalidTransition
    ) -> JSONResponse:
        logger.error("illegal transition: %s", exc)
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")
