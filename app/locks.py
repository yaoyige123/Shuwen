"""会话级锁（P2+）。

为什么需要：LangGraph 的 checkpoint 是「每个 thread_id 一条时间线」，
同一会话并发跑两轮会互相覆盖状态。单进程内用 asyncio.Lock 就够，
但多副本部署时请求会落到不同实例，必须用 Redis 分布式锁。

两种实现，同一套 `async with session_lock(sid)` 接口：
    memory —— 进程内 asyncio.Lock（默认，单机开发）
    redis  —— SET NX PX 抢占 + 乐观删除（多副本）

拿不到锁时抛 ``SessionBusy``，任务直接判失败而不是排队到天荒地老——
对医疗问诊来说，"明确失败"比"半小时后突然继续上一轮"更安全。
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.config import settings

_local_locks: dict[str, asyncio.Lock] = {}
_redis_client = None


class SessionBusy(RuntimeError):
    """会话正被另一轮对话占用。"""


def _get_redis():
    global _redis_client
    if _redis_client is None:
        import redis.asyncio as aioredis

        _redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


@asynccontextmanager
async def _local_lock(session_id: str) -> AsyncIterator[None]:
    lock = _local_locks.setdefault(session_id, asyncio.Lock())
    try:
        await asyncio.wait_for(
            lock.acquire(), timeout=settings.session_lock_timeout_seconds
        )
    except TimeoutError:
        raise SessionBusy(f"session {session_id} is busy") from None

    try:
        yield
    finally:
        lock.release()


@asynccontextmanager
async def _redis_lock(session_id: str) -> AsyncIterator[None]:
    client = _get_redis()
    key = f"lock:session:{session_id}"
    token = uuid.uuid4().hex
    lease_ms = int(settings.session_lock_lease_seconds * 1000)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.session_lock_timeout_seconds

    while True:
        acquired = await client.set(key, token, nx=True, px=lease_ms)
        if acquired:
            break
        if loop.time() >= deadline:
            raise SessionBusy(f"session {session_id} is busy (distributed lock)")
        await asyncio.sleep(0.05)

    try:
        yield
    finally:
        # 只删自己的锁：值被换掉说明租约已过期、别人接手了
        pipe = client.pipeline(transaction=True)
        try:
            await pipe.watch(key)
            current = await pipe.get(key)  # type: ignore[misc]
            if current == token:
                pipe.multi()
                pipe.delete(key)
                await pipe.execute()
        except Exception:  # noqa: BLE001 - 释放失败不应影响调用方
            pass
        finally:
            with contextlib.suppress(Exception):
                await pipe.reset()


@asynccontextmanager
async def session_lock(session_id: str) -> AsyncIterator[None]:
    """按配置选择本地锁或分布式锁。"""
    backend = settings.session_lock_backend
    if backend == "redis":
        async with _redis_lock(session_id):
            yield
    else:
        async with _local_lock(session_id):
            yield


def reset() -> None:
    """测试用：清空本地锁与 Redis 客户端缓存。"""
    global _redis_client
    _local_locks.clear()
    _redis_client = None
