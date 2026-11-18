"""限流（P2）。

两种实现同一接口：``acquire(key)`` 返回 ``None`` 放行，返回 float 表示还要等多少秒。

* ``MemoryRateLimiter``：进程内固定窗口，单机开发用；
* ``RedisRateLimiter``：借 Redis ``INCR`` + ``EXPIRE`` 做跨副本的固定窗口。

为什么要有 Redis 版：进程内计数在多副本下等于把额度放大 N 倍，
客户端只要轮流打到不同实例就能绕过限流。
"""

from __future__ import annotations

import time
from typing import Any


class RateLimiter:
    async def acquire(self, key: str) -> float | None:  # pragma: no cover - 协议
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - 协议
        return None


class MemoryRateLimiter(RateLimiter):
    def __init__(self, limit: int, window_seconds: int) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self._limit = limit
        self._window = float(window_seconds)
        self._hits: dict[str, list[float]] = {}

    async def acquire(self, key: str) -> float | None:
        now = time.monotonic()
        cutoff = now - self._window
        hits = self._hits.setdefault(key, [])

        while hits and hits[0] <= cutoff:
            hits.pop(0)

        if len(hits) >= self._limit:
            return max(hits[0] + self._window - now, 0.0)

        hits.append(now)
        return None

    def reset(self) -> None:
        self._hits.clear()


class RedisRateLimiter(RateLimiter):
    """跨副本固定窗口：``INCR`` 计数器 + ``EXPIRE`` 自动过期。"""

    def __init__(self, client: Any, *, limit: int, window_seconds: int) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self._client = client
        self._limit = limit
        self._window = float(window_seconds)

    def _bucket_key(self, key: str, now: float) -> str:
        bucket = int(now // self._window)
        return f"ratelimit:{key}:{bucket}"

    async def acquire(self, key: str) -> float | None:
        now = time.monotonic()
        redis_key = self._bucket_key(key, now)

        count = await self._client.incr(redis_key)  # type: ignore[misc]
        if count == 1:
            await self._client.expire(redis_key, int(self._window) + 1)  # type: ignore[misc]

        if count > self._limit:
            ttl = await self._client.ttl(redis_key)  # type: ignore[misc]
            return float(max(int(ttl), 1))
        return None

    async def close(self) -> None:
        await self._client.aclose()
