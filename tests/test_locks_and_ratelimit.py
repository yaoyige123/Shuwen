"""分布式会话锁与 Redis 限流（不需要真实 Redis，用 fakeredis）。"""

from __future__ import annotations

import asyncio

import pytest

from app import locks
from app.config import settings
from app.ratelimit import MemoryRateLimiter, RedisRateLimiter

fakeredis_aioredis = pytest.importorskip("fakeredis.aioredis")


@pytest.fixture(autouse=True)
def _reset_locks():
    locks.reset()
    yield
    locks.reset()


def _fake_redis():
    return fakeredis_aioredis.FakeRedis(decode_responses=True)


# ---------------- 会话锁 ----------------


def test_local_lock_serialises_same_session(monkeypatch):
    monkeypatch.setattr(settings, "session_lock_backend", "memory", raising=False)

    order: list[str] = []

    async def worker(name: str) -> None:
        async with locks.session_lock("s1"):
            order.append(f"{name}-in")
            await asyncio.sleep(0.05)
            order.append(f"{name}-out")

    async def scenario():
        await asyncio.gather(worker("a"), worker("b"))

    asyncio.run(scenario())

    # 必须成对出现，不能交错
    assert order in (
        ["a-in", "a-out", "b-in", "b-out"],
        ["b-in", "b-out", "a-in", "a-out"],
    )


def test_local_lock_times_out_with_session_busy(monkeypatch):
    monkeypatch.setattr(settings, "session_lock_backend", "memory", raising=False)
    monkeypatch.setattr(settings, "session_lock_timeout_seconds", 0.05, raising=False)

    async def scenario():
        async with locks.session_lock("s-busy"):
            with pytest.raises(locks.SessionBusy):
                async with locks.session_lock("s-busy"):
                    pass

    asyncio.run(scenario())


def test_redis_lock_only_one_holder(monkeypatch):
    monkeypatch.setattr(settings, "session_lock_backend", "redis", raising=False)
    monkeypatch.setattr(settings, "session_lock_timeout_seconds", 0.1, raising=False)

    client = _fake_redis()
    monkeypatch.setattr(locks, "_get_redis", lambda: client)

    async def scenario():
        acquired: list[str] = []

        async def holder(name: str) -> None:
            try:
                async with locks.session_lock("s-redis"):
                    acquired.append(name)
                    await asyncio.sleep(0.2)
            except locks.SessionBusy:
                pass

        await asyncio.gather(holder("a"), holder("b"))
        await client.aclose()
        return acquired

    acquired = asyncio.run(scenario())

    # 第一个持有期间，第二个应当在超时后抛 SessionBusy，不会同时进入
    assert len(acquired) == 1


def test_redis_lock_released_after_exit(monkeypatch):
    monkeypatch.setattr(settings, "session_lock_backend", "redis", raising=False)
    client = _fake_redis()
    monkeypatch.setattr(locks, "_get_redis", lambda: client)

    async def scenario():
        async with locks.session_lock("s-free"):
            pass
        # 释放之后可以立刻再次获取
        async with locks.session_lock("s-free"):
            pass
        keys = await client.keys("lock:session:*")
        await client.aclose()
        return keys

    assert asyncio.run(scenario()) == []


def test_redis_lock_does_not_delete_foreign_token(monkeypatch):
    """租约过期被别人接手时，旧持有者退出不能删掉新锁。"""
    monkeypatch.setattr(settings, "session_lock_backend", "redis", raising=False)
    client = _fake_redis()
    monkeypatch.setattr(locks, "_get_redis", lambda: client)

    async def scenario():
        key = "lock:session:s-stale"
        async with locks.session_lock("s-stale"):
            # 模拟：租约过期 + 另一个执行者抢到了锁
            await client.set(key, "someone-else")
        return await client.get(key)

    assert asyncio.run(scenario()) == "someone-else"


# ---------------- Redis 限流 ----------------


def test_redis_rate_limiter_enforces_limit():
    client = _fake_redis()
    limiter = RedisRateLimiter(client, limit=2, window_seconds=60)

    async def scenario():
        assert await limiter.acquire("k") is None
        assert await limiter.acquire("k") is None
        blocked = await limiter.acquire("k")
        await client.aclose()
        return blocked

    blocked = asyncio.run(scenario())
    assert blocked is not None and blocked > 0


def test_redis_rate_limiter_isolates_keys():
    client = _fake_redis()
    limiter = RedisRateLimiter(client, limit=1, window_seconds=60)

    async def scenario():
        assert await limiter.acquire("a") is None
        assert await limiter.acquire("b") is None
        assert await limiter.acquire("a") is not None
        await client.aclose()

    asyncio.run(scenario())


def test_redis_rate_limiter_sets_expiry():
    client = _fake_redis()
    limiter = RedisRateLimiter(client, limit=5, window_seconds=30)

    async def scenario():
        await limiter.acquire("ttl")
        keys = await client.keys("ratelimit:ttl:*")
        assert len(keys) == 1
        ttl = await client.ttl(keys[0])
        await client.aclose()
        return ttl

    assert 0 < asyncio.run(scenario()) <= 31


def test_memory_limiter_close_is_noop():
    limiter = MemoryRateLimiter(limit=1, window_seconds=60)
    asyncio.run(limiter.close())
