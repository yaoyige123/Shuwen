"""任务租约回收 + trace_id 跨进程传递。

两种存储实现（内存 / Redis）跑同一组断言，保证语义一致。
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from app.config import settings
from app.observability import get_trace_id
from app.tasks import InMemoryTaskStore, TaskStatus

fakeredis_aioredis = pytest.importorskip("fakeredis.aioredis")

from app.store_redis import RedisTaskStore  # noqa: E402

KINDS = ["memory", "redis"]


def _make(kind: str):
    if kind == "memory":
        return InMemoryTaskStore()
    return RedisTaskStore(fakeredis_aioredis.FakeRedis(decode_responses=True))


async def _close(store) -> None:
    closer = getattr(store, "close", None)
    if closer is not None:
        await closer()


@pytest.mark.parametrize("kind", KINDS)
def test_trace_id_roundtrip(kind):
    async def scenario():
        store = _make(kind)
        try:
            record, _ = await store.create("s1", "hi", trace_id="trace-xyz")
            assert record.trace_id == "trace-xyz"

            fetched = await store.get(record.task_id)
            assert fetched is not None
            assert fetched.trace_id == "trace-xyz"
            assert fetched.to_dict()["trace_id"] == "trace-xyz"
        finally:
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", KINDS)
def test_lease_is_set_on_running_and_cleared_on_terminal(kind, monkeypatch):
    monkeypatch.setattr(settings, "task_lease_seconds", 60, raising=False)

    async def scenario():
        store = _make(kind)
        try:
            record, _ = await store.create("s1", "hi")
            assert record.lease_expires_at is None

            await store.transition(record.task_id, TaskStatus.RUNNING)
            running = await store.get(record.task_id)
            assert running is not None
            assert running.lease_expires_at is not None

            done = await store.transition(
                record.task_id, TaskStatus.SUCCEEDED, result={"reply": "ok"}
            )
            assert done.lease_expires_at is None
        finally:
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", KINDS)
def test_reaper_fails_expired_running_task(kind, monkeypatch):
    monkeypatch.setattr(settings, "task_lease_seconds", 60, raising=False)

    async def scenario():
        store = _make(kind)
        try:
            record, _ = await store.create("s1", "hi")
            await store.transition(record.task_id, TaskStatus.RUNNING)

            # 还没到期：不该被回收
            assert await store.reap_expired() == []

            running = await store.get(record.task_id)
            assert running is not None and running.lease_expires_at is not None
            future = running.lease_expires_at + timedelta(seconds=1)

            reaped = await store.reap_expired(now=future)
            assert reaped == [record.task_id]

            final = await store.get(record.task_id)
            assert final is not None
            assert final.status is TaskStatus.FAILED
            assert "lease expired" in (final.error or "")
            assert final.lease_expires_at is None

            # 幂等：再扫一次不会重复回收
            assert await store.reap_expired(now=future) == []
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_reaper_ignores_queued_and_finished_tasks(monkeypatch):
    monkeypatch.setattr(settings, "task_lease_seconds", 1, raising=False)

    async def scenario():
        store = InMemoryTaskStore()

        queued, _ = await store.create("s1", "still queued")
        done, _ = await store.create("s1", "done")
        await store.transition(done.task_id, TaskStatus.RUNNING)
        await store.transition(done.task_id, TaskStatus.SUCCEEDED, result={})

        far_future = done.finished_at + timedelta(days=1)
        assert await store.reap_expired(now=far_future) == []

        assert (await store.get(queued.task_id)).status is TaskStatus.QUEUED
        assert (await store.get(done.task_id)).status is TaskStatus.SUCCEEDED

    asyncio.run(scenario())


def test_runner_restores_trace_id_from_task_record(monkeypatch):
    """无论 inline 还是 queue，runner 都必须续上创建任务时的 trace。"""
    from app import runner as runner_module
    from app.runner import run_turn

    captured: dict[str, str] = {}

    def fake_invoke(session_id: str, message: str) -> dict:
        captured["trace"] = get_trace_id()
        return {"reply": "ok"}

    monkeypatch.setattr(runner_module, "invoke_graph_sync", fake_invoke)

    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi", trace_id="trace-from-http")
        await run_turn(store, record.task_id, "s1", "hi")

    asyncio.run(scenario())

    assert captured["trace"] == "trace-from-http"
