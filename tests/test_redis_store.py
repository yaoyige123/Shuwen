"""Redis 任务存储测试。

用 ``fakeredis`` 在进程内模拟 Redis，**不需要真实服务端**——
这样 CI 里也能验证 CAS 与 Pub/Sub 逻辑。
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fakeredis")

import fakeredis.aioredis  # noqa: E402

from app.store_redis import RedisTaskStore  # noqa: E402
from app.tasks import InvalidTransition, TaskNotFound, TaskStatus  # noqa: E402


def _store() -> RedisTaskStore:
    return RedisTaskStore(fakeredis.aioredis.FakeRedis(decode_responses=True))


def test_create_and_get_roundtrip():
    async def scenario():
        store = _store()
        record, replayed = await store.create("s1", "你好")

        assert replayed is False
        assert record.status is TaskStatus.QUEUED
        assert record.version == 0

        fetched = await store.get(record.task_id)
        assert fetched is not None
        assert fetched.message == "你好"
        assert fetched.session_id == "s1"

        await store.close()

    asyncio.run(scenario())


def test_transition_persists_fields_and_bumps_version():
    async def scenario():
        store = _store()
        record, _ = await store.create("s1", "hi")

        running = await store.transition(record.task_id, TaskStatus.RUNNING)
        assert running.status is TaskStatus.RUNNING
        assert running.started_at is not None
        assert running.version == 1

        done = await store.transition(
            record.task_id, TaskStatus.SUCCEEDED, result={"reply": "ok"}
        )
        assert done.status is TaskStatus.SUCCEEDED
        assert done.result == {"reply": "ok"}
        assert done.finished_at is not None
        assert done.version == 2

        await store.close()

    asyncio.run(scenario())


def test_illegal_transition_is_rejected():
    async def scenario():
        store = _store()
        record, _ = await store.create("s1", "hi")

        with pytest.raises(InvalidTransition):
            await store.transition(record.task_id, TaskStatus.SUCCEEDED)

        current = await store.get(record.task_id)
        assert current is not None and current.status is TaskStatus.QUEUED

        await store.close()

    asyncio.run(scenario())


def test_unknown_task_raises():
    async def scenario():
        store = _store()
        with pytest.raises(TaskNotFound):
            await store.transition("missing", TaskStatus.RUNNING)
        await store.close()

    asyncio.run(scenario())


def test_idempotency_key_replays_same_task():
    async def scenario():
        store = _store()
        first, replayed = await store.create("s1", "hi", idempotency_key="k1")
        assert replayed is False

        second, replayed2 = await store.create("s1", "hi", idempotency_key="k1")
        assert replayed2 is True
        assert second.task_id == first.task_id

        other, replayed3 = await store.create("s2", "hi", idempotency_key="k1")
        assert replayed3 is False
        assert other.task_id != first.task_id

        await store.close()

    asyncio.run(scenario())


def test_concurrent_terminal_transitions_only_one_wins():
    """两个执行者同时抢同一个任务时，CAS 只能让一个成功。"""

    async def scenario():
        store = _store()
        record, _ = await store.create("s1", "hi")
        await store.transition(record.task_id, TaskStatus.RUNNING)

        async def succeed():
            return await store.transition(
                record.task_id, TaskStatus.SUCCEEDED, result={"by": "A"}
            )

        async def fail():
            return await store.transition(
                record.task_id, TaskStatus.FAILED, error="by B"
            )

        results = await asyncio.gather(succeed(), fail(), return_exceptions=True)
        winners = [r for r in results if not isinstance(r, BaseException)]
        losers = [r for r in results if isinstance(r, InvalidTransition)]

        assert len(winners) == 1, results
        assert len(losers) == 1, results

        final = await store.get(record.task_id)
        assert final is not None
        assert final.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED)

        await store.close()

    asyncio.run(scenario())


def test_wait_change_returns_immediately_when_version_moved():
    async def scenario():
        store = _store()
        record, _ = await store.create("s1", "hi")
        await store.transition(record.task_id, TaskStatus.RUNNING)

        changed = await store.wait_change(record.task_id, 0, timeout=0.1)
        assert changed is not None
        assert changed.version == 1

        await store.close()

    asyncio.run(scenario())


def test_wait_change_wakes_up_on_transition():
    async def scenario():
        store = _store()
        record, _ = await store.create("s1", "hi")

        async def later():
            await asyncio.sleep(0.05)
            await store.transition(record.task_id, TaskStatus.RUNNING)

        task = asyncio.create_task(later())
        changed = await store.wait_change(record.task_id, 0, timeout=3.0)
        await task

        assert changed is not None
        assert changed.status is TaskStatus.RUNNING

        await store.close()

    asyncio.run(scenario())


def test_wait_change_times_out():
    async def scenario():
        store = _store()
        record, _ = await store.create("s1", "hi")

        assert await store.wait_change(record.task_id, 0, timeout=0.05) is None

        await store.close()

    asyncio.run(scenario())
