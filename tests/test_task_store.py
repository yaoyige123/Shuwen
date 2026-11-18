"""任务状态机与内存存储的单元测试。

不依赖 pytest-asyncio：每个用例用 ``asyncio.run`` 驱动协程，
保证只要装了 pytest 就能跑。
"""

from __future__ import annotations

import asyncio

import pytest

from app.tasks import (
    InMemoryTaskStore,
    InvalidTransition,
    TaskNotFound,
    TaskStatus,
)


def test_create_returns_queued_task():
    async def scenario():
        store = InMemoryTaskStore()
        record, replayed = await store.create("s1", "hello")

        assert record.status is TaskStatus.QUEUED
        assert record.version == 0
        assert replayed is False
        assert await store.get(record.task_id) is record

    asyncio.run(scenario())


def test_happy_path_transitions():
    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi")

        running = await store.transition(record.task_id, TaskStatus.RUNNING)
        assert running.status is TaskStatus.RUNNING
        assert running.started_at is not None
        assert running.finished_at is None
        assert running.version == 1

        done = await store.transition(
            record.task_id, TaskStatus.SUCCEEDED, result={"reply": "ok"}
        )
        assert done.status is TaskStatus.SUCCEEDED
        assert done.result == {"reply": "ok"}
        assert done.finished_at is not None
        assert done.version == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (TaskStatus.QUEUED, TaskStatus.SUCCEEDED),
        (TaskStatus.QUEUED, TaskStatus.FAILED),
        (TaskStatus.QUEUED, TaskStatus.TIMEOUT),
    ],
)
def test_illegal_transition_from_queued(start, target):
    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi")

        with pytest.raises(InvalidTransition):
            await store.transition(record.task_id, target)

        # 失败不能改变状态
        assert (await store.get(record.task_id)).status is start

    asyncio.run(scenario())


def test_terminal_state_is_frozen():
    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi")
        await store.transition(record.task_id, TaskStatus.RUNNING)
        await store.transition(record.task_id, TaskStatus.SUCCEEDED, result={})

        for target in (TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.TIMEOUT):
            with pytest.raises(InvalidTransition):
                await store.transition(record.task_id, target)

    asyncio.run(scenario())


def test_unknown_task_raises_not_found():
    async def scenario():
        store = InMemoryTaskStore()
        with pytest.raises(TaskNotFound):
            await store.transition("nope", TaskStatus.RUNNING)
        assert await store.get("nope") is None

    asyncio.run(scenario())


def test_idempotency_key_replays_same_task():
    async def scenario():
        store = InMemoryTaskStore()
        first, replayed = await store.create("s1", "hi", idempotency_key="k1")
        assert replayed is False

        second, replayed2 = await store.create("s1", "hi again", idempotency_key="k1")
        assert replayed2 is True
        assert second.task_id == first.task_id

        # 幂等键按会话隔离
        other, replayed3 = await store.create("s2", "hi", idempotency_key="k1")
        assert replayed3 is False
        assert other.task_id != first.task_id

    asyncio.run(scenario())


def test_without_idempotency_key_each_call_is_new():
    async def scenario():
        store = InMemoryTaskStore()
        a, _ = await store.create("s1", "hi")
        b, _ = await store.create("s1", "hi")
        assert a.task_id != b.task_id

    asyncio.run(scenario())


def test_wait_change_returns_immediately_on_newer_version():
    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi")
        await store.transition(record.task_id, TaskStatus.RUNNING)

        seen = await store.wait_change(record.task_id, 0, timeout=0.1)
        assert seen is not None
        assert seen.version == 1

    asyncio.run(scenario())


def test_wait_change_wakes_up_on_transition():
    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi")

        async def later():
            await asyncio.sleep(0.02)
            await store.transition(record.task_id, TaskStatus.RUNNING)

        waiter = asyncio.create_task(later())
        changed = await store.wait_change(record.task_id, 0, timeout=1.0)
        await waiter

        assert changed is not None
        assert changed.status is TaskStatus.RUNNING

    asyncio.run(scenario())


def test_wait_change_returns_none_on_timeout():
    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi")
        assert await store.wait_change(record.task_id, 0, timeout=0.02) is None

    asyncio.run(scenario())
