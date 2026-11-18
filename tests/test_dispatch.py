"""调度器测试：inline 执行与 ARQ 投递（用假连接池，无需 Redis）。"""

from __future__ import annotations

import asyncio

from app.dispatch import JOB_NAME, ArqDispatcher, InlineDispatcher
from app.tasks import InMemoryTaskStore, TaskStatus


def test_inline_dispatcher_runs_turn():
    async def scenario():
        store = InMemoryTaskStore()
        seen: list[tuple[str, str, str]] = []

        async def turn(st, task_id, session_id, message):
            seen.append((task_id, session_id, message))
            await st.transition(task_id, TaskStatus.RUNNING)
            await st.transition(task_id, TaskStatus.SUCCEEDED, result={"reply": "x"})

        dispatcher = InlineDispatcher(store, turn)
        record, _ = await store.create("s1", "hi")

        await dispatcher.submit(record)
        for _ in range(100):
            current = await store.get(record.task_id)
            if current is not None and current.is_terminal:
                break
            await asyncio.sleep(0.01)

        assert seen == [(record.task_id, "s1", "hi")]
        final = await store.get(record.task_id)
        assert final is not None and final.status is TaskStatus.SUCCEEDED
        assert dispatcher.pending() == 0

    asyncio.run(scenario())


def test_inline_dispatcher_cancel_stops_task():
    async def scenario():
        store = InMemoryTaskStore()

        async def slow_turn(st, task_id, session_id, message):
            await st.transition(task_id, TaskStatus.RUNNING)
            await asyncio.sleep(10)

        dispatcher = InlineDispatcher(store, slow_turn)
        record, _ = await store.create("s1", "hi")
        await dispatcher.submit(record)
        await asyncio.sleep(0.05)

        await dispatcher.cancel(record.task_id)
        await asyncio.sleep(0.05)

        assert dispatcher.pending() == 0

    asyncio.run(scenario())


class _FakeJob:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id


class _FakePool:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, tuple, dict]] = []
        self.aborted: list[str] = []

    async def enqueue_job(self, name, *args, **kwargs):
        self.enqueued.append((name, args, kwargs))
        return _FakeJob("job-1")

    async def abort_job(self, job_id: str) -> None:
        self.aborted.append(job_id)


def test_arq_dispatcher_enqueues_job():
    async def scenario():
        store = InMemoryTaskStore()
        pool = _FakePool()
        dispatcher = ArqDispatcher(pool)

        record, _ = await store.create("s1", "hi")
        await dispatcher.submit(record)

        name, args, kwargs = pool.enqueued[0]
        assert name == JOB_NAME
        assert args == (record.task_id, "s1", "hi")
        # 不能再夹带 _job_timeout：ARQ 会把它当业务参数传给任务函数
        assert kwargs == {}

    asyncio.run(scenario())


def test_arq_dispatcher_cancel_aborts_job():
    async def scenario():
        store = InMemoryTaskStore()
        pool = _FakePool()
        dispatcher = ArqDispatcher(pool)

        record, _ = await store.create("s1", "hi")
        await dispatcher.submit(record)
        await dispatcher.cancel(record.task_id)

        assert pool.aborted == ["job-1"]

    asyncio.run(scenario())


def test_arq_dispatcher_cancel_unknown_is_noop():
    async def scenario():
        pool = _FakePool()
        dispatcher = ArqDispatcher(pool)
        await dispatcher.cancel("never-submitted")
        assert pool.aborted == []

    asyncio.run(scenario())
