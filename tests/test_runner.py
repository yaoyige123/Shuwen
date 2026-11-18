"""执行器测试：成功 / 超时 / 异常 / 取消 四条路径。

``invoke_graph_sync`` 被替换成替身，不会真的调用 LLM 或加载图谱。
"""

from __future__ import annotations

import asyncio
import time

from app import runner as runner_module
from app.config import settings
from app.runner import run_turn
from app.tasks import InMemoryTaskStore, TaskStatus


def _boom(*_args, **_kwargs):
    raise RuntimeError("graph blew up")


def test_success_path(monkeypatch):
    monkeypatch.setattr(
        runner_module, "invoke_graph_sync", lambda sid, msg: {"reply": f"echo:{msg}"}
    )

    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "你好")

        done = await run_turn(store, record.task_id, "s1", "你好")

        assert done is not None
        assert done.status is TaskStatus.SUCCEEDED
        assert done.result is not None
        assert done.result["reply"] == "echo:你好"
        assert done.result["attempts"] == 1
        assert done.started_at is not None
        assert done.finished_at is not None

    asyncio.run(scenario())


def test_timeout_marks_task_timeout(monkeypatch):
    monkeypatch.setattr(settings, "turn_timeout_seconds", 0.05, raising=False)
    monkeypatch.setattr(
        runner_module,
        "invoke_graph_sync",
        lambda sid, msg: time.sleep(0.5) or {"reply": "too late"},
    )

    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi")

        done = await run_turn(store, record.task_id, "s1", "hi")

        assert done is not None
        assert done.status is TaskStatus.TIMEOUT
        assert "exceeded" in (done.error or "")

    asyncio.run(scenario())


def test_exception_marks_task_failed(monkeypatch):
    monkeypatch.setattr(runner_module, "invoke_graph_sync", _boom)

    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi")

        done = await run_turn(store, record.task_id, "s1", "hi")

        assert done is not None
        assert done.status is TaskStatus.FAILED
        assert "RuntimeError" in (done.error or "")

    asyncio.run(scenario())


def test_cancellation_marks_task_cancelled(monkeypatch):
    monkeypatch.setattr(
        runner_module,
        "invoke_graph_sync",
        lambda sid, msg: time.sleep(0.3) or {"reply": "late"},
    )

    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi")

        task = asyncio.create_task(run_turn(store, record.task_id, "s1", "hi"))
        await asyncio.sleep(0.05)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        final = await store.get(record.task_id)
        assert final is not None
        assert final.status is TaskStatus.CANCELLED

    asyncio.run(scenario())


def test_cancelled_before_start_does_not_run(monkeypatch):
    """已取消（QUEUED -> CANCELLED）的任务不应再被拿起来执行。"""
    called = False

    def _should_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"reply": "should not happen"}

    monkeypatch.setattr(runner_module, "invoke_graph_sync", _should_not_run)

    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi")
        await store.transition(record.task_id, TaskStatus.CANCELLED, error="user aborted")

        result = await run_turn(store, record.task_id, "s1", "hi")

        assert result is None
        assert called is False
        assert (await store.get(record.task_id)).status is TaskStatus.CANCELLED

    asyncio.run(scenario())


class TransientError(Exception):
    """模拟 503：带 status_code，应被判为可重试。"""

    status_code = 503


def test_retries_transient_error_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    def flaky(_sid, _msg):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientError("503 service unavailable")
        return {"reply": "third time lucky"}

    monkeypatch.setattr(runner_module, "invoke_graph_sync", flaky)
    monkeypatch.setattr(settings, "turn_retry_base_seconds", 0.01, raising=False)
    monkeypatch.setattr(settings, "turn_max_attempts", 3, raising=False)

    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi")

        done = await run_turn(store, record.task_id, "s1", "hi")

        assert done is not None
        assert done.status is TaskStatus.SUCCEEDED
        assert done.result is not None
        assert done.result["attempts"] == 3
        assert attempts["n"] == 3

    asyncio.run(scenario())


def test_non_retryable_error_fails_fast(monkeypatch):
    attempts = {"n": 0}

    def boom(_sid, _msg):
        attempts["n"] += 1
        raise ValueError("bad request, do not retry")

    monkeypatch.setattr(runner_module, "invoke_graph_sync", boom)
    monkeypatch.setattr(settings, "turn_max_attempts", 3, raising=False)

    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi")

        done = await run_turn(store, record.task_id, "s1", "hi")

        assert done is not None
        assert done.status is TaskStatus.FAILED
        assert attempts["n"] == 1
        assert "ValueError" in (done.error or "")

    asyncio.run(scenario())


def test_exhausted_retries_mark_failed(monkeypatch):
    attempts = {"n": 0}

    def always_transient(_sid, _msg):
        attempts["n"] += 1
        raise TransientError("503 still down")

    monkeypatch.setattr(runner_module, "invoke_graph_sync", always_transient)
    monkeypatch.setattr(settings, "turn_retry_base_seconds", 0.01, raising=False)
    monkeypatch.setattr(settings, "turn_max_attempts", 2, raising=False)

    async def scenario():
        store = InMemoryTaskStore()
        record, _ = await store.create("s1", "hi")

        done = await run_turn(store, record.task_id, "s1", "hi")

        assert done is not None
        assert done.status is TaskStatus.FAILED
        assert attempts["n"] == 2
        assert "503" in (done.error or "")

    asyncio.run(scenario())
