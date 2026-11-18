"""任务调度抽象（P2）。

两种执行模式，同一份路由代码：
    inline —— 在 API 进程内用 ``asyncio.create_task`` 跑（P1 的行为，单机开发）；
    queue  —— 投递到 ARQ worker，API 进程不再承担推理。

为什么还保留 inline：
    没有 Redis 时仍能本地起服务、跑测试；同时它是 queue 模式出问题时的降级路径。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from app.tasks import TaskRecord, TaskStore

logger = logging.getLogger(__name__)

TurnCallable = Callable[[TaskStore, str, str, str], Awaitable[Any]]
JOB_NAME = "run_turn_job"


class Dispatcher(Protocol):
    """把"某个任务该被执行"这件事分发出去。"""

    async def submit(self, record: TaskRecord) -> None: ...

    async def cancel(self, task_id: str) -> None: ...


class InlineDispatcher:
    """进程内执行：一个任务一个 asyncio.Task。"""

    def __init__(self, store: TaskStore, turn: TurnCallable) -> None:
        self._store = store
        self._turn = turn
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    async def submit(self, record: TaskRecord) -> None:
        task = asyncio.create_task(self._run(record), name=f"turn-{record.task_id}")
        self._tasks[record.task_id] = task

        def _forget(_t: asyncio.Task[Any], tid: str = record.task_id) -> None:
            self._tasks.pop(tid, None)

        task.add_done_callback(_forget)

    async def _run(self, record: TaskRecord) -> None:
        try:
            await self._turn(
                self._store, record.task_id, record.session_id, record.message
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - 后台任务必须自兜底
            logger.exception("inline turn crashed for task %s", record.task_id)

    async def cancel(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if task is not None and not task.done():
            task.cancel()

    async def drain(self, timeout: float = 5.0) -> int:
        """等待在途任务收敛（优雅关闭用），返回仍未完成的数量。"""
        pending = [task for task in self._tasks.values() if not task.done()]
        if not pending:
            return 0
        done, still = await asyncio.wait(pending, timeout=timeout)
        return len(still)

    def pending(self) -> int:
        return len(self._tasks)


class ArqDispatcher:
    """投递到 ARQ worker（需要 Redis）。

    注意：不要往 ``enqueue_job`` 里传 ``_job_timeout``——ARQ 0.28 会把它当成
    业务 kwargs 一起传进任务函数（实测报 ``unexpected keyword argument``）。
    单任务的超时统一由 ``WorkerSettings.job_timeout`` 控制。
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._jobs: dict[str, str] = {}

    async def submit(self, record: TaskRecord) -> None:
        job = await self._pool.enqueue_job(
            JOB_NAME,
            record.task_id,
            record.session_id,
            record.message,
        )
        if job is not None:
            self._jobs[record.task_id] = job.job_id
        logger.info(
            "task %s enqueued to arq (job=%s)",
            record.task_id,
            getattr(job, "job_id", None),
        )

    async def cancel(self, task_id: str) -> None:
        """尽力而为的取消。

        ARQ 没有"从队列里删除"的通用语义，真正的取消以存储层的状态为准：
        任务被置为 CANCELLED 后，runner 的 ``_settle`` 不会再把结果写回去。
        """
        job_id = self._jobs.pop(task_id, None)
        if job_id is None:
            return
        abort = getattr(self._pool, "abort_job", None)
        if abort is None:
            logger.info("task %s: pool has no abort_job, rely on store state", task_id)
            return
        try:
            await abort(job_id)
        except Exception:  # noqa: BLE001 - 取消失败不影响状态机
            logger.warning("task %s: abort_job failed", task_id, exc_info=True)
