"""任务状态机与存储抽象（P1）。

设计要点
--------
1. 状态迁移是**受控**的：非法迁移直接抛 ``InvalidTransition``，而不是悄悄改坏状态。
2. 存储层用 Protocol 抽象，当前实现是进程内内存版；P2 换成 Redis/PostgreSQL 时
   只替换实现，路由与 runner 不用改。
3. 幂等：``create()`` 接受 ``idempotency_key``，重复提交返回同一 task 并标记 replay。
4. 等待变更用 ``asyncio.Event``，SSE 端点不需要轮询数据库（对比旧版每 100ms 轮询）。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from app.config import settings


def utcnow() -> datetime:
    return datetime.now(UTC)


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED}
)

# 合法迁移表：任何未列出的迁移都会被拒绝
ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.QUEUED: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED}),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.TIMEOUT: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


class InvalidTransition(RuntimeError):
    """尝试执行非法状态迁移。"""

    def __init__(self, task_id: str, current: TaskStatus, target: TaskStatus) -> None:
        self.task_id = task_id
        self.current = current
        self.target = target
        super().__init__(f"task {task_id}: illegal transition {current.value} -> {target.value}")


class TaskNotFound(KeyError):
    """任务不存在。"""


@dataclass
class TaskRecord:
    task_id: str
    session_id: str
    message: str
    status: TaskStatus = TaskStatus.QUEUED
    created_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    version: int = 0
    # 创建该任务的请求的 trace_id，跨进程传递（worker 侧据此续上日志链路）
    trace_id: str = ""
    # 租约到期时间：RUNNING 时设置，worker 崩溃后由 reaper 回收
    lease_expires_at: datetime | None = None
    # SSE 等待用；repr=False 避免日志里出现对象地址
    waiter: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "version": self.version,
            "trace_id": self.trace_id,
        }


class TaskStore(Protocol):
    """任务存储接口；P2 的 Redis/PostgreSQL 实现需满足同一契约。"""

    async def create(
        self,
        session_id: str,
        message: str,
        *,
        idempotency_key: str | None = None,
        trace_id: str = "",
    ) -> tuple[TaskRecord, bool]:
        """创建任务。返回 (记录, 是否为幂等重放)。"""
        ...

    async def get(self, task_id: str) -> TaskRecord | None: ...

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> TaskRecord: ...

    async def wait_change(
        self, task_id: str, seen_version: int, timeout: float
    ) -> TaskRecord | None: ...

    async def reap_expired(self, now: datetime | None = None) -> list[str]:
        """回收租约过期的 RUNNING 任务（worker 崩溃兜底）。"""
        ...


class InMemoryTaskStore:
    """进程内实现（P1）。多副本部署必须换成 Redis/PostgreSQL（P2）。"""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._idempotency: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        session_id: str,
        message: str,
        *,
        idempotency_key: str | None = None,
        trace_id: str = "",
    ) -> tuple[TaskRecord, bool]:
        async with self._lock:
            if idempotency_key:
                scoped = f"{session_id}:{idempotency_key}"
                existing_id = self._idempotency.get(scoped)
                if existing_id is not None:
                    existing = self._tasks.get(existing_id)
                    if existing is not None:
                        return existing, True

            record = TaskRecord(
                task_id=uuid.uuid4().hex,
                session_id=session_id,
                message=message,
                trace_id=trace_id,
            )
            self._tasks[record.task_id] = record
            if idempotency_key:
                self._idempotency[f"{session_id}:{idempotency_key}"] = record.task_id
            return record, False

    async def get(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> TaskRecord:
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise TaskNotFound(task_id)

            if to not in ALLOWED_TRANSITIONS[record.status]:
                raise InvalidTransition(task_id, record.status, to)

            record.status = to
            record.version += 1
            if to is TaskStatus.RUNNING:
                record.started_at = utcnow()
                # 租约：worker 崩溃后由 reaper 依据它回收
                record.lease_expires_at = record.started_at + timedelta(
                    seconds=settings.task_lease_seconds
                )
            if to in TERMINAL_STATUSES:
                record.finished_at = utcnow()
                record.lease_expires_at = None
            if result is not None:
                record.result = result
            if error is not None:
                record.error = error

            record.waiter.set()
            return record

    async def reap_expired(self, now: datetime | None = None) -> list[str]:
        """回收租约过期的 RUNNING 任务（模拟 worker 崩溃后的兜底）。"""
        moment = now or utcnow()

        async with self._lock:
            expired = [
                record.task_id
                for record in self._tasks.values()
                if record.status is TaskStatus.RUNNING
                and record.lease_expires_at is not None
                and record.lease_expires_at <= moment
            ]

        reaped: list[str] = []
        for task_id in expired:
            try:
                await self.transition(
                    task_id, TaskStatus.FAILED, error="lease expired: worker likely died"
                )
            except InvalidTransition:
                continue
            reaped.append(task_id)
        return reaped

    async def wait_change(
        self, task_id: str, seen_version: int, timeout: float
    ) -> TaskRecord | None:
        record = self._tasks.get(task_id)
        if record is None:
            return None
        if record.version != seen_version:
            return record

        record.waiter.clear()
        # 清除标记后再次确认，避免「读完就变」导致漏掉一次唤醒
        if record.version != seen_version:
            return record

        try:
            await asyncio.wait_for(record.waiter.wait(), timeout=timeout)
        except TimeoutError:
            return None
        return self._tasks.get(task_id)
