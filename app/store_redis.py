"""Redis 版任务存储（P2）。

为什么需要它：
    P1 的 ``InMemoryTaskStore`` 只活在单个进程里——重启即丢、多副本之间
    互不可见。把任务状态外置到 Redis 后，API 可以水平扩容，worker 也能
    在别的进程/机器上执行同一批任务。

一致性设计：
    * 任务本体存在 Hash ``task:{id}`` 里，``version`` 单调递增；
    * 状态迁移用 **WATCH / MULTI 乐观锁** 实现 CAS：
      读版本 → 校验迁移合法性 → MULTI 写回；被别的执行者抢先则重试。
      这样两个 worker 同时抢同一任务时，只有一个能成功迁移。
    * 变更通知走 Pub/Sub ``task:{id}:changes``，SSE 靠它唤醒而不是轮询；
      为防丢消息，等待循环里每 0.5s 仍会重读一次版本兜底。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any

import redis.asyncio as aioredis
from redis.exceptions import WatchError

from app.config import settings
from app.tasks import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    InvalidTransition,
    TaskNotFound,
    TaskRecord,
    TaskStatus,
    utcnow,
)

_TASK_KEY = "task:{task_id}"
_IDEM_KEY = "idem:{session_id}:{key}"
_CHANNEL = "task:{task_id}:changes"
# 租约集合：score = 到期时间戳，member = task_id（reaper 按分数区间扫）
_LEASE_KEY = "tasks:leases"

_CAS_MAX_ATTEMPTS = 50


def _iso(value: datetime | None) -> str:
    return value.isoformat() if value else ""


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class RedisTaskStore:
    """基于 Redis 的 TaskStore 实现。"""

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    # ---------- 序列化 ----------

    def _to_record(self, task_id: str, raw: dict[str, str]) -> TaskRecord:
        result_raw = raw.get("result") or ""
        return TaskRecord(
            task_id=task_id,
            session_id=raw.get("session_id", ""),
            message=raw.get("message", ""),
            status=TaskStatus(raw.get("status", TaskStatus.QUEUED.value)),
            created_at=_parse_dt(raw.get("created_at")) or utcnow(),
            started_at=_parse_dt(raw.get("started_at")),
            finished_at=_parse_dt(raw.get("finished_at")),
            result=json.loads(result_raw) if result_raw else None,
            error=raw.get("error") or None,
            version=int(raw.get("version") or 0),
            trace_id=raw.get("trace_id", ""),
            lease_expires_at=_parse_dt(raw.get("lease_expires_at")),
        )

    # ---------- 读 ----------

    async def get(self, task_id: str) -> TaskRecord | None:
        # redis-py 的 Pipeline/hgetall 在类型上是 union，await 后才是 dict
        raw = await self._client.hgetall(_TASK_KEY.format(task_id=task_id))  # type: ignore[misc]
        if not raw:
            return None
        return self._to_record(task_id, raw)

    # ---------- 写 ----------

    async def create(
        self,
        session_id: str,
        message: str,
        *,
        idempotency_key: str | None = None,
        trace_id: str = "",
    ) -> tuple[TaskRecord, bool]:
        task_id = uuid.uuid4().hex

        if idempotency_key:
            idem_key = _IDEM_KEY.format(session_id=session_id, key=idempotency_key)
            acquired = await self._client.set(idem_key, task_id, nx=True)
            if not acquired:
                existing_id = await self._client.get(idem_key)  # type: ignore[misc]
                if existing_id:
                    existing = await self.get(existing_id)
                    if existing is not None:
                        return existing, True
                # 极端竞态：对方刚 SET 还没写 Hash，稍等再读
                for _ in range(20):
                    await asyncio.sleep(0.01)
                    existing_id = await self._client.get(idem_key)
                    if existing_id:
                        existing = await self.get(existing_id)
                        if existing is not None:
                            return existing, True
                raise RuntimeError("idempotency key exists but task record missing")

        now = utcnow()
        await self._client.hset(  # type: ignore[misc]
            _TASK_KEY.format(task_id=task_id),
            mapping={
                "session_id": session_id,
                "message": message,
                "status": TaskStatus.QUEUED.value,
                "created_at": _iso(now),
                "started_at": "",
                "finished_at": "",
                "result": "",
                "error": "",
                "version": 0,
                # 跨进程传递 trace：worker 拿到任务后据此续上日志链路
                "trace_id": trace_id,
                "lease_expires_at": "",
            },
        )
        record = await self.get(task_id)
        if record is None:  # pragma: no cover - 写后即读，理论上不会发生
            raise TaskNotFound(task_id)
        return record, False

    async def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> TaskRecord:
        key = _TASK_KEY.format(task_id=task_id)
        last_error: Exception | None = None

        for _ in range(_CAS_MAX_ATTEMPTS):
            pipe = self._client.pipeline(transaction=True)
            try:
                await pipe.watch(key)
                raw: dict[str, str] = await pipe.hgetall(key) or {}  # type: ignore[misc]
                if not raw:
                    raise TaskNotFound(task_id)

                current = TaskStatus(raw["status"])
                if to not in ALLOWED_TRANSITIONS[current]:
                    raise InvalidTransition(task_id, current, to)

                version = int(raw.get("version") or 0) + 1
                now = utcnow()
                patch: dict[str, str] = {"status": to.value, "version": str(version)}
                lease_score: float | None = None
                if to is TaskStatus.RUNNING:
                    patch["started_at"] = _iso(now)
                    expires = now + timedelta(seconds=settings.task_lease_seconds)
                    patch["lease_expires_at"] = _iso(expires)
                    lease_score = expires.timestamp()
                if to in TERMINAL_STATUSES:
                    patch["finished_at"] = _iso(now)
                    patch["lease_expires_at"] = ""
                if result is not None:
                    patch["result"] = json.dumps(result, ensure_ascii=False, default=str)
                if error is not None:
                    patch["error"] = error

                pipe.multi()
                pipe.hset(key, mapping=patch)
                await pipe.execute()
            except WatchError as exc:  # 被别的执行者抢先，重试
                last_error = exc
                await asyncio.sleep(0.005)
                continue
            finally:
                await pipe.reset()

            # 租约索引：RUNNING 写入，终态移除（放在事务外，失败最多导致一次误回收）
            if lease_score is not None:
                await self._client.zadd(  # type: ignore[misc]
                    _LEASE_KEY, {task_id: lease_score}
                )
            elif to in TERMINAL_STATUSES:
                await self._client.zrem(_LEASE_KEY, task_id)  # type: ignore[misc]

            await self._client.publish(_CHANNEL.format(task_id=task_id), task_id)
            record = await self.get(task_id)
            if record is None:  # pragma: no cover
                raise TaskNotFound(task_id)
            return record

        raise RuntimeError(
            f"task {task_id}: CAS retries exhausted after {_CAS_MAX_ATTEMPTS} attempts"
        ) from last_error

    # ---------- 变更等待 ----------

    async def wait_change(
        self, task_id: str, seen_version: int, timeout: float
    ) -> TaskRecord | None:
        """等待版本变化；超时返回 None（调用方据此发心跳）。"""
        record = await self.get(task_id)
        if record is None:
            return None
        if record.version != seen_version:
            return record

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        pubsub = self._client.pubsub()
        try:
            await pubsub.subscribe(_CHANNEL.format(task_id=task_id))
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return None
                # 每 0.5s 醒一次兜底，避免漏消息
                chunk = min(remaining, 0.5)
                try:
                    await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=chunk
                    )
                except Exception:  # noqa: BLE001 - pubsub 异常降级为轮询
                    await asyncio.sleep(chunk)

                current = await self.get(task_id)
                if current is None:
                    return None
                if current.version != seen_version:
                    return current
        finally:
            try:
                await pubsub.unsubscribe(_CHANNEL.format(task_id=task_id))
                await pubsub.aclose()
            except Exception:  # noqa: BLE001 - 关闭失败不应影响主流程
                pass

    async def reap_expired(self, now: datetime | None = None) -> list[str]:
        """回收租约过期的 RUNNING 任务（worker 崩溃兜底）。"""
        moment = now or utcnow()
        candidates = await self._client.zrangebyscore(  # type: ignore[misc]
            _LEASE_KEY, "-inf", moment.timestamp()
        )

        reaped: list[str] = []
        for task_id in candidates:
            record = await self.get(task_id)
            if record is None or record.status is not TaskStatus.RUNNING:
                # 任务已不在或已收敛，只需从租约集合里清掉
                await self._client.zrem(_LEASE_KEY, task_id)  # type: ignore[misc]
                continue

            try:
                await self.transition(
                    task_id, TaskStatus.FAILED, error="lease expired: worker likely died"
                )
            except InvalidTransition:
                continue
            reaped.append(task_id)

        return reaped

    async def close(self) -> None:
        await self._client.aclose()
