"""HTTP 端点：提交消息、查询任务、订阅 SSE 进度、取消任务。

契约（P1/P2）：
    POST /api/v1/sessions/{session_id}/messages   -> 202 + task_id
    GET  /api/v1/tasks/{task_id}                  -> 任务状态
    GET  /api/v1/tasks/{task_id}/events           -> text/event-stream
    POST /api/v1/tasks/{task_id}/cancel           -> 取消（幂等）
    GET  /api/v1/health                           -> 存活探针

执行由 ``Dispatcher`` 决定：inline（进程内 asyncio）或 queue（ARQ worker）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse

from app.api.deps import Dispatcher, get_dispatcher, get_store
from app.api.security import enforce_rate_limit, require_api_key
from app.config import settings
from app.observability import get_trace_id
from app.schemas import HealthOut, MessageIn, TaskAccepted, TaskOut, TtsIn
from app.tasks import InvalidTransition, TaskStatus, TaskStore
from app.tts import get_tts_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

VERSION = "0.2.0"

# 用 Annotated 声明依赖，避免把函数调用写进参数默认值（ruff B008）
StoreDep = Annotated[TaskStore, Depends(get_store)]
DispatcherDep = Annotated[Dispatcher, Depends(get_dispatcher)]
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


@router.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    return HealthOut(status="ok", version=VERSION)


@router.post(
    "/sessions/{session_id}/messages",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TaskAccepted,
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def post_message(
    session_id: str,
    payload: MessageIn,
    store: StoreDep,
    dispatcher: DispatcherDep,
    idempotency_key: IdempotencyKey = None,
) -> TaskAccepted:
    """提交一条用户消息，立即返回任务 ID（不等待推理完成）。"""
    record, replayed = await store.create(
        session_id,
        payload.message,
        idempotency_key=idempotency_key,
        # 把 trace 写进任务记录：worker 在另一个进程里据此续上同一条链路
        trace_id=get_trace_id(),
    )

    if not replayed:
        await dispatcher.submit(record)
        logger.info(
            "task %s queued (session=%s, len=%d)",
            record.task_id,
            session_id,
            len(payload.message),
        )
    else:
        logger.info("task %s replayed from idempotency key", record.task_id)

    return TaskAccepted(
        task_id=record.task_id,
        session_id=record.session_id,
        status=record.status.value,
        idempotent_replay=replayed,
    )


@router.post(
    "/tts",
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def synthesize_speech(payload: TtsIn) -> Response:
    """把回复文本合成为 wav，供页面播放。

    未配置讯飞凭据时返回 503——前端据此把语音按钮置灰，而不是静默失败。
    """
    service = get_tts_service()
    if not service.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS 未配置：请在 .env 设置 XUNFEI_APP_ID / XUNFEI_API_KEY / XUNFEI_API_SECRET",
        )

    audio = await asyncio.to_thread(service.synthesize, payload.text)
    if audio is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="语音合成失败，请稍后重试"
        )

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get(
    "/tasks/{task_id}",
    response_model=TaskOut,
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def get_task(
    task_id: str,
    store: StoreDep,
) -> TaskOut:
    record = await store.get(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"task {task_id} not found"
        )
    return TaskOut(**record.to_dict())


@router.post(
    "/tasks/{task_id}/cancel",
    response_model=TaskOut,
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def cancel_task(
    task_id: str,
    store: StoreDep,
    dispatcher: DispatcherDep,
) -> TaskOut:
    """取消任务；已处于终态则原样返回（幂等）。"""
    record = await store.get(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"task {task_id} not found"
        )
    if record.is_terminal:
        return TaskOut(**record.to_dict())

    # 先让执行侧停手，再把状态机收敛到终态
    await dispatcher.cancel(task_id)

    try:
        record = await store.transition(
            task_id, TaskStatus.CANCELLED, error="cancelled by client"
        )
    except InvalidTransition:
        # runner 抢先收敛到了别的终态，以存储里的最新状态为准
        latest = await store.get(task_id)
        if latest is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"task {task_id} not found",
            ) from None
        record = latest

    logger.info("task %s cancelled by client (status=%s)", task_id, record.status.value)
    return TaskOut(**record.to_dict())


@router.get(
    "/tasks/{task_id}/events",
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def task_events(
    task_id: str,
    request: Request,
    store: StoreDep,
) -> StreamingResponse:
    """SSE 推送任务状态变更。

    用存储层的变更通知唤醒而不是定时轮询：内存实现用 ``asyncio.Event``，
    Redis 实现用 Pub/Sub。空闲时靠 keep-alive 注释保活，避免反向代理掐断。
    """
    record = await store.get(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"task {task_id} not found"
        )

    async def generate() -> AsyncIterator[str]:
        seen = -1
        while True:
            if await request.is_disconnected():
                logger.debug("sse client disconnected from task %s", task_id)
                break

            current = await store.get(task_id)
            if current is None:
                break

            if current.version != seen:
                seen = current.version
                yield _sse("status", current.to_dict())
                if current.is_terminal:
                    yield _sse(
                        "done", {"task_id": task_id, "status": current.status.value}
                    )
                    break
                continue

            changed = await store.wait_change(
                task_id, seen, timeout=float(settings.sse_keepalive_seconds)
            )
            if changed is None:
                yield ": keep-alive\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Nginx 必须关掉缓冲，否则 SSE 会被攒成一批
            "X-Accel-Buffering": "no",
        },
    )
