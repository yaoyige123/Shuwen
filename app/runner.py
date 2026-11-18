"""单轮对话执行器（P1）。

为什么要把 LangGraph 放进线程：
    LangGraph 的 ``invoke`` 是同步阻塞 API（内部是 LLM/HTTP 调用），
    直接在协程里调用会把事件循环堵死——一个会话就会拖垮整个进程。
    这里用 ``asyncio.to_thread`` 把它挪到默认线程池，再套一层总超时。

已知边界：
    ``to_thread`` 无法真正杀死已经在跑的线程，超时只是**放弃等待**并在状态机
    里标记 TIMEOUT。要拿到硬中断能力，得在 P2 把执行搬到 Celery worker 进程里
    （进程可以被 revoke/kill）。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import threading
from typing import Any

from app.config import settings
from app.locks import session_lock
from app.observability import set_trace_id
from app.tasks import InvalidTransition, TaskRecord, TaskStatus, TaskStore

logger = logging.getLogger(__name__)

# 每个会话只预取一次图谱结果
_prefetch_started: set[str] = set()
_prefetch_lock = threading.Lock()


def _prefetch_graph_qa(session_id: str, state: dict[str, Any]) -> None:
    """在镇痛宣教阶段后台预热 GraphRAG，供后面的风险告知使用。

    结果写进 app.events，而不是从线程里改 LangGraph 的 state——
    跨线程写图状态不安全，也会让 checkpoint 出现意外版本。

    检索本身按 global → local → 词面 分层降级，每层都有超时上限
    （见 app/graphqa.py），所以这里不需要再包一层超时。
    """
    from app import graphqa
    from app.events import register, set_result

    ev = register(session_id)
    payload = {
        "messages": state.get("messages", []),
        "user_information": state.get("user_information", ""),
        "medical_history": state.get("medical_history", []),
        "medicine_taking": state.get("medicine_taking", []),
    }
    try:
        # 先放一个"快速兜底"占位：GraphRAG 的 global 检索常要十几秒，
        # 而词面检索只要几十毫秒。先让风险环节有东西可用，慢层到了再覆盖。
        fast = graphqa.retrieve_keyword(payload)
        if fast:
            set_result(session_id, fast, graphqa.PATH_KEYWORD)

        text, path = graphqa.retrieve(payload)
        if text:
            set_result(session_id, text, path)
            logger.info(
                "graph qa prefetch ready for %s via %s (%d chars)",
                session_id,
                path,
                len(text),
            )
        elif not fast:
            logger.warning("graph qa prefetch 全部降级（%s），风险阶段将使用档案兜底", session_id)
    except Exception:  # noqa: BLE001 - 预取失败不影响主流程
        logger.exception("graph qa prefetch failed for %s", session_id)
    finally:
        ev.set()


def _ensure_prefetch(session_id: str, state: dict[str, Any]) -> None:
    with _prefetch_lock:
        if session_id in _prefetch_started:
            return
        _prefetch_started.add(session_id)

    threading.Thread(
        target=_prefetch_graph_qa,
        args=(session_id, state),
        name=f"graph-qa-{session_id}",
        daemon=True,
    ).start()
    logger.info("graph qa prefetch started for %s", session_id)


async def _settle(
    store: TaskStore,
    task_id: str,
    status: TaskStatus,
    **fields: Any,
) -> TaskRecord | None:
    """收敛到终态，并记录指标。

    任务可能已被 /cancel 提前置为 CANCELLED——此时再写 SUCCEEDED 是非法迁移，
    这里降级为「保持现状」，不把异常抛给后台任务。
    """
    try:
        record = await store.transition(task_id, status, **fields)
    except InvalidTransition:
        logger.warning(
            "task %s already left RUNNING, keep current state (wanted %s)",
            task_id,
            status.value,
        )
        # 已经被别人收敛过，不再重复计数，避免指标翻倍
        return await store.get(task_id)

    _observe_turn(record)
    return record


def _observe_turn(record: TaskRecord | None) -> None:
    """记录单轮对话的终态与耗时（耗时取 checkpointer 里记的时间戳）。"""
    if record is None:
        return

    from app.metrics import TURN_DURATION, TURNS

    TURNS.labels(status=record.status.value).inc()
    if record.started_at and record.finished_at:
        seconds = (record.finished_at - record.started_at).total_seconds()
        TURN_DURATION.labels(status=record.status.value).observe(seconds)


def invoke_graph_sync(session_id: str, message: str) -> dict[str, Any]:
    """在线程中同步跑一轮 LangGraph 对话。

    只做 import 的延迟加载：模块级导入会把 LangGraph 及其全部 Chain
    一次性载入，拖慢 API 冷启动。
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from app.checkpoint import get_graph

    # 由 checkpoint 工厂决定用哪种存储（memory / sqlite / postgres）
    graph = get_graph()

    config = {"configurable": {"thread_id": session_id}}

    # 新会话：把 dialog_state 置为信息核对阶段（与旧 Flask 版行为一致）
    snapshot = graph.get_state(config)
    if not snapshot.values.get("dialog_state"):
        graph.update_state(
            config,
            {"dialog_state": "verify_information", "session_id": session_id},
        )

    state = graph.invoke({"messages": [HumanMessage(content=message)]}, config)

    ai_messages = [
        item
        for item in state.get("messages", [])
        if isinstance(item, AIMessage) and item.content
    ]
    dialog_state = state.get("dialog_state") or []
    current_stage = dialog_state[-1] if dialog_state else None

    # 进入镇痛宣教阶段就后台预热图谱：患者在听宣教时把检索做掉，
    # 等到风险告知时结果通常已经就绪。
    if current_stage == "analgesia":
        _ensure_prefetch(session_id, state)

    return {
        "reply": ai_messages[-1].content if ai_messages else "",
        "dialog_state": current_stage,
        "message_count": len(state.get("messages", [])),
    }


_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "temporarily",
    "connection",
    "reset by peer",
    "rate limit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "unavailable",
)


def is_retryable(exc: BaseException) -> bool:
    """判断异常是否值得重试。

    默认**不**重试：宁可快速失败，也不要因为重复调用而重复计费。
    只有明确的瞬时错误（连接类、限流、5xx）才放宽。
    """
    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status_code, int):
        return status_code == 429 or 500 <= status_code < 600

    if isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return True

    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if any(marker in name for marker in ("connection", "ratelimit", "timeout")):
        return True
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _backoff_delay(attempt: int) -> float:
    """指数退避 + 抖动，避免多个任务同时重试造成尖峰。"""
    base = settings.turn_retry_base_seconds
    return base * (2 ** (attempt - 1)) * (0.5 + random.random())


async def run_turn(
    store: TaskStore,
    task_id: str,
    session_id: str,
    message: str,
) -> TaskRecord | None:
    """执行一轮对话并把结果写回任务状态机。

    重试只覆盖瞬时错误；**超时不重试**——``to_thread`` 杀不掉已在跑的线程，
    重试只会让同一轮推理重复计费。
    """
    try:
        await store.transition(task_id, TaskStatus.RUNNING)
    except Exception:
        # 任务可能已被取消（QUEUED -> CANCELLED），此时不该继续执行
        logger.exception("task %s: cannot enter RUNNING, abort", task_id)
        return None

    # 续上创建任务时的 trace：inline 模式是同一个进程，queue 模式则是另一个进程，
    # 但 trace_id 随任务记录一起走，日志链路不会断。
    current = await store.get(task_id)
    if current is not None and current.trace_id:
        set_trace_id(current.trace_id)

    max_attempts = max(1, settings.turn_max_attempts)
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        try:
            async with session_lock(session_id):
                result = await asyncio.wait_for(
                    asyncio.to_thread(invoke_graph_sync, session_id, message),
                    timeout=settings.turn_timeout_seconds,
                )
            return await _settle(
                store,
                task_id,
                TaskStatus.SUCCEEDED,
                result={**result, "attempts": attempt},
            )

        except TimeoutError:
            logger.warning(
                "task %s timed out after %ss (attempt %d/%d)",
                task_id,
                settings.turn_timeout_seconds,
                attempt,
                max_attempts,
            )
            return await _settle(
                store,
                task_id,
                TaskStatus.TIMEOUT,
                error=f"turn exceeded {settings.turn_timeout_seconds}s",
            )

        except asyncio.CancelledError:
            logger.info("task %s cancelled", task_id)
            with contextlib.suppress(Exception):
                await store.transition(task_id, TaskStatus.CANCELLED, error="cancelled")
            raise

        except Exception as exc:  # noqa: BLE001 - 兜底，保证状态机一定收敛
            last_error = f"{type(exc).__name__}: {exc}"

            if attempt >= max_attempts or not is_retryable(exc):
                logger.exception(
                    "task %s failed (attempt %d/%d)", task_id, attempt, max_attempts
                )
                return await _settle(
                    store, task_id, TaskStatus.FAILED, error=last_error
                )

            delay = _backoff_delay(attempt)
            from app.metrics import RETRIES

            RETRIES.inc()
            logger.warning(
                "task %s transient failure (attempt %d/%d): %s; retry in %.2fs",
                task_id,
                attempt,
                max_attempts,
                last_error,
                delay,
            )
            await asyncio.sleep(delay)

    # 理论上不可达：循环内要么 return 要么重试
    return await _settle(store, task_id, TaskStatus.FAILED, error=last_error)
