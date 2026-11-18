"""可观测性基础设施：trace_id 贯穿 + 访问指标中间件（P4）。

设计取舍
--------
* trace_id 用 ``contextvars``：同一次请求里的业务代码、后台 asyncio 任务
  （``asyncio.create_task`` 会复制当前上下文）都自动带同一个 id，无需层层传参。
* 日志格式统一在这里配置一次，业务代码只管 ``logger.info``，不用手动拼 trace。
* 中间件不 reset contextvar——SSE 的响应体是在 ``call_next`` 返回之后才生成的，
  reset 会让流式日志丢 trace；每个请求本身是独立 task，不会串味。
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")

LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(trace_id)s] %(name)s: %(message)s"


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def set_trace_id(value: str) -> contextvars.Token[str]:
    return _trace_id.set(value)


def get_trace_id() -> str:
    return _trace_id.get()


class TraceIdFilter(logging.Filter):
    """给每条日志补上当前 trace_id。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = get_trace_id() or "-"
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """统一日志格式（幂等，可重复调用）。"""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(TraceIdFilter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    # 第三方库降噪：httpx 每个请求打一行，会把业务日志淹掉
    for noisy in ("httpx", "httpcore", "urllib3", "nano-graphrag", "nano-vectordb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """一个中间件同时负责 trace_id 传播与 HTTP 指标。

    放在一个中间件里是为了控制顺序：先定 trace，再计时，
    失败时也能记录到（异常路径下 status 记为 500）。
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        from app.metrics import HTTP_LATENCY, HTTP_REQUESTS

        trace_id = request.headers.get("x-request-id") or new_trace_id()
        set_trace_id(trace_id)

        started = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = trace_id
            return response
        finally:
            elapsed = time.perf_counter() - started
            route = request.scope.get("route")
            # 用路由模板而不是原始 URL，避免 /tasks/{id} 把基数打爆
            path = getattr(route, "path", None) or "unmatched"
            HTTP_LATENCY.labels(method=request.method, path=path).observe(elapsed)
            HTTP_REQUESTS.labels(
                method=request.method, path=path, status=str(status_code)
            ).inc()
