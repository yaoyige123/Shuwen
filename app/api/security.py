"""鉴权与限流的 FastAPI 依赖（P2）。

鉴权策略：
    ``API_KEY`` 为空时视为开发模式——不校验（但要打日志提醒）；
    配置后就要求 ``X-API-Key`` 头，用常量时间比较避免时序侧信道。

限流键：优先用 api key，没有就用客户端 IP。
"""

from __future__ import annotations

import logging
import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import settings
from app.ratelimit import RateLimiter

logger = logging.getLogger(__name__)


async def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    expected = settings.api_key
    if not expected:
        # 开发模式：不拦截，但每次启动提醒一次（见 main.py 的日志）
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-API-Key",
        )


def _client_key(request: Request) -> str:
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"key:{api_key}"
    client = request.client
    return f"ip:{client.host}" if client else "anonymous"


async def enforce_rate_limit(request: Request) -> None:
    limiter: RateLimiter = request.app.state.rate_limiter
    retry_after = await limiter.acquire(_client_key(request))
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={"Retry-After": str(max(int(retry_after) + 1, 1))},
        )


AuthDep = Annotated[None, Depends(require_api_key)]
RateLimitDep = Annotated[None, Depends(enforce_rate_limit)]
