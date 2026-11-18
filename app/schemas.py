"""API 的数据契约（Pydantic v2）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MessageIn(BaseModel):
    """一次用户发言。"""

    message: str = Field(min_length=1, max_length=4000, description="用户输入文本")
    client_message_id: str | None = Field(
        default=None, description="前端生成的客户端消息 ID，便于对账（可选）"
    )


class TaskAccepted(BaseModel):
    """202 响应：任务已入队。"""

    task_id: str
    session_id: str
    status: str
    idempotent_replay: bool = Field(
        default=False, description="True 表示命中幂等键，返回的是已有任务"
    )


class TaskOut(BaseModel):
    """任务状态查询结果。"""

    task_id: str
    session_id: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class HealthOut(BaseModel):
    status: str
    version: str


class TtsIn(BaseModel):
    """语音合成请求。"""

    text: str = Field(min_length=1, max_length=2000, description="要合成的文本")
