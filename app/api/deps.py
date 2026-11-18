"""依赖注入 provider（P1）。

统一从 ``request.app.state`` 取，这样测试可以用 ``create_app(store=...,
turn_runner=...)`` 注入替身，无需 monkeypatch 模块级全局变量。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request

from app.dispatch import Dispatcher
from app.tasks import TaskStore

TurnRunner = Callable[..., Awaitable[Any]]


def get_store(request: Request) -> TaskStore:
    return request.app.state.task_store


def get_runner(request: Request) -> TurnRunner:
    return request.app.state.turn_runner


def get_dispatcher(request: Request) -> Dispatcher:
    return request.app.state.dispatcher
