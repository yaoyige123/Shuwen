"""会话状态（checkpoint）持久化工厂（P2）。

三种后端，只影响这一个模块：

    memory   —— ``MemorySaver``，进程内，重启即丢（默认，开发用）
    sqlite   —— ``SqliteSaver``，单文件持久化，**无需服务端**，便于自动化验证
    postgres —— ``PostgresSaver``，多副本共享，生产用

为什么做成工厂：LangGraph 的 checkpointer 是在 ``compile()`` 时注入的，
换后端不该动 ``Graph/`` 的定义，也不该动业务代码。

注意：PostgresSaver 需要 ``langgraph-checkpoint-postgres``，它要求
langchain-core >= 0.2（即 langgraph 1.x 栈）。在本项目升级到 1.x 之前
这个后端是无法安装的。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_checkpointer: Any | None = None
_graph: Any | None = None


def build_checkpointer() -> Any | None:
    """按配置创建 checkpointer；memory 后端返回 None。"""
    global _checkpointer

    with _lock:
        if _checkpointer is not None:
            return _checkpointer

        backend = settings.checkpoint_backend

        if backend == "sqlite":
            from langgraph.checkpoint.sqlite import SqliteSaver

            conn: Any = sqlite3.connect(
                settings.checkpoint_sqlite_path, check_same_thread=False
            )
            saver: Any = SqliteSaver(conn)
            saver.setup()
            _checkpointer = saver
            logger.info(
                "checkpoint backend = sqlite (%s)", settings.checkpoint_sqlite_path
            )

        elif backend == "postgres":
            if not settings.database_url:
                raise RuntimeError(
                    "CHECKPOINT_BACKEND=postgres 时必须配置 DATABASE_URL"
                )

            import psycopg
            from langgraph.checkpoint.postgres import PostgresSaver

            conn = psycopg.connect(settings.database_url, autocommit=True)
            saver = PostgresSaver(conn)
            saver.setup()
            _checkpointer = saver
            logger.info("checkpoint backend = postgres")

        else:
            logger.info("checkpoint backend = memory（进程内，重启即丢）")
            return None

        return _checkpointer


def get_graph() -> Any:
    """返回编译好的图（复用同一个 checkpointer）。"""
    global _graph

    with _lock:
        if _graph is not None:
            return _graph

    from Graph.graph import build_graph

    graph = build_graph(build_checkpointer())
    with _lock:
        _graph = graph
    return graph


def reset() -> None:
    """清空缓存（测试用：模拟进程重启）。"""
    global _checkpointer, _graph
    with _lock:
        _checkpointer = None
        _graph = None
