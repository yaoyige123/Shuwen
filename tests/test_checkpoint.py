"""checkpoint 持久化测试（SQLite 后端，无需任何服务端）。

模拟「进程重启」的方式是清空 checkpoint 模块的缓存，让下一次 get_graph()
重新建立连接并重新编译图——如果状态还在，说明它真的落在文件里。
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph.checkpoint.sqlite")

from app import checkpoint as checkpoint_module  # noqa: E402
from app.config import settings  # noqa: E402

THREAD_ID = "t-sqlite-verify"


@pytest.fixture()
def sqlite_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_backend", "sqlite")
    monkeypatch.setattr(settings, "checkpoint_sqlite_path", str(tmp_path / "cp.sqlite"))
    checkpoint_module.reset()
    yield tmp_path
    checkpoint_module.reset()


def test_state_survives_rebuild(sqlite_backend):
    config = {"configurable": {"thread_id": THREAD_ID}}

    graph = checkpoint_module.get_graph()
    graph.update_state(
        config,
        {
            "dialog_state": "verify_information",
            "session_id": THREAD_ID,
            "user_information": "姓名: 张三",
        },
    )

    # 模拟进程重启：丢掉缓存，重新建连接、重新编译图
    checkpoint_module.reset()

    graph2 = checkpoint_module.get_graph()
    values = graph2.get_state(config).values

    assert values.get("session_id") == THREAD_ID
    assert values.get("user_information") == "姓名: 张三"


def test_unknown_thread_is_empty(sqlite_backend):
    config = {"configurable": {"thread_id": "never-written"}}
    graph = checkpoint_module.get_graph()
    values = graph.get_state(config).values
    assert not values.get("session_id")


def test_memory_backend_returns_plain_graph(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_backend", "memory")
    checkpoint_module.reset()
    graph = checkpoint_module.get_graph()
    assert graph is not None
    checkpoint_module.reset()
