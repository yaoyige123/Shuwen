"""会话级事件注册表。

背景
----
重构前 ``Graph/nodes.py`` 在函数体里 ``from web.app import graph_qa_events``，
即领域层（Agent 编排）反向依赖 Flask 应用模块。除了分层不干净，它还会在
``web.app → Graph.graph → Graph.nodes`` 的导入链上埋下循环导入风险。

现在注册表下沉到这里：Web 层负责 ``register()/signal()``，领域层只
``event()/wait()``，两侧都不再互相 import。

已知缺陷（P0 保持行为不变，P1 修复）
----------------------------------
``web/app.py`` 目前只在查询完成时 ``signal()``，从未 ``register()``，
所以 ``Risk_Agent`` 里的 ``wait(timeout=30)`` 实际上从未阻塞过 —— 事件字典
永远是空的。P1 把任务编排搬到 Redis/Celery 时用真正的消息总线替换这里。
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_events: dict[str, threading.Event] = {}
# 预取结果：GraphRAG 查询完成后先放这里，由 Risk_Agent 取用，
# 这样不必从后台线程去改 LangGraph 的 state（跨线程写图状态不安全）。
_results: dict[str, str] = {}
# 结果来自哪一层检索（global / local / keyword），用于打点和溯源
_paths: dict[str, str] = {}
# 已经等过预取的会话：每会话只等一次，超时后不再重复等
_waited: set[str] = set()


def register(session_id: str) -> threading.Event:
    """为会话创建（或复用）事件对象。"""
    with _lock:
        return _events.setdefault(session_id, threading.Event())


def event(session_id: str) -> threading.Event | None:
    """取会话事件；未注册返回 None。"""
    return _events.get(session_id)


def wait(session_id: str, timeout: float | None = None) -> bool:
    """等待会话事件；未注册视为已完成（返回 False，不阻塞）。"""
    ev = _events.get(session_id)
    if ev is None:
        return False
    return ev.wait(timeout=timeout)


def signal(session_id: str) -> None:
    """唤醒等待方；未注册时为空操作。"""
    ev = _events.get(session_id)
    if ev is not None:
        ev.set()


def set_result(session_id: str, value: str, path: str = "global") -> None:
    """保存 GraphRAG 预取结果及其来源层级。"""
    with _lock:
        _results[session_id] = value
        _paths[session_id] = path


def get_result(session_id: str) -> str:
    """取预取结果；没有则返回空串。"""
    return _results.get(session_id, "")


def get_path(session_id: str) -> str:
    """取预取结果的来源层级；没有则返回空串。"""
    return _paths.get(session_id, "")


def claim_wait(session_id: str) -> bool:
    """申请"等待预取"的资格：每个会话只有第一次返回 True。

    避免预取一直失败时，风险阶段每一轮都各等一次超时。
    """
    with _lock:
        if session_id in _waited:
            return False
        _waited.add(session_id)
        return True


def pop_result(session_id: str) -> str:
    """取出并删除（用一次就够，避免串到下一轮）。"""
    with _lock:
        _paths.pop(session_id, None)
        return _results.pop(session_id, "")


def reset(session_id: str) -> None:
    """会话结束时清理，避免字典无限增长。"""
    with _lock:
        _events.pop(session_id, None)
        _results.pop(session_id, None)
        _paths.pop(session_id, None)
        _waited.discard(session_id)


def clear() -> None:
    """清空全部（测试用）。"""
    with _lock:
        _events.clear()
        _results.clear()
        _paths.clear()
        _waited.clear()


def sessions() -> list[str]:
    """当前已注册的会话 ID。"""
    return list(_events)
