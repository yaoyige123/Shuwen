"""P0：会话事件注册表的行为测试。

这个注册表是 web 层与领域层解耦的关键：Graph/nodes.py 只读它，
web/app.py 只发信号，两边不再互相 import。
"""

import threading
import time

import pytest

from app import events


@pytest.fixture(autouse=True)
def _clean_registry():
    events.clear()
    yield
    events.clear()


def test_event_returns_none_when_not_registered():
    assert events.event("no-such-session") is None
    # 未注册时 wait 必须立即返回 False，绝不阻塞
    started = time.monotonic()
    assert events.wait("no-such-session", timeout=5) is False
    assert time.monotonic() - started < 0.5


def test_register_is_idempotent():
    first = events.register("s1")
    second = events.register("s1")
    assert first is second
    assert events.sessions() == ["s1"]


def test_signal_unblocks_waiter():
    ev = events.register("s2")

    threading.Timer(0.05, events.signal, args=("s2",)).start()

    assert ev.wait(timeout=2) is True


def test_signal_without_registration_is_noop():
    """未注册时发信号不应抛错（web 层可能先于注册就完成查询）。"""
    events.signal("ghost")
    assert events.event("ghost") is None


def test_reset_removes_session():
    events.register("s3")
    events.reset("s3")
    assert events.event("s3") is None
    assert events.sessions() == []


def test_clear_removes_all():
    events.register("a")
    events.register("b")
    events.clear()
    assert events.sessions() == []
