"""HTTP 契约测试：202 + task_id、幂等重放、状态查询、SSE 收尾、404。"""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.tasks import InMemoryTaskStore, TaskStatus

TERMINAL = {"succeeded", "failed", "timeout", "cancelled"}


async def fake_runner(store, task_id, session_id, message):
    await store.transition(task_id, TaskStatus.RUNNING)
    await store.transition(task_id, TaskStatus.SUCCEEDED, result={"reply": f"echo:{message}"})


@pytest.fixture()
def client() -> TestClient:
    store = InMemoryTaskStore()
    # 必须用 with：TestClient 只有在进入上下文时才持有长期运行的事件循环，
    # 否则请求结束后后台任务会被取消，状态永远停在 queued。
    with TestClient(create_app(task_store=store, turn_runner=fake_runner)) as test_client:
        yield test_client


def _submit(client: TestClient, session_id: str = "s1", message: str = "你好", key: str | None = None):
    headers = {"Idempotency-Key": key} if key else {}
    return client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": message},
        headers=headers,
    )


def _wait_terminal(client: TestClient, task_id: str, timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in TERMINAL:
            return body
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} did not reach a terminal state in {timeout}s")


def test_health(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_redirects_to_chat(client: TestClient):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)
    assert response.headers["location"] == "/chat"


def test_chat_page_is_served(client: TestClient):
    response = client.get("/chat")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "CMeKG" in response.text
    # 页面通过这两个接口工作
    assert "/messages" in response.text
    assert "/events" in response.text


def test_post_message_returns_202_with_task_id(client: TestClient):
    response = _submit(client)

    assert response.status_code == 202
    body = response.json()
    assert body["task_id"]
    assert body["session_id"] == "s1"
    assert body["idempotent_replay"] is False


def test_task_completes_and_carries_result(client: TestClient):
    task_id = _submit(client, message="我姓张").json()["task_id"]

    body = _wait_terminal(client, task_id)

    assert body["status"] == "succeeded"
    assert body["result"]["reply"] == "echo:我姓张"
    assert body["started_at"] and body["finished_at"]


def test_idempotency_key_replays_existing_task(client: TestClient):
    first = _submit(client, key="abc")
    second = _submit(client, key="abc")

    assert first.status_code == second.status_code == 202
    assert second.json()["task_id"] == first.json()["task_id"]
    assert second.json()["idempotent_replay"] is True


def test_unknown_task_returns_404(client: TestClient):
    assert client.get("/api/v1/tasks/does-not-exist").status_code == 404
    assert client.get("/api/v1/tasks/does-not-exist/events").status_code == 404


def test_empty_message_is_rejected(client: TestClient):
    response = client.post("/api/v1/sessions/s1/messages", json={"message": ""})
    assert response.status_code == 422


def test_events_stream_delivers_status_and_done(client: TestClient):
    task_id = _submit(client).json()["task_id"]

    with client.stream("GET", f"/api/v1/tasks/{task_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: status" in body
    assert "event: done" in body
    assert '"status": "succeeded"' in body or '"status":"succeeded"' in body


async def slow_runner(store, task_id, session_id, message):
    await store.transition(task_id, TaskStatus.RUNNING)
    await asyncio.sleep(5)
    await store.transition(task_id, TaskStatus.SUCCEEDED, result={"reply": "late"})


@pytest.fixture()
def slow_client() -> TestClient:
    store = InMemoryTaskStore()
    with TestClient(create_app(task_store=store, turn_runner=slow_runner)) as test_client:
        yield test_client


def test_cancel_running_task(slow_client: TestClient):
    task_id = _submit(slow_client).json()["task_id"]

    response = slow_client.post(f"/api/v1/tasks/{task_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_cancel_is_idempotent(slow_client: TestClient):
    task_id = _submit(slow_client).json()["task_id"]

    first = slow_client.post(f"/api/v1/tasks/{task_id}/cancel")
    second = slow_client.post(f"/api/v1/tasks/{task_id}/cancel")

    assert first.json()["status"] == "cancelled"
    assert second.json()["status"] == "cancelled"


def test_cancel_after_success_keeps_result(client: TestClient):
    task_id = _submit(client, message="已完成").json()["task_id"]
    _wait_terminal(client, task_id)

    response = client.post(f"/api/v1/tasks/{task_id}/cancel")

    body = response.json()
    assert body["status"] == "succeeded"
    assert body["result"]["reply"] == "echo:已完成"


def test_cancel_unknown_task_returns_404(client: TestClient):
    assert client.post("/api/v1/tasks/nope/cancel").status_code == 404


def test_auth_required_when_api_key_configured(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "api_key", "s3cret")

    store = InMemoryTaskStore()
    with TestClient(create_app(task_store=store, turn_runner=fake_runner)) as guarded:
        assert guarded.get("/api/v1/health").status_code == 200

        denied = guarded.post("/api/v1/sessions/s1/messages", json={"message": "hi"})
        assert denied.status_code == 401

        allowed = guarded.post(
            "/api/v1/sessions/s1/messages",
            json={"message": "hi"},
            headers={"X-API-Key": "s3cret"},
        )
        assert allowed.status_code == 202


def test_rate_limit_returns_429_with_retry_after():
    from app.ratelimit import MemoryRateLimiter

    store = InMemoryTaskStore()
    limiter = MemoryRateLimiter(limit=2, window_seconds=60)

    with TestClient(
        create_app(task_store=store, turn_runner=fake_runner, rate_limiter=limiter)
    ) as limited:
        first = limited.post("/api/v1/sessions/s1/messages", json={"message": "a"})
        second = limited.post("/api/v1/sessions/s1/messages", json={"message": "b"})
        third = limited.post("/api/v1/sessions/s1/messages", json={"message": "c"})

        assert first.status_code == 202
        assert second.status_code == 202
        assert third.status_code == 429
        assert "retry-after" in {key.lower() for key in third.headers}


def test_rate_limiter_window_slides(monkeypatch):
    from app.ratelimit import MemoryRateLimiter

    limiter = MemoryRateLimiter(limit=2, window_seconds=0.05)

    async def scenario():
        assert await limiter.acquire("k") is None
        assert await limiter.acquire("k") is None
        blocked = await limiter.acquire("k")
        assert blocked is not None

        await asyncio.sleep(0.06)
        assert await limiter.acquire("k") is None

    asyncio.run(scenario())
