"""P4：trace_id 传播与 Prometheus 指标。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.tasks import InMemoryTaskStore, TaskStatus


async def fake_runner(store, task_id, session_id, message):
    await store.transition(task_id, TaskStatus.RUNNING)
    await store.transition(task_id, TaskStatus.SUCCEEDED, result={"reply": "ok"})


@pytest.fixture()
def client() -> TestClient:
    with TestClient(
        create_app(task_store=InMemoryTaskStore(), turn_runner=fake_runner)
    ) as test_client:
        yield test_client


def test_request_id_is_generated(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.headers.get("x-request-id")


def test_provided_request_id_is_preserved(client: TestClient):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "trace-abc-123"})
    assert response.headers["x-request-id"] == "trace-abc-123"


def test_metrics_endpoint_exposes_http_counters(client: TestClient):
    client.get("/api/v1/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "cmekg_http_requests_total" in response.text
    # 标签用路由模板，不是原始 URL
    assert 'path="/api/v1/health"' in response.text


def test_turn_metrics_are_recorded_by_settle():
    """指标是在 _settle 里记的；这里直接测那一层，避免依赖假 runner。"""
    from prometheus_client import REGISTRY

    from app.runner import _observe_turn
    from app.tasks import TaskRecord

    record = TaskRecord(task_id="t", session_id="s", message="m")
    record.status = TaskStatus.SUCCEEDED

    def sample() -> float:
        value = REGISTRY.get_sample_value("cmekg_turns_total", {"status": "succeeded"})
        return value or 0.0

    before = sample()
    _observe_turn(record)
    assert sample() == before + 1


def test_turn_metrics_family_is_exposed(client: TestClient):
    body = client.get("/metrics").text
    assert "cmekg_turns_total" in body
    assert "cmekg_turn_duration_seconds" in body
    assert "cmekg_turn_retries_total" in body


def test_health_is_not_traced_as_unmatched(client: TestClient):
    client.get("/api/v1/health")
    body = client.get("/metrics").text
    assert 'path="unmatched"' not in body
