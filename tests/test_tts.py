"""TTS 服务与 /api/v1/tts 端点。

真正调用讯飞需要凭据，这里只覆盖不触网的部分：
* 未配置凭据时服务禁用、端点返回 503；
* 缓存键对空白不敏感、命中缓存不发网络请求；
* 端点成功/失败分支用替身注入。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import routes as routes_module
from app.main import create_app
from app.tasks import InMemoryTaskStore
from app.tts import TtsService


def _service(tmp_path, *, enabled: bool) -> TtsService:
    return TtsService(
        "app-id" if enabled else "",
        "key" if enabled else "",
        "secret" if enabled else "",
        tmp_path,
    )


class _FakeService:
    def __init__(self, payload: bytes | None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._payload = payload

    def synthesize(self, _text: str) -> bytes | None:
        return self._payload


@pytest.fixture()
def client() -> TestClient:
    with TestClient(create_app(task_store=InMemoryTaskStore())) as test_client:
        yield test_client


# ---------------- 服务本身 ----------------


def test_disabled_without_credentials(tmp_path):
    service = _service(tmp_path, enabled=False)
    assert service.enabled is False
    assert service.synthesize("你好") is None


def test_cache_path_is_stable_and_whitespace_insensitive(tmp_path):
    service = _service(tmp_path, enabled=True)

    assert service.text_digest("你好 世界") == service.text_digest("  你好   世界  ")
    assert service.cache_path("你好 世界") == service.cache_path("你好   世界")
    assert service.cache_path("A") != service.cache_path("B")


def test_cached_file_is_returned_without_network(tmp_path):
    service = _service(tmp_path, enabled=True)

    cached = service.cache_path("已经合成过")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"RIFFfakewav")

    def _boom(*_args, **_kwargs):
        raise AssertionError("命中缓存时不应发起网络请求")

    service._fetch_pcm = _boom  # type: ignore[method-assign]

    assert service.synthesize("已经合成过") == b"RIFFfakewav"


def test_empty_text_returns_none(tmp_path):
    service = _service(tmp_path, enabled=True)
    assert service.synthesize("   ") is None


# ---------------- HTTP 端点 ----------------


def test_tts_endpoint_returns_audio(client: TestClient, monkeypatch):
    # 注意：routes 里是 `from app.tts import get_tts_service`，
    # 所以必须打在 routes 模块上，打 app.tts 无效。
    monkeypatch.setattr(
        routes_module, "get_tts_service", lambda: _FakeService(b"RIFFdata")
    )

    response = client.post("/api/v1/tts", json={"text": "你好"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFFdata"


def test_tts_endpoint_503_when_not_configured(client: TestClient, monkeypatch):
    monkeypatch.setattr(
        routes_module,
        "get_tts_service",
        lambda: _FakeService(None, enabled=False),
    )

    response = client.post("/api/v1/tts", json={"text": "你好"})

    assert response.status_code == 503
    assert "XUNFEI" in response.json()["detail"]


def test_tts_endpoint_502_when_synthesis_fails(client: TestClient, monkeypatch):
    monkeypatch.setattr(
        routes_module, "get_tts_service", lambda: _FakeService(None)
    )

    response = client.post("/api/v1/tts", json={"text": "你好"})

    assert response.status_code == 502


def test_tts_endpoint_rejects_empty_text(client: TestClient):
    assert client.post("/api/v1/tts", json={"text": ""}).status_code == 422
