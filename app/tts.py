"""讯飞语音合成服务（P2+，供 FastAPI 使用）。

与旧的流式回调实现相比：
* **有超时**：WebSocket 收音频最多等 ``timeout`` 秒，不会把请求挂死；
* **有缓存**：按 ``sha1(规范化文本)`` 落盘成 wav，重复请求直接读盘；
* **无框架依赖**：不再绑 Flask，FastAPI 与 worker 都能用。

合成是阻塞的（websocket-client），调用方需放进线程，例如
``await asyncio.to_thread(service.synthesize, text)``。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import ssl
import threading
import wave
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import websocket

from app.config import settings

logger = logging.getLogger(__name__)

_TTS_URL = "wss://tts-api.xfyun.cn/v2/tts"
_TTS_HOST = "ws-api.xfyun.cn"


class TtsService:
    def __init__(
        self,
        app_id: str,
        api_key: str,
        api_secret: str,
        cache_dir: Path,
        *,
        voice: str = "xiaoyan",
        speed: int = 70,
        timeout: float = 15.0,
    ) -> None:
        self._app_id = app_id
        self._api_key = api_key
        self._api_secret = api_secret
        self._cache_dir = cache_dir
        self._voice = voice
        self._speed = speed
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self._app_id and self._api_key and self._api_secret)

    # ---------- 缓存 ----------

    @staticmethod
    def text_digest(text: str) -> str:
        normalized = " ".join(text.split())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20]

    def cache_path(self, text: str) -> Path:
        return self._cache_dir / f"tts_{self.text_digest(text)}.wav"

    # ---------- 对外接口 ----------

    def synthesize(self, text: str) -> bytes | None:
        """返回 wav 字节；未配置凭据或合成失败时返回 None。"""
        if not self.enabled or not text.strip():
            return None

        cached = self.cache_path(text)
        if cached.exists() and cached.stat().st_size > 0:
            return cached.read_bytes()

        pcm = self._fetch_pcm(text)
        if not pcm:
            return None

        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".part")
        with wave.open(str(tmp), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(pcm)
        os.replace(tmp, cached)
        return cached.read_bytes()

    # ---------- 讯飞 WebSocket ----------

    def _ws_url(self) -> str:
        date = format_date_time(mktime(datetime.now().timetuple()))
        origin = f"host: {_TTS_HOST}\ndate: {date}\nGET /v2/tts HTTP/1.1"
        digest = hmac.new(
            self._api_secret.encode("utf-8"),
            origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature = base64.b64encode(digest).decode()
        authorization = base64.b64encode(
            (
                f'api_key="{self._api_key}", algorithm="hmac-sha256", '
                f'headers="host date request-line", signature="{signature}"'
            ).encode()
        ).decode()
        query = urlencode(
            {"authorization": authorization, "date": date, "host": _TTS_HOST}
        )
        return f"{_TTS_URL}?{query}"

    def _fetch_pcm(self, text: str) -> bytes | None:
        chunks = bytearray()
        state: dict[str, str] = {}

        def on_message(_ws, message) -> None:  # type: ignore[no-untyped-def]
            try:
                data = json.loads(message)
            except Exception:  # noqa: BLE001
                return
            if data.get("code") != 0:
                state["error"] = f"code={data.get('code')} {data.get('message')}"
                _ws.close()
                return
            payload = data.get("data") or {}
            if payload.get("audio"):
                chunks.extend(base64.b64decode(payload["audio"]))
            if payload.get("status") == 2:
                _ws.close()

        def on_error(_ws, error) -> None:  # type: ignore[no-untyped-def]
            state["error"] = str(error)

        def on_open(ws) -> None:  # type: ignore[no-untyped-def]
            frame = {
                "common": {"app_id": self._app_id},
                "business": {
                    "aue": "raw",
                    "auf": "audio/L16;rate=16000",
                    "vcn": self._voice,
                    "tte": "utf8",
                    "speed": self._speed,
                },
                "data": {
                    "status": 2,
                    "text": base64.b64encode(text.encode("utf-8")).decode(),
                },
            }
            ws.send(json.dumps(frame))

        websocket.enableTrace(False)
        app = websocket.WebSocketApp(
            self._ws_url(),
            on_message=on_message,
            on_error=on_error,
            on_close=lambda *args, **kwargs: None,
        )
        app.on_open = on_open

        worker = threading.Thread(
            target=app.run_forever,
            kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}},
            daemon=True,
        )
        worker.start()
        worker.join(self._timeout)

        if worker.is_alive():
            logger.warning("TTS timeout after %ss, closing socket", self._timeout)
            try:
                app.close()
            except Exception:  # noqa: BLE001
                pass
            return None

        if state.get("error"):
            logger.warning("TTS failed: %s", state["error"])
            return None

        return bytes(chunks) if chunks else None


@lru_cache(maxsize=1)
def get_tts_service() -> TtsService:
    cache_dir = Path(settings.audio_dir)
    if not cache_dir.is_absolute():
        cache_dir = Path(__file__).resolve().parents[1] / cache_dir

    service = TtsService(
        settings.xunfei_app_id,
        settings.xunfei_api_key,
        settings.xunfei_api_secret,
        cache_dir,
        voice=settings.xunfei_tts_voice,
        speed=settings.xunfei_tts_speed,
    )
    if not service.enabled:
        logger.warning("未配置讯飞凭据，/api/v1/tts 将返回 503（页面语音按钮会提示）")
    return service
