"""讯飞 TTS 连通性探针。

用来区分三种失败：凭据无效 / 签名构造错误 / 网络不可达。

用法：
    python scripts/probe_tts.py --app-id X --api-key Y --api-secret Z
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import ssl
import sys
import threading
from datetime import datetime
from pathlib import Path
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import websocket

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEXT = "你好，我是术前访视助手。"


def build_url(app_id: str, api_key: str, api_secret: str, sign_host: str, url_host: str) -> str:
    date = format_date_time(mktime(datetime.now().timetuple()))
    origin = f"host: {sign_host}\ndate: {date}\nGET /v2/tts HTTP/1.1"
    digest = hmac.new(
        api_secret.encode(), origin.encode(), digestmod=hashlib.sha256
    ).digest()
    signature = base64.b64encode(digest).decode()
    authorization = base64.b64encode(
        (
            f'api_key="{api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature}"'
        ).encode()
    ).decode()
    query = urlencode({"authorization": authorization, "date": date, "host": sign_host})
    return f"wss://{url_host}/v2/tts?{query}"


def probe(app_id: str, api_key: str, api_secret: str, sign_host: str, url_host: str) -> dict:
    result: dict = {"sign_host": sign_host, "url_host": url_host, "code": None, "message": "", "bytes": 0, "error": ""}

    def on_message(ws, message):
        try:
            data = json.loads(message)
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"parse: {exc}"
            return
        if data.get("code") != 0:
            result["code"] = data.get("code")
            result["message"] = data.get("message", "")
            ws.close()
            return
        payload = data.get("data") or {}
        if payload.get("audio"):
            result["bytes"] += len(base64.b64decode(payload["audio"]))
        if payload.get("status") == 2:
            ws.close()

    def on_error(_ws, error):
        result["error"] = str(error)

    def on_open(ws):
        ws.send(
            json.dumps(
                {
                    "common": {"app_id": app_id},
                    "business": {
                        "aue": "raw",
                        "auf": "audio/L16;rate=16000",
                        "vcn": "xiaoyan",
                        "tte": "utf8",
                        "speed": 70,
                    },
                    "data": {
                        "status": 2,
                        "text": base64.b64encode(TEXT.encode()).decode(),
                    },
                }
            )
        )

    app = websocket.WebSocketApp(
        build_url(app_id, api_key, api_secret, sign_host, url_host),
        on_message=on_message,
        on_error=on_error,
        on_close=lambda *a, **k: None,
    )
    app.on_open = on_open
    worker = threading.Thread(
        target=app.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}}, daemon=True
    )
    worker.start()
    worker.join(12)
    if worker.is_alive():
        result["error"] = result["error"] or "timeout"
        try:
            app.close()
        except Exception:  # noqa: BLE001
            pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-secret", required=True)
    args = parser.parse_args()

    combos = [
        ("tts-api.xfyun.cn", "tts-api.xfyun.cn"),
        ("ws-api.xfyun.cn", "tts-api.xfyun.cn"),
        ("ws-api.xfyun.cn", "ws-api.xfyun.cn"),
    ]

    ok = False
    for sign_host, url_host in combos:
        res = probe(args.app_id, args.api_key, args.api_secret, sign_host, url_host)
        status = "OK" if res["bytes"] > 0 else "FAIL"
        print(
            f"[{status}] sign={sign_host:20s} url={url_host:20s} "
            f"code={res['code']} bytes={res['bytes']} msg={res['message'] or res['error']}"
        )
        if res["bytes"] > 0:
            ok = True

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
