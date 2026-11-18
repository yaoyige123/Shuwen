"""端到端：Redis 任务存储 + ARQ worker（真正跨进程）。

前置：Redis 已在 REDIS_URL 上运行（``docker compose up -d redis``）。

流程：
    1. 起 ARQ worker 子进程；
    2. 起 uvicorn 子进程（TASK_BACKEND=redis, EXECUTION_MODE=queue）；
    3. 经 HTTP 提交消息，轮询任务到终态；
    4. **直接读 Redis 里的任务 Hash**，确认状态是 worker 写进去的；
    5. 收尾：停掉两个子进程。

用法：
    python scripts/e2e_queue.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PORT = 8020
BASE = f"http://127.0.0.1:{PORT}"
# 每次跑用全新的会话与幂等键，避免命中上一轮遗留的 Redis 记录
RUN_ID = uuid.uuid4().hex[:8]
SESSION_ID = f"e2e-queue-{RUN_ID}"
IDEMPOTENCY_KEY = f"e2e-queue-{RUN_ID}"
TERMINAL = {"succeeded", "failed", "timeout", "cancelled"}
TURN_TIMEOUT = 240
LOG_DIR = ROOT / "logs"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TASK_BACKEND": "redis",
            "EXECUTION_MODE": "queue",
            "CHECKPOINT_BACKEND": "memory",
            "PYTHONPATH": str(ROOT),
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return env


def _spawn(args: list[str], log_name: str) -> tuple[subprocess.Popen, Path]:
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / log_name
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        args,
        cwd=str(ROOT),
        env=_env(),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._log_handle = handle  # type: ignore[attr-defined]
    return proc, log_path


def _tail(path: Path, lines: int = 25) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return f"(no log at {path})"
    return "\n".join(content[-lines:])


async def _redis_ping() -> bool:
    import redis.asyncio as aioredis

    from app.config import settings

    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        return bool(await asyncio.wait_for(client.ping(), timeout=3))
    finally:
        await client.aclose()


async def _task_hash(task_id: str) -> dict[str, str]:
    import redis.asyncio as aioredis

    from app.config import settings

    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        return await client.hgetall(f"task:{task_id}")
    finally:
        await client.aclose()


async def _drive(client: httpx.AsyncClient) -> dict:
    response = await client.post(
        f"/api/v1/sessions/{SESSION_ID}/messages",
        json={"message": "你好，我姓张"},
        headers={"Idempotency-Key": IDEMPOTENCY_KEY},
    )
    print(f"[api ] POST -> {response.status_code} {response.text}")
    if response.status_code != 202:
        raise SystemExit(f"expected 202, got {response.status_code}")

    task_id = response.json()["task_id"]
    deadline = time.monotonic() + TURN_TIMEOUT
    body: dict = {}

    while time.monotonic() < deadline:
        poll = await client.get(f"/api/v1/tasks/{task_id}")
        body = poll.json()
        if body["status"] in TERMINAL:
            break
        await asyncio.sleep(1.0)

    print(f"[api ] status       = {body.get('status')}")
    if body.get("result"):
        print(f"[api ] reply        = {body['result'].get('reply', '')[:60]}")
        print(f"[api ] attempts     = {body['result'].get('attempts')}")
    if body.get("error"):
        print(f"[api ] error        = {body['error']}")

    record = await _task_hash(task_id)
    print(f"[redis] task hash    = {len(record)} 个字段")
    print(f"[redis] status field = {record.get('status')!r}")

    body["_redis_ok"] = record.get("status") == "succeeded"
    return body


async def _wait_health(timeout: float = 40.0) -> bool:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(base_url=BASE, timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get("/api/v1/health")
                if response.status_code == 200:
                    return True
            except Exception:  # noqa: BLE001 - 启动期连接失败是正常的
                pass
            await asyncio.sleep(0.5)
    return False


def main() -> int:
    if not asyncio.run(_redis_ping()):
        print("Redis 不可用：请先 docker compose up -d redis")
        return 2

    worker, worker_log = _spawn(
        [sys.executable, "-m", "arq", "app.worker.WorkerSettings"],
        "e2e-worker.log",
    )
    api, api_log = _spawn(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        "e2e-api.log",
    )

    ok = False
    try:
        if not asyncio.run(_wait_health()):
            print("[api ] 启动失败")
            return 1

        print("[api ] healthy")
        time.sleep(3)  # 给 worker 一点注册时间

        async def run() -> dict:
            async with httpx.AsyncClient(base_url=BASE, timeout=30.0) as client:
                return await _drive(client)

        body = asyncio.run(run())
        ok = (
            body.get("status") == "succeeded"
            and bool((body.get("result") or {}).get("reply"))
            and bool(body.get("_redis_ok"))
        )
    finally:
        for proc, _log_path in ((api, api_log), (worker, worker_log)):
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()
            handle = getattr(proc, "_log_handle", None)
            if handle is not None:
                handle.close()

        if not ok:
            print("\n--- api log tail ---")
            print(_tail(api_log))
            print("\n--- worker log tail ---")
            print(_tail(worker_log))

    print()
    print("RESULT:", "PASS - Redis + ARQ 跨进程链路打通" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
