"""端到端验证：FastAPI + Postgres checkpoint + 真实一轮对话。

做的事：
    1. 用 pgserver 起一个自带 PostgreSQL（无需 Docker）；
    2. 配置 CHECKPOINT_BACKEND=postgres，用 ASGI 传输跑完整 HTTP 链路；
    3. 提交消息 → 轮询任务到终态 → 断言回复非空；
    4. **关掉连接**，用新连接 + 新 saver 重新读会话状态 → 断言确实落库。

用法：
    python scripts/e2e_postgres.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TERMINAL = {"succeeded", "failed", "timeout", "cancelled"}
SESSION_ID = "e2e-pg-session"
TURN_TIMEOUT_SECONDS = 240


def _boot_postgres() -> tuple[object, str]:
    import pgserver

    data_dir = Path(tempfile.mkdtemp(prefix="cmekg-e2e-pg-"))
    server = pgserver.get_server(str(data_dir))
    return server, server.get_uri()


def _configure(dsn: str) -> None:
    # 必须在导入 app.config 之前设置：Settings 是进程内缓存的单例
    os.environ["DATABASE_URL"] = dsn
    os.environ["CHECKPOINT_BACKEND"] = "postgres"
    os.environ["TASK_BACKEND"] = "memory"      # 任务状态先走内存；Redis 另测
    os.environ["EXECUTION_MODE"] = "inline"


async def _run_turn_via_api() -> dict:
    import httpx

    from app.main import create_app

    app = create_app()

    # 手动进入 lifespan：ASGITransport 不会自动触发它
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://e2e",
            timeout=30.0,
        ) as client:
            health = await client.get("/api/v1/health")
            print(f"[api  ] health {health.status_code} {health.json()}")

            response = await client.post(
                f"/api/v1/sessions/{SESSION_ID}/messages",
                json={"message": "你好，我姓张，要做胆囊切除手术"},
                headers={"Idempotency-Key": "e2e-pg-1"},
            )
            print(f"[api  ] POST messages -> {response.status_code} {response.text}")
            if response.status_code != 202:
                raise SystemExit(f"expected 202, got {response.status_code}")

            task_id = response.json()["task_id"]

            deadline = time.monotonic() + TURN_TIMEOUT_SECONDS
            body: dict = {}
            while time.monotonic() < deadline:
                poll = await client.get(f"/api/v1/tasks/{task_id}")
                body = poll.json()
                if body["status"] in TERMINAL:
                    break
                await asyncio.sleep(1.0)

            print(f"[api  ] task status = {body.get('status')}")
            if body.get("result"):
                print(f"[api  ] reply       = {body['result'].get('reply', '')[:60]}")
                print(f"[api  ] dialog_state= {body['result'].get('dialog_state')}")
            if body.get("error"):
                print(f"[api  ] error       = {body['error']}")
            return body


def _read_checkpoint_from_new_connection(dsn: str) -> bool:
    """关掉旧连接后，用全新连接读状态——证明数据在 PG 里。"""
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver

    from Graph.graph import build_graph

    conn = psycopg.connect(dsn, autocommit=True)
    try:
        saver = PostgresSaver(conn)
        saver.setup()
        graph = build_graph(saver)
        snapshot = graph.get_state({"configurable": {"thread_id": SESSION_ID}})
        values = snapshot.values or {}
        messages = values.get("messages") or []

        print(f"[db   ] session_id   = {values.get('session_id')!r}")
        print(f"[db   ] dialog_state = {values.get('dialog_state')!r}")
        print(f"[db   ] messages     = {len(messages)} 条")
        if messages:
            last = messages[-1]
            print(f"[db   ] last message = {type(last).__name__}: {str(last.content)[:50]}")

        return values.get("session_id") == SESSION_ID and len(messages) > 0
    finally:
        conn.close()


def main() -> int:
    server, dsn = _boot_postgres()
    print(f"[pg   ] {dsn}")
    _configure(dsn)

    try:
        body = asyncio.run(_run_turn_via_api())
    except SystemExit as exc:
        print(f"[api  ] aborted: {exc}")
        return 1

    ok_api = body.get("status") == "succeeded" and bool(
        (body.get("result") or {}).get("reply")
    )

    # 清掉进程内缓存，模拟"重启后再读"
    from app import checkpoint as checkpoint_module

    checkpoint_module.reset()
    ok_db = _read_checkpoint_from_new_connection(dsn)

    print()
    print(f"API 链路      : {'PASS' if ok_api else 'FAIL'}")
    print(f"PG 持久化     : {'PASS' if ok_db else 'FAIL'}")
    print("RESULT:", "PASS" if (ok_api and ok_db) else "FAIL")

    try:
        server.cleanup()  # 停掉内置 PG
    except Exception:  # noqa: BLE001 - 清理失败不影响结论
        pass

    return 0 if (ok_api and ok_db) else 1


if __name__ == "__main__":
    raise SystemExit(main())
