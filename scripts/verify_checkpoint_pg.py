"""验证 checkpoint 是否真的落盘（不需要 LLM）。

思路：
    1. 用 PostgresSaver 编译图，写入一段状态；
    2. **关掉连接**，再用新连接 + 新 saver 重新编译；
    3. 读回来的状态如果还在，说明数据在 PG 里，而不是进程内存里。

用法：
    python scripts/verify_checkpoint_pg.py                # 自带 pgserver 起临时 PG
    python scripts/verify_checkpoint_pg.py --dsn <DSN>    # 用已有 PostgreSQL
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

THREAD_ID = "checkpoint-verify"
SESSION_ID = "checkpoint-verify"


def _write_state(dsn: str) -> None:
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver

    from Graph.graph import build_graph

    conn = psycopg.connect(dsn, autocommit=True)
    try:
        saver = PostgresSaver(conn)
        saver.setup()
        graph = build_graph(saver)
        graph.update_state(
            {"configurable": {"thread_id": THREAD_ID}},
            {
                "dialog_state": "risk_assessment",
                "session_id": SESSION_ID,
                "user_information": "姓名: 张三\n手术: 胆囊切除",
            },
        )
        print("[write] state written with PostgresSaver")
    finally:
        conn.close()  # 关键：连接关掉，内存里的东西一起没


def _read_state(dsn: str) -> bool:
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver

    from Graph.graph import build_graph

    conn = psycopg.connect(dsn, autocommit=True)
    try:
        saver = PostgresSaver(conn)
        saver.setup()
        graph = build_graph(saver)
        snapshot = graph.get_state({"configurable": {"thread_id": THREAD_ID}})
        values = snapshot.values or {}
        print(f"[read ] session_id    = {values.get('session_id')!r}")
        print(f"[read ] dialog_state  = {values.get('dialog_state')!r}")
        print(f"[read ] user_info     = {values.get('user_information')!r}")
        return values.get("session_id") == SESSION_ID and bool(
            values.get("user_information")
        )
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default="", help="已有 PostgreSQL 连接串")
    args = parser.parse_args()

    if args.dsn:
        dsn = args.dsn
        _write_state(dsn)
        ok = _read_state(dsn)
    else:
        try:
            import pgserver
        except ImportError:
            print("pgserver 未安装：pip install pgserver，或用 --dsn 指定现成数据库")
            return 2

        data_dir = Path(tempfile.mkdtemp(prefix="cmekg-pg-"))
        print(f"[pg    ] starting bundled postgres at {data_dir}")
        server = pgserver.get_server(str(data_dir))
        dsn = server.get_uri()
        print(f"[pg    ] {dsn}")
        _write_state(dsn)
        ok = _read_state(dsn)

    print()
    print("RESULT:", "PASS - checkpoint 跨连接持久化" if ok else "FAIL - 状态没有落库")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
