"""基础设施连通性自检。

用法：
    python scripts/check_infra.py

用于在启用 ``TASK_BACKEND=redis`` / ``EXECUTION_MODE=queue`` 之前
确认依赖真的可用；缺失项会给出可执行的下一步，而不是等到运行时炸。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402


async def check_redis() -> tuple[bool, str]:
    try:
        import redis.asyncio as aioredis
    except ImportError:
        return False, "redis not installed (pip install redis)"

    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        pong = await asyncio.wait_for(client.ping(), timeout=3)
        return bool(pong), "PING ok"
    except Exception as exc:  # noqa: BLE001 - 自检脚本只关心可用性
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            pass


async def check_postgres() -> tuple[bool, str]:
    if not settings.database_url:
        return False, "DATABASE_URL not set"
    try:
        import psycopg
    except ImportError:
        return False, "psycopg not installed (pip install 'psycopg[binary]')"

    try:
        conn = await asyncio.wait_for(
            psycopg.AsyncConnection.connect(settings.database_url), timeout=3
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"

    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            row = await cur.fetchone()
        return row == (1,), "SELECT 1 ok"
    finally:
        await conn.close()


async def main() -> int:
    redis_ok, redis_msg = await check_redis()
    pg_ok, pg_msg = await check_postgres()

    print(f"redis    {'OK  ' if redis_ok else 'FAIL'}  {settings.redis_url}  {redis_msg}")
    print(f"postgres {'OK  ' if pg_ok else 'FAIL'}  {settings.database_url or '-'}  {pg_msg}")

    if not redis_ok:
        print()
        print("hint: keep the defaults when Redis is unavailable")
        print("      TASK_BACKEND=memory")
        print("      EXECUTION_MODE=inline")
        print("      or start the container: docker compose up -d redis")

    # 两种后端都不可用不算失败——memory/inline 是受支持的单机模式
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
