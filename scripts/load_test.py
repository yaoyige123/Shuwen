"""API 层压测（推理用桩替代，只测 HTTP + 任务状态写入的开销）。

为什么用桩：真实 LLM 一轮 5~15s，压测会变成"测 OpenAI 的延迟"。这里把 runner
换成瞬间完成的桩，测的是本项目自己的路径：中间件 → 依赖注入 → 限流 → 建任务 →
分发 → 返回 202。这个数字和模型无关，可复现。

用法：
    python scripts/load_test.py --scenario memory --concurrency 50 --total 500
    python scripts/load_test.py --scenario redis  --concurrency 50 --total 500
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import create_app  # noqa: E402
from app.ratelimit import MemoryRateLimiter  # noqa: E402
from app.tasks import InMemoryTaskStore, TaskStatus  # noqa: E402


async def stub_runner(store, task_id, session_id, message):
    """瞬间完成的执行器：只走状态机，不调用 LLM。"""
    await store.transition(task_id, TaskStatus.RUNNING)
    await store.transition(task_id, TaskStatus.SUCCEEDED, result={"reply": "stub"})


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(q * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]


async def _build_store(scenario: str):
    if scenario == "memory":
        return InMemoryTaskStore(), None

    import redis.asyncio as aioredis

    from app.config import settings
    from app.store_redis import RedisTaskStore

    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    return RedisTaskStore(client), client


async def run(scenario: str, concurrency: int, total: int) -> int:
    store, client = await _build_store(scenario)
    # 压测要放大限流阈值，否则会被 429 挡住
    limiter = MemoryRateLimiter(limit=total * 2, window_seconds=60)
    app = create_app(task_store=store, turn_runner=stub_runner, rate_limiter=limiter)

    latencies: list[float] = []
    errors = 0
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int) -> None:
        nonlocal errors
        async with semaphore:
            payload = {"message": f"load-{index}"}
            headers = {"Idempotency-Key": f"load-{scenario}-{index}"}
            started = time.perf_counter()
            response = await client_http.post(
                f"/api/v1/sessions/load-{index % 10}/messages",
                json=payload,
                headers=headers,
            )
            elapsed = time.perf_counter() - started
            if response.status_code == 202:
                latencies.append(elapsed)
            else:
                errors += 1

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://load", timeout=60.0
        ) as client_http:
            # 预热：把中间件/依赖解析的首次开销排除掉
            await client_http.get("/api/v1/health")

            wall_start = time.perf_counter()
            await asyncio.gather(*(one(i) for i in range(total)))
            wall = time.perf_counter() - wall_start

    if client is not None:
        await client.aclose()

    ok = len(latencies)
    print(f"scenario    = {scenario}")
    print(f"concurrency = {concurrency}")
    print(f"total       = {total}")
    print(f"success     = {ok}   errors = {errors}")
    print(f"wall        = {wall:.2f}s   throughput = {ok / wall:.0f} req/s")
    if ok:
        p50 = _percentile(latencies, 0.50) * 1000
        p95 = _percentile(latencies, 0.95) * 1000
        p99 = _percentile(latencies, 0.99) * 1000
        mx = max(latencies) * 1000
        print(f"latency     = p50 {p50:.1f}ms   p95 {p95:.1f}ms   p99 {p99:.1f}ms   max {mx:.1f}ms")

    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["memory", "redis"], default="memory")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--total", type=int, default=500)
    args = parser.parse_args()

    return asyncio.run(run(args.scenario, args.concurrency, args.total))


if __name__ == "__main__":
    raise SystemExit(main())
