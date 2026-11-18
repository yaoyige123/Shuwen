"""仿真病人：端到端跑一遍完整术前访视流程。

用一个脚本化的"假病人"回答，验证：
    * 信息核对 → 病史采集(×3) → 镇痛宣教 → 风险评估 → 总结 能否走完；
    * 每轮任务是否都能成功收敛；
    * 是否出现复述、卡死、报错。

不占端口：直接用 ASGI 传输在进程内跑，存储用内存后端。

用法：
    python scripts/simulate_patient.py
    python scripts/simulate_patient.py --max-turns 40
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TASK_BACKEND", "memory")
os.environ.setdefault("EXECUTION_MODE", "inline")
os.environ.setdefault("CHECKPOINT_BACKEND", "memory")
os.environ.setdefault("RATE_LIMIT_REQUESTS", "100000")
os.environ.setdefault("REAPER_INTERVAL_SECONDS", "0")

import httpx  # noqa: E402

from app.main import create_app  # noqa: E402
from app.tasks import InMemoryTaskStore  # noqa: E402

TERMINAL = {"succeeded", "failed", "timeout", "cancelled"}

# 假病人：正则 → 回答。按顺序匹配，第一条命中即用。
SCRIPT: list[tuple[str, str]] = [
    # 追问"具体数值"类，放在通用规则前面
    (r"血压值|血压多少|控制得怎么样|控制情况", "最近量是 140 到 90 左右，吃药控制得还行"),
    (r"血糖|糖化", "空腹血糖 7 左右，打胰岛素"),
    (r"怎么称呼|您贵姓|姓名|叫什么", "我叫李建国"),
    (r"先生|女士", "是的，我是李建国"),
    (r"年龄|多少岁|多大", "我今年 65 岁"),
    (r"身高|体重", "身高 172 厘米，体重 80 公斤"),
    (r"血型", "不知道血型"),
    (r"手术", "胆囊切除手术"),
    (r"麻醉方式|什么麻醉|全麻", "全身麻醉"),
    (r"高血压|血压", "有高血压十年了，一直吃硝苯地平"),
    (r"糖尿病|血糖", "糖尿病五年，吃二甲双胍"),
    (r"心脏|冠心病|心律", "没有心脏病"),
    (r"过敏", "青霉素过敏，别的没有"),
    (r"吸烟|抽烟", "抽烟三十年，一天一包"),
    (r"饮酒|喝酒", "偶尔喝一点，不常喝"),
    (r"手术史|做过手术|麻醉过", "十年前做过阑尾炎手术，也是全麻，术后有点恶心"),
    (r"晕车|晕船|晕动", "有点晕车"),
    (r"输血", "没有输过血"),
    (r"放疗|化疗", "没有"),
    (r"植入物|支架|起搏器", "没有"),
    (r"睡眠|饮食|大小便", "睡得一般，胃口还可以，大小便正常"),
    (r"假牙|牙齿|松动", "有两颗牙有点松，没戴假牙"),
    (r"张口|颈椎|气道|打鼾", "张嘴能放三根手指，脖子活动正常，有点打鼾"),
    (r"禁食|禁水|吃东西|喝水", "从昨晚十点后就没吃东西了"),
    (r"镇痛泵|镇痛", "想用镇痛泵"),
    (r"费用|多少钱|医保", "能报销吗？不能报销的话也要用"),
    (r"病|家族|遗传", "家里人没有遗传病"),
    (r"咳嗽|发热|感冒|发烧", "没有"),
    (r"出血|抗凝", "没有异常出血"),
    (r"青光眼", "没有"),
    (r"怀孕|月经", "我是男的"),
    (r"活动|卧床", "平时能自己活动，买菜遛弯都行"),
    (r"前列腺", "前列腺有点增生，在吃药"),
]

DEFAULT_ANSWER = "没有其他了，都正常。"
# 同一个问题被反复追问时的补充回答（模拟"记不清/说不准"的真实病人）
FALLBACKS = [
    "具体数字记不太清了，大概还算正常。",
    "这个我说不准，平时没太注意。",
    "真不记得了，要不先跳过这一项吧。",
    "医生您就按没问题的算吧。",
]
MAX_REPEAT = 3


def pick_answer(reply: str, usage: dict[str, int], repeat_idx: int = 0) -> str:
    # 同一句话被连续追问：换成补充性回答，避免我们自己的脚本把 AI 卡死
    if repeat_idx > 0:
        return FALLBACKS[min(repeat_idx - 1, len(FALLBACKS) - 1)]

    # AI 现在会一条消息合并问多个问题，病人也要一次性都答上
    answers: list[str] = []
    for pattern, answer in SCRIPT:
        if len(answers) >= 4:
            break
        if re.search(pattern, reply) and usage.get(pattern, 0) < MAX_REPEAT:
            usage[pattern] = usage.get(pattern, 0) + 1
            if answer not in answers:
                answers.append(answer)

    return "；".join(answers) if answers else DEFAULT_ANSWER


def is_finished(reply: str) -> bool:
    return "签字" in reply or "术前谈话已完成" in reply


async def wait_terminal(client: httpx.AsyncClient, task_id: str, timeout: float = 300.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = await client.get(f"/api/v1/tasks/{task_id}")
        body = resp.json()
        if body["status"] in TERMINAL:
            return body
        await asyncio.sleep(0.4)
    return {"status": "client-timeout", "error": "poll timeout", "result": None}


async def run(max_turns: int) -> int:
    app = create_app(task_store=InMemoryTaskStore())
    session_id = f"sim-{uuid.uuid4().hex[:8]}"
    usage: dict[str, int] = {}
    transcript: list[tuple[str, str]] = []
    problems: list[str] = []
    stages: set[str] = set()
    failures = 0

    print(f"[sim ] session = {session_id}", flush=True)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://sim", timeout=300.0) as client:
            async def send(text: str, key: str) -> str:
                nonlocal failures
                started = time.monotonic()
                resp = await client.post(
                    f"/api/v1/sessions/{session_id}/messages",
                    json={"message": text},
                    headers={"Idempotency-Key": key},
                )
                if resp.status_code != 202:
                    problems.append(f"提交失败 HTTP {resp.status_code}: {resp.text[:120]}")
                    failures += 1
                    return ""

                task = await wait_terminal(client, resp.json()["task_id"])
                elapsed = time.monotonic() - started
                if task["status"] != "succeeded":
                    problems.append(f"任务 {task['status']}: {task.get('error')}")
                    failures += 1
                    return ""

                result = task.get("result") or {}
                state = result.get("dialog_state")
                if state:
                    stages.add(state)
                reply = result.get("reply") or ""
                print(
                    f"[t{len(transcript) + 1:02d}] {(state or '-'):<18} {elapsed:5.1f}s  用户: {text}",
                    flush=True,
                )
                print(f"      AI: {reply}", flush=True)
                transcript.append((text, reply))
                return reply

            reply = await send("开始术前访视", "k-0")
            if not reply:
                problems.append("开场失败")

            last_reply = reply
            repeat_idx = 0
            for turn in range(1, max_turns + 1):
                if not reply or is_finished(reply):
                    break
                repeat_idx = repeat_idx + 1 if reply == last_reply else 0
                last_reply = reply
                answer = pick_answer(reply, usage, repeat_idx)
                reply = await send(answer, f"k-{turn}")

    # 汇总
    finished = is_finished(reply) if reply else False
    print()
    print("=" * 60, flush=True)
    print(f"轮数        : {len(transcript)}")
    print(f"经历阶段    : {sorted(stages)}")
    print(f"任务失败数  : {failures}")
    print(f"是否走到结尾: {'是' if finished else '否'}")
    if problems:
        print("问题:")
        for item in dict.fromkeys(problems):
            print(f"  - {item}")

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"sim-{stamp}.txt"
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"session={session_id}\n轮数={len(transcript)} 阶段={sorted(stages)}\n")
        fh.write(f"失败数={failures} 走到结尾={finished}\n\n")
        for idx, (user, ai) in enumerate(transcript, 1):
            fh.write(f"[{idx:02d}] 患者: {user}\n     助手: {ai}\n")
    print(f"transcript  : {log_path}")

    ok = finished and failures == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-turns", type=int, default=80)
    args = parser.parse_args()
    return asyncio.run(run(args.max_turns))


if __name__ == "__main__":
    raise SystemExit(main())
