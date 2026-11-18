# 异步任务 API（P1）

FastAPI 入口：`app/main.py`，默认监听 `0.0.0.0:8000`。

```bash
python -m app.main
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

浏览器直接打开 **http://127.0.0.1:8000/chat** 就是聊天页（`/` 会跳过去），
接口文档在 http://127.0.0.1:8000/docs。

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/health` | 存活探针 |
| POST | `/api/v1/sessions/{session_id}/messages` | 提交一条用户消息，**立即**返回 `202 + task_id` |
| GET | `/api/v1/tasks/{task_id}` | 查询任务状态与结果 |
| GET | `/api/v1/tasks/{task_id}/events` | SSE 订阅状态变更（`status` / `done`） |
| POST | `/api/v1/tasks/{task_id}/cancel` | 取消任务（幂等，已终态则原样返回） |
| POST | `/api/v1/tts` | 文本转语音，返回 `audio/wav`；未配置讯飞凭据返回 503 |
| GET | `/chat` | 自带聊天页（`/` 会跳到这里） |
| GET | `/docs` | Swagger 接口文档 |
| GET | `/metrics` | Prometheus 指标 |

### 提交消息

```bash
curl -i -X POST http://127.0.0.1:8000/api/v1/sessions/s-001/messages \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: 7f3c...' \
  -d '{"message":"你好，我姓张"}'
```

```http
HTTP/1.1 202 Accepted
{"task_id":"eaf3...","session_id":"s-001","status":"queued","idempotent_replay":false}
```

同一个 `Idempotency-Key` 重复提交不会产生新任务，返回原 `task_id` 且
`idempotent_replay=true`。幂等键按 `session_id` 隔离。

### 查询任务

```bash
curl http://127.0.0.1:8000/api/v1/tasks/eaf3...
```

```json
{
  "task_id": "eaf3...",
  "session_id": "s-001",
  "status": "succeeded",
  "created_at": "...", "started_at": "...", "finished_at": "...",
  "result": {"reply": "请问您是 某某 先生吗？", "dialog_state": "verify_information", "message_count": 2},
  "error": null
}
```

### 订阅进度（SSE）

```bash
curl -N http://127.0.0.1:8000/api/v1/tasks/eaf3.../events
```

```
event: status
data: {"task_id":"eaf3...","status":"running",...}

event: status
data: {"task_id":"eaf3...","status":"succeeded",...}

event: done
data: {"task_id":"eaf3...","status":"succeeded"}
```

实现用存储层的 `asyncio.Event` 唤醒，**不是定时轮询**；空闲期每
`SSE_KEEPALIVE_SECONDS`（默认 15s）发一次注释心跳。反向代理必须关闭缓冲
（响应头已带 `X-Accel-Buffering: no`）。

### 语音合成

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"请问您今天有没有吃降压药？"}' \
  -o reply.wav
```

- 返回 `audio/wav`（16kHz 单声道），带 `Cache-Control: max-age=3600`；
- 按 `sha1(规范化文本)` 落盘缓存到 `app/static/audio/`，同一句话只调一次讯飞；
- 未配置 `XUNFEI_APP_ID/API_KEY/API_SECRET` 时返回 **503**，页面会自动降级到
  浏览器内置语音（Web Speech API），因此**不配凭据也能朗读**；
- 合成失败返回 **502**；文本长度限制 1~2000 字。

## 任务状态机
```
queued ──► running ──► succeeded
   │           ├──────► failed
   │           ├──────► timeout
   └───────────┴──────► cancelled
```

- 迁移受控：未列出的迁移抛 `InvalidTransition`，HTTP 层映射为 `409`；
- 终态不可再变（`succeeded/failed/timeout/cancelled`）；
- 取消是幂等的：已终态直接返回当前状态。

## 设计取舍

| 决定 | 原因 |
|---|---|
| LangGraph 跑在 `asyncio.to_thread` | `graph.invoke` 是同步阻塞 API，直接 await 会堵死事件循环 |
| 总超时 `TURN_TIMEOUT_SECONDS` | 超时置 `timeout` 并放弃等待；线程杀不掉，硬中断留给 P2 的 worker 进程 |
| 同一 `session_id` 串行 | LangGraph checkpoint 是「每个 thread_id 一条时间线」，并发 turn 会互相覆盖 |
| `create_app(store=..., turn_runner=...)` | 工厂 + 依赖注入，测试可注入替身，不必 monkeypatch 全局变量 |
| 内存 `TaskStore` | P1 先跑通链路；P2 换成 Redis/PostgreSQL，只替换实现，路由与 runner 不动 |

## 已知边界（P2 处理）

- 任务状态在进程内存里，**重启即丢**、多副本不共享 → Redis + PostgreSQL；
- 推理仍在 API 进程的线程池里，无法随 API 水平扩容 → Celery worker；
- 无鉴权、无限流 → 与 P2 的 Redis 限流一起做；
- 无 trace_id 与指标 → P4 可观测性。
