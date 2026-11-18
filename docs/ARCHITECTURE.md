# 架构说明

面向"要改这个仓库的人"，讲清分层、数据模型、并发与故障处理。
接口细节见 [API.md](API.md)，分阶段演进见 [P2.md](P2.md) / [P4.md](P4.md)。

---

## 1. 分层与依赖方向

```
            ┌──────────────────────────────┐
            │  app/static/chat.html        │  自带聊天页（浏览器）
            │  api_server.py               │  遗留 REST 入口（Flask），仅作对照
            └──────────────┬───────────────┘
                           │
   ┌───────────────────────▼───────────────────────┐
   │  app/  平台层：配置·任务·调度·持久化·可观测      │
   └───────────────────────┬───────────────────────┘
                           │ 调用
   ┌───────────────────────▼───────────────────────┐
   │  Graph/ 图定义 ·  Chains/ 各专科 Agent          │
   └───────────────────────┬───────────────────────┘
                           │
                  LLM  ·  GraphRAG  ·  讯飞 TTS
```

规则：

- `app/` 不依赖任何 UI 层（P0 之前反过来：`Graph/nodes.py` 直接
  `from web.app import graph_qa_events`，形成循环依赖；旧 Flask UI 已在 P4 后移除）；
- `Graph/` 允许依赖 `app.events` 这类基础设施，但不依赖 `app.api`；
- `Chains/` 不依赖 `Graph/`，唯一例外是共用的 `Graph/router.py` 里的工具模型；
- `Prompts/`、`Indexes/`、`Tools/` 是历史遗留，主流程不引用。

---

## 2. 数据模型

### 任务记录

`TaskRecord`（`app/tasks.py`）：

| 字段 | 说明 |
|---|---|
| `task_id` | uuid4 hex |
| `session_id` | 会话（= LangGraph 的 `thread_id`） |
| `message` | 待处理的用户输入 |
| `status` | `queued / running / succeeded / failed / timeout / cancelled` |
| `created_at / started_at / finished_at` | 用于算耗时（指标取自这里，不是函数内计时） |
| `result` | 成功时 `{reply, dialog_state, message_count, attempts}` |
| `error` | 失败原因（`Type: message`） |
| `version` | 单调递增，SSE 靠它判断"有没有变化" |

### Redis 键布局

```
task:{task_id}            Hash    上面这些字段（result 存 JSON 字符串）
idem:{session_id}:{key}   String  → task_id（SET NX 实现幂等）
task:{task_id}:changes    Pub/Sub 状态变更通知
```

> 幂等键按 `session_id` 隔离：不同会话用同一个 key 不会互相干扰。

---

## 3. 并发模型

| 层面 | 机制 | 作用 |
|---|---|---|
| HTTP | uvicorn 单事件循环，每请求一个 task | 高并发下不被单请求阻塞 |
| Agent 调用 | `asyncio.to_thread` + `wait_for` | LangGraph 是同步阻塞 API，必须出事件循环 |
| 同会话 | `session_lock`：本地 `asyncio.Lock` 或 Redis `SET NX PX` | LangGraph checkpoint 是"每 thread 一条时间线"，并发会互相覆盖 |
| 跨进程 | Redis `WATCH/MULTI` 乐观锁 CAS | 两个 worker 抢同一任务时只有一方能迁移状态 |
| SSE | `asyncio.Event`（内存）/ Pub/Sub（Redis） | 状态变更即推送，不用轮询 |

**多副本下的会话串行**：`SESSION_LOCK_BACKEND=redis` 时，同一会话的锁是
`lock:session:{id}` 上的 `SET NX PX`，租约默认 300s（大于单轮超时 180s）。
拿不到锁会在 `SESSION_LOCK_TIMEOUT_SECONDS` 后抛 `SessionBusy`，任务直接判失败——
对医疗问诊来说，"明确失败"比"半小时后突然接着上一轮"更安全。

---

## 4. 故障处理

| 故障 | 行为 | 依据 |
|---|---|---|
| LLM 超时 | 任务置 `timeout`，**不重试** | `to_thread` 杀不掉线程，重试会重复计费 |
| 连接类 / 429 / 5xx | 指数退避 + 抖动重试，最多 `TURN_MAX_ATTEMPTS` 次 | 瞬时故障自愈 |
| 业务错误（4xx、解析失败） | 立即 `failed` | 重试无意义 |
| 客户端取消 | 取消 asyncio 任务 → `cancelled`；已终态则幂等返回 | |
| 任务已取消但 runner 仍在跑 | `_settle` 捕获 `InvalidTransition`，保持现状并告警 | 不把异常抛给后台任务 |
| API 进程崩溃 | 任务留在 `running`，需人工/定时任务回收 | **未实现**：缺 lease/心跳 |
| Redis 不可用 | 启动时降级为 memory（仅当配置为 inline） | 生产应直接失败而非静默降级 |
| 讯飞凭据缺失 | TTS 整体禁用，只出文字 | 每句重试 3 次会拖慢整轮对话 |

---

## 5. 版本升级：langchain 0.1 → 1.x

升级不是为了追新，而是因为 **Postgres checkpointer 只存在于 1.x 生态**
（`langgraph-checkpoint-postgres` 要求 `langchain-core>=0.2`）。代价是三处适配：

| 位置 | 改动 | 原因 |
|---|---|---|
| `Graph/state.py` | `AnyMessage` 改从 `langchain_core.messages` 导入 | `langgraph.graph.message` 不再导出 |
| `Graph/router.py` + 4 个 chain | `pydantic.v1` → `pydantic` | 1.x 的 `bind_tools` / `PydanticOutputParser` 只接受 v2 模型 |
| `Chains/sequence_primary_agent.py` | `AIMessage(tool_calls=[...])` | 旧的 `additional_kwargs` 兼容路径已废弃 |

> 这三处正是 P1 时为 0.1.x 做的适配，升级时反向改回。

---

## 6. 决策日志

**为什么用 ARQ 而不是 Celery？**
代码全是 async，ARQ 原生异步且依赖只有 redis；Celery 是同步模型，要额外引入
线程池/gevent。代价是 ARQ 生态小、没有 Flower 那种成熟面板。

**为什么用 WATCH/MULTI 而不是 Lua 脚本？**
两者都能做 CAS。WATCH/MULTI 的迁移逻辑留在 Python 里可读可调试，且**能在
fakeredis 里完整测试**（不依赖 Lua 运行时）。代价是冲突时多一次往返。

**为什么不把 Agent 改写成 async？**
Chain 层深度依赖 LangChain 的同步接口，重写面太大。`to_thread` 的代价是
超时后线程仍在跑（无法硬中断），这一点已在文档和代码注释里写明。

**为什么 checkpointer 做成工厂？**
LangGraph 的 checkpointer 是 `compile()` 时注入的。做成工厂后，切 memory/sqlite/postgres
只影响 `app/checkpoint.py`，`Graph/` 与业务代码零改动。

**为什么默认是 memory + inline？**
让"clone 下来就能跑"成立——不需要 Redis/PG，也不必先理解队列。外部依赖是**可选加速项**
而不是门槛。

---

## 7. 部署拓扑

**开发（默认）**

```
uvicorn (API, inline 执行)  → 只依赖 LLM API
```

**生产**

```
Nginx (TLS, SSE 关缓冲)
   ├── api × N        TASK_BACKEND=redis  EXECUTION_MODE=queue
   └── worker × M     python -m arq app.worker.WorkerSettings
Redis (任务状态 + 队列)
PostgreSQL (会话 checkpoint)
```

Nginx 必须给 SSE 关缓冲（响应头已带 `X-Accel-Buffering: no`），
并把读超时调到大于 `SSE_KEEPALIVE_SECONDS`。

---

## 8. 还没做的

- **端到端压测**：只测了 API 层（`scripts/load_test.py`，推理用桩），没有带真实 LLM 的压测；
- **锁租约续期**：单轮超过 `SESSION_LOCK_LEASE_SECONDS` 时锁会被后来者抢占（默认配置下不会触发）；
- **告警与 OTel 导出**：指标有了，但没有告警规则与跨服务追踪。

已经补上的（曾在此列出）：

- **任务租约**：`RUNNING` 时写 `lease_expires_at` 并进 Redis ZSet，后台 reaper
  每 30s 扫一次过期项并收敛为 `FAILED`；worker 崩溃不再留下永久 `running`；
- **trace 跨进程**：`trace_id` 随任务记录存进 Redis，worker 取到任务后
  `set_trace_id()`，日志链路跨进程连续；
- **TTS 副作用幂等**：片段按 `sha1(规范化文本)` 命名，重试/重放直接复用；
  删除时检查引用计数，不会误删被复用的文件；
- **优雅关闭**：关闭时先 `dispatcher.drain()`（默认 5s）再关存储，
  避免后台协程往已关闭的连接里写；
- **分布式会话锁**：`app/locks.py`，Redis `SET NX PX` + 乐观删除（只删自己的 token）；
- **Redis 版限流**：`app/ratelimit.py`，`INCR` + `EXPIRE` 的跨副本固定窗口。
