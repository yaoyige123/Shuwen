"""集中式配置：环境变量与凭据的唯一入口。

设计要点
--------
1. 导入期就 ``load_dotenv``：``Chains/*.py`` 在 import 阶段直接读
   ``os.environ["OPENAI_API_KEY"]``，所以 .env 必须早于这些模块进入
   ``os.environ``。这里保留"一次加载、全局可见"的旧行为，但把读取入口
   统一收敛到 ``Settings``，业务代码不再散落 ``os.environ.get(...)``。
2. 字段名小写、环境变量名自动大写匹配（``openai_api_key`` ← ``OPENAI_API_KEY``）。
3. 不在代码里保留任何默认密钥：缺失即视为未配置，由启动自检报错。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（本文件位于 <root>/app/config.py）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

# 必须在任何 Chains/Graph 模块被导入之前执行
load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    """所有可配置项的单一来源。"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    openai_api_base: str = ""
    openai_api_key: str = ""
    tavily_api_key: str = ""
    chat_model: str = "gpt-4o"
    extract_model: str = "gpt-4o-mini"

    # --- 科大讯飞语音（TTS / IAT） ---
    xunfei_app_id: str = ""
    xunfei_api_key: str = ""
    xunfei_api_secret: str = ""
    xunfei_tts_voice: str = "xiaoyan"
    xunfei_tts_speed: int = 70

    # --- 鉴权 ---
    api_key: str = ""  # 非空即要求 X-API-Key

    # --- 存储路径（相对项目根目录） ---
    graphrag_working_dir: str = "dickens"
    chat_logs_dir: str = "chat_logs"
    audio_dir: str = "app/static/audio"  # TTS 缓存目录

    # --- 异步任务（P1） ---
    turn_timeout_seconds: int = 180
    sse_keepalive_seconds: int = 15

    # --- 重试与限流（P2） ---
    turn_max_attempts: int = 3          # 可重试错误的最大尝试次数
    turn_retry_base_seconds: float = 0.5  # 指数退避基数（秒）
    rate_limit_requests: int = 120      # 每个窗口允许的请求数
    rate_limit_window_seconds: int = 60

    # --- 任务租约（P2+） ---
    # RUNNING 任务的租约时长：超过它仍未被收敛，说明 worker 可能已死
    task_lease_seconds: int = 300
    # 回收扫描间隔（秒）；0 表示关闭后台回收
    reaper_interval_seconds: int = 30

    # --- 并发保护与限流的后端（P2+） ---
    # memory: 进程内（单机）；redis: 跨副本
    session_lock_backend: Literal["memory", "redis"] = "memory"
    session_lock_timeout_seconds: float = 30.0
    session_lock_lease_seconds: float = 300.0
    rate_limit_backend: Literal["memory", "redis"] = "memory"

    # --- 任务后端（P2） ---
    # memory: 进程内（默认，单机开发）；redis: 跨进程/多副本共享
    task_backend: Literal["memory", "redis"] = "memory"
    # inline: API 进程内 asyncio 执行；queue: 投递到 ARQ worker
    execution_mode: Literal["inline", "queue"] = "inline"
    redis_url: str = "redis://127.0.0.1:6379/0"
    database_url: str = ""

    # --- 会话状态持久化（P2） ---
    # memory: 进程内；sqlite: 单文件（无需服务端，便于验证）；postgres: 多副本共享
    checkpoint_backend: Literal["memory", "sqlite", "postgres"] = "memory"
    checkpoint_sqlite_path: str = "checkpoints.sqlite"

    # --- GraphRAG 分层降级 ---
    # 风险环节最多等预取多久（秒）；每个会话只等一次，超时后直接降级
    graphqa_wait_seconds: float = 2.0
    # 各层的超时预算：global 是社区摘要检索（最慢），local 是实体检索（仍要调 LLM）
    graphqa_global_timeout: float = 20.0
    graphqa_local_timeout: float = 8.0
    # 纯本地词面检索（不调 LLM），只做兜底
    graphqa_keyword_timeout: float = 0.5

    @property
    def tts_available(self) -> bool:
        """讯飞 TTS 凭据是否齐全；不齐时语音降级为纯文字。"""
        return bool(
            self.xunfei_app_id and self.xunfei_api_key and self.xunfei_api_secret
        )

    def missing(self) -> list[str]:
        """返回缺失的必需环境变量名，供启动自检使用。"""
        required = {
            "OPENAI_API_KEY": self.openai_api_key,
            "OPENAI_API_BASE": self.openai_api_base,
        }
        return [name for name, value in required.items() if not value]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例；后续 FastAPI 可直接作为依赖注入的 provider。"""
    return Settings()


settings = get_settings()
