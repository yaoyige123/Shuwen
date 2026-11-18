"""GraphRAG 检索的分层降级。

风险告知不能因为图谱检索慢或失败而卡住，也不能在没有依据时"装作知道"。
所以按代价从高到低分层，逐层降级：

======================  ==============================  ===========
层                      做法                            延迟量级
======================  ==============================  ===========
L0  global              社区摘要检索（质量最好）        10~30s
L1  local               实体 + 邻居检索，仍要调一次 LLM  3~8s
L2  keyword             直接读分块文本做词面匹配，不调 LLM  < 50ms
L3  profile_only        只用患者档案 + 内置风险清单       0
======================  ==============================  ===========

为什么 L2 是"词面"而不是"向量"：本项目的 nano-graphrag 索引里
``vdb_entities.json`` 只持久化了 ``__id__`` 和 ``entity_name``，
**没有存向量矩阵**。要做纯向量降级就得把分块重新 embedding 一遍，
成本和冷启动都不划算；而中文医疗词（病名、药名）的词面重合度本身
就很高，L2 用词面检索足够当兜底。将来要换向量，只需替换
``_keyword_lookup`` 一个函数。

每层都包了超时：``ragllm.query`` 是同步阻塞调用，卡住不会自己返回，
所以用守护线程 + Event 等待，超时即放弃（线程仍在后台跑完即弃）。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app import events
from app.config import settings
from app.metrics import GRAPH_QA_DURATION, GRAPH_QA_RESOLVED, GRAPH_QA_TOTAL

logger = logging.getLogger(__name__)

# 检索路径标识
PATH_GLOBAL = "global"
PATH_LOCAL = "local"
PATH_KEYWORD = "keyword"
PATH_PROFILE = "profile_only"
PATH_NONE = "none"

# L4：明确告诉模型"这批结论没有图谱依据"，让它把局限说出来，
# 而不是把通用清单当成个性化结论。
PROFILE_ONLY_TEXT = (
    "知识图谱未检索到与本问题直接相关的依据。请仅依据患者档案与通用麻醉风险知识作答，"
    "并明确告知患者：这部分内容需要与麻醉医生当面确认。"
)

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD = re.compile(r"[a-z0-9]{2,}")


class GraphQaTimeout(TimeoutError):
    """某层检索超时。"""

    def __init__(self, path: str, seconds: float) -> None:
        super().__init__(f"{path} 检索超时（{seconds}s）")
        self.path = path


def run_with_timeout(fn: Callable[[], Any], seconds: float, *, path: str) -> Any:
    """在守护线程里跑 fn，最多等 seconds 秒。

    注意：超时只是"不再等待"，并不能中断已经在跑的线程
    （同步 SDK 没有提供取消手段）。所以这里只保证调用方不被拖住。
    """
    box: dict[str, Any] = {}
    done = threading.Event()

    def worker() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - 原样抛回调用方
            box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=worker, name=f"graphqa-{path}", daemon=True)
    thread.start()

    if not done.wait(seconds):
        raise GraphQaTimeout(path, seconds)
    if "error" in box:
        raise box["error"]
    return box["value"]


# ---------------------------------------------------------------- 分层检索


_chain_cache: dict[str, Any] = {}
_chain_lock = threading.Lock()


def _chain(mode: str) -> Any:
    """按模式缓存检索链（避免每轮重建）。"""
    with _chain_lock:
        chain = _chain_cache.get(mode)
        if chain is None:
            from Chains.graph_qa_chain import get_graph_qa_chain

            chain = get_graph_qa_chain(mode=mode)
            _chain_cache[mode] = chain
        return chain


def _as_text(result: Any) -> str:
    """统一检索链的返回形态。

    ``get_graph_qa_chain`` 返回的是 ``{"risk_analysis": ...}`` 这样的字典，
    而测试替身和未来可能的实现可能直接返回字符串，这里统一成 str。
    """
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("risk_analysis") or "").strip()
    return str(result or "").strip()


def _attempt(payload: dict[str, Any], mode: str, timeout: float) -> str:
    """跑一层检索；超时/异常都转成空串，由调用方决定是否继续降级。"""
    started = time.monotonic()
    outcome = "hit"
    try:
        text = _as_text(
            run_with_timeout(lambda: _chain(mode).invoke(payload), timeout, path=mode)
        )
        if not text:
            outcome = "empty"
        return text
    except GraphQaTimeout:
        outcome = "timeout"
        logger.warning("graphqa %s 超时（%.1fs）", mode, timeout)
        return ""
    except Exception:  # noqa: BLE001 - 检索失败必须降级，不能打断对话
        outcome = "error"
        logger.exception("graphqa %s 失败", mode)
        return ""
    finally:
        GRAPH_QA_DURATION.labels(path=mode).observe(time.monotonic() - started)
        if outcome != "hit":
            GRAPH_QA_TOTAL.labels(outcome=outcome, path=mode).inc()


def retrieve_keyword(payload: dict[str, Any]) -> str:
    """只跑词面兜底（毫秒级），用于"先占位、后覆盖"。"""
    query = _build_query(payload)
    if not query:
        return ""
    started = time.monotonic()
    try:
        return run_with_timeout(
            lambda: _keyword_lookup(query),
            settings.graphqa_keyword_timeout,
            path=PATH_KEYWORD,
        )
    except GraphQaTimeout:
        GRAPH_QA_TOTAL.labels(outcome="timeout", path=PATH_KEYWORD).inc()
        return ""
    except Exception:  # noqa: BLE001
        GRAPH_QA_TOTAL.labels(outcome="error", path=PATH_KEYWORD).inc()
        logger.exception("graphqa keyword 兜底失败")
        return ""
    finally:
        GRAPH_QA_DURATION.labels(path=PATH_KEYWORD).observe(time.monotonic() - started)


def retrieve(payload: dict[str, Any]) -> tuple[str, str]:
    """按 L0 → L1 → L2 依次尝试，返回 (文本, 路径)。

    全部失败时返回 ("", "none")，由调用方决定用 L3 的档案兜底。
    """
    for mode, timeout in (
        (PATH_GLOBAL, settings.graphqa_global_timeout),
        (PATH_LOCAL, settings.graphqa_local_timeout),
    ):
        text = _attempt(payload, mode, timeout)
        if text:
            GRAPH_QA_TOTAL.labels(outcome="hit", path=mode).inc()
            return text, mode

    snippet = retrieve_keyword(payload)
    if snippet:
        GRAPH_QA_TOTAL.labels(outcome="hit", path=PATH_KEYWORD).inc()
        return snippet, PATH_KEYWORD
    GRAPH_QA_TOTAL.labels(outcome="empty", path=PATH_KEYWORD).inc()

    return "", PATH_NONE


# ---------------------------------------------------------------- L2：词面检索


_chunks: list[tuple[frozenset[str], str]] | None = None
_chunks_lock = threading.Lock()


def _terms(text: str) -> frozenset[str]:
    """中文用二元字组、英文用单词，够用且不依赖分词库。"""
    lowered = text.lower()
    grams: set[str] = set()
    for run in _CJK_RUN.findall(lowered):
        for i in range(len(run) - 1):
            grams.add(run[i : i + 2])
    grams.update(_ASCII_WORD.findall(lowered))
    return frozenset(grams)


def _load_chunks() -> list[tuple[frozenset[str], str]]:
    global _chunks
    with _chunks_lock:
        if _chunks is not None:
            return _chunks

        path = Path(settings.graphrag_working_dir) / "kv_store_text_chunks.json"
        if not path.exists():
            logger.warning("graphqa 找不到分块文件：%s", path)
            _chunks = []
            return _chunks

        raw = json.loads(path.read_text(encoding="utf-8"))
        loaded: list[tuple[frozenset[str], str]] = []
        for value in raw.values():
            content = (value or {}).get("content") or ""
            if content:
                loaded.append((_terms(content), content))
        _chunks = loaded
        logger.info("graphqa 载入 %d 个分块用于兜底检索", len(loaded))
        return _chunks


def _keyword_lookup(query: str, top_k: int = 3, max_chars: int = 1800) -> str:
    """按词面重合度排序，返回最相关的若干分块。"""
    wanted = _terms(query)
    if not wanted:
        return ""

    scored: list[tuple[float, str]] = []
    for terms, content in _load_chunks():
        overlap = len(wanted & terms)
        if overlap:
            scored.append((overlap / max(len(terms), 1) ** 0.5, content))

    if not scored:
        return ""

    scored.sort(key=lambda item: item[0], reverse=True)
    picked: list[str] = []
    used = 0
    for _score, content in scored[:top_k]:
        remaining = max_chars - used
        if remaining <= 0:
            break
        picked.append(content[:remaining])
        used += min(len(content), remaining)
    return "\n\n".join(picked)


def _build_query(payload: dict[str, Any]) -> str:
    """把患者档案拼成检索语句。"""
    parts = [
        str(payload.get("user_information") or ""),
        " ".join(map(str, payload.get("medical_history") or [])),
        " ".join(map(str, payload.get("medicine_taking") or [])),
    ]
    return " ".join(part for part in parts if part).strip()


# ---------------------------------------------------------------- 风险节点入口


def resolve_for_risk(session_id: str, state: dict[str, Any]) -> tuple[str, str]:
    """风险告知取数：最多等一次预取，然后按已有结果或降级返回。"""
    existing = str(state.get("graph_qa_result") or "").strip()
    if state.get("graph_is_qa") and existing:
        GRAPH_QA_RESOLVED.labels(path=PATH_GLOBAL).inc()
        return existing, PATH_GLOBAL

    event = events.event(session_id)
    if event is not None and events.claim_wait(session_id):
        # 每会话只等一次；超时后该会话后续轮次不再等
        if not event.wait(timeout=settings.graphqa_wait_seconds):
            logger.info("graphqa 等待预取超时（%ss）：%s", settings.graphqa_wait_seconds, session_id)
    elif event is not None:
        logger.debug("graphqa 本会话已等待过，跳过：%s", session_id)

    text = events.get_result(session_id).strip()
    if text:
        path = events.get_path(session_id) or PATH_GLOBAL
        GRAPH_QA_RESOLVED.labels(path=path).inc()
        return text, path

    GRAPH_QA_RESOLVED.labels(path=PATH_PROFILE).inc()
    return PROFILE_ONLY_TEXT, PATH_PROFILE


def reset() -> None:
    """清空缓存（测试用）。"""
    global _chunks
    with _chain_lock:
        _chain_cache.clear()
    with _chunks_lock:
        _chunks = None
