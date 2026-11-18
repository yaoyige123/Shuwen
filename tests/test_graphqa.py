"""GraphRAG 分层降级的行为测试。

全部离线：不调用真实 LLM / 真实 GraphRAG，用替身验证
「超时 → 降级 → 兜底 → 打点」这条链路。
"""

from __future__ import annotations

import time

import pytest
from prometheus_client import REGISTRY

from app import events, graphqa
from app.config import settings


class _FakeChain:
    def __init__(self, *, text: str = "", delay: float = 0.0, boom: bool = False) -> None:
        self.text = text
        self.delay = delay
        self.boom = boom

    def invoke(self, _payload):
        if self.delay:
            time.sleep(self.delay)
        if self.boom:
            raise RuntimeError("boom")
        return self.text


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    events.clear()
    graphqa.reset()
    # 把各层预算压到毫秒级，避免测试变慢
    monkeypatch.setattr(settings, "graphqa_wait_seconds", 0.05)
    monkeypatch.setattr(settings, "graphqa_global_timeout", 0.05)
    monkeypatch.setattr(settings, "graphqa_local_timeout", 0.05)
    monkeypatch.setattr(settings, "graphqa_keyword_timeout", 0.5)
    yield
    events.clear()
    graphqa.reset()


def _patch_chains(monkeypatch, mapping: dict[str, _FakeChain]) -> None:
    monkeypatch.setattr(graphqa, "_chain", lambda mode: mapping[mode])


PAYLOAD = {
    "messages": [],
    "user_information": "男，65 岁，高血压",
    "medical_history": ["高血压"],
    "medicine_taking": ["硝苯地平"],
}


# ---------------------------------------------------------------- 超时工具


def test_run_with_timeout_returns_value():
    assert graphqa.run_with_timeout(lambda: "ok", 1.0, path="global") == "ok"


def test_run_with_timeout_raises_graphqa_timeout():
    with pytest.raises(graphqa.GraphQaTimeout) as err:
        graphqa.run_with_timeout(lambda: time.sleep(0.2), 0.02, path="global")
    assert err.value.path == "global"


def test_run_with_timeout_propagates_exception():
    def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        graphqa.run_with_timeout(boom, 1.0, path="global")


# ---------------------------------------------------------------- 分层检索


def test_retrieve_uses_global_when_it_answers(monkeypatch):
    _patch_chains(monkeypatch, {"global": _FakeChain(text="图谱结论"), "local": _FakeChain()})

    text, path = graphqa.retrieve(PAYLOAD)

    assert (text, path) == ("图谱结论", graphqa.PATH_GLOBAL)


def test_retrieve_falls_back_to_local_on_global_timeout(monkeypatch):
    _patch_chains(
        monkeypatch,
        {
            "global": _FakeChain(text="太慢的结果", delay=0.2),
            "local": _FakeChain(text="局部检索结论"),
        },
    )

    text, path = graphqa.retrieve(PAYLOAD)

    assert (text, path) == ("局部检索结论", graphqa.PATH_LOCAL)


def test_retrieve_falls_back_to_keyword_when_llm_layers_fail(monkeypatch):
    _patch_chains(
        monkeypatch,
        {"global": _FakeChain(boom=True), "local": _FakeChain(text="")},
    )
    monkeypatch.setattr(graphqa, "_keyword_lookup", lambda q: "高血压 术前 继续用药")

    text, path = graphqa.retrieve(PAYLOAD)

    assert path == graphqa.PATH_KEYWORD
    assert "高血压" in text


def test_retrieve_returns_none_when_every_layer_fails(monkeypatch):
    _patch_chains(
        monkeypatch,
        {"global": _FakeChain(boom=True), "local": _FakeChain(boom=True)},
    )
    monkeypatch.setattr(graphqa, "_keyword_lookup", lambda q: "")

    text, path = graphqa.retrieve(PAYLOAD)

    assert (text, path) == ("", graphqa.PATH_NONE)


def test_attempt_normalizes_dict_result(monkeypatch):
    """真实检索链返回的是 {risk_analysis: ...} 字典，不能被当成字符串处理。"""

    class _DictChain:
        def invoke(self, _payload):
            return {"risk_analysis": "图谱给出的风险结论", "messages": []}

    monkeypatch.setattr(graphqa, "_chain", lambda mode: _DictChain())

    text, path = graphqa.retrieve(PAYLOAD)

    assert (text, path) == ("图谱给出的风险结论", graphqa.PATH_GLOBAL)


def test_keyword_lookup_hits_chunks_from_index():
    """真实读取 dickens 分块做词面匹配（不调模型）。"""
    hits = graphqa._keyword_lookup("高血压 术前 停药", top_k=2)

    assert isinstance(hits, str)
    # 索引里必然有高血压相关内容；至少不应为空
    assert hits


def test_retrieve_keyword_returns_snippet_without_llm(monkeypatch):
    monkeypatch.setattr(graphqa, "_keyword_lookup", lambda q: "片段内容")

    assert graphqa.retrieve_keyword(PAYLOAD) == "片段内容"


# ---------------------------------------------------------------- 风险节点取数


def test_resolve_for_risk_waits_once_then_gives_up(monkeypatch):
    monkeypatch.setattr(settings, "graphqa_wait_seconds", 0.2)
    events.register("s-wait")

    started = time.monotonic()
    text, path = graphqa.resolve_for_risk("s-wait", {})
    first = time.monotonic() - started

    started = time.monotonic()
    graphqa.resolve_for_risk("s-wait", {})
    second = time.monotonic() - started

    assert path == graphqa.PATH_PROFILE
    assert text == graphqa.PROFILE_ONLY_TEXT
    assert first >= 0.15, "第一次应该真的等了一会儿"
    assert second < 0.05, "第二次不应再等（每会话只等一次）"


def test_resolve_for_risk_uses_prefetched_result(monkeypatch):
    events.register("s-hit")
    events.set_result("s-hit", "图谱结论", graphqa.PATH_LOCAL)

    text, path = graphqa.resolve_for_risk("s-hit", {})

    assert (text, path) == ("图谱结论", graphqa.PATH_LOCAL)


def test_resolve_for_risk_prefers_state_result():
    text, path = graphqa.resolve_for_risk(
        "s-state", {"graph_qa_result": "节点里已有的结论", "graph_is_qa": True}
    )

    assert (text, path) == ("节点里已有的结论", graphqa.PATH_GLOBAL)


def test_resolve_for_risk_without_event_does_not_block():
    started = time.monotonic()
    _text, path = graphqa.resolve_for_risk("s-noevent", {})
    elapsed = time.monotonic() - started

    assert path == graphqa.PATH_PROFILE
    assert elapsed < 0.05


# ---------------------------------------------------------------- 打点


def _sample(name: str, labels: dict[str, str]) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return float(value) if value is not None else 0.0


def test_metrics_record_outcome_and_path(monkeypatch):
    _patch_chains(
        monkeypatch,
        {"global": _FakeChain(boom=True), "local": _FakeChain(text="局部结论")},
    )
    # 计数器是进程级累加的，断言增量而不是绝对值
    before_error = _sample("cmekg_graphqa_total", {"outcome": "error", "path": "global"})
    before_hit = _sample("cmekg_graphqa_total", {"outcome": "hit", "path": "local"})

    graphqa.retrieve(PAYLOAD)

    assert _sample("cmekg_graphqa_total", {"outcome": "error", "path": "global"}) == before_error + 1
    assert _sample("cmekg_graphqa_total", {"outcome": "hit", "path": "local"}) == before_hit + 1
    # 直方图必须能采样到（有样本即说明观测发生了）
    assert _sample("cmekg_graphqa_seconds_count", {"path": "local"}) >= 1.0
