"""Prometheus 指标定义（P4）。

命名遵循 Prometheus 惯例：``<namespace>_<subsystem>_<name>_<unit>``。
标签基数刻意保持低——路径用路由模板，不把 task_id 放进标签。
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

_HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_TURN_BUCKETS = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 240.0)

HTTP_REQUESTS = Counter(
    "cmekg_http_requests_total",
    "HTTP 请求总数",
    ["method", "path", "status"],
)

HTTP_LATENCY = Histogram(
    "cmekg_http_request_duration_seconds",
    "HTTP 请求耗时",
    ["method", "path"],
    buckets=_HTTP_BUCKETS,
)

TURNS = Counter(
    "cmekg_turns_total",
    "Agent 单轮对话的终态分布",
    ["status"],
)

TURN_DURATION = Histogram(
    "cmekg_turn_duration_seconds",
    "Agent 单轮对话耗时（从 RUNNING 到终态）",
    ["status"],
    buckets=_TURN_BUCKETS,
)

RETRIES = Counter(
    "cmekg_turn_retries_total",
    "因瞬时错误而重试的次数",
)

_GRAPHQA_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 30.0)

GRAPH_QA_TOTAL = Counter(
    "cmekg_graphqa_total",
    "GraphRAG 各层检索的结果分布",
    ["outcome", "path"],
)

GRAPH_QA_DURATION = Histogram(
    "cmekg_graphqa_seconds",
    "GraphRAG 各层检索耗时",
    ["path"],
    buckets=_GRAPHQA_BUCKETS,
)

GRAPH_QA_RESOLVED = Counter(
    "cmekg_graphqa_resolved_total",
    "风险告知最终采用的依据来源（profile_only 表示已降级）",
    ["path"],
)


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
