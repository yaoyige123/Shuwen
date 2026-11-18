"""Import smoke test: check whether every project module imports cleanly.

Usage:
    python scripts/smoke_imports.py

Exit code 0 = all good, 1 = at least one failure.
Import only: no network calls, no LLM invocation.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    "app.config",
    "app.events",
    "app.schemas",
    "app.tasks",
    "app.store_redis",
    "app.dispatch",
    "app.checkpoint",
    "app.locks",
    "app.ratelimit",
    "app.runner",
    "app.worker",
    "app.tts",
    "app.graphqa",
    "app.api.deps",
    "app.api.routes",
    "app.main",
    "Graph.router",
    "Graph.state",
    "Chains.extract_info_chain",
    "Chains.assistant2agent_chain",
    "Chains.information_chain",
    "Chains.conclusion_chain",
    "Chains.sequence_primary_agent",
    "Chains.history_chain",
    "Chains.risk_chain",
    "Chains.analgesia_chain",
    "Chains.graph_qa_chain",
    "Graph.nodes",
    "Graph.graph",
    "api_server",
]

KEY_DISTS = [
    "langchain-core",
    "langchain-openai",
    "langchain-community",
    "langgraph",
    "pydantic",
    "pydantic-settings",
    "nano-graphrag",
    "lightrag",
    "numpy",
    "flask",
    "openai",
    "sounddevice",
    "soundfile",
    "websocket-client",
]


def main() -> int:
    print(f"python {sys.version.split()[0]}  ({sys.executable})")

    import os

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    from importlib.metadata import PackageNotFoundError, version

    print("--- packages ---")
    for dist in KEY_DISTS:
        try:
            print(f"  {dist:22s} {version(dist)}")
        except PackageNotFoundError:
            print(f"  {dist:22s} MISSING")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  {dist:22s} ERROR {exc}")

    print("--- imports ---")
    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"  OK   {name}")
        except Exception as exc:
            failed.append(name)
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")

    print("--- summary ---")
    print(f"{len(MODULES) - len(failed)}/{len(MODULES)} modules imported")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
