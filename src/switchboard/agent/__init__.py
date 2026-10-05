"""Agents: the tool-calling LLM agent and the deterministic workflow baseline."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from switchboard import config
from switchboard.agent.loop import Agent, RunResult
from switchboard.agent.workflow import WorkflowAgent
from switchboard.retrieval import Retriever


class SupportAgent(Protocol):
    def run(self, customer_id: str, message: str, conn=None) -> RunResult: ...  # noqa: ANN001

    def config_dict(self) -> dict: ...


@lru_cache(maxsize=1)
def _qwen():  # noqa: ANN202 - loaded once per process; a 3B model is expensive
    from switchboard.agent.llm import QwenChat

    return QwenChat()


def build_agent(
    kind: str | None = None,
    retriever: Retriever | None = None,
    guardrails_on: bool | None = None,
    exposure: str | None = None,
) -> SupportAgent:
    kind = (kind or config.AGENT).lower()
    if kind == "workflow":
        return WorkflowAgent(retriever)
    if kind == "llm":
        from switchboard.router import load_router

        exposure = exposure or config.TOOL_EXPOSURE
        router = load_router() if exposure == "routed" else None
        if exposure == "routed" and router is None:
            raise RuntimeError("Routed exposure needs a trained router at models/router.joblib (not included).")
        return Agent(_qwen(), retriever, guardrails_on, exposure, router)
    raise ValueError(f"Unknown agent {kind!r} (expected 'workflow' or 'llm').")


__all__ = ["Agent", "RunResult", "SupportAgent", "WorkflowAgent", "build_agent"]
