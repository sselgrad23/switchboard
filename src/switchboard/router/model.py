"""The intent router: a classic text classifier that narrows the LLM's tool list.

The router predicts one of Bitext telco's 26 intents from the customer's message and
maps it to the tools that intent can need. Fewer tools means a shorter prompt and
fewer wrong-tool choices for a small model, at the risk of hiding a tool the
conversation turns out to need. Whether that trade is worth it is an experiment
(``eval/experiments``), not an assumption.

Confidence handling: if the top intent is below ``ROUTER_THRESHOLD`` the second
intent's tools are added; below ``ROUTER_FLOOR`` the router abstains and every tool
is exposed. The core tools (account, help search, escalation) are always exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from switchboard import config

CORE_TOOLS = ("get_account", "search_help", "escalate")
_BILLING = ("get_bill", "query_billing_data")
_FAULT = ("check_outage", "run_line_test", "book_engineer")

INTENT_TOOLS: dict[str, tuple[str, ...]] = {
    "dispute_invoice": _BILLING,
    "invoices": _BILLING,
    "check_mobile_payments": _BILLING,
    "pay": _BILLING,
    "payment_methods": _BILLING,
    "schedule_payments": _BILLING,
    "check_excess_data_charges": _BILLING,
    "check_usage": _BILLING,
    "set_usage_limits": _BILLING,
    "get_compensation": ("check_outage", "apply_credit", "get_bill"),
    "report_problem": _FAULT,
    "report_poor_signal_coverage": _FAULT,
    "check_signal_coverage": _FAULT,
    "install_internet": _FAULT,
    "customer_service": (),
    "human_agent": (),
    "cancel_plan": (),
    "change_plan": (),
    "change_provider": (),
    "check_cancellation_fee": (),
    "sign_up_for_plan": (),
    "activate_call_management_services": (),
    "deactivate_call_management_services": (),
    "activate_phone": (),
    "deactivate_phone": (),
    "activate_roaming": (),
}


@dataclass(frozen=True)
class RouteDecision:
    tools: tuple[str, ...] | None  # None = abstain, expose everything
    intents: tuple[tuple[str, float], ...]


class IntentRouter:
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.vectorizer = bundle["vectorizer"]
        self.model = bundle["model"]
        self.labels: list[str] = list(bundle["labels"])
        self.kind: str = bundle.get("kind", "unknown")

    def rank(self, text: str) -> list[tuple[str, float]]:
        probs = self.model.predict_proba(self.vectorizer.transform([text]))[0]
        order = probs.argsort()[::-1]
        return [(self.labels[i], float(probs[i])) for i in order]

    def route(self, text: str) -> RouteDecision:
        ranked = self.rank(text)
        top = tuple(ranked[:2])
        (first, p1), (second, _) = ranked[0], ranked[1]
        if p1 < config.ROUTER_FLOOR:
            return RouteDecision(None, top)
        names = set(CORE_TOOLS) | set(INTENT_TOOLS.get(first, ()))
        if p1 < config.ROUTER_THRESHOLD:
            names |= set(INTENT_TOOLS.get(second, ()))
        return RouteDecision(tuple(sorted(names)), top)


@lru_cache(maxsize=1)
def load_router(path: Path | None = None) -> IntentRouter | None:
    """The trained router, or None if it has not been trained (then: all tools)."""
    import joblib

    p = Path(path or config.ROUTER_PATH)
    if not p.exists():
        return None
    return IntentRouter(joblib.load(p))
