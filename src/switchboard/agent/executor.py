"""Run one tool call: validate, apply policy, execute, sanitise, trace.

Shared by the LLM agent and the deterministic workflow so both are traced and
policed identically. The order matters:

1. unknown or unexposed tool -> error back to the model
2. argument validation (Pydantic) -> readable error back to the model
3. policy check against the evidence gathered *before* this call
   - guardrails on: a blocked call is not executed; the reason goes back
   - guardrails off: the call executes; the decision is kept for auditing
4. execute, then scan the output for injected instructions (guardrails on)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from switchboard import guardrails
from switchboard.tools import TOOLS, Session, validation_message
from switchboard.tracing import Trace


@dataclass
class ToolCallRecord:
    step: int
    name: str
    arguments: dict[str, Any]
    status: str  # ok | blocked | invalid_args | unknown_tool | error
    result: dict[str, Any] = field(default_factory=dict)
    rule: str = ""  # the policy rule that blocked it (or would have, in audit mode)
    would_block: bool = False  # audit mode: executed although the policy says no


def _sanitise(obj: Any) -> tuple[Any, bool]:
    """Neutralise injection-looking strings anywhere in a tool result."""
    if isinstance(obj, str):
        if guardrails.detect_injection(obj):
            return guardrails.neutralise(obj), True
        return obj, False
    if isinstance(obj, dict):
        flagged = False
        out = {}
        for k, v in obj.items():
            out[k], f = _sanitise(v)
            flagged |= f
        return out, flagged
    if isinstance(obj, list):
        items = [_sanitise(v) for v in obj]
        return [v for v, _ in items], any(f for _, f in items)
    return obj, False


class ToolExecutor:
    def __init__(self, session: Session, trace: Trace, exposed: list[str], guardrails_on: bool):
        self.session = session
        self.trace = trace
        self.exposed = set(exposed)
        self.guardrails_on = guardrails_on
        self.records: list[ToolCallRecord] = []
        self.injection_in_tool_output = False

    def __call__(self, step: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        rec = self._run(step, name, arguments)
        self.records.append(rec)
        return rec.result

    def _run(self, step: int, name: str, arguments: dict[str, Any]) -> ToolCallRecord:
        spec = TOOLS.get(name)
        if spec is None or name not in self.exposed:
            result = {"error": f"Unknown tool {name!r}. Available tools: "
                      f"{', '.join(sorted(self.exposed))}."}
            self.trace.event("tool", name, "invalid", reason="unknown_tool", arguments=arguments)
            return ToolCallRecord(step, name, arguments, "unknown_tool", result)
        try:
            args = spec.parse_args(arguments)
        except ValidationError as err:
            result = {"error": validation_message(err)}
            self.trace.event("tool", name, "invalid", reason="invalid_args",
                             arguments=arguments, error=result["error"])
            return ToolCallRecord(step, name, arguments, "invalid_args", result)

        decision = guardrails.check(name, args, self.session)
        if not decision.allowed:
            self.trace.event("guard", f"policy.{decision.rule}",
                             "blocked" if self.guardrails_on else "flagged",
                             tool=name, arguments=arguments, message=decision.message)
            if self.guardrails_on:
                return ToolCallRecord(step, name, arguments, "blocked",
                                      {"error": decision.message}, rule=decision.rule)

        with self.trace.span("tool", name, arguments=arguments) as span:
            try:
                result = spec.run(self.session, args)
                status = "error" if "error" in result else "ok"
            except Exception as err:  # noqa: BLE001 - a tool bug must not crash the loop
                result, status = {"error": f"Tool failed: {err}"}, "error"
            if self.guardrails_on:
                result, flagged = _sanitise(result)
                if flagged:
                    self.injection_in_tool_output = True
                    self.trace.event("guard", "tool_output.injection", "flagged", tool=name)
            span.status = status
            span.attributes["result"] = result
        return ToolCallRecord(step, name, arguments, status, result,
                              rule=decision.rule, would_block=not decision.allowed)
