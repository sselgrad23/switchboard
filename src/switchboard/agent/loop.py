"""The tool-calling agent loop.

One customer message in, one reply out. Each step: the model sees the conversation
so far plus the exposed tool schemas, and either calls tools (executed through
``ToolExecutor`` and fed back as ``tool`` messages) or writes the final reply. The
loop stops at the reply or at ``MAX_STEPS`` model calls.

Guardrails on the way in (PII redaction, injection notice), around every tool call
(policy, output sanitising) and on the way out (grounding check with one revision,
cross-customer leak check) are all switchable with one flag, so their effect is
measurable.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from switchboard import config, guardrails
from switchboard.agent.executor import ToolCallRecord, ToolExecutor
from switchboard.agent.llm import ChatModel
from switchboard.agent.parse import parse
from switchboard.agent.prompts import INJECTION_NOTICE, system_prompt
from switchboard.retrieval import Retriever, build_retriever
from switchboard.router import IntentRouter
from switchboard.tools import TOOLS, Session
from switchboard.tracing import Trace
from switchboard.world.db import session_db, side_effects

REPAIR_PROMPT = (
    "Your tool call could not be parsed. Reply with exactly one JSON object "
    '{"name": ..., "arguments": {...}} inside <tool_call></tool_call> tags, '
    "or answer the customer directly."
)
STEP_LIMIT_REPLY = (
    "Sorry, I wasn't able to finish that. Please try again, or ask me to pass this to "
    "one of our team."
)
LEAK_REPLY = "Sorry, I can only discuss the account you are signed in to."


@dataclass
class RunResult:
    answer: str
    tool_calls: list[ToolCallRecord]
    side_effects: dict[str, list[dict[str, Any]]]
    flags: dict[str, Any]
    steps: int
    hit_step_limit: bool
    malformed_calls: int
    tokens_in: int
    tokens_out: int
    latency_ms: float
    tools_exposed: list[str]
    route: dict[str, Any] | None
    trace: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class _TurnState:
    """Counters accumulated across the turns of one conversation."""

    steps: int = 0
    malformed: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    hit_limit: bool = False


class Agent:
    """LLM agent: model + tools + guardrails, optionally with the intent router."""

    def __init__(
        self,
        llm: ChatModel,
        retriever: Retriever | None = None,
        guardrails_on: bool | None = None,
        exposure: str | None = None,
        router: IntentRouter | None = None,
        max_steps: int | None = None,
        prompt_variant: str = "base",
        force_action: bool = False,
    ) -> None:
        self.llm = llm
        self.retriever = retriever or build_retriever()
        self.guardrails_on = config.GUARDRAILS if guardrails_on is None else guardrails_on
        self.exposure = exposure or config.TOOL_EXPOSURE
        self.router = router
        self.max_steps = max_steps or config.MAX_STEPS
        self.prompt_variant = prompt_variant
        self.force_action = force_action
        self.name = f"llm[{getattr(llm, 'name', 'model')}]"

    def config_dict(self) -> dict[str, Any]:
        return {"agent": "llm", "model": getattr(self.llm, "name", "?"),
                "guardrails": self.guardrails_on, "exposure": self.exposure,
                "retriever": self.retriever.name, "prompt": self.prompt_variant,
                "force_action": self.force_action}

    def _exposed(self, message: str) -> tuple[list[str], dict[str, Any] | None]:
        if self.exposure != "routed" or self.router is None:
            return sorted(TOOLS), None
        decision = self.router.route(message)
        route = {"intents": [list(i) for i in decision.intents],
                 "abstained": decision.tools is None}
        return sorted(decision.tools or TOOLS), route

    @staticmethod
    def _unbacked(answer: str, executor: ToolExecutor) -> set[str]:
        done = {r.name for r in executor.records if r.status == "ok"}
        return guardrails.claimed_actions(answer) - done

    @staticmethod
    def _unpromised(answer: str, executor: ToolExecutor) -> set[str]:
        done = {r.name for r in executor.records if r.status == "ok"}
        return guardrails.promised_actions(answer) - done

    def run(self, customer_id: str, message: str, conn: sqlite3.Connection | None = None,
            user_sim: Any = None, max_customer_turns: int = 3) -> RunResult:
        """Handle one customer message, or a whole conversation if ``user_sim`` is given.

        With a simulated customer, each agent reply goes to ``user_sim.reply(...)``;
        the conversation ends when it returns None (goal met or given up) or after
        ``max_customer_turns`` further customer messages. ``answer`` is then every agent
        reply joined, so facts said in any turn count.
        """
        conn = conn or session_db()
        session = Session(customer_id, conn, self.retriever, config.TODAY)
        trace = Trace(customer_id, message, self.config_dict())
        flags: dict[str, Any] = {}
        found = conn.execute("SELECT postcode FROM customers WHERE customer_id = ?",
                             (customer_id,)).fetchone()
        system = system_prompt(customer_id, config.TODAY, found[0] if found else "unknown",
                               self.prompt_variant)
        flags["injection_input"] = guardrails.detect_injection(message)
        if self.guardrails_on and flags["injection_input"]:
            system += "\n\n" + INJECTION_NOTICE
            trace.event("guard", "input.injection", "flagged")

        exposed, route = self._exposed(message)
        schemas = [TOOLS[n].json_schema() for n in exposed]
        executor = ToolExecutor(session, trace, exposed, self.guardrails_on)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        st = _TurnState()
        customer_texts: list[str] = []
        answers: list[str] = []
        text = message
        for turn in range(1 + (max_customer_turns if user_sim is not None else 0)):
            user_text = text
            if self.guardrails_on:
                user_text, redacted = guardrails.redact_pii(text)
                flags["pii_redacted"] = flags.get("pii_redacted", False) or redacted
            customer_texts.append(text)
            messages.append({"role": "user", "content": user_text})
            answer = self._turn(messages, schemas, executor, trace, st, system, customer_texts,
                                conn, customer_id, flags)
            answers.append(answer)
            if user_sim is None:
                break
            messages.append({"role": "assistant", "content": answer})
            with trace.span("sim", "customer", turn=turn) as span:
                nxt = user_sim.reply(customer_texts, answers)
                span.attributes["output"] = nxt
            if nxt is None:
                break
            text = nxt
        flags["customer_turns"] = len(customer_texts)
        flags["customer_messages"] = customer_texts
        flags["injection_tool_output"] = executor.injection_in_tool_output
        flags["unbacked_claims"] = sorted(self._unbacked(answers[-1], executor))
        final = "\n\n".join(answers)
        trace.finish(final)
        return RunResult(
            answer=final, tool_calls=executor.records, side_effects=side_effects(conn),
            flags=flags, steps=st.steps, hit_step_limit=st.hit_limit,
            malformed_calls=st.malformed, tokens_in=st.tokens_in, tokens_out=st.tokens_out,
            latency_ms=trace.duration_ms, tools_exposed=exposed, route=route,
            trace=trace.to_dict(),
        )

    def _chat(self, messages: list[dict[str, Any]], schemas: list[dict[str, Any]],
              trace: Trace, st: _TurnState, name: str = "chat") -> str:
        with trace.span("llm", name, step=st.steps) as span:
            comp = self.llm.chat(messages, schemas)
            span.attributes.update(tokens_in=comp.tokens_in, tokens_out=comp.tokens_out,
                                   output=comp.text)
        st.tokens_in += comp.tokens_in
        st.tokens_out += comp.tokens_out
        return comp.text

    def _turn(self, messages: list[dict[str, Any]], schemas: list[dict[str, Any]],
              executor: ToolExecutor, trace: Trace, st: _TurnState, system: str,
              customer_texts: list[str], conn: sqlite3.Connection, customer_id: str,
              flags: dict[str, Any]) -> str:
        """One agent turn: tool calls until a reply, then the output checks."""
        answer, hit_limit, nudges = "", True, 0
        max_nudges = 2 if self.force_action else 1
        for i in range(self.max_steps):
            st.steps += 1
            raw = self._chat(messages, schemas, trace, st)
            parsed = parse(raw)
            if parsed.errors and not parsed.calls:
                st.malformed += 1
                trace.event("llm", "parse", "invalid", raw=parsed.errors)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": REPAIR_PROMPT})
                continue
            if not parsed.calls:
                answer, hit_limit = parsed.content, False
                # Action verifier. Default: past-tense claims ("I've booked") with no
                # backing tool call get one nudge (guardrails on). Force-action also
                # counts commitments ("I'll book") and allows two nudges.
                pending = (self._unpromised(answer, executor) if self.force_action
                           else self._unbacked(answer, executor) if self.guardrails_on
                           else set())
                if pending and nudges < max_nudges and i + 1 < self.max_steps:
                    nudges += 1
                    hit_limit = True
                    trace.event("guard", "output.unbacked_claim", "flagged",
                                tools=sorted(pending))
                    messages.append({"role": "assistant", "content": answer})
                    messages.append({"role": "user", "content": (
                        "System check: your reply says you have used or will use "
                        f"{', '.join(sorted(pending))}, but no such tool call succeeded. If "
                        "the policies allow the action, call the tool now. Otherwise, "
                        "correct your reply so it does not promise the action.")})
                    continue
                break
            messages.append({
                "role": "assistant", "content": parsed.content,
                "tool_calls": [{"type": "function",
                                "function": {"name": c.name, "arguments": c.arguments}}
                               for c in parsed.calls],
            })
            for call in parsed.calls:
                result = executor(st.steps, call.name, call.arguments)
                messages.append({"role": "tool", "name": call.name,
                                 "content": json.dumps(result, ensure_ascii=False)})
        if hit_limit:
            answer = STEP_LIMIT_REPLY
            st.hit_limit = True
            trace.event("llm", "step_limit", "error", max_steps=self.max_steps)

        # Output checks: every £ amount must come from a tool result or the prompt.
        sources = [system, *customer_texts] + [json.dumps(r.result) for r in executor.records]
        ungrounded = guardrails.ungrounded_amounts(answer, sources)
        if ungrounded and self.guardrails_on and not hit_limit:
            trace.event("guard", "output.ungrounded_amount", "flagged", amounts=ungrounded)
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": (
                f"System check: your reply mentions {', '.join('£' + a for a in ungrounded)}, "
                "which no tool returned. Rewrite the reply using only amounts from the tool "
                "results. Do not call tools.")})
            revised = parse(self._chat(messages, [], trace, st, name="revise"))
            if revised.content and not revised.calls:
                answer = revised.content
            ungrounded = guardrails.ungrounded_amounts(answer, sources)
        flags["ungrounded_amounts"] = sorted(set(flags.get("ungrounded_amounts", [])) |
                                             set(ungrounded))
        leaked = guardrails.leaked_customers(answer, conn, customer_id,
                                             ignore=" ".join(customer_texts))
        flags["leaked_customers"] = sorted(set(flags.get("leaked_customers", [])) | set(leaked))
        if leaked and self.guardrails_on:
            trace.event("guard", "output.cross_customer_leak", "blocked", customers=leaked)
            answer = LEAK_REPLY
        return answer
