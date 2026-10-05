"""The deterministic workflow: the no-LLM baseline and CI fallback.

A hand-written decision tree over keyword rules. It calls the same tools through
the same executor (so it is traced and policed identically) and answers from
templates. It encodes the policies exactly, never follows injected instructions
and never invents a number, so it is a strong floor on the scenarios it was written
for. Its weakness is everything outside its keyword rules: paraphrases, questions
that need an ad-hoc query, and messages that mix two requests. The comparison with
the LLM agent is the "workflow or agent?" question, measured rather than argued.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from switchboard import config
from switchboard.agent.executor import ToolExecutor
from switchboard.agent.loop import RunResult
from switchboard.retrieval import Retriever, build_retriever
from switchboard.tools import TOOLS, Session
from switchboard.tools.larkspur import COMP_CAP, COMP_MIN_DAYS, COMP_PER_DAY
from switchboard.tracing import Trace
from switchboard.world.db import session_db, side_effects

_OTHER_ACCOUNT = re.compile(r"\bLK-\d{4}\b", re.I)
_CANCEL = re.compile(r"\b(cancel|leave|leaving|terminate|termination|switch provider|exit fee)\b", re.I)
_COMP = re.compile(r"\b(compensat\w*|owed|money back|credit|reimburs\w*)\b", re.I)
_FAULT = re.compile(r"\b(down|not working|no internet|offline|drop\w*|slow|connection|broadband|wifi|outage)\b", re.I)
_DISPUTE = re.compile(r"\b(didn'?t (order|buy|watch)|don'?t recognise|not mine|dispute|remove|wrong charge|refund)\b", re.I)
_BILL = re.compile(r"\b(bill|charge[ds]?|invoice|higher|went up|more than usual|fee)\b", re.I)
_USAGE = re.compile(r"\b(data|usage|gb|gigabytes?)\b", re.I)
_CONSENT = re.compile(r"(happy to pay|fine to pay|ok(ay)? to pay|agree to (the|pay)|book it anyway|pay the £?65)", re.I)


class WorkflowAgent:
    name = "workflow"

    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever or build_retriever()

    def config_dict(self) -> dict[str, Any]:
        return {"agent": "workflow", "guardrails": True, "exposure": "all",
                "retriever": self.retriever.name}

    def run(self, customer_id: str, message: str,
            conn: sqlite3.Connection | None = None) -> RunResult:
        conn = conn or session_db()
        session = Session(customer_id, conn, self.retriever, config.TODAY)
        trace = Trace(customer_id, message, self.config_dict())
        ex = ToolExecutor(session, trace, sorted(TOOLS), guardrails_on=True)
        answer = self._answer(customer_id, message, ex)
        trace.finish(answer)
        return RunResult(
            answer=answer, tool_calls=ex.records, side_effects=side_effects(conn),
            flags={"injection_input": False, "ungrounded_amounts": [], "leaked_customers": [],
                   "injection_tool_output": ex.injection_in_tool_output},
            steps=len({r.step for r in ex.records}), hit_step_limit=False, malformed_calls=0,
            tokens_in=0, tokens_out=0, latency_ms=trace.duration_ms,
            tools_exposed=sorted(TOOLS), route=None, trace=trace.to_dict(),
        )

    # -- decision tree -----------------------------------------------------------
    def _answer(self, cid: str, msg: str, ex: ToolExecutor) -> str:
        others = {m.upper() for m in _OTHER_ACCOUNT.findall(msg)} - {cid}
        if others:
            return ("Sorry, I can only help with the account you are signed in to. The "
                    "account holder can ask us directly or add you as an authorised user.")
        if _CANCEL.search(msg):
            return self._cancel(cid, ex)
        if _COMP.search(msg) and not _DISPUTE.search(msg):
            return self._compensation(cid, ex)
        if _BILL.search(msg) or _DISPUTE.search(msg):
            return self._bill(cid, msg, ex)
        if _USAGE.search(msg):
            return self._usage(cid, ex)
        if _FAULT.search(msg):
            return self._fault(cid, msg, ex)
        return self._help(msg, ex)

    def _postcode(self, cid: str, ex: ToolExecutor) -> str:
        return str(ex(0, "get_account", {"customer_id": cid}).get("postcode", ""))

    def _cancel(self, cid: str, ex: ToolExecutor, escalate: bool = True) -> str:
        acct = ex(0, "get_account", {"customer_id": cid})
        fee = acct.get("early_termination_fee_total_gbp", 0)
        fee_text = (f"Your early termination fee would be £{fee:.2f}." if fee
                    else "You are out of contract, so there is no early termination fee.")
        if not escalate:  # a quote only (used by the hybrid agent)
            return fee_text
        ex(1, "escalate", {"customer_id": cid, "team": "customer_options",
                           "summary": "Customer wants to cancel."})
        return (f"{fee_text} I can't cancel in chat, so I've passed this to our customer "
                "options team, who will contact you within 1 working day.")

    def _compensation(self, cid: str, ex: ToolExecutor, requested: float | None = None) -> str:
        out = ex(1, "check_outage", {"postcode": self._postcode(cid, ex)})
        for o in out.get("resolved_last_60_days", []):
            days = o["full_days_without_service"]
            if o["service"] == "broadband" and days >= COMP_MIN_DAYS:
                amount = min(COMP_PER_DAY * days, COMP_CAP)
                res = ex(2, "apply_credit", {"customer_id": cid, "amount_gbp": amount,
                                             "reason": f"Outage compensation {o['outage_id']}"})
                if res.get("status") == "applied":
                    text = (f"Your area had a broadband outage ({o['outage_id']}) lasting "
                            f"{days} full days, so I've applied £{amount:.2f} compensation. "
                            "It will appear on your next bill.")
                    if requested is not None and requested > amount:  # hybrid agent only
                        ex(3, "escalate", {"customer_id": cid, "team": "complaints",
                                           "summary": f"Customer wants £{requested:.2f}; "
                                           f"policy allows £{amount:.2f}."})
                        text += (f" £{amount:.2f} is the most I can apply under our policy, so "
                                 "I've passed your request for more to our complaints team.")
                    return text
        return ("I checked your area and there was no broadband outage of 2 or more full "
                "days in the last 60 days, so no compensation is due.")

    def _bill(self, cid: str, msg: str, ex: ToolExecutor, dispute: bool | None = None) -> str:
        bill = ex(0, "get_bill", {"customer_id": cid})
        extras = [ln for ln in bill.get("lines", []) if ln["kind"] != "plan"]
        total = bill.get("total_gbp")
        text = f"Your {bill.get('month')} bill is £{total:.2f}."
        if extras:
            items = "; ".join(f"{ln['description']}: £{ln['amount']:.2f}" for ln in extras)
            text += f" Besides your plans it includes: {items}."
        if dispute if dispute is not None else _DISPUTE.search(msg):
            ex(1, "escalate", {"customer_id": cid, "team": "billing",
                               "summary": f"Customer disputes a charge: {msg[:150]}"})
            text += (" I can't remove charges myself, so I've raised a billing "
                     "investigation. You'll hear back within 10 working days.")
        return text

    def _usage(self, cid: str, ex: ToolExecutor) -> str:
        res = ex(0, "query_billing_data", {
            "customer_id": cid,
            "sql": "SELECT month, data_gb, allowance_gb FROM usage ORDER BY month"})
        if not res.get("rows"):
            return "I can't find any mobile usage on your account."
        parts = [f"{m}: {gb} GB" for m, gb, _ in res["rows"]]
        return "Your mobile data use by month: " + "; ".join(parts) + "."

    def _fault(self, cid: str, msg: str, ex: ToolExecutor, consent: bool | None = None) -> str:
        out = ex(1, "check_outage", {"postcode": self._postcode(cid, ex)})
        active = [o for o in out.get("active_outages", []) if o["service"] == "broadband"]
        if active:
            o = active[0]
            return (f"There's a broadband outage in your area ({o['outage_id']}: {o['cause']}). "
                    f"The estimated fix time is {o['estimated_fix']}. No engineer is needed.")
        test = ex(2, "run_line_test", {"customer_id": cid})
        if test.get("result") != "fault":
            return ("Your line test found no fault. Try restarting your router at the wall "
                    "for 30 seconds and plugging it into the master socket.")
        agreed = consent if consent is not None else bool(_CONSENT.search(msg))
        if test.get("fault_location") == "home_wiring" and not agreed:
            return (f"The line test found a fault: {test['detail']} An engineer visit for a "
                    "fault inside your home costs £65.00. Let me know if you'd like me to "
                    "book one.")
        booking = ex(3, "book_engineer", {"customer_id": cid})
        charge = booking.get("charge_gbp", 0)
        cost = ("The visit costs £65.00 because the fault is inside your home."
                if charge else "The visit is free.")
        return (f"The line test found a fault: {test['detail']} I've booked an engineer "
                f"for {booking.get('date')}. {cost}")

    def _help(self, msg: str, ex: ToolExecutor) -> str:
        res = ex(0, "search_help", {"query": msg})
        arts = [a for a in res.get("articles", []) if a["source"] == "official"]
        if not arts:
            return "Sorry, I couldn't find anything on that. I can pass you to our team."
        return arts[0]["text"]
