"""Guardrails: input/output text checks and the pre-execution tool policy."""

from __future__ import annotations

import sqlite3

from switchboard.guardrails.policy import Decision, check, entitlement
from switchboard.guardrails.text import (
    claimed_actions,
    detect_injection,
    neutralise,
    promised_actions,
    redact_pii,
    ungrounded_amounts,
)


def leaked_customers(
    answer: str, conn: sqlite3.Connection, signed_in: str, ignore: str = ""
) -> list[str]:
    """Other customers whose name, email or postcode appears in the reply.

    Anything the customer typed themselves (``ignore``) does not count: repeating the
    account id they asked about in a refusal is not a leak.
    """
    low, seen = answer.lower(), ignore.lower()
    leaked = []
    rows = conn.execute("SELECT customer_id, name, email, postcode FROM customers")
    for cid, name, email, postcode in rows:
        if cid == signed_in:
            continue
        for value in (name, email, postcode):
            v = value.lower()
            if v in low and v not in seen:
                leaked.append(cid)
                break
    return leaked


__all__ = [
    "Decision", "check", "claimed_actions", "detect_injection", "entitlement", "leaked_customers",
    "neutralise", "promised_actions", "redact_pii", "ungrounded_amounts",
]
