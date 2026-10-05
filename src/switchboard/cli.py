"""Talk to the agent from the command line.

    python -m switchboard.cli --customer LK-1006 "My broadband isn't working"
    python -m switchboard.cli --customer LK-1024 --agent hybrid --backend mistral \\
        --model ministral-3b-2512 "I want compensation for the August outage"
    python -m switchboard.cli --list-customers

Each call runs against a fresh copy of the seeded database, prints the reply, the
tool calls (with any guardrail blocks) and what was written to the back office.
"""

from __future__ import annotations

import argparse
import json

from switchboard.agent.loop import RunResult
from switchboard.eval.run_suite import build
from switchboard.retrieval import build_retriever
from switchboard.world.db import rows, session_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one message to the Larkspur agent.")
    parser.add_argument("message", nargs="?")
    parser.add_argument("--customer", default="LK-1006", help="Signed-in customer id.")
    parser.add_argument("--agent", choices=["workflow", "llm", "hybrid"], default="workflow")
    parser.add_argument("--backend", choices=["qwen-local", "groq", "mistral"],
                        default="qwen-local")
    parser.add_argument("--model", help="Model id (required for hosted backends).")
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    parser.add_argument("--no-guardrails", action="store_true")
    parser.add_argument("--list-customers", action="store_true")
    args = parser.parse_args()

    if args.list_customers:
        conn = session_db()
        for cust in rows(conn, "SELECT customer_id, name, postcode FROM customers"):
            products = [r["product"] for r in rows(
                conn, "SELECT product FROM subscriptions WHERE customer_id = ?",
                (cust["customer_id"],))]
            print(f"{cust['customer_id']}  {cust['name']:<16} {cust['postcode']:<9} "
                  f"{', '.join(products)}")
        return
    if not args.message:
        parser.error("a message is required")

    opts = {"agent": args.agent, "backend": args.backend, "model": args.model,
            "guardrails": not args.no_guardrails}
    agent, _ = build(opts, build_retriever(args.retriever))
    result: RunResult = agent.run(args.customer, args.message, conn=session_db())

    print(result.answer, "\n")
    for c in result.tool_calls:
        note = f"  [{c.status}: {c.rule}]" if c.status == "blocked" else f"  [{c.status}]"
        print(f"  {c.name}({json.dumps(c.arguments)}){note}")
    effects = {k: v for k, v in result.side_effects.items() if v}
    if effects:
        print("\nwrites:", json.dumps(effects, indent=2))


if __name__ == "__main__":
    main()
