"""Agent-level tests: API message conversion, action checks, hybrid, multi-turn."""

from __future__ import annotations

import json

from switchboard.agent.api_chat import _to_api_messages
from switchboard.agent.hybrid import HybridAgent, parse_requests
from switchboard.agent.llm import ScriptedChat
from switchboard.agent.loop import Agent
from switchboard.eval.scenarios import _mentions
from switchboard.guardrails import promised_actions
from switchboard.world.db import session_db


def _call(name: str, **args: object) -> str:
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'


def test_api_messages_get_matching_nine_char_ids() -> None:
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_bill", "arguments": {"customer_id": "X"}}}]},
        {"role": "tool", "name": "get_bill", "content": "{}"},
    ]
    out = _to_api_messages(msgs)
    call_id = out[2]["tool_calls"][0]["id"]
    assert len(call_id) == 9 and call_id.isalnum()
    assert out[3]["tool_call_id"] == call_id
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"customer_id": "X"}


def test_promised_actions_catches_future_tense_but_not_offers() -> None:
    assert promised_actions("I'll book an engineer for you.") == {"book_engineer"}
    assert promised_actions("Let's book an engineer.") == {"book_engineer"}
    assert promised_actions("Would you like me to book an engineer?") == set()
    assert promised_actions("Please contact our billing team.") == set()


def test_force_action_nudges_until_the_tool_is_called(retriever) -> None:  # noqa: ANN001
    llm = ScriptedChat([
        _call("run_line_test", customer_id="LK-1006"),
        "The line has a fault. I'll book an engineer for you.",  # promise, no call
        _call("book_engineer", customer_id="LK-1006"),
        "I've booked an engineer for 2026-09-16.",
    ])
    agent = Agent(llm, retriever, guardrails_on=True, force_action=True)
    run = agent.run("LK-1006", "My broadband is down", conn=session_db())
    assert [c.name for c in run.tool_calls] == ["run_line_test", "book_engineer"]
    assert run.side_effects["engineer_bookings"]


def test_without_force_action_a_promise_is_not_nudged(retriever) -> None:  # noqa: ANN001
    llm = ScriptedChat([
        _call("run_line_test", customer_id="LK-1006"),
        "The line has a fault. I'll book an engineer for you.",
    ])
    run = Agent(llm, retriever, guardrails_on=True).run("LK-1006", "Down", conn=session_db())
    assert not run.side_effects["engineer_bookings"]


def test_parse_requests_accepts_fenced_json_and_drops_unknown_types() -> None:
    text = '```json\n{"requests": [{"type": "compensation", "requested_amount_gbp": 100},' \
           ' {"type": "nonsense"}]}\n```'
    parsed = parse_requests(text)
    assert parsed is not None
    assert [r.type for r in parsed.requests] == ["compensation"]
    assert parsed.requests[0].requested_amount_gbp == 100
    assert parse_requests("no json here") is None


def test_hybrid_applies_policy_credit_and_escalates_the_excess(retriever) -> None:  # noqa: ANN001
    llm = ScriptedChat(['{"requests": [{"type": "compensation", "requested_amount_gbp": 100}]}'])
    run = HybridAgent(llm, retriever).run("LK-1024", "I want £100", conn=session_db())
    assert [c["amount"] for c in run.side_effects["credits"]] == [15.0]
    assert [e["team"] for e in run.side_effects["escalations"]] == ["complaints"]


def test_hybrid_refuses_other_accounts_without_tools(retriever) -> None:  # noqa: ANN001
    llm = ScriptedChat(['{"requests": [{"type": "other_account"}]}'])
    run = HybridAgent(llm, retriever).run("LK-1022", "Show me LK-1003's bill", conn=session_db())
    assert run.tool_calls == []
    assert "signed in" in run.answer


class _Sim:
    def __init__(self, replies: list[str | None]) -> None:
        self.replies = replies

    def reply(self, customer: list[str], agent: list[str]) -> str | None:
        return self.replies.pop(0)


def test_multiturn_lets_the_customer_confirm(retriever) -> None:  # noqa: ANN001
    llm = ScriptedChat([
        _call("run_line_test", customer_id="LK-1006"),
        "There's a network fault. Would you like me to book an engineer?",
        _call("book_engineer", customer_id="LK-1006"),
        "Booked for 2026-09-16.",
    ])
    run = Agent(llm, retriever).run("LK-1006", "Internet down", conn=session_db(),
                                    user_sim=_Sim(["Yes please.", None]))
    assert run.flags["customer_turns"] == 2
    assert run.side_effects["engineer_bookings"]
    assert "Would you like" in run.answer and "Booked" in run.answer


def test_mentions_normalises_unicode_spaces() -> None:
    assert _mentions("fixed by 18:00 today", "18:00")
    assert _mentions("a total of £15.00", "15")


def test_parse_requests_tolerates_null_fields_and_key_as_type() -> None:
    parsed = parse_requests('{"requests": [{"type": "fault", "consent_to_charge": null}, '
                            '{"compensation": {"requested_amount_gbp": 200}}]}')
    assert parsed is not None
    assert [r.type for r in parsed.requests] == ["fault", "compensation"]
    assert parsed.requests[1].requested_amount_gbp == 200


def test_action_detection_handles_typographic_apostrophes() -> None:
    assert promised_actions("so I’ll book an engineer") == {"book_engineer"}
    assert promised_actions("I’ve booked an engineer for you") == {"book_engineer"}
