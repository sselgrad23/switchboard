from __future__ import annotations

from switchboard.guardrails import check
from switchboard.tools import TOOLS, Session


def _args(name: str, **args: object):  # noqa: ANN202
    return TOOLS[name].parse_args(args)


def test_other_customers_are_blocked(session: Session) -> None:
    d = check("get_bill", _args("get_bill", customer_id="LK-1011"), session)
    assert not d.allowed and d.rule == "cross_account"


def test_credit_needs_a_confirmed_outage(session: Session) -> None:
    credit = _args("apply_credit", customer_id="LK-1003", amount_gbp=15, reason="x")
    assert check("apply_credit", credit, session).rule == "credit_evidence"
    TOOLS["check_outage"].run(session, _args("check_outage", postcode="ME4 6LP"))
    assert check("apply_credit", credit, session).allowed


def test_credit_is_capped_at_the_entitlement(session: Session) -> None:
    TOOLS["check_outage"].run(session, _args("check_outage", postcode="ME4 6LP"))
    over = _args("apply_credit", customer_id="LK-1003", amount_gbp=20, reason="x")
    d = check("apply_credit", over, session)
    assert not d.allowed and d.rule == "credit_amount" and "15.00" in d.message


def test_engineer_needs_a_failed_line_test(session: Session, retriever) -> None:  # noqa: ANN001
    s = Session("LK-1006", session.conn, retriever, "2026-09-14")
    book = _args("book_engineer", customer_id="LK-1006")
    assert check("book_engineer", book, s).rule == "engineer_evidence"
    TOOLS["run_line_test"].run(s, _args("run_line_test", customer_id="LK-1006"))
    assert check("book_engineer", book, s).allowed


def test_no_engineer_during_an_area_outage(session: Session, retriever) -> None:  # noqa: ANN001
    s = Session("LK-1001", session.conn, retriever, "2026-09-14")
    TOOLS["run_line_test"].run(s, _args("run_line_test", customer_id="LK-1001"))
    d = check("book_engineer", _args("book_engineer", customer_id="LK-1001"), s)
    assert d.rule == "engineer_outage"
