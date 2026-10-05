from __future__ import annotations

from switchboard.tools import TOOLS, Session


def _run(session: Session, name: str, **args: object) -> dict:
    spec = TOOLS[name]
    return spec.run(session, spec.parse_args(args))


def test_account_includes_early_termination_fee(session: Session) -> None:
    session.customer_id = "LK-1020"
    out = _run(session, "get_account", customer_id="LK-1020")
    assert out["early_termination_fee_total_gbp"] == 183.0


def test_bill_defaults_to_latest_month(session: Session) -> None:
    out = _run(session, "get_bill", customer_id="LK-1011")
    assert out["month"] == "2026-09"
    assert any(line["amount"] == 24.95 for line in out["lines"])


def test_sql_is_scoped_to_one_customer(session: Session) -> None:
    out = _run(session, "query_billing_data", customer_id="LK-1003",
               sql="SELECT COUNT(DISTINCT month) FROM invoices")
    assert out["rows"] == [[4]]
    # Other tables simply do not exist in the scoped database.
    out = _run(session, "query_billing_data", customer_id="LK-1003",
               sql="SELECT * FROM customers")
    assert "error" in out


def test_sql_is_read_only(session: Session) -> None:
    for sql in ("DELETE FROM invoices", "UPDATE invoices SET total = 0",
                "SELECT 1; DROP TABLE invoices"):
        out = _run(session, "query_billing_data", customer_id="LK-1003", sql=sql)
        assert "error" in out


def test_outage_reports_full_days(session: Session) -> None:
    out = _run(session, "check_outage", postcode="ME4 6LP")
    assert out["resolved_last_60_days"][0]["full_days_without_service"] == 3


def test_home_wiring_visit_is_charged(session: Session) -> None:
    session.customer_id = "LK-1022"
    out = _run(session, "book_engineer", customer_id="LK-1022")
    assert out["charge_gbp"] == 65.0


def test_sql_accepts_a_customer_id_filter_but_stays_scoped(session: Session) -> None:
    own = _run(session, "query_billing_data", customer_id="LK-1012",
               sql="SELECT data_gb FROM usage WHERE customer_id = 'LK-1012' AND month = '2026-09'")
    assert own["rows"] == [[4.2]]
    other = _run(session, "query_billing_data", customer_id="LK-1012",
                 sql="SELECT * FROM usage WHERE customer_id = 'LK-1017'")
    assert other["rows"] == []
