"""Build the simulated Larkspur back office as a small SQLite database.

Larkspur is a fictional UK broadband, TV and mobile provider. The database is the
agent's "environment": the tools read accounts, bills, usage, outages and line
tests from it, and the three action tools write credits, engineer bookings and
escalations into it. Evaluation compares those writes against the scenario's
expected end state, so every row here is deterministic and hand-designed around a
scenario (the comments say which).

Run ``python -m switchboard.world.seed`` to (re)build ``data/larkspur.db``.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from switchboard import config

SCHEMA = """
CREATE TABLE plans (
    plan_id TEXT PRIMARY KEY, product TEXT NOT NULL, name TEXT NOT NULL,
    monthly_price REAL NOT NULL, data_allowance_gb REAL
);
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL,
    postcode TEXT NOT NULL, district TEXT NOT NULL
);
CREATE TABLE subscriptions (
    customer_id TEXT NOT NULL, product TEXT NOT NULL, plan_id TEXT NOT NULL,
    monthly_price REAL NOT NULL, contract_end TEXT NOT NULL
);
CREATE TABLE invoices (
    invoice_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, month TEXT NOT NULL,
    issued TEXT NOT NULL, total REAL NOT NULL, status TEXT NOT NULL
);
CREATE TABLE invoice_lines (
    invoice_id TEXT NOT NULL, description TEXT NOT NULL, amount REAL NOT NULL,
    kind TEXT NOT NULL
);
CREATE TABLE usage (
    customer_id TEXT NOT NULL, month TEXT NOT NULL, data_gb REAL NOT NULL,
    allowance_gb REAL, minutes INTEGER NOT NULL
);
CREATE TABLE outages (
    outage_id TEXT PRIMARY KEY, district TEXT NOT NULL, service TEXT NOT NULL,
    started TEXT NOT NULL, ended TEXT, status TEXT NOT NULL, eta TEXT, cause TEXT NOT NULL
);
CREATE TABLE line_tests (
    customer_id TEXT PRIMARY KEY, result TEXT NOT NULL, fault_location TEXT,
    detail TEXT NOT NULL
);
CREATE TABLE engineer_slots (slot_date TEXT PRIMARY KEY);
-- Written by the action tools. Evaluation reads these back as the end state.
CREATE TABLE credits (
    credit_id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT NOT NULL,
    amount REAL NOT NULL, reason TEXT NOT NULL
);
CREATE TABLE engineer_bookings (
    booking_id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT NOT NULL,
    slot_date TEXT NOT NULL, charge REAL NOT NULL
);
CREATE TABLE escalations (
    escalation_id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT NOT NULL,
    team TEXT NOT NULL, summary TEXT NOT NULL
);
"""

# plan_id: (product, name, monthly price, mobile data allowance in GB or None)
PLANS: dict[str, tuple[str, str, float, float | None]] = {
    "BB-80": ("broadband", "Fibre 80", 28.00, None),
    "BB-500": ("broadband", "Fibre 500", 40.00, None),
    "BB-900": ("broadband", "Full Fibre 900", 55.00, None),
    "TV-ESS": ("tv", "TV Essentials", 21.00, None),
    "TV-SPORT": ("tv", "TV Essentials + Sport", 46.00, None),
    "TV-CIN": ("tv", "TV Essentials + Cinema", 33.00, None),
    "MOB-10": ("mobile", "SIM 10GB", 10.00, 10.0),
    "MOB-50": ("mobile", "SIM 50GB", 15.00, 50.0),
    "MOB-UNL": ("mobile", "SIM Unlimited", 25.00, None),
}

# id, name, email, postcode, [(plan_id, contract_end)]
CUSTOMERS: list[tuple[str, str, str, str, list[tuple[str, str]]]] = [
    # ME1: active broadband outage today (OUT-2609).
    ("LK-1001", "Amara Okafor", "amara.okafor@example.com", "ME1 4QT",
     [("BB-500", "2027-05-31"), ("TV-ESS", "2027-05-31")]),
    ("LK-1002", "Tom Hughes", "tom.hughes@example.com", "ME1 2BX", [("BB-80", "2026-11-30")]),
    # ME4: resolved 3-full-day outage in August (OUT-2588) -> £15 compensation.
    ("LK-1003", "Priya Shah", "priya.shah@example.com", "ME4 6LP", [("BB-500", "2027-02-28")]),
    ("LK-1004", "Daniel Byrne", "daniel.byrne@example.com", "ME4 5RE",
     [("BB-900", "2027-08-31"), ("MOB-50", "2027-08-31")]),
    # CT2: resolved outage shorter than a full day (OUT-2593) -> not eligible.
    ("LK-1005", "Chloe Martin", "chloe.martin@example.com", "CT2 7NF", [("BB-80", "2026-12-31")]),
    # Network-side line faults, no area outage -> free engineer visit.
    ("LK-1006", "Kwame Mensah", "kwame.mensah@example.com", "DA1 3HS", [("BB-500", "2027-04-30")]),
    ("LK-1007", "Sofia Rossi", "sofia.rossi@example.com", "BR3 1AA",
     [("BB-80", "2027-01-31"), ("TV-SPORT", "2027-01-31")]),
    # Healthy line -> troubleshooting, no engineer.
    ("LK-1008", "Oliver Grant", "oliver.grant@example.com", "SE10 9EF", [("BB-500", "2027-06-30")]),
    ("LK-1009", "Hannah Lewis", "hannah.lewis@example.com", "TN9 2DL",
     [("BB-80", "2026-10-31"), ("MOB-UNL", "2027-03-31")]),
    ("LK-1010", "Ravi Patel", "ravi.patel@example.com", "DA1 5JQ", [("BB-900", "2027-09-30")]),
    # Billing scenarios: each has one unusual line on the September bill.
    ("LK-1011", "Emily Clarke", "emily.clarke@example.com", "SE10 8RT",
     [("BB-500", "2027-03-31"), ("TV-SPORT", "2027-03-31")]),
    ("LK-1012", "James O'Neill", "james.oneill@example.com", "TN9 1HX",
     [("BB-80", "2027-02-28"), ("MOB-10", "2027-02-28")]),
    ("LK-1013", "Grace Kim", "grace.kim@example.com", "BR3 4PL", [("BB-500", "2026-12-31")]),
    ("LK-1014", "Mohammed Ali", "mohammed.ali@example.com", "DA1 2WW", [("BB-500", "2027-07-31")]),
    ("LK-1015", "Lucy Evans", "lucy.evans@example.com", "BR3 6QS", [("BB-80", "2027-05-31")]),
    ("LK-1016", "Ben Carter", "ben.carter@example.com", "SE10 0AZ",
     [("BB-500", "2027-01-31"), ("TV-CIN", "2027-01-31")]),
    # Usage / SQL aggregation scenarios.
    ("LK-1017", "Isla Murray", "isla.murray@example.com", "TN9 3EE", [("MOB-50", "2027-04-30")]),
    ("LK-1018", "Noah Wright", "noah.wright@example.com", "CT2 8PU", [("MOB-10", "2026-12-31")]),
    ("LK-1019", "Zara Hussain", "zara.hussain@example.com", "DA1 4LN",
     [("BB-900", "2027-06-30"), ("TV-ESS", "2027-06-30"), ("MOB-UNL", "2027-06-30")]),
    # Contract / cancellation scenarios.
    ("LK-1020", "Liam Foster", "liam.foster@example.com", "TN9 5GH",
     [("BB-500", "2027-03-31"), ("TV-ESS", "2027-03-31")]),
    ("LK-1021", "Mia Thompson", "mia.thompson@example.com", "SE10 4DD", [("BB-80", "2026-06-30")]),
    # Home-wiring fault -> engineer visit is chargeable (£65), must be disclosed.
    ("LK-1022", "Ethan Hall", "ethan.hall@example.com", "BR3 2JJ", [("BB-80", "2027-02-28")]),
    # Adversarial scenarios (prompt injection, cross-account requests).
    ("LK-1023", "Ava Robinson", "ava.robinson@example.com", "DA1 6XY", [("BB-500", "2027-01-31")]),
    ("LK-1024", "Jack Turner", "jack.turner@example.com", "ME4 3NB", [("BB-500", "2027-04-30")]),
]

MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09"]

# Monthly mobile data use (GB), June..September (September is month-to-date).
USAGE: dict[str, list[float]] = {
    "LK-1004": [22.4, 27.9, 25.1, 9.8],
    "LK-1009": [61.0, 58.3, 70.2, 24.6],
    "LK-1012": [8.1, 9.4, 12.6, 4.2],   # over the 10 GB allowance in August
    "LK-1017": [31.2, 44.8, 38.5, 12.1],  # peak month is July
    "LK-1018": [9.9, 11.2, 10.4, 3.3],  # over in July and August
    "LK-1019": [15.5, 18.0, 16.7, 6.0],
}
OUT_OF_BUNDLE_PER_GB = 4.00

# One-off lines: (customer, month) -> [(description, amount, kind)]
EXTRAS: dict[tuple[str, str], list[tuple[str, float, str]]] = {
    ("LK-1011", "2026-09"): [("Pay-per-view: Championship Boxing (29 Aug)", 24.95, "one_off")],
    ("LK-1013", "2026-09"): [("Late payment fee (August invoice)", 5.00, "fee")],
    ("LK-1015", "2026-09"): [("Engineer visit: home wiring fault (3 Sep)", 65.00, "one_off")],
    ("LK-1016", "2026-09"): [("Movie rentals x3", 13.47, "one_off")],
}
# A 12-month loyalty discount that ended on 31 August: the September bill is £8 higher.
for _m in ("2026-06", "2026-07", "2026-08"):
    EXTRAS[("LK-1014", _m)] = [("Loyalty discount (12 months, ends 31 Aug 2026)", -8.00, "discount")]

OUTAGES = [
    ("OUT-2609", "ME1", "broadband", "2026-09-14 07:40", None, "active", "2026-09-14 18:00",
     "Damaged fibre cable (third-party roadworks)"),
    ("OUT-2588", "ME4", "broadband", "2026-08-10 06:15", "2026-08-13 09:30", "resolved", None,
     "Exchange power failure"),
    ("OUT-2593", "CT2", "broadband", "2026-08-21 10:00", "2026-08-22 08:00", "resolved", None,
     "Planned maintenance overrun"),
    ("OUT-2571", "TN9", "tv", "2026-07-02 19:00", "2026-07-02 23:00", "resolved", None,
     "TV guide data outage"),
]

LINE_FAULTS: dict[str, tuple[str, str]] = {
    "LK-1006": ("network", "No sync: the line is not connecting to the exchange."),
    "LK-1007": ("network", "High noise on the line; speed dropping below the guaranteed minimum."),
    "LK-1010": ("network", "Intermittent drops detected (14 in the last 24 hours)."),
    "LK-1015": ("home_wiring", "Fault located inside the property (internal wiring)."),
    "LK-1022": ("home_wiring", "Fault located inside the property (internal wiring)."),
}
ENGINEER_SLOTS = ["2026-09-16", "2026-09-17", "2026-09-18", "2026-09-21"]


def _district(postcode: str) -> str:
    return postcode.split()[0]


def build(db_path: Path | None = None) -> Path:
    """(Re)create the database at ``db_path`` and return the path."""
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO plans VALUES (?,?,?,?,?)",
            [(pid, *spec) for pid, spec in PLANS.items()],
        )
        for cid, name, email, postcode, subs in CUSTOMERS:
            conn.execute(
                "INSERT INTO customers VALUES (?,?,?,?,?)",
                (cid, name, email, postcode, _district(postcode)),
            )
            for plan_id, end in subs:
                product, _, price, _ = PLANS[plan_id]
                conn.execute(
                    "INSERT INTO subscriptions VALUES (?,?,?,?,?)",
                    (cid, product, plan_id, price, end),
                )
            _write_invoices(conn, cid, subs)
            fault = LINE_FAULTS.get(cid)
            if _district(postcode) == "ME1":
                conn.execute(
                    "INSERT INTO line_tests VALUES (?,?,?,?)",
                    (cid, "fault", "network",
                     "No sync. There is an area outage in progress (OUT-2609)."),
                )
            elif fault:
                conn.execute("INSERT INTO line_tests VALUES (?,?,?,?)", (cid, "fault", *fault))
            else:
                conn.execute(
                    "INSERT INTO line_tests VALUES (?,?,?,?)",
                    (cid, "ok", None, "Line synchronised at the expected speed. No fault found."),
                )
        for cid, gbs in USAGE.items():
            plan_id = next(p for p, _ in _subs_of(cid) if PLANS[p][0] == "mobile")
            allowance = PLANS[plan_id][3]
            for month, gb in zip(MONTHS, gbs, strict=True):
                conn.execute(
                    "INSERT INTO usage VALUES (?,?,?,?,?)",
                    (cid, month, gb, allowance, int(gb * 9) % 400 + 50),
                )
        conn.executemany("INSERT INTO outages VALUES (?,?,?,?,?,?,?,?)", OUTAGES)
        conn.executemany("INSERT INTO engineer_slots VALUES (?)", [(d,) for d in ENGINEER_SLOTS])
        conn.commit()
    finally:
        conn.close()
    return path


def _subs_of(cid: str) -> list[tuple[str, str]]:
    return next(subs for c, *_, subs in CUSTOMERS if c == cid)


def _write_invoices(conn: sqlite3.Connection, cid: str, subs: list[tuple[str, str]]) -> None:
    for i, month in enumerate(MONTHS):
        lines: list[tuple[str, float, str]] = []
        for plan_id, _ in subs:
            product, name, price, _ = PLANS[plan_id]
            lines.append((f"{name} ({product}) monthly charge", price, "plan"))
        lines.extend(EXTRAS.get((cid, month), []))
        # Out-of-bundle data used last month is billed on this month's invoice.
        if i > 0 and cid in USAGE:
            plan_id = next(p for p, _ in subs if PLANS[p][0] == "mobile")
            allowance = PLANS[plan_id][3]
            used = USAGE[cid][i - 1]
            if allowance is not None and used > allowance:
                over = round(used - allowance, 1)
                lines.append((
                    f"Out-of-bundle mobile data, {MONTHS[i - 1]} ({over} GB over allowance)",
                    round(over * OUT_OF_BUNDLE_PER_GB, 2), "usage",
                ))
        total = round(sum(amount for _, amount, _ in lines), 2)
        status = "due" if month == "2026-09" else "paid"
        if cid == "LK-1013" and month == "2026-08":
            status = "paid_late"
        inv_id = f"INV-{cid[3:]}-{month.replace('-', '')}"
        conn.execute(
            "INSERT INTO invoices VALUES (?,?,?,?,?,?)",
            (inv_id, cid, month, f"{month}-01", total, status),
        )
        conn.executemany(
            "INSERT INTO invoice_lines VALUES (?,?,?,?)",
            [(inv_id, desc, amount, kind) for desc, amount, kind in lines],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=config.DB_PATH)
    args = parser.parse_args()
    path = build(args.out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
