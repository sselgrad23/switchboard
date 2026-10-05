"""Database access for one agent session.

Every conversation runs against its own in-memory copy of the seeded database, so
the action tools can write freely and each evaluation scenario starts from the same
state. ``side_effects`` reads back what the session wrote: that is the "end state"
the evaluation compares against the scenario's expected state.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from switchboard import config
from switchboard.world import seed

# The tables the action tools write. Everything else is read-only reference data.
ACTION_TABLES = ("credits", "engineer_bookings", "escalations")


def ensure_db(path: Path | None = None) -> Path:
    """Return the seeded database path, building it on first use."""
    db = Path(path or config.DB_PATH)
    if not db.exists():
        seed.build(db)
    return db


def session_db(path: Path | None = None) -> sqlite3.Connection:
    """A fresh in-memory copy of the seeded database for one conversation."""
    src = sqlite3.connect(ensure_db(path))
    mem = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        src.backup(mem)
    finally:
        src.close()
    mem.row_factory = sqlite3.Row
    return mem


def rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Run a query and return plain dicts (JSON-serialisable tool output)."""
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def side_effects(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    """Everything the session wrote through the action tools."""
    return {t: rows(conn, f"SELECT * FROM {t}") for t in ACTION_TABLES}  # noqa: S608 - fixed names
