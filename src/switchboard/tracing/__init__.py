"""Structured traces: one ``Trace`` per conversation, one ``Span`` per step.

Every model call, tool call and guardrail decision becomes a span with its inputs,
outputs, status and timing. Traces are plain JSON (one line per conversation in
``reports/traces/*.jsonl``), so they can be analysed with pandas or loaded into
any tracing backend. The same fields an OpenTelemetry/LangSmith span would carry
(kind, name, start, duration, attributes, status) are used, without the dependency.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    kind: str  # "llm" | "tool" | "guard"
    name: str
    start_ms: float
    duration_ms: float = 0.0
    status: str = "ok"  # ok | error | blocked | flagged | invalid
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    customer_id: str
    message: str
    config: dict[str, Any]
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    spans: list[Span] = field(default_factory=list)
    answer: str = ""
    duration_ms: float = 0.0
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def _now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    @contextmanager
    def span(self, kind: str, name: str, **attributes: Any) -> Iterator[Span]:
        s = Span(kind=kind, name=name, start_ms=round(self._now_ms(), 1), attributes=attributes)
        t = time.perf_counter()
        try:
            yield s
        except Exception as err:
            s.status = "error"
            s.attributes["error"] = repr(err)
            raise
        finally:
            s.duration_ms = round((time.perf_counter() - t) * 1000, 1)
            self.spans.append(s)

    def event(self, kind: str, name: str, status: str, **attributes: Any) -> None:
        """A zero-duration span (guardrail decisions, parse failures)."""
        self.spans.append(Span(kind, name, round(self._now_ms(), 1), 0.0, status, attributes))

    def finish(self, answer: str) -> None:
        self.answer = answer
        self.duration_ms = round(self._now_ms(), 1)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
