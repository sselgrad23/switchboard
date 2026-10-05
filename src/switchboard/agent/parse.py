"""Parse tool calls out of raw model output.

Qwen2.5's chat template asks for each call as JSON inside ``<tool_call>`` tags. A 3B
model mostly complies, but also produces prose before the call, several calls in
one turn, a missing closing tag at the end of generation, arguments serialised as a
string, or a bare JSON object with no tags at all. This parser accepts those, and
reports anything it could not read as an error so the loop can ask for a retry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


@dataclass
class ParsedCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class Parsed:
    content: str  # prose outside the tool calls
    calls: list[ParsedCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _to_call(obj: Any) -> ParsedCall | None:
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return ParsedCall(str(obj["name"]), args)


def parse(text: str) -> Parsed:
    if "<tool_call>" in text:
        out = Parsed(content=text.split("<tool_call>")[0].strip())
        for raw in _CALL_RE.findall(text):
            try:
                call = _to_call(json.loads(raw))
            except json.JSONDecodeError:
                call = None
            if call is None:
                out.errors.append(raw[:200])
            else:
                out.calls.append(call)
        return out
    # Untagged fallback: the whole reply is one JSON call object (possibly fenced).
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    if stripped.startswith("{") and '"name"' in stripped:
        try:
            call = _to_call(json.loads(stripped))
        except json.JSONDecodeError:
            call = None
        if call is not None:
            return Parsed(content="", calls=[call])
    return Parsed(content=text.strip())
