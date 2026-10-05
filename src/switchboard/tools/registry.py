"""Tool specs, the per-conversation session, and JSON-schema export.

A tool is a typed function: a Pydantic model for its arguments (which doubles as
the JSON schema the LLM sees), a plain Python implementation, and a flag for
whether it changes state. Argument validation happens here, before any policy check
or execution, so a malformed call from the model becomes a readable error message
the model can recover from rather than an exception.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from switchboard.retrieval import Retriever


@dataclass
class Session:
    """State for one conversation: who is signed in, their DB copy, and evidence.

    ``evidence`` records what the tools have actually returned in this conversation
    (outages checked, line tests run, credits applied). The guardrail policy reads it,
    so an action is only allowed when the facts that justify it were looked up first.
    """

    customer_id: str
    conn: sqlite3.Connection
    retriever: Retriever
    today: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Session, Any], dict[str, Any]]
    side_effect: bool = False

    def json_schema(self) -> dict[str, Any]:
        """OpenAI/Qwen-style function schema, without Pydantic's title noise."""
        schema = self.args_model.model_json_schema()
        props = {
            k: {kk: vv for kk, vv in v.items() if kk != "title"}
            for k, v in schema.get("properties", {}).items()
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": schema.get("required", []),
                },
            },
        }

    def parse_args(self, raw: dict[str, Any]) -> BaseModel:
        return self.args_model.model_validate(raw)

    def run(self, session: Session, args: BaseModel) -> dict[str, Any]:
        return self.fn(session, args)


def validation_message(err: ValidationError) -> str:
    """A short, model-readable summary of what was wrong with the arguments."""
    parts = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "arguments"
        parts.append(f"{loc}: {e['msg']}")
    return "Invalid arguments. " + "; ".join(parts)
