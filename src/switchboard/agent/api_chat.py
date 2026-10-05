"""Hosted models through OpenAI-compatible chat APIs (Groq, Mistral).

Both providers accept the same request shape (``messages`` + ``tools``) and return
native tool calls. ``APIChat`` converts the agent's internal message list into that
shape and turns the returned tool calls back into ``<tool_call>`` text, so the agent
loop, parser, guardrails and traces are identical for local and hosted models.

Free tiers are rate limited, so the client paces itself (requests and tokens per
minute, per provider), retries 429/5xx with backoff, and raises ``QuotaExhausted``
on a daily cap so a run can stop cleanly and resume the next day.

Reproducibility: hosted models can change behind an alias, so always request a
pinned, dated model id (``mistral-small-2603``, not ``mistral-small-latest``) and keep
the model id the API reports it actually served (``served_models``). Both end up in
the run metadata with the date.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from switchboard import config
from switchboard.agent.llm import Completion

PROVIDERS: dict[str, dict[str, Any]] = {
    # Free-tier limits as of Sep 2026 (check the provider console; they change often).
    "groq": {"base_url": "https://api.groq.com/openai/v1", "env": "GROQ_API_KEY",
             "prefix": "gsk_", "rpm": 28, "tpm": 7500},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "env": "MISTRAL_API_KEY",
                "prefix": "", "rpm": 150, "tpm": 500_000},
}
KEYS_FILE = config.ROOT_DIR / "keys.env"


class QuotaExhausted(RuntimeError):
    """The provider's daily cap is hit: stop the run and resume later."""


def load_key(provider: str, keys_file: Path = KEYS_FILE) -> str:
    """Key from the environment, else from keys.env.

    keys.env may hold ``GROQ_API_KEY=...`` / ``MISTRAL_API_KEY=...``, ``Groq=...`` /
    ``Mistral=...``, or bare key lines (a Groq key starts with ``gsk_``).
    """
    spec = PROVIDERS[provider]
    if os.getenv(spec["env"]):
        return os.environ[spec["env"]]
    if keys_file.exists():
        lines = [ln.strip() for ln in keys_file.read_text().splitlines() if ln.strip()]
        for ln in lines:
            if "=" in ln:
                name, _, value = ln.partition("=")
                # Accept GROQ_API_KEY=..., Groq=..., groq=... and the Mistral equivalents.
                if name.strip().lower() in (spec["env"].lower(), provider):
                    return value.strip().strip("'\"")
        bare = [ln for ln in lines if "=" not in ln]
        if provider == "groq":
            found = [ln for ln in bare if ln.startswith("gsk_")]
        else:
            found = [ln for ln in bare if not ln.startswith("gsk_")]
        if found:
            return found[0]
    raise RuntimeError(f"No API key for {provider}: set {spec['env']} or add it to keys.env.")


def _to_api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Internal messages -> OpenAI format, with generated tool-call ids.

    Mistral requires 9-character alphanumeric ids; Groq accepts anything, so one
    scheme serves both. Tool results are matched to calls in order.
    """
    out: list[dict[str, Any]] = []
    pending: deque[str] = deque()
    n = 0
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                n += 1
                cid = f"c{n:08d}"
                pending.append(cid)
                calls.append({"id": cid, "type": "function", "function": {
                    "name": tc["function"]["name"],
                    "arguments": json.dumps(tc["function"]["arguments"])}})
            out.append({"role": "assistant", "content": m.get("content") or "",
                        "tool_calls": calls})
        elif m["role"] == "tool":
            cid = pending.popleft() if pending else f"c{0:08d}"
            out.append({"role": "tool", "tool_call_id": cid, "name": m.get("name", ""),
                        "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class APIChat:
    """One hosted model. ``name`` is the pinned model id that was requested."""

    def __init__(self, provider: str, model: str, temperature: float = 0.0,
                 extra: dict[str, Any] | None = None, timeout: float = 120.0) -> None:
        import httpx

        self.provider = provider
        self.name = model
        self.temperature = temperature
        self.extra = extra or {}
        spec = PROVIDERS[provider]
        self._rpm, self._tpm = spec["rpm"], spec["tpm"]
        self._client = httpx.Client(
            base_url=spec["base_url"], timeout=timeout,
            headers={"Authorization": f"Bearer {load_key(provider)}"},
        )
        self._window: deque[tuple[float, int]] = deque()  # (time, tokens) in the last 60 s
        self.served_models: set[str] = set()
        self.calls = 0

    def meta(self) -> dict[str, Any]:
        return {"provider": self.provider, "model_requested": self.name,
                "models_served": sorted(self.served_models), "temperature": self.temperature,
                "extra_params": self.extra, "api_calls": self.calls}

    def _pace(self, est_tokens: int) -> None:
        while True:
            now = time.time()
            while self._window and now - self._window[0][0] > 60:
                self._window.popleft()
            used = sum(t for _, t in self._window)
            if not self._window or (
                len(self._window) < self._rpm and used + min(est_tokens, self._tpm) <= self._tpm
            ):
                return
            time.sleep(max(0.5, 60 - (now - self._window[0][0]) + 0.1))

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Completion:
        payload: dict[str, Any] = {"model": self.name, "messages": _to_api_messages(messages),
                                   "temperature": self.temperature, **self.extra}
        if tools:
            payload["tools"] = tools
        est = len(json.dumps(payload)) // 3  # rough token estimate for pacing
        for attempt in range(8):
            self._pace(est)
            resp = self._client.post("/chat/completions", json=payload)
            self._window.append((time.time(), est))
            if resp.status_code == 200:
                break
            body = resp.text[:500]
            if resp.status_code == 429 and resp.headers.get("x-ratelimit-limit-req-minute") == "0":
                raise RuntimeError(f"{self.provider}: {self.name} is not available on this "
                                   "API tier (request limit is 0 per minute).")
            if resp.status_code == 429 and ("per day" in body or "TPD" in body or "RPD" in body):
                raise QuotaExhausted(f"{self.provider} daily limit: {body}")
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = float(resp.headers.get("retry-after") or 0) or min(60, 2 ** attempt * 2)
                time.sleep(wait)
                continue
            raise RuntimeError(f"{self.provider} error {resp.status_code}: {body}")
        else:
            raise RuntimeError(f"{self.provider}: gave up after repeated rate limiting")

        data = resp.json()
        self.calls += 1
        if data.get("model"):
            self.served_models.add(data["model"])
        msg = data["choices"][0]["message"]
        text = msg.get("content") or ""
        if isinstance(text, list):  # some APIs return content chunks
            text = "".join(c.get("text", "") for c in text if isinstance(c, dict))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
                call = json.dumps({"name": fn.get("name"), "arguments": args})
            except json.JSONDecodeError:
                call = raw  # left unparseable on purpose: the loop counts it as malformed
            text += f"\n<tool_call>\n{call}\n</tool_call>"
        usage = data.get("usage") or {}
        tin, tout = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        if self._window:
            self._window[-1] = (self._window[-1][0], tin + tout)
        return Completion(text=text.strip(), tokens_in=tin, tokens_out=tout)
