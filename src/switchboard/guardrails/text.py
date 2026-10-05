"""Text guardrails: prompt-injection detection, PII redaction and the grounding check.

These are deliberately simple, deterministic and explainable. They are a first
layer, not a complete defence: a regex will not catch every paraphrased injection,
and the evaluation reports how often they fire and what gets past them.

- ``detect_injection`` flags instruction-like text aimed at the assistant, in the
  customer's message (direct injection) or in a tool result such as a community
  help article (indirect injection).
- ``neutralise`` removes the offending sentences from a tool result before the model
  reads it.
- ``redact_pii`` masks card numbers and other secrets before they reach the model or
  the trace log.
- ``ungrounded_amounts`` lists £ amounts in the final reply that no tool returned and
  the prompt does not state: the cheap faithfulness check for hallucinated numbers.
- ``claimed_actions`` finds replies that say an action was taken ("I've booked an
  engineer"), so the loop can check a matching tool call actually succeeded.
"""

from __future__ import annotations

import re

_INJECTION_RE = re.compile(
    r"(ignore (all |any |your )?(previous|prior|above) (instructions|rules)|"
    r"disregard (all |your )?(previous |prior )?(instructions|rules|policies)|"
    r"system (note|prompt|message)\s*(to assistant)?\s*:|"
    r"you are now|developer mode|new instructions\s*:|act as (an? )?(admin|supervisor))",
    re.I,
)
# 13-19 digit card numbers, optionally spaced/dashed in groups of four.
_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_PASSWORD_RE = re.compile(r"(password|passcode)\s*(is|:)\s*\S+", re.I)
_AMOUNT_RE = re.compile(r"£\s?(\d+(?:,\d{3})*(?:\.\d{1,2})?)")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def detect_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def neutralise(text: str) -> str:
    """Drop sentences that look like instructions to the assistant."""
    kept = [s for s in _SENTENCE_RE.split(text) if not _INJECTION_RE.search(s)]
    removed = len(_SENTENCE_RE.split(text)) - len(kept)
    if removed:
        kept.append("[Removed instruction-like text from an untrusted source.]")
    return " ".join(kept)


def redact_pii(text: str) -> tuple[str, bool]:
    redacted = _CARD_RE.sub("[REDACTED CARD NUMBER]", text)
    redacted = _PASSWORD_RE.sub(r"\1 [REDACTED]", redacted)
    return redacted, redacted != text


def _norm(num: str) -> str:
    """'15', '15.0' and '15.00' are the same amount."""
    value = float(num.replace(",", ""))
    return f"{value:.2f}"


def numbers_in(text: str) -> set[str]:
    return {_norm(n) for n in _NUMBER_RE.findall(text)}


def ungrounded_amounts(answer: str, sources: list[str]) -> list[str]:
    """£ amounts in ``answer`` that appear in none of the ``sources``."""
    known: set[str] = set()
    for s in sources:
        known |= numbers_in(s)
    return [m for m in _AMOUNT_RE.findall(answer) if _norm(m) not in known]


# Past-tense claims that an action happened, keyed by the tool that must back them.
_CLAIMS: dict[str, re.Pattern[str]] = {
    "book_engineer": re.compile(
        r"(I('ve| have)|we('ve| have)) (booked|scheduled|arranged)|"
        r"(engineer|visit|appointment) (has been|is|was) (booked|scheduled|arranged)", re.I),
    "apply_credit": re.compile(
        r"(I('ve| have)|we('ve| have)) (applied|added|credited|issued)|"
        r"(credit|compensation) (has been|was|is being) (applied|added|issued)", re.I),
    "escalate": re.compile(
        r"(I('ve| have)|we('ve| have)) (escalated|passed|raised|forwarded|referred)|"
        r"(has been|was) (escalated|passed|raised|forwarded|referred)", re.I),
}


# Models often write typographic apostrophes ("I’ll"); the patterns use ASCII ones.
_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})


def claimed_actions(answer: str) -> set[str]:
    """Tools whose effect the reply claims already happened."""
    answer = answer.translate(_QUOTES)
    return {tool for tool, pattern in _CLAIMS.items() if pattern.search(answer)}


# Future-tense commitments ("I'll book an engineer"). The force-action check treats these
# like claims: a reply that commits to an action must be backed by the tool call.
_PROMISES: dict[str, re.Pattern[str]] = {
    "book_engineer": re.compile(
        r"(I'?ll|I will|we'?ll|we will|let me|let's|I'?m going to|I am going to|we'?re going to)"
        r" (now )?(go ahead and )?(book|schedule|arrange)|"
        r"(I'?m|I am|we'?re|we are) (now )?(booking|scheduling|arranging)|proceed with booking",
        re.I),
    "apply_credit": re.compile(
        r"(I'?ll|I will|we'?ll|we will|let me|I'?m going to|I am going to)"
        r" (now )?(go ahead and )?(apply|add|credit|issue)", re.I),
    "escalate": re.compile(
        r"(I'?ll|I will|we'?ll|we will|let me|I'?m going to|I am going to)"
        r" (now )?(go ahead and )?(escalate|pass|raise|forward|refer)|"
        r"(will be|is being) (escalated|passed|forwarded|referred)", re.I),
}


def promised_actions(answer: str) -> set[str]:
    """Tools the reply commits to using (future tense) or claims it already used."""
    answer = answer.translate(_QUOTES)
    return claimed_actions(answer) | {t for t, p in _PROMISES.items() if p.search(answer)}
