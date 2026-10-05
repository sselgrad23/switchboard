"""Build the evaluation report from every saved run.

    python -m switchboard.eval.report

Reads ``reports/runs/*.jsonl`` (plus ``*.meta.json``) and writes
``reports/agent_eval.md`` / ``.json``: success with 95% bootstrap intervals by split
and segment, safety, efficiency, failure taxonomy, the model/date of every run, and
paired McNemar tests for each experiment (on the scenarios both runs share).

Splits: **v1** = the original 45 scenarios; **v1-held-out** = v1 minus the 9 used to
debug the prompt; **v2** = 15 scenarios written on 24 Sep 2026 before any run that
used the second-round techniques. v2 is the fairest test of those techniques,
because they were designed after seeing v1 results.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import numpy as np

from switchboard import config
from switchboard.eval.metrics import mcnemar_exact, percentile, rate_ci
from switchboard.eval.scenarios import SEGMENTS

# Runs made before the apostrophe and hybrid-parser fixes are in runs/archive-before-fixes/
# and are not read here (the glob is not recursive).
DEV_SLICE = frozenset({"S01", "S03", "S09", "S13", "S18", "S20", "S28", "S39", "S41"})

# Display order and a readable label for each known run.
RUNS: dict[str, str] = {
    "workflow": "Workflow (rules, no LLM)",
    "llm": "Qwen2.5-3B local",
    "qwen3b-fewshot": "Qwen2.5-3B + worked examples (pre SQL fix, re-run pending)",
    "qwen3b-force": "Qwen2.5-3B + force-action",
    "qwen3b-hybrid": "Qwen2.5-3B hybrid",
    "qwen7b": "Qwen2.5-7B local (4-bit)",
    "m3b": "Ministral-3B",
    "m3b-fewshot": "Ministral-3B + worked examples",
    "m3b-force": "Ministral-3B + force-action",
    "m3b-hybrid": "Ministral-3B hybrid",
    "m8b": "Ministral-8B",
    "m14b": "Ministral-14B",
    "m14b-multiturn": "Ministral-14B multi-turn",
    "groq-qwen27b": "Qwen3.8-27B (Groq)",
    "groq-gptoss120b": "GPT-OSS-120B (Groq)",
    "llm_noguard.adversarial": "Qwen2.5-3B, guardrails OFF (adversarial only)",
}

# (A, B, question) pairs for paired tests.
PAIRS = [
    ("workflow", "llm", "rules vs 3B agent"),
    ("llm", "qwen7b", "size 3B -> 7B (Qwen2.5, local)"),
    ("workflow", "qwen7b", "rules vs 7B agent"),
    ("qwen7b", "qwen3b-hybrid", "7B agent vs 3B hybrid"),
    ("llm", "qwen3b-fewshot", "worked examples (Qwen-3B)"),
    ("llm", "qwen3b-force", "force-action (Qwen-3B)"),
    ("llm", "qwen3b-hybrid", "hybrid vs agent (Qwen-3B)"),
    ("workflow", "qwen3b-hybrid", "hybrid vs rules (Qwen-3B)"),
    ("m3b", "m3b-fewshot", "worked examples (Ministral-3B)"),
    ("m3b", "m3b-force", "force-action (Ministral-3B)"),
    ("m3b", "m3b-hybrid", "hybrid vs agent (Ministral-3B)"),
    ("workflow", "m3b-hybrid", "hybrid vs rules (Ministral-3B)"),
    ("m3b", "m14b", "size 3B -> 14B (Ministral)"),
    ("m3b", "m8b", "size 3B -> 8B (Ministral)"),
    ("workflow", "m14b", "rules vs 14B agent"),
    ("llm", "m3b", "Qwen2.5-3B vs Ministral-3B (same size)"),
    ("workflow", "groq-gptoss120b", "rules vs 120B agent"),
    ("workflow", "groq-qwen27b", "rules vs 27B agent"),
    ("m14b", "m14b-multiturn", "single vs multi-turn (Ministral-14B)"),
]


_REPEAT_RE = re.compile(r"^(?P<base>.+)\.r(?P<k>\d+)$")


def load_runs(include_repeats: bool = False) -> dict[str, list[dict[str, Any]]]:
    """Main runs (repeat 1 of each configuration), or every file with repeats."""
    runs = {}
    for path in sorted((config.REPORT_DIR / "runs").glob("*.jsonl")):
        name = path.stem
        if name.endswith(".partial"):
            continue
        if _REPEAT_RE.match(name) and not include_repeats:
            continue
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        for r in rows:
            r.setdefault("split", "v2" if r["scenario_id"].startswith("V") else "v1")
        runs[name] = sorted(rows, key=lambda r: r["scenario_id"])
    order = list(RUNS)
    return dict(sorted(runs.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))


def load_meta(name: str) -> dict[str, Any]:
    path = config.REPORT_DIR / "runs" / f"{name}.meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _ci(values: list[Any]) -> dict[str, float]:
    mean, lo, hi = rate_ci(values)
    return {"rate": mean, "lo": lo, "hi": hi, "n": len(values)}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    v1 = [r for r in rows if r["split"] == "v1"]
    return {
        "n": len(rows),
        "all": _ci([r["success"] for r in rows]),
        "v1": _ci([r["success"] for r in v1]),
        "v1_held_out": _ci([r["success"] for r in v1 if r["scenario_id"] not in DEV_SLICE]),
        "v2": _ci([r["success"] for r in rows if r["split"] == "v2"]),
        "by_segment": {s: _ci([r["success"] for r in rows if r["segment"] == s])
                       for s in SEGMENTS},
        "state_accuracy": _ci([r["state_ok"] for r in rows]),
        "fact_accuracy": _ci([r["facts_ok"] for r in rows]),
        "tool_recall": sum(r["tool_recall"] for r in rows) / max(1, len(rows)),
        "unsafe_scenarios": sum(r["unsafe"] for r in rows),
        "unsafe_rules": dict(Counter(rule for r in rows for rule in r["unsafe_rules"])),
        "blocked_calls": sum(r["blocked_calls"] for r in rows),
        "blocked_rules": dict(Counter(rule for r in rows for rule in r["blocked_rules"])),
        "ungrounded_answers": sum(1 for r in rows if r["ungrounded_amounts"]),
        "malformed_calls": sum(r["malformed_calls"] for r in rows),
        "step_limit_hits": sum(r["hit_step_limit"] for r in rows),
        "tool_calls_mean": sum(len(r["tool_calls"]) for r in rows) / max(1, len(rows)),
        "llm_calls_mean": sum(r["steps"] for r in rows) / max(1, len(rows)),
        "customer_turns_mean": sum(r.get("customer_turns", 1) for r in rows) / max(1, len(rows)),
        "latency_s_p50": percentile([r["latency_ms"] / 1000 for r in rows], 50),
        "latency_s_p95": percentile([r["latency_ms"] / 1000 for r in rows], 95),
        "tokens_in_mean": sum(r["tokens_in"] for r in rows) / max(1, len(rows)),
        "tokens_out_mean": sum(r["tokens_out"] for r in rows) / max(1, len(rows)),
        "failures": dict(Counter(r["primary_failure"] for r in rows if not r["success"])),
        "failed_ids": [r["scenario_id"] for r in rows if not r["success"]],
    }


def _fmt(ci: dict[str, float]) -> str:
    if not ci["n"]:
        return "-"
    k = round(ci["rate"] * ci["n"])
    return f"{k}/{ci['n']} = {ci['rate']:.2f} [{ci['lo']:.2f}, {ci['hi']:.2f}]"


def paired(runs: dict[str, list[dict[str, Any]]], a: str, b: str) -> dict[str, Any] | None:
    if a not in runs or b not in runs:
        return None
    ra = {r["scenario_id"]: r["success"] for r in runs[a]}
    rb = {r["scenario_id"]: r["success"] for r in runs[b]}
    ids = sorted(set(ra) & set(rb))
    if not ids:
        return None
    res = mcnemar_exact([ra[i] for i in ids], [rb[i] for i in ids])
    return {"a": a, "b": b, "n": len(ids), "a_rate": sum(ra[i] for i in ids) / len(ids),
            "b_rate": sum(rb[i] for i in ids) / len(ids), **res}


def group_repeats(all_runs: dict[str, list[dict[str, Any]]]) -> dict[str, list[list[dict[str, Any]]]]:
    """Base name -> list of runs (repeat 1 is the file without a suffix)."""
    groups: dict[str, list[list[dict[str, Any]]]] = {}
    for name, rows in all_runs.items():
        m = _REPEAT_RE.match(name)
        groups.setdefault(m.group("base") if m else name, []).append(rows)
    return groups


def _scenario_means(reps: list[list[dict[str, Any]]]) -> dict[str, float]:
    ids = set.intersection(*[{r["scenario_id"] for r in rows} for rows in reps])
    return {i: float(np.mean([next(r["success"] for r in rows if r["scenario_id"] == i)
                              for rows in reps])) for i in sorted(ids)}


def repeat_summary(reps: list[list[dict[str, Any]]]) -> dict[str, Any]:
    totals = [sum(r["success"] for r in rows) for rows in reps]
    means = _scenario_means(reps)
    flips = sum(1 for v in means.values() if 0 < v < 1)
    v2 = [v for i, v in means.items() if i.startswith("V")]
    return {"repeats": len(reps), "n": len(means), "totals": totals,
            "mean_rate": float(np.mean(list(means.values()))),
            "v2_mean_rate": float(np.mean(v2)) if v2 else float("nan"),
            "flip_rate": flips / max(1, len(means)), "flipped": flips}


def paired_means(a: list[list[dict[str, Any]]], b: list[list[dict[str, Any]]],
                 n_boot: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """Difference in success (B - A) on shared scenarios, averaging over repeats.

    Paired bootstrap CI over scenarios, and a two-sided sign-flip randomization test
    (the repeat-aware analogue of McNemar's test).
    """
    ma, mb = _scenario_means(a), _scenario_means(b)
    ids = sorted(set(ma) & set(mb))
    d = np.array([mb[i] - ma[i] for i in ids])
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    signs = rng.choice([-1.0, 1.0], size=(n_boot * 2, d.size))
    null = np.abs((signs * d).mean(axis=1))
    p = float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (null.size + 1))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"n": len(ids), "a_rate": float(np.mean([ma[i] for i in ids])),
            "b_rate": float(np.mean([mb[i] for i in ids])), "diff": float(d.mean()),
            "lo": float(lo), "hi": float(hi), "p_value": p,
            "a_repeats": len(a), "b_repeats": len(b)}


def _model_line(meta: dict[str, Any]) -> tuple[str, str, str]:
    sessions = meta.get("sessions") or [{}]
    m = sessions[-1].get("model") or {}
    served = ", ".join(m.get("models_served") or []) or m.get("hf_revision", "")[:12]
    dates = sorted({(s.get("started_utc") or "")[:10] for s in sessions if s.get("started_utc")})
    status = sessions[-1].get("status", "?")
    return (m.get("model_requested", "-") or "-", served or "-",
            f"{', '.join(dates)} ({status})")


def to_markdown(runs: dict[str, list[dict[str, Any]]], summary: dict[str, Any],
                tests: list[dict[str, Any]], repeat_rows: dict[str, Any] | None = None,
                mean_tests: list[dict[str, Any]] | None = None) -> str:
    label = {n: RUNS.get(n, n) for n in summary}
    out = ["# Agent evaluation", "",
           "Success = end state matches, required facts present, nothing forbidden, nothing "
           "unsafe. 95% percentile-bootstrap intervals over scenarios. **v1** = original 45; "
           "**v1 held-out** = v1 without the 9 prompt-debugging scenarios; **v2** = 15 "
           "scenarios written before any second-round run (the fairest test of the "
           "second-round techniques). Hosted-model rows in the first tables are repeat 1; "
           "see *Repeat runs* for averages over repeats.", "",
           "## Success", "",
           "| Run | All | v1 (45) | v1 held-out (36) | v2 (15) | Core | Long-tail | Adversarial |",
           "|---|---|---|---|---|---|---|---|"]
    for n, s in summary.items():
        seg = s["by_segment"]
        out.append(f"| {label[n]} | {_fmt(s['all'])} | {_fmt(s['v1'])} | "
                   f"{_fmt(s['v1_held_out'])} | {_fmt(s['v2'])} | {_fmt(seg['core'])} | "
                   f"{_fmt(seg['long_tail'])} | {_fmt(seg['adversarial'])} |")
    out += ["", "## Paired comparisons (McNemar exact, on shared scenarios)", "",
            "| Question | A | B | n | A rate | B rate | only A | only B | p |",
            "|---|---|---|---|---|---|---|---|---|"]
    for t in tests:
        out.append(f"| {t['question']} | {label.get(t['a'], t['a'])} | {label.get(t['b'], t['b'])} "
                   f"| {t['n']} | {t['a_rate']:.2f} | {t['b_rate']:.2f} | {t['a_only']} | "
                   f"{t['b_only']} | {t['p_value']:.3f} |")
    if repeat_rows:
        out += ["", "## Repeat runs (hosted models are not deterministic at temperature 0)", "",
                "Each hosted configuration was run several times on identical inputs. "
                "Flip rate = share of scenarios whose pass/fail differed between repeats.", "",
                "| Run | Repeats | Successes per repeat | Mean rate | v2 mean rate | Flip rate |",
                "|---|---|---|---|---|---|"]
        for n, r in repeat_rows.items():
            out.append(f"| {RUNS.get(n, n)} | {r['repeats']} | "
                       f"{', '.join(str(t) for t in r['totals'])} of {r['n']} | "
                       f"{r['mean_rate']:.2f} | {r['v2_mean_rate']:.2f} | "
                       f"{r['flip_rate']:.2f} ({r['flipped']}) |")
    if mean_tests:
        out += ["", "## Paired comparisons using all repeats", "",
                "Per-scenario success averaged over repeats; B - A with a paired bootstrap 95% "
                "interval and a sign-flip randomization test. Single-run (local) "
                "configurations count as one repeat.", "",
                "| Question | A (repeats) | B (repeats) | n | A | B | B - A [95% CI] | p |",
                "|---|---|---|---|---|---|---|---|"]
        for t in mean_tests:
            out.append(f"| {t['question']} | {RUNS.get(t['a'], t['a'])} ({t['a_repeats']}) | "
                       f"{RUNS.get(t['b'], t['b'])} ({t['b_repeats']}) | {t['n']} | "
                       f"{t['a_rate']:.2f} | {t['b_rate']:.2f} | {t['diff']:+.2f} "
                       f"[{t['lo']:+.2f}, {t['hi']:+.2f}] | {t['p_value']:.3f} |")
    out += ["", "## Safety and reliability", "",
            "| Run | Unsafe scenarios | Unsafe rules | Blocked calls | Ungrounded £ | "
            "Malformed calls | Step-limit hits |", "|---|---|---|---|---|---|---|"]
    for n, s in summary.items():
        out.append(f"| {label[n]} | {s['unsafe_scenarios']} | {s['unsafe_rules'] or '-'} | "
                   f"{s['blocked_calls']} {s['blocked_rules'] or ''} | "
                   f"{s['ungrounded_answers']} | {s['malformed_calls']} | {s['step_limit_hits']} |")
    out += ["", "## Efficiency", "",
            "| Run | State acc | Fact acc | Tool recall | Tool calls | LLM calls | Customer turns "
            "| Tokens in | Tokens out | Latency p50 / p95 (s) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in summary.items():
        out.append(f"| {label[n]} | {s['state_accuracy']['rate']:.2f} | "
                   f"{s['fact_accuracy']['rate']:.2f} | {s['tool_recall']:.2f} | "
                   f"{s['tool_calls_mean']:.1f} | {s['llm_calls_mean']:.1f} | "
                   f"{s['customer_turns_mean']:.1f} | {s['tokens_in_mean']:.0f} | "
                   f"{s['tokens_out_mean']:.0f} | {s['latency_s_p50']:.1f} / "
                   f"{s['latency_s_p95']:.1f} |")
    out += ["", "## Failure taxonomy (primary reason per failed scenario)", ""]
    for n, s in summary.items():
        out.append(f"- **{label[n]}**: {s['failures'] or 'none'}; failed: "
                   f"{', '.join(s['failed_ids']) or '-'}")
    out += ["", "## Models and dates", "",
            "| Run | Model requested | Served / revision | Run dates (status) |",
            "|---|---|---|---|"]
    for n in summary:
        req, served, dates = _model_line(load_meta(n))
        out.append(f"| {label[n]} | {req} | {served} | {dates} |")
    return "\n".join(out) + "\n"


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("No runs found. Run `python -m switchboard.eval.run_suite` first.")
    summary = {name: summarise(rows) for name, rows in runs.items()}
    tests = []
    for a, b, question in PAIRS:
        res = paired(runs, a, b)
        if res:
            tests.append({"question": question, **res})
    groups = group_repeats(load_runs(include_repeats=True))
    repeat_rows = {n: repeat_summary(g) for n, g in groups.items() if len(g) > 1}
    mean_tests = []
    for a, b, question in PAIRS:
        if a in groups and b in groups and (len(groups[a]) > 1 or len(groups[b]) > 1):
            mean_tests.append({"question": question, "a": a, "b": b,
                               **paired_means(groups[a], groups[b])})
    out = config.REPORT_DIR
    (out / "agent_eval.json").write_text(json.dumps(
        {"summary": summary, "tests": tests, "repeats": repeat_rows,
         "repeat_tests": mean_tests}, indent=2))
    md = to_markdown(runs, summary, tests, repeat_rows, mean_tests)
    (out / "agent_eval.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
