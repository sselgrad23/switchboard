"""Run the scenario suite under one configuration and save results, traces and metadata.

Named presets (the original experiments):

    python -m switchboard.eval.run_suite --config workflow
    python -m switchboard.eval.run_suite --config llm            # local Qwen2.5-3B, guardrails on
    python -m switchboard.eval.run_suite --config llm_noguard    # guardrails off (audit mode)

Or build a configuration from flags:

    python -m switchboard.eval.run_suite --name m14b --agent llm \\
        --backend mistral --model ministral-14b-2512
    python -m switchboard.eval.run_suite --name qwen3b-fewshot --agent llm --prompt fewshot
    python -m switchboard.eval.run_suite --name qwen3b-hybrid --agent hybrid
    python -m switchboard.eval.run_suite --name m14b-multiturn --backend mistral \\
        --model ministral-14b-2512 --multiturn --sim-model ministral-14b-2512

Writes ``reports/runs/<name>.jsonl`` (one scored row per scenario),
``reports/traces/<name>.jsonl`` (one full trace per scenario) and
``reports/runs/<name>.meta.json`` (model ids requested and served, provider, dates,
code hash, library versions). ``--resume`` skips scenarios already in the run file,
which is how a run continues after a provider's daily quota stops it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from switchboard import config
from switchboard.eval.scenarios import load_scenarios, score
from switchboard.retrieval import build_retriever
from switchboard.tracing import append_jsonl
from switchboard.world.db import ensure_db, session_db

PRESETS: dict[str, dict[str, Any]] = {
    "workflow": {"agent": "workflow"},
    "llm": {"agent": "llm", "backend": "qwen-local", "guardrails": True},
    "llm_noguard": {"agent": "llm", "backend": "qwen-local", "guardrails": False},
    "llm_routed": {"agent": "llm", "backend": "qwen-local", "exposure": "routed"},
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_hash() -> str:
    """Hash of the code, scenarios and help centre: identifies exactly what was run."""
    h = hashlib.sha256()
    pkg = Path(__file__).resolve().parents[1]
    for p in sorted(pkg.rglob("*")):
        if p.suffix in {".py", ".json", ".jsonl"} and "__pycache__" not in p.parts:
            h.update(p.relative_to(pkg).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _local_model_meta(model_name: str) -> dict[str, Any]:
    meta: dict[str, Any] = {"provider": "local", "model_requested": model_name,
                            "load_4bit": config.LLM_LOAD_4BIT,
                            "dtype": "bitsandbytes nf4, fp16 compute" if config.LLM_LOAD_4BIT
                            else "float16"}
    ref = (Path.home() / ".cache/huggingface/hub" / f"models--{model_name.replace('/', '--')}"
           / "refs" / "main")
    if ref.exists():
        meta["hf_revision"] = ref.read_text().strip()
    try:
        import torch
        import transformers

        meta.update(torch=torch.__version__, transformers=transformers.__version__,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:  # noqa: BLE001, S110 - metadata is best effort
        pass
    return meta


def build_chat(backend: str, model: str | None, extra: dict[str, Any] | None):  # noqa: ANN201
    if backend == "qwen-local":
        from switchboard.agent import _qwen
        from switchboard.agent.llm import QwenChat

        if model and model != config.LLM_MODEL:
            return QwenChat(model)
        return _qwen()
    from switchboard.agent.api_chat import APIChat

    if not model:
        raise SystemExit(f"--model is required for backend {backend!r}")
    return APIChat(backend, model, extra=extra)


def build(opts: dict[str, Any], retriever):  # noqa: ANN001, ANN201
    agent_kind = opts.get("agent", "llm")
    if agent_kind == "workflow":
        from switchboard.agent import WorkflowAgent

        return WorkflowAgent(retriever), None
    chat = build_chat(opts.get("backend", "qwen-local"), opts.get("model"), opts.get("extra"))
    if agent_kind == "hybrid":
        from switchboard.agent.hybrid import HybridAgent

        return HybridAgent(chat, retriever), chat
    from switchboard.agent.loop import Agent

    router = None
    if opts.get("exposure") == "routed":
        from switchboard.router import load_router

        router = load_router()
        if router is None:
            raise SystemExit("Routed exposure needs a trained router at models/router.joblib (not included).")
    agent = Agent(chat, retriever, guardrails_on=opts.get("guardrails", True),
                  exposure=opts.get("exposure", "all"), router=router,
                  prompt_variant=opts.get("prompt", "base"),
                  force_action=opts.get("force_action", False))
    return agent, chat


def run(name: str, opts: dict[str, Any], only: set[str] | None = None,
        split: str | None = None, resume: bool = False) -> None:
    from switchboard.agent.api_chat import QuotaExhausted

    config.ensure_dirs()
    ensure_db()
    retriever = build_retriever(opts.get("retriever"))
    agent, chat = build(opts, retriever)
    sim_chat = None
    if opts.get("multiturn"):
        sim_chat = build_chat(opts.get("sim_backend", "mistral"), opts.get("sim_model"), None)

    # Partial (debug) runs go to a separate file so they never mix with full runs.
    stem = f"{name}.partial" if only else name
    runs_path = config.REPORT_DIR / "runs" / f"{stem}.jsonl"
    traces_path = config.REPORT_DIR / "traces" / f"{stem}.jsonl"
    meta_path = config.REPORT_DIR / "runs" / f"{stem}.meta.json"
    done: set[str] = set()
    if resume and runs_path.exists():
        done = {json.loads(line)["scenario_id"] for line in runs_path.open() if line.strip()}
    else:
        for p in (runs_path, traces_path, meta_path):
            p.unlink(missing_ok=True)
    meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "name": name, "options": opts, "agent_config": agent.config_dict(), "sessions": []}
    session_meta: dict[str, Any] = {"started_utc": _utc(), "code_sha256_16": code_hash(),
                                    "python": platform.python_version(),
                                    "today_in_world": config.TODAY, "status": "running"}
    meta["sessions"].append(session_meta)

    def model_meta() -> dict[str, Any] | None:
        if chat is None:
            return None
        return chat.meta() if hasattr(chat, "meta") else _local_model_meta(chat.name)

    def save_meta(status: str) -> None:
        session_meta.update(status=status, finished_utc=_utc(), model=model_meta(),
                            simulator=sim_chat.meta() if sim_chat is not None and
                            hasattr(sim_chat, "meta") else None)
        meta_path.write_text(json.dumps(meta, indent=2))

    scenarios = [s for s in load_scenarios()
                 if (not only or s.id in only) and (not split or s.split == split)
                 and s.id not in done]
    t0 = t_prev = time.perf_counter()
    recent: list[float] = []
    try:
        for i, scn in enumerate(scenarios, 1):
            user_sim = None
            if sim_chat is not None:
                from switchboard.eval.user_sim import SimulatedCustomer

                user_sim = SimulatedCustomer(sim_chat, scn)
            if user_sim is not None:
                result = agent.run(scn.customer_id, scn.message, conn=session_db(),
                                   user_sim=user_sim)
            else:
                result = agent.run(scn.customer_id, scn.message, conn=session_db())
            row = score(scn, result)
            mm = model_meta() or {}
            row.update({
                "config": name,
                "answer": result.answer,
                "tool_calls": [
                    {"name": c.name, "arguments": c.arguments, "status": c.status,
                     "rule": c.rule, "would_block": c.would_block}
                    for c in result.tool_calls
                ],
                "side_effects": {k: v for k, v in result.side_effects.items() if v},
                "steps": result.steps,
                "malformed_calls": result.malformed_calls,
                "hit_step_limit": result.hit_step_limit,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "latency_ms": result.latency_ms,
                "n_tools_exposed": len(result.tools_exposed),
                "route": result.route,
                "trace_id": result.trace["trace_id"],
                "retriever": retriever.name,
                "model_requested": mm.get("model_requested"),
                "models_served": mm.get("models_served"),
                "run_utc": _utc(),
                "customer_messages": result.flags.get("customer_messages"),
                "requests": result.flags.get("requests"),
            })
            append_jsonl(runs_path, row)
            append_jsonl(traces_path, {"scenario_id": scn.id, **result.trace})
            mark = "PASS" if row["success"] else f"FAIL ({row['primary_failure']})"
            # Rolling average over the last 10 scenarios, so a slowdown shows up quickly
            # in both the average and the ETA. Counts include scenarios done before --resume.
            recent.append(time.perf_counter() - t_prev)
            t_prev = time.perf_counter()
            avg = sum(recent[-10:]) / len(recent[-10:])
            eta_min = avg * (len(scenarios) - i) / 60
            print(f"[{name}] {len(done) + i:>2}/{len(done) + len(scenarios)} {scn.id} {mark} "
                  f"{result.latency_ms / 1000:.1f}s  avg {avg:.0f}s  ETA {eta_min:.0f} min  "
                  f"tools={[c['name'] for c in row['tool_calls']]}",
                  flush=True)
    except QuotaExhausted as err:
        save_meta("stopped_quota")
        print(f"[{name}] stopped: {err}. Re-run the same command with --resume later.")
        return
    except BaseException:
        save_meta("crashed")
        raise
    save_meta("complete")
    print(f"Done in {time.perf_counter() - t0:.0f}s -> {runs_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=sorted(PRESETS), help="A named preset.")
    parser.add_argument("--name", help="Output name for a flag-built configuration.")
    parser.add_argument("--agent", choices=["llm", "workflow", "hybrid"])
    parser.add_argument("--backend", choices=["qwen-local", "groq", "mistral"])
    parser.add_argument("--model", help="Model id; pin a dated version for hosted models.")
    parser.add_argument("--extra", help='Extra API params as JSON, e.g. \'{"reasoning_effort": "low"}\'')
    parser.add_argument("--prompt", choices=["base", "fewshot"])
    parser.add_argument("--force-action", action="store_true")
    parser.add_argument("--no-guardrails", action="store_true")
    parser.add_argument("--multiturn", action="store_true")
    parser.add_argument("--sim-backend", choices=["groq", "mistral"], default="mistral")
    parser.add_argument("--sim-model", default="ministral-14b-2512")
    parser.add_argument("--only", help="Comma-separated scenario ids (debugging).")
    parser.add_argument("--split", choices=["v1", "v2"], help="Run one split only.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="dense")
    args = parser.parse_args()

    if args.config:
        name, opts = args.config, dict(PRESETS[args.config])
    elif args.name:
        name, opts = args.name, {"agent": args.agent or "llm"}
    else:
        raise SystemExit("Give --config PRESET or --name NAME.")
    for key, value in {"backend": args.backend, "model": args.model, "prompt": args.prompt}.items():
        if value:
            opts[key] = value
    if args.extra:
        opts["extra"] = json.loads(args.extra)
    if args.force_action:
        opts["force_action"] = True
    if args.no_guardrails:
        opts["guardrails"] = False
    if args.multiturn:
        opts.update(multiturn=True, sim_backend=args.sim_backend, sim_model=args.sim_model)
    opts["retriever"] = args.retriever
    only = set(args.only.split(",")) if args.only else None
    run(name, opts, only, args.split, args.resume)


if __name__ == "__main__":
    main()
