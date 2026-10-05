"""Central configuration for switchboard.

Every path resolves relative to the repository root so the code behaves the same on
a laptop, in a Docker container, or in CI. Anything that differs between local
development and a deployed Space is overridable with an environment variable.

The simulated back office (``world/``), the help-centre knowledge base (``kb/``) and
the scenario suite (``eval/scenarios.jsonl``) are data, not code: pointing the agent
at a different provider means new data files and new tool implementations, not a
new agent loop.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths ------------------------------------------------------------------
# ``config.py`` lives at ``<root>/src/switchboard/config.py`` -> three parents up.
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = Path(os.getenv("SWB_DATA_DIR", ROOT_DIR / "data"))
MODEL_DIR: Path = Path(os.getenv("SWB_MODEL_DIR", ROOT_DIR / "models"))
REPORT_DIR: Path = Path(os.getenv("SWB_REPORT_DIR", ROOT_DIR / "reports"))

# The seeded SQLite back office. Rebuilt deterministically by ``world.seed`` when it
# is missing, so a fresh clone needs no download to run the agent.
DB_PATH: Path = Path(os.getenv("SWB_DB_PATH", DATA_DIR / "larkspur.db"))

# --- The simulated world ----------------------------------------------------
# A fixed "today" makes every run reproducible: outage windows, contract ends and
# engineer slots are all relative to it. Larkspur is a fictional UK provider.
TODAY: str = os.getenv("SWB_TODAY", "2026-09-14")
COMPANY: str = "Larkspur"

# --- Retrieval over the help centre -----------------------------------------
# "bm25" (lexical, no download, the CI baseline) or "dense" (sentence-transformers
# embeddings in a Chroma collection).
RETRIEVER: str = os.getenv("SWB_RETRIEVER", "bm25")
EMBED_MODEL: str = os.getenv("SWB_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# CPU by default: the help centre is tiny, and it keeps the 8 GB GPU for the LLM.
EMBED_DEVICE: str = os.getenv("SWB_EMBED_DEVICE", "cpu")
HELP_TOP_K: int = int(os.getenv("SWB_HELP_TOP_K", "3"))

# --- Intent router (classic ML) ---------------------------------------------
# Trained on the Bitext telco dataset (CDLA-Sharing-1.0). Used to narrow the tool
# list the LLM sees; falls back to all tools when missing or unsure.
BITEXT_DATASET: str = os.getenv(
    "SWB_BITEXT_DATASET", "bitext/Bitext-telco-llm-chatbot-training-dataset"
)
ROUTER_PATH: Path = Path(os.getenv("SWB_ROUTER_PATH", MODEL_DIR / "router.joblib"))
# Below this top-1 probability the router adds its second-best intent's tools; below
# ROUTER_FLOOR it gives up and exposes every tool.
ROUTER_THRESHOLD: float = float(os.getenv("SWB_ROUTER_THRESHOLD", "0.6"))
ROUTER_FLOOR: float = float(os.getenv("SWB_ROUTER_FLOOR", "0.3"))
SEED: int = int(os.getenv("SWB_SEED", "20260914"))

# --- Agent ------------------------------------------------------------------
# "llm" (tool-calling Qwen2.5-3B) or "workflow" (the deterministic hand-written
# workflow: the CI/no-GPU fallback and the evaluation floor).
AGENT: str = os.getenv("SWB_AGENT", "workflow")
LLM_MODEL: str = os.getenv("SWB_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
# fp16 fits on an 8 GB Turing card (about 6.2 GB peak) and is much faster than
# bitsandbytes 4-bit generation there. Set SWB_LLM_4BIT=1 on smaller cards.
LLM_LOAD_4BIT: bool = os.getenv("SWB_LLM_4BIT", "0") == "1"
LLM_MAX_NEW_TOKENS: int = int(os.getenv("SWB_LLM_MAX_NEW_TOKENS", "256"))
# Hard cap on model calls per conversation turn: stops tool-call loops.
MAX_STEPS: int = int(os.getenv("SWB_MAX_STEPS", "6"))
# "all" tools, or "routed" (the intent router picks a subset).
TOOL_EXPOSURE: str = os.getenv("SWB_TOOL_EXPOSURE", "all")
GUARDRAILS: bool = os.getenv("SWB_GUARDRAILS", "1") == "1"


def ensure_dirs() -> None:
    """Create the data/model/report directories if they do not exist."""
    for directory in (DATA_DIR, MODEL_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
