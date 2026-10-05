#!/usr/bin/env bash
# The second round of experiments (24 Sep 2026). Three independent chains that can run
# in parallel: hosted Mistral models, hosted Groq models, and the local GPU.
# Every run writes reports/runs/<name>.jsonl + .meta.json (model ids, dates, code hash).
set -uo pipefail
cd "$(dirname "$0")/.."
R=".venv/bin/python -u -m switchboard.eval.run_suite"

chain_mistral() {
  $R --name m3b    --backend mistral --model ministral-3b-2512
  $R --name m8b    --backend mistral --model ministral-8b-2512
  $R --name m14b   --backend mistral --model ministral-14b-2512
  $R --name m3b-fewshot --backend mistral --model ministral-3b-2512 --prompt fewshot
  $R --name m3b-force   --backend mistral --model ministral-3b-2512 --force-action
  $R --name m3b-hybrid  --agent hybrid --backend mistral --model ministral-3b-2512
  $R --name m14b-multiturn --backend mistral --model ministral-14b-2512 \
     --multiturn --sim-backend mistral --sim-model ministral-14b-2512
}

chain_groq() {
  # Free tier: ~200K tokens/day per model. If a run stops on quota, re-run with --resume.
  $R --name groq-qwen27b --backend groq --model qwen/qwen3.8-27b --resume
  $R --name groq-gptoss120b --backend groq --model openai/gpt-oss-120b \
     --extra '{"reasoning_effort": "low"}' --resume
}

chain_local() {
  $R --config workflow --resume                 # adds the 15 v2 scenarios
  $R --config llm --resume                      # Qwen2.5-3B base on the 15 v2 scenarios
  $R --name qwen3b-fewshot --prompt fewshot
  $R --name qwen3b-force --force-action
  $R --name qwen3b-hybrid --agent hybrid
}

case "${1:-all}" in
  mistral) chain_mistral ;;
  groq) chain_groq ;;
  local) chain_local ;;
  all) chain_mistral & chain_groq & chain_local & wait ;;
esac
