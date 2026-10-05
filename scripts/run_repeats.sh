#!/usr/bin/env bash
# Repeat runs for the hosted Mistral configurations. The Mistral API is not
# deterministic at temperature 0 (about 15% of scenario outcomes flip between
# identical runs), so each configuration is run 3 times: <name> is repeat 1,
# <name>.r2 and <name>.r3 the others. The report aggregates them.
set -uo pipefail
cd "$(dirname "$0")/.."
R=".venv/bin/python -u -m switchboard.eval.run_suite"
M="--backend mistral"

rep() {  # rep <name> <args...>: runs repeats 2 and 3
  local name=$1; shift
  for r in 2 3; do $R --name "$name.r$r" "$@"; done
}

chain_a() {
  rep m3b         $M --model ministral-3b-2512
  rep m3b-fewshot $M --model ministral-3b-2512 --prompt fewshot
  rep m3b-force   $M --model ministral-3b-2512 --force-action
}
chain_b() {
  rep m8b         $M --model ministral-8b-2512
  rep m14b        $M --model ministral-14b-2512
  rep m3b-hybrid  --agent hybrid $M --model ministral-3b-2512
}
chain_c() {
  rep m14b-multiturn $M --model ministral-14b-2512 --multiturn \
      --sim-backend mistral --sim-model ministral-14b-2512
}

case "${1:-all}" in
  a) chain_a ;; b) chain_b ;; c) chain_c ;;
  all) chain_a & chain_b & chain_c & wait ;;
esac
