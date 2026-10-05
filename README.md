# switchboard

A tool-using customer-service agent for a simulated UK broadband, TV and mobile provider, plus the evaluation suite used to test it.

The agent reads a customer message, calls tools against a SQLite back office (accounts, bills, usage, outages, line tests, help centre) and can take actions: apply a credit, book an engineer, escalate to a human team. Every tool call passes through a policy layer before it runs. Runs are scored on what ends up in the database, not on how the reply reads.

## How it works

```
message -> input checks (PII redaction, injection detection)
        -> LLM with tool schemas, up to 6 steps
             each tool call: validate args -> policy check -> execute -> sanitise output -> trace
        -> output checks (unbacked action claims, ungrounded £ amounts, cross-customer leaks)
        -> reply + trace + database writes
```

- **Tools** (`tools/larkspur.py`): 6 read tools, including `query_billing_data`, which runs model-written SQL against a throwaway database holding only the signed-in customer's rows, with a read-only authorizer. 3 action tools write credits, bookings and escalations.
- **Policy** (`guardrails/policy.py`): actions are gated on evidence gathered in the same conversation. A credit needs an eligible outage returned by `check_outage`, and it is capped at the policy amount. An engineer booking needs a failed line test and no active area outage. Other customers' accounts are off limits. With guardrails off, the same checks run in audit mode, so unsafe actions are counted rather than prevented.
- **Agents**: a plain tool-calling agent (`agent/loop.py`), a rule-based workflow with no LLM (`agent/workflow.py`), and a hybrid (`agent/hybrid.py`), where one LLM call turns the message into typed requests and deterministic handlers do the work.
- **Models**: Qwen2.5-3B-Instruct locally (fp16, fits an 8 GB GPU), or hosted models through Groq and Mistral (`agent/api_chat.py`, OpenAI-compatible, with rate limiting and resume on daily quota).

## Evaluation

60 scenarios in `src/switchboard/eval/scenarios.jsonl`: 45 original (core requests, long-tail phrasing, adversarial) and 15 written later as a held-out set. Each scenario specifies the expected database writes, the facts the reply must contain and anything it must not contain. A run passes if the end state matches, the facts are present and nothing unsafe happened. Rates come with 95% bootstrap intervals, and configurations are compared with McNemar's exact test on paired scenarios. Full tables: [`reports/agent_eval.md`](reports/agent_eval.md).

Local models decode greedily and are reproducible. The hosted Mistral models are not: at temperature 0, 10 to 32% of scenario outcomes changed between identical runs. So each Mistral configuration was run 3 times, and the table shows the mean with the per-run counts.

| Configuration | All 60 | Held-out 15 |
|---|---|---|
| Rule-based workflow | 40/60 (0.67) | 6/15 |
| Qwen2.5-3B agent | 29/60 (0.48) | 7/15 |
| Qwen2.5-7B agent (4-bit) | 30/60 (0.50) | 5/15 |
| Qwen2.5-3B hybrid | 52/60 (0.87) | 12/15 |
| Ministral-3B agent | 0.63 (38, 38, 38) | 0.67 |
| Ministral-3B + worked examples | 0.76 (47, 46, 44) | 0.76 |
| Ministral-3B + "call the tool you promised" check | 0.74 (44, 42, 47) | 0.80 |
| Ministral-3B hybrid | 0.95 (57, 57, 57) | 1.00 |
| Ministral-8B agent | 0.72 (42, 44, 44) | 0.69 |
| Ministral-14B agent | 0.77 (47, 46, 46) | 0.76 |
| Ministral-14B, multi-turn (simulated customer) | 0.78 (48, 47, 46) | 0.78 |
| Qwen3.8-27B agent (Groq), 50 of 60 run | 48/50 (0.96) | 5/5 |
| GPT-OSS-120B agent (Groq) | 53/60 (0.88) | 12/15 |

Findings:

- Restricting the small model to parsing requests had the biggest effect. The same local 3B model went from 0.48 as a plain agent to 0.87 as a hybrid (p < 0.001) and beat the rules it reuses (p = 0.004). The Ministral-3B hybrid scored 57/60 in all three runs, and only 3% of its scenario outcomes varied between runs, against 13 to 32% for the plain agents.
- The rules were fitted to the original scenarios: 34/45 on those, 6/15 on the held-out set.
- Worked examples (+0.13, p = 0.025) and a check that makes the model call a tool it has promised to use (+0.11, p = 0.048) helped Ministral-3B. Neither effect was visible in single runs because of the run-to-run variance, and with about a dozen comparisons in the report the second one is borderline.
- Size helped within Ministral (3B to 14B: +0.14, p = 0.009) but not within Qwen2.5 (3B and 7B both about 0.5). At 3B, Ministral beat Qwen2.5 (0.63 vs 0.48, p = 0.034).
- Large hosted models work well as plain agents (0.88 and 0.96, both above the rules with p <= 0.002), but the 3B hybrid matched them at a fraction of the latency.
- The most common plain-agent failure was offering or announcing an action ("Would you like me to book an engineer?", "Let me check your account") without making the tool call.
- With guardrails off, Qwen2.5-3B executed 3 unsafe actions in the 8 adversarial scenarios: a £200 credit requested by a prompt injection, another customer's bill, and a £100 credit over a £30 cap. With guardrails on there were no unsafe actions in any run.

Each run writes `reports/runs/<name>.jsonl` (one scored row per scenario) and `<name>.meta.json` (requested and served model ids, dates, code hash, library versions). Runs were redone after three bugs surfaced in the traces: a strict JSON schema in the hybrid parser, curly apostrophes defeating the action check, and a SQL tool whose scoped tables lacked the `customer_id` column that models filter on. The original runs are kept in `reports/runs/archive-before-fixes/`. One local run (Qwen2.5-3B with worked examples) predates the SQL fix and is marked as such in the report.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # rule-based agent, BM25 search, tests
pip install -e ".[dev,dense,llm]"  # + Chroma/MiniLM search and the local LLM

python -m switchboard.cli --list-customers
python -m switchboard.cli --customer LK-1006 "My broadband isn't working at all"
python -m switchboard.cli --customer LK-1024 --agent hybrid --backend mistral \
    --model ministral-3b-2512 "I want £100 for the outage in August"
```

Hosted backends read `GROQ_API_KEY` / `MISTRAL_API_KEY` from the environment, or from a `keys.env` file in the repo root with lines like `Groq=...` and `Mistral=...` (ignored by git).

## Running the evaluation

```bash
make eval-workflow    # rule-based baseline, CPU only
make eval-local       # Qwen2.5-3B agent on a local GPU
python -m switchboard.eval.run_suite --name m3b-hybrid --agent hybrid \
    --backend mistral --model ministral-3b-2512
python -m switchboard.eval.run_suite --name m14b-mt --backend mistral \
    --model ministral-14b-2512 --multiturn
make report           # rebuild reports/agent_eval.md
```

`scripts/run_experiments.sh` reproduces every run in the table and `scripts/run_repeats.sh` adds the Mistral repeats. Free API tiers have daily token caps, so add `--resume` to continue a run that stopped on quota.

## Layout

```
src/switchboard/
  world/        seeded SQLite back office (24 customers, bills, usage, outages)
  tools/        tool schemas and implementations
  guardrails/   pre-execution policy, injection/PII/grounding checks
  agent/        tool-calling loop, workflow baseline, hybrid, local and API backends
  retrieval/    help-centre search (BM25, MiniLM embeddings in Chroma)
  eval/         scenarios, scoring, metrics, runner, report, simulated customer
  router/       intent router for narrowing the tool list (not trained yet)
tests/          unit tests (no GPU or network needed)
```

## Limitations

- 60 hand-written scenarios, one annotator. Differences of a few scenarios are within noise, which is why the report shows intervals and paired tests.
- The hybrid's handlers were written by the same person who wrote the scenarios, so the held-out set is the fairer comparison.
- The multi-turn customer is an LLM and sometimes drifts (for example, it agrees to escalations it never asked for), which accounts for most multi-turn failures.
- In the multi-turn runs the same model plays the agent and the customer.
- p-values are not corrected for multiple comparisons.
- The guardrails-off comparison covers only the adversarial scenarios.
- All customers, policies and data are fictional.

## License

MIT
