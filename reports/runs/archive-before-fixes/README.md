Runs from 24 Sep 2026 made before two bugs were fixed. Kept as evidence; not in the main report.

- Apostrophe bug: the action verifier (and force-action) matched only ASCII apostrophes, so replies like "I’ll book" / "I’ve booked" from Mistral models were never checked. Affects every Mistral run here.
- Hybrid parser bug (m3b-hybrid only): `"consent_to_charge": null` failed validation and sank the whole parse. m3b-hybrid-fixed has the parser fix but not the apostrophe fix.

Effect of fixing (same model, same scenarios): m3b-hybrid 51/60 -> m3b-hybrid-fixed 58/60 (parser); m3b-force 40/60 -> 44/60 and m14b 43/60 -> 47/60 (apostrophes).

- SQL tool schema (25 Sep 2026, files `*.before-sql-fix.*`): the scoped tables behind `query_billing_data` omitted `customer_id`, so model-written queries that filtered on it (a reasonable habit) failed with "no such column". The fix keeps the column (it only ever holds the signed-in customer's id); the tool description the model sees is unchanged. Affected runs, re-run on the fixed tool: qwen7b (3 scenarios hit the error), m3b and m3b-fewshot (1 each), and qwen3b-fewshot (used `SELECT *`, whose output changes with the extra column). Other runs never filtered on customer_id or used `SELECT *`.
