.PHONY: install install-all test lint typecheck seed eval-workflow eval-local report

install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[dev,dense,llm]"

test:
	pytest -q

lint:
	ruff check src tests

typecheck:
	mypy src

seed:
	python -m switchboard.world.seed

# Rule-based baseline: CPU only, no downloads.
eval-workflow:
	python -m switchboard.eval.run_suite --config workflow --retriever bm25

# Local Qwen2.5-3B agent (needs a GPU and the llm + dense extras).
eval-local:
	python -m switchboard.eval.run_suite --config llm

report:
	python -m switchboard.eval.report
