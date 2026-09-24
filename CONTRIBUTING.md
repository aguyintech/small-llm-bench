# Contributing to small-llm-bench

Thanks for your interest in contributing!

## Setup

```bash
git clone https://github.com/aguyintech/small-llm-bench
cd small-llm-bench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q
```

The test suite is air-gapped: mock tools, no model calls, no network.

## Guidelines

- **Keep it lightweight.** Dependencies are limited to `httpx`, `rich`,
  `pydantic`, `pydantic-settings`, `typer`, `jinja2`, `pyyaml`, and
  `python-levenshtein`. PRs adding new dependencies need a strong rationale.
- **Scoring must stay deterministic.** Any nondeterministic evaluation belongs
  in the optional LLM judge pass, never in the default scorers.
- **Tasks are declarative.** New tasks go in the YAML files under `tasks/`,
  with no Python logic. Mock tools must return static, deterministic data.
- **Checks fail closed.** A constraint or file check that names nothing to
  test must fail, not pass. The schema tests enforce this for known keys.
- **Tests required.** New scorers, tools, or task fields need test coverage.
  All tests must pass: `pytest tests/`.
- **Style.** Type hints on every signature, one-line docstrings on public
  functions and classes, functions under 80 lines.

## Adding a task

1. Write the candidate into a scratch bank at `.scratch/probe/tasks/<module>.yaml`
   and screen it with `sllmb probe --module <module> --task <id>`. A task that
   does not separate the probe trio is not worth a fleet sweep.
2. Move a survivor into the relevant file in `tasks/`, following the existing
   schema. Keep prompts realistic and self-contained, and do not ask for output
   nothing grades.
3. Update the count assertions in `tests/test_tasks_schema.py` and the task
   counts in `README.md`.
4. Run `pytest tests/`.

## Adding a model to the registry

Add an entry to `models.yaml` with `params_b` (and `active_b` for a
mixture-of-experts model) taken from the model card, not from the model's name.

## Reporting issues

Open a GitHub issue with the model, endpoint type (llama.cpp, Ollama, vLLM,
...), the command used, and the saved results file if possible.
