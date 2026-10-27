# Contributing to FailureLens

Thanks for your interest in improving FailureLens! This guide covers the
practical workflow. For the deeper "rules of the repo" — especially the
**honesty rules** every contributor (human or AI agent) must follow — read
[AGENTS.md](AGENTS.md).

## Ground rules (short version)

- **Never fabricate** models, weights, benchmark numbers, or accuracy claims.
- Keep the **baseline runnable** with no network and only core deps
  (`pydantic`, `typer`). Heavy dependencies belong in the `ml`/`train` extras.
- Clearly separate **working** code from **planned** design.
- Respect licenses and never commit secrets or personal data.

## Development setup

Python **3.11+** is required (CI also runs 3.12 and 3.13).

```bash
git clone https://github.com/LevelCapTech/FailureLens.git
cd FailureLens
pip install -e "packages/python[dev]"      # or: make install
```

Optionally enable git hooks that mirror CI locally:

```bash
pip install pre-commit
pre-commit install
```

## Quality gates (run before every PR)

These are exactly what CI enforces. Run them all with `make check`:

```bash
ruff check .            # lint            (make lint)
ruff format --check .   # formatting      (make format-check)
mypy                    # strict types    (make typecheck)
pytest                  # tests           (make test)
```

Fix formatting automatically with `ruff format .` (or `make format`).

## Making changes

1. Create a branch for your change.
2. Keep changes **small and focused**; prefer the smallest correct change.
3. Add or update **tests** consistent with the existing suite (unit tests mirror
   `src/` modules; repo-level tests live in `tests/`).
4. Update docs/READMEs when behaviour or interfaces change.
5. Ensure `make check` passes.
6. Open a PR using the template and fill in the checklist.

### Where things go

- Library/CLI code: `packages/python/src/failurelens/`.
- Tests: `packages/python/tests/` (unit) and `tests/` (integration/e2e/regression).
- Evaluation harness/benchmarks: `evaluation/`.
- Data schemas/design: `datasets/`.
- Planned model/training code: `models/`, `training/` (clearly marked planned).
- Specs and references: `docs/`.

See [AGENTS.md §7](AGENTS.md) for step-by-step recipes (adding a parser, a
backend, an evaluation case, a dataset record).

## Coding conventions

- Full type annotations; `mypy` runs in **strict** mode over the library.
- `ruff` is the linter and formatter (config in the root `pyproject.toml`).
- Use `from __future__ import annotations` in library modules **except**
  `cli.py` (Typer needs runtime annotations). Prefer `X | None` over `Optional`.
- Use `StrEnum` for string enums; use Pydantic models for structured data.
- Keep core import paths free of heavy dependencies (import `torch`/
  `transformers` lazily, only inside the `ml` backend).
- ASCII only in source (ruff flags fancy dashes). Keep docstrings factual.

## Commit & PR guidance

- Write clear, imperative commit messages ("Add JS traceback parser").
- Reference related issues (e.g. `Closes #123`).
- The PR template includes a self-review checklist — complete it.

## Reporting bugs / requesting features

Use the issue templates under **New issue**. For security issues, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.

## License of contributions

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.
