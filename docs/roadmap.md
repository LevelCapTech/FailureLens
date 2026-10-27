# Roadmap

Honest, phased plan. Items under **Done** actually work today; everything else is
explicitly planned and must not be presented as finished. Priorities and ordering
may change as experiments produce evidence.

## Guiding priority

> Complete a genuinely working minimal baseline first; then move to dedicated
> model training and evaluation. Do not build a large, un-runnable codebase.

## Done (works today)

- Monorepo structure and lightweight, installable `failurelens` package.
- Python traceback parser; AST repository indexer (Python).
- Baseline retrieval + ranking (stack-trace + lexical), deterministic.
- Public API (`from_baseline`, `diagnose`, `index`) and `failurelens` CLI with
  stable exit codes and versioned JSON output shared by both.
- Optional embedding reranker backend (uses an external base model; opt-in).
- Reusable, tested evaluation metrics + a runnable baseline harness with honest,
  measured results; dataset record schema + conforming fixture.
- Quality gates: ruff, mypy (strict), pytest; CI (3.11–3.13), lint, and package
  build workflows; documentation set.

## Near term

- More log formats behind the parser interface (JavaScript/TypeScript, Java).
- Richer retrieval signals (call/dependency relations; test-failure info).
- Larger, license-cleared evaluation benchmarks and committed reports.
- BM25 and off-the-shelf-embedding comparison baselines in `evaluation/`.

## Medium term

- Data-generation pipeline (mutation/bug injection) honoring the
  pass-before/fail-after validity rule and data-leak splits (`docs/datasets.md`).
- Training pipeline with a CPU-only smoke test and one ranking objective
  (`docs/training.md`); reproducible experiment records.
- First small fine-tuned bi-encoder model, evaluated against the baselines.

## Longer term / research

- Probability **calibration** so scores can be interpreted as likelihoods
  (then revisit `score_type`).
- Success-vs-failure execution-diff signals; ablation studies per signal.
- Additional languages (Go, Rust, C/C++) as parsers/indexers mature.
- Distribution: PyPI release of the library and Hugging Face publication of any
  trained model (with a correct, verified license) — only with explicit approval.

## Explicit non-goals

- No fabricated models, benchmarks, or accuracy claims.
- No mandatory network access, external APIs, or model downloads for the baseline.
- No premature claims of multi-language support or superiority over other tools
  without measured evidence.
