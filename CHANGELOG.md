# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial FailureLens monorepo bootstrap.
- `failurelens` Python package (core deps: `pydantic`, `typer`):
  - Public API: `FailureLens.from_baseline()` / `from_pretrained()`, `diagnose()`,
    `index()`.
  - Versioned output schema (`DiagnosisResult`, `Candidate`, `Evidence`,
    `StackFrame`, `ParsedLog`) shared by the API and CLI JSON output
    (`SCHEMA_VERSION = 0.1.0`).
  - Python traceback parser and AST-based repository indexer (behind extension
    interfaces for future languages).
  - Dependency-free baseline backend (`baseline-stacktrace-lexical`): stack-trace
    + lexical ranking with auditable evidence; heuristic scores (not
    probabilities).
  - Optional embedding reranker backend (via the `ml` extra; uses an external
    base model, nothing bundled or auto-downloaded).
  - `failurelens` CLI (`version`, `index`, `diagnose`, `models`) with stable exit
    codes and clean JSON output.
- Reusable, tested ranking metrics (`failurelens.evaluation.metrics`) and a
  runnable baseline evaluation harness (`evaluation/run_baseline_eval.py`) that
  reports measured results only.
- Dataset record JSON Schema + conforming fixture and a regression test.
- Example fixture project and end-to-end tests.
- Documentation set under `docs/`, plus `AGENTS.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- CI workflows (test matrix on Python 3.11–3.13, lint, and package build with
  `twine check`; no publishing), issue/PR templates, and developer tooling
  (`Makefile`, `.pre-commit-config.yaml`, `.gitattributes`).
- AI-model **design-gate** documents (design only; no training/data/model release
  authorized): `docs/requirements.md`, `docs/data-contract.md` (source/rights
  matrix, rights-cleared pilot, semantic quality gates), `docs/evaluation-protocol.md`
  (candidate parity, metrics, reproducible reporting), and `docs/decision-gate.md`
  (Go/No-Go sequencing and handoff; numeric thresholds left as `TBD` pending
  pilot measurement and human approval).
- `failurelens.quality`: pure, dependency-free **semantic quality gates** for
  dataset records (beyond JSON Schema), implementing the `docs/data-contract.md`
  table. `validate_record` / `validate_record_set` enforce the license,
  candidate (positive + negative), rationale, pre/post validity,
  environment-failure, secrets/PII, and duplicate gates, accumulating **all**
  violations and recording a `GateRejectionReason` for each. Includes
  deterministic `check_secret` / `check_pii` screeners and a regression test
  that the committed fixture passes every gate.

### Notes
- No trained FailureLens model is published yet; the baseline is fully functional.
- No benchmark or accuracy claims are made beyond what the evaluation harness
  measures on committed data.

[Unreleased]: https://github.com/LevelCapTech/FailureLens/commits/main
