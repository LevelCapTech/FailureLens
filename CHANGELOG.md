# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Call-graph baseline (opt-in strategy)
- `baseline-stacktrace-callgraph` analyzer: ranks every function and method from
  the stack-trace depth plus static, name-based call relations (up to two hops)
  to functions in the stack. No model, offline, no new dependencies.
  Select it with `FailureLens.from_baseline(strategy="stacktrace-callgraph")` or
  `failurelens diagnose --baseline stacktrace-callgraph`.
- Output schema `0.2.0`: new evidence kind `call_relation` (additive).

#### Monorepo bootstrap and core library
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
- AI-model **design-gate** documents (design only): `docs/requirements.md`,
  `docs/data-contract.md`, `docs/evaluation-protocol.md`, `docs/decision-gate.md`.
- `failurelens.quality`: pure, dependency-free semantic quality gates for dataset
  records. `validate_record` / `validate_record_set` enforce license, candidate,
  rationale, pre/post validity, environment-failure, secrets/PII, and duplicate
  gates with full violation accumulation and `GateRejectionReason` per violation.

#### AI model design gate (Gate A + B)
- Gate A completed (2026-10-27): pilot feasibility run on rights-cleared in-repo
  fixture; baseline measurement with `evaluation/run_baseline_eval.py`. Results:
  file Top-1/5/10 = 1.0, MRR = 1.0; function Top-1 = 0.0, MRR = 0.5 (n=1).
- Gate B completed (2026-10-29, human-owner approved): frozen acceptance criteria:
  function Top-1 ≥ 0.4, MRR ≥ 0.6, absolute improvements ≥ 0.4 / ≥ 0.1,
  ≥ 30 cases across ≥ 3 repos, p95 latency ≤ 5 s, peak ≤ 8 GiB, ≤ 1B params.
- Implementation Go authorized by human owner (2026-10-29).

#### Private training data rights (PR #18 — data/issue17-private-training)
- Rights matrix and aggregate-only handoff for the `failurelens-train-issue17-v2`
  dataset (12 synthetic records, `pallets/click` + `python-attrs/attrs`).
- Data not in Git; SHA-256 pinned: `74909f51...`.

#### NomicBERT transformers-5.x compat + inference (PR #20 — feature/issue19-coderankeembed-smoke)
- `failurelens.inference`: `load_sentence_transformer` with audited-revision pin
  and safetensors-only enforcement; `NomicBertCompatShim` for transformers 5.x
  Flash-Attention API change. `einops` added as `ml` extra dependency.
- CPU smoke test; ML integration tests behind `--run-ml-integration`.

#### Stage 2 CodeRankEmbed fine-tuning (PR #25 — feature/issue19-stage2-coderankeembed-finetune)
- `training/finetune.py`: InfoNCE fine-tuning of the last NomicBertBlock
  (~9.4M / 136.7M params); AdamW + CosineAnnealingLR; checkpoint/resume with
  exact loss reproducibility; safetensors checkpoints and final artifact.
- `training/data_stage2.py`: source-code reconstruction from private records;
  `_strip_generator_marker` sanitizes all generator markers, private absolute
  paths, and injected exception messages from training features.
- Private artifact v3 (Gate C candidate):
  SHA-256 `506492b034fadb3d7ee6e7288d04b66dfa477c25b1de41d9d70166353cd343d4`,
  code SHA `795b0a1`, seed 42, 3 epochs; not in Git or HF Hub.
  Supersedes tainted v1 (`7520d7aa...`) and v2 (`e91955b4...`) artifacts.

#### Private evaluation adapter (PR #22 — eval/issue21-private-adapter)
- `evaluation/private_adapter.py`: opt-in private case loader with exact record
  digest, schema/quality gates, candidate-snapshot checks, rights assertion,
  30-case / 3-repo minimum, and train/eval disjointness.

#### Independent Gate C runner (PR #24 — eval/issue23-independent-runner)
- `evaluation/independent_runner.py`: evaluator-owned library for three-seed
  paired bootstrap evaluation (function Top-1, MRR, delta, intervals).
  No model or held-out data bundled; CI uses synthetic fixtures only.

#### Opt-in v3 evaluation bridge (PR #27 — eval/issue26-v3-bridge)
- `evaluation/v3_bridge.py`: evaluator-owned bridge that loads the v3 artifact
  (exact SHA-256 + provenance check), applies the Stage 2 query transform, and
  produces a `model_factory` compatible with `run_independent`.
  Gate C is blocked pending comparator protocol review.

#### Shipped-baseline common pool bridge (PR #29 — eval/issue28-shipped-baseline-pool)
- `evaluation/product_pool.py`: opt-in adapter using shipped
  `FailureLens.from_baseline()` retrieval with complete failing-tree manifests;
  passes exact ordered candidate pool to v3 reranker.
  Resolves the comparator limitation for a valid shipped-baseline Gate C delta.
  Gate C hold: independent evaluator must review provenance and run.

#### Non-leaking split check (d44f5a4)
- `failurelens.split_leakage`: detect repositories and record ids shared across
  dataset splits; fails closed on missing `source_repository`.
- `datasets/validation/check_splits.py`: offline validation script; JSON report;
  exit 1 on leak or rejected record.
- `docs/data-contract.md` updated with non-leaking split requirements.

### Notes
- Gate C (independent held-out evaluation) not yet run; Gate D (publication)
  not yet approved. See `docs/decision-gate.md` and `docs/model-card.md`.
- Trained v3 artifact exists locally; not in Git or HF Hub.
- No benchmark or accuracy claims beyond Gate A baseline measurement on n=1 fixture.

[Unreleased]: https://github.com/LevelCapTech/FailureLens/commits/main
