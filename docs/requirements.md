# Requirements specification (AI-model design gate)

This document is a **design-gate deliverable** for the AI-model work (issue #3).
It fixes the agreed requirements *before* any large training/data/benchmark
implementation. It is **not** authorization to train at scale or publish a model
(see `docs/decision-gate.md`). It builds on the library delivered by the
bootstrap (issue #2) and the product definition in issue #1.

> **Status legend.** **Works today** = shipped and runnable in this repository.
> **Planned** = agreed design, not implemented; must never be presented as
> finished. The same honesty rules as `AGENTS.md` apply here.

## Goal and users

For **developers**, **CI systems**, and **AI coding agents**: given a Python
failure log/traceback and a **pinned** source repository, rank the evidence-
backed code locations most likely **responsible** for the failure, and clearly
distinguish the **exception site** (where the error surfaced) from the likely
**cause**. Return ranked candidates with observed evidence, the analysis
method/version, and an explicit **inconclusive** status when evidence is
insufficient.

An uncalibrated score is **not** a probability (`score_type = heuristic` today;
see `docs/architecture.md` and `docs/model-card.md`).

## Scope

- **Python-first.** The parser/indexer target Python tracebacks and source.
  Additional languages and log formats are **later decisions**, not part of this
  gate (the interfaces already allow them — see `docs/architecture.md`).
- **Offline, CPU baseline stays working.** No mandatory external LLM/API, model
  download, or network access for the baseline path.
- **No automatic-fix guarantee.** FailureLens *suggests* candidate locations with
  evidence; it does not promise a correct fix.

## Inputs and outputs

**Inputs**

- A failure log / traceback: exactly one of a log file (`error_log`) or raw text
  (`error_text`).
- A source repository, referenced at a **pinned** commit for reproducibility.

**Outputs** (the versioned `DiagnosisResult` contract; see `docs/api.md`)

- Ranked `candidates`, each with `rank`, `file`, `function`, `start_line`/
  `end_line`, `score`, `score_type`, and observed `evidence`.
- `analysis_method` (and `model_id`/`model_revision` for model backends) so every
  result records *how* it was produced.
- `status` of `ok | no_candidates | insufficient_evidence | error` — the
  **inconclusive** states are first-class, not failures.
- Location fields are `null` when they cannot be determined — never guessed.

## Existing features (work today)

- Python traceback parser; AST repository indexer (Python).
- Dependency-light baseline backend (`baseline-stacktrace-lexical`): deterministic
  stack-trace + lexical ranking with auditable evidence.
- Public API (`from_baseline`, `diagnose`, `index`) and `failurelens` CLI with
  stable exit codes and a versioned JSON schema shared by both.
- Optional, opt-in embedding reranker that uses an **external** base model (the
  `ml` extra). It is **not** a FailureLens-trained model.
- Reusable ranking metrics and a runnable baseline evaluation harness that reports
  **measured** results only; a dataset record schema with a conforming fixture.

## Planned features (not implemented)

- A FailureLens-**trained** model (first planned as a small bi-encoder; see
  `docs/training.md`). **No trained model exists yet.**
- A rights-cleared data pilot and the generation/collection pipeline
  (`docs/data-contract.md`, `docs/datasets.md`).
- Larger, license-cleared evaluation benchmarks and comparison baselines
  (stack-only, BM25, off-the-shelf embeddings) under `docs/evaluation-protocol.md`.
- Score **calibration** so scores may later be read as likelihoods.

## Exclusions / non-goals

- Not an LLM API wrapper, generic code search, or error-message summariser.
- No mandatory network/model download for the baseline.
- No multi-language or superiority claims without **measured** evidence.
- No public dataset/model release or full-scale training authorized here.

## Risks

- **Label validity.** An injected-bug location is not automatically the root
  cause; weak labels mislead training and evaluation.
- **Data rights and leakage.** Candidate corpora (e.g. SWE-smith, SWE-bench) are
  **not** pre-cleared; naive splits leak the same repo/fix into train and eval.
- **Baseline bias.** The heuristic over-weights the exception site, so the
  surfacing function can outrank the true cause.
- **Fabrication risk.** Pressure to show results must never produce invented
  models, weights, or accuracy numbers.
- **Resource limits.** Training/inference must fit stated compute budgets (frozen
  at the decision gate, not assumed up front).

## Decision owners

| Role | Owner | Responsibility |
|------|-------|----------------|
| Coordinator | `failurelens-coordinator` | Issue owner/review; recommends Go/No-Go. |
| Data | `failurelens-data-steward` | Data rights, pilot, label review, splits. |
| Model feasibility | `failurelens-model-developer` | Base-model/training feasibility. |
| Independent evaluation | `failurelens-independent-evaluator` | Parity, metrics, reports. |
| Human owner | Project maintainer | Approves the design gate and later adoption/publication. |

## Related design-gate documents

- `docs/data-contract.md` — data sources/rights matrix, pilot, quality gates.
- `docs/evaluation-protocol.md` — independent evaluation protocol.
- `docs/decision-gate.md` — Go/No-Go sequencing and handoff (the hub).
