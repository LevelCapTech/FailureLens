# Model card — FailureLens

> **Status: there is no trained FailureLens model yet.** This card honestly
> documents what exists today (a heuristic baseline) and the intended direction.
> It will be replaced by a proper per-model card when weights are published. Do
> not cite any accuracy numbers that are not produced by the evaluation harness.

## Summary

| Field | Value |
|-------|-------|
| Name | FailureLens (baseline) |
| Task | Fault localization — rank candidate code locations for a failure |
| What ships today | Heuristic baseline (`analysis_method = baseline-stacktrace-lexical`) |
| Trained weights | **None published** |
| Runtime | CPU, offline, no heavy dependencies |
| Source license | Apache-2.0 (weights, when they exist, may differ — see below) |

## Intended use

- Help developers, CI systems, and AI coding agents **narrow down** where a
  failure likely originates, with auditable evidence.
- Input: an error log / stack trace + a source repository. Output: ranked
  candidate locations (file / function / line span) with evidence.

### Out of scope

- Not an LLM API wrapper, generic code search, or error-message summariser.
- Does **not** guarantee a fix is correct — it suggests candidates.
- Not a multi-language tool yet (Python traceback support first).

## How the baseline works

Deterministic scoring over parsed stack frames and indexed code units:
`score = 0.6·depth + 0.2·exception_site + 0.4·lexical`. See
`docs/architecture.md` for details. `score` is a **relative heuristic, not a
probability** (`score_type = heuristic`).

## Limitations & biases

- The baseline **over-weights the exception site**, so the function where the
  error surfaced can outrank the true root cause. (On the bundled fixture, the
  root-cause function ranks 2nd — see `docs/evaluation.md`.)
- Only Python tracebacks are parsed today; other formats return no candidates
  rather than guessing.
- Lexical matching is shallow (token overlap), not semantic.
- No learned relevance yet; an optional embedding reranker exists but uses an
  **external** base model and is not fine-tuned for fault localization.

## Evaluation

No benchmark claims are made here. Reproduce the only published numbers with:

```bash
python evaluation/run_baseline_eval.py
```

Results are computed from the committed benchmark, never hand-written. See
`docs/evaluation.md`.

## Planned model

A small **bi-encoder** fine-tuned for fault localization (see `docs/training.md`).
Concrete size/latency/accuracy will be determined by experiment.

## Licensing of future weights

The GitHub source is Apache-2.0, but a released model's license depends on its
**base model and training data**. The correct license will be determined before
any weights are published; Apache-2.0 is **not** assumed to apply to weights.

## Provenance

- Code: this repository (`LevelCapTech/FailureLens`), Apache-2.0.
- Optional base models referenced (not bundled): `nomic-ai/CodeRankEmbed` (MIT,
  verify), `Salesforce/SweRankEmbed-Small` (non-commercial, verify).
