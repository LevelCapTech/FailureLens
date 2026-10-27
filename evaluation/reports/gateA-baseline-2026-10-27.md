# Gate A baseline measurement — 2026-10-27

Real, harness-produced measurement of the stack-trace-lexical baseline on the
bundled demonstration benchmark. No numbers in this file or the accompanying
JSON were hand-written: the JSON is the verbatim output of
`evaluation/run_baseline_eval.py`.

## Environment

| Item | Value |
|------|-------|
| Commit | `61af5a5c04a3eb94b55471c3208a3a5452c18aa2` (origin/main) |
| Host | macOS 26.5 arm64 (Apple M4 Mini), CPU-only |
| Python | 3.14.9 (venv at `/tmp/fl-venv`) |
| Install | `pip install -e "packages/python[dev]"` inside the venv |
| Command | `python evaluation/run_baseline_eval.py --benchmark <abs>/evaluation/benchmarks/sample_cases.json --output <abs>/evaluation/reports/gateA-baseline-2026-10-27.json` |
| Date (JST) | 2026-10-27 |

Note: the harness resolves `--benchmark` via `Path.cwd()`, so a relative path
fails with `ValueError`; an absolute path is required. (Harness bug, not a
measurement problem — see limitations.)

## Measured results (num_cases = 1)

| Level | Top-1 | Top-5 | Top-10 | MRR | Recall@1 | Recall@5 | Recall@10 |
|-------|-------|-------|--------|-----|----------|----------|-----------|
| file    | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| function| 0.0 | 1.0 | 1.0 | 0.5 | 0.0 | 1.0 | 1.0 |

The baseline method (`baseline-stacktrace-lexical`) is deterministic:
`failurelens.evaluation.metrics` contains no randomness, and the retrieval
path has no stochastic components.

## Stack-trace-only baseline status

The harness baseline **is** the stack-trace-only baseline defined by the
evaluation protocol (traceback frames + lexical matching, no embeddings, no
ML). No additional comparison baseline was run; `evaluation/baselines/`
(BM25, embeddings) remains planned and was not implemented here.

## Limitations (required for any Gate B decision)

- **n = 1 case.** `sample_cases.json` exists to exercise the harness end to
  end; these numbers are a reproducibility check, **not** a performance claim
  and not generalizable.
- The single case is also the fixture used by the data pilot (PR #8), so
  file-level 1.0 partly reflects that the baseline can see the exact
  traceback of a case it was demonstrably wired up on.
- Function-level Top-1 = 0.0: the lexical baseline ranks `divide()` (the
  throwing frame) above `average()` (the labelled root cause). This is the
  expected behavior of a stack-trace-only method and is exactly the gap the
  model is meant to close — but with n=1 it is an observation, not evidence.
- No train/eval split is possible with a single repository of cases.

## Gate A conclusion (baseline side)

The evaluation harness runs end-to-end on the target host and produces real,
reproducible numbers in the committed `evaluation/reports/`. Baseline
measurement capability is therefore **feasible**; a generalizable baseline
number requires a multi-repo labelled benchmark (rights-cleared), which is a
Gate B input.
