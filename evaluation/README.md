# Evaluation

This directory holds the **independent** evaluation harness for FailureLens. It
measures how well an analyzer localises faults, so that any accuracy claim is
backed by reproducible, real measurements.

> **No fabricated results.** Every number in `reports/` must be produced by
> running the harness. Do not hand-write metrics, and do not claim an analyzer
> is better than another without a committed report that reproduces it.

## Layout

| Path | Purpose | Status |
|------|---------|--------|
| `run_baseline_eval.py` | Runnable harness: evaluates the baseline on a benchmark and prints real metrics. | **Working** |
| `benchmarks/` | Labelled cases (query + ground-truth locations). | Seed case only |
| `metrics/` | Notes on the metric definitions. The reusable implementations live in the library at `failurelens.evaluation.metrics`. | **Working** |
| `baselines/` | Notes/configs for comparison baselines (stack-trace only, BM25, embeddings). | Planned |
| `reports/` | Committed JSON/Markdown reports produced by the harness. | Generated |

### Why the metric code lives in the library

The ranking metrics (`MRR`, `recall@k`, top-k accuracy) are small, pure
functions that are useful at runtime too, so they are implemented and unit
tested inside the installable package (`failurelens.evaluation.metrics`) and
simply *used* by the harness here. This keeps a single tested implementation and
avoids a second import path. (The issue's initial layout suggested a standalone
`evaluation/metrics/` package; this is the documented adjustment.)

## Run it

```bash
pip install -e "packages/python[dev]"

# Evaluate the baseline on the bundled demonstration benchmark.
python evaluation/run_baseline_eval.py

# Use your own benchmark and also write a report file.
python evaluation/run_baseline_eval.py \
  --benchmark evaluation/benchmarks/sample_cases.json \
  --output evaluation/reports/baseline.json
```

The harness only *measures* analyzers. It never edits target repositories and
performs no network access.

## Metrics reported

- **File-level** and **function-level** Top-1 / Top-5 / Top-10 accuracy
- **MRR** (mean reciprocal rank)
- **Recall@k**

Function-level metrics are only reported for cases that actually provide
function labels; file labels are never reused as function labels (see the issue
requirements in `docs/evaluation.md`).

## Adding cases

See `benchmarks/README.md`. Prioritise **label reliability** over quantity, keep
train/eval data split by repository, and record each case's `label_rationale`.
