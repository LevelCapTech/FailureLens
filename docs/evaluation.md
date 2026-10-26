# Evaluation

FailureLens's value depends on being **measurably** useful, not merely on having
a model. This document describes the independent evaluation setup (project
charter §13). **The harness and metrics below work today.**

## What runs today

- A reusable, unit-tested metrics library: `failurelens.evaluation.metrics`.
- A runnable baseline harness: `evaluation/run_baseline_eval.py`.
- A seed benchmark: `evaluation/benchmarks/sample_cases.json` (one reproducible
  demonstration case — **not** a performance claim).

```bash
pip install -e "packages/python[dev]"
python evaluation/run_baseline_eval.py
```

Example output on the seed benchmark (produced by the harness, not hand-written):
file-level Top-1 = 1.0, MRR = 1.0; function-level Top-1 = 0.0, MRR = 0.5,
Top-5 = 1.0. The function-level numbers honestly reflect that the baseline ranks
the exception site (`divide`) above the root cause (`average`).

## Metrics (§13.2)

Implemented in `failurelens.evaluation.metrics` and documented in
`evaluation/metrics/README.md`:

- **File-level** Top-1 / Top-5 (and Top-k) accuracy
- **Function-level** Top-k accuracy
- **MRR** (mean reciprocal rank)
- **Recall@k**

Planned additional measurements: inference latency, memory usage, model size
(these require a model backend to compare).

### Honesty rules for metrics

- Only report location-level metrics for cases that actually have location
  labels.
- **Never** reuse a file-level label as a function-level label.
- Function-level metrics are computed only for cases providing function labels.

## Comparison targets (§13.1)

The point of evaluation is to show whether a trained model beats simpler methods.
Planned comparisons (see `evaluation/baselines/README.md`):

1. Stack-trace-only ranking
2. Lexical search (BM25)
3. CodeRankEmbed (off-the-shelf embeddings)
4. SweRankEmbed-Small (check non-commercial license; do not redistribute)
5. A FailureLens trained model — **once one exists**

## Ablations (§13.3)

Design metrics so the contribution of each signal can be isolated: stack-trace
info, error message, semantic search, call relations, success/failure diff, and
dedicated fine-tuning. Not all are implemented yet, but added features must be
measurable.

## Handling of results (§13.4)

- Record **measured** values only; keep the comparison conditions and the number
  of evaluated cases.
- Do not cherry-pick successes; keep failures.
- Never treat an unrun experiment as a success.
- Never mix training and evaluation data.
- Everything must be re-runnable under the same conditions.

> Until real benchmark results exist, **do not** put accuracy guarantees in the
> README or marketing copy. Committed reports in `evaluation/reports/` are the
> only source of such numbers.

## Adding cases

See `evaluation/benchmarks/README.md`: provide ground-truth
`relevant_files`/`relevant_functions` and a `label_rationale`, keep benchmark
repositories out of any training split, and prefer well-labelled cases over many
weak ones.
