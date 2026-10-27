# Metrics

The reusable metric implementations live in the installable library at
[`failurelens.evaluation.metrics`](../../packages/python/src/failurelens/evaluation/metrics.py)
and are unit tested in
[`test_evaluation_metrics.py`](../../packages/python/tests/test_evaluation_metrics.py).

This directory is for **documentation** of the metric definitions so the
benchmark reports are unambiguous.

## Definitions

Given a case with a ranked list of predicted identifiers `predicted` (best
first) and a set of ground-truth `relevant` identifiers:

- **Reciprocal Rank (RR)** - `1 / rank` of the first relevant prediction, or `0`
  if none of the predictions are relevant.
- **Mean Reciprocal Rank (MRR)** - the mean of RR over all cases.
- **Hit@k** - `True` if any relevant identifier appears in the top `k`.
- **Top-k Accuracy** - the fraction of cases where Hit@k is true.
- **Recall@k** - `|relevant ∩ top-k| / |relevant|`, averaged over cases.

Identifiers are opaque strings: file paths (`pkg/mod.py`) for file-level
evaluation, or `pkg/mod.py::function` for function-level evaluation.

## Conventions

- When a case has no relevant identifiers, RR and Recall are defined as `0`.
- `k` must be a positive integer.
- Function-level metrics are only computed for cases that provide function
  labels; file labels are never substituted for missing function labels.
