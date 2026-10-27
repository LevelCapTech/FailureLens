# Independent evaluation protocol

A **design-gate deliverable** for the AI-model work (issue #3). It defines how any
analyzer — the existing baseline, simpler baselines, and a *future* trained model
— is compared **fairly**, which metrics decide the outcome, and what must be true
before any quality or superiority claim is made. It governs the harness and
metrics described in `docs/evaluation.md` (which **work today**).

> **Independence.** Evaluation is owned by `failurelens-independent-evaluator`.
> No quality/superiority claim is valid without an **independent report** that
> reproduces its numbers from committed data. No trained model exists yet, so no
> such claim is made.

## Candidate parity (apples to apples)

Every method is evaluated under **identical conditions**:

- **Same cases** — the same benchmark set and ground-truth labels.
- **Same candidate sets** — each method ranks the *same* pool of candidate code
  units per case; differences come from ordering, not from who got to see which
  candidates.
- **Same resource conditions** — the same `top_k`, hardware regime (CPU/GPU), and
  time/memory limits, recorded alongside results.

Any deviation is a protocol violation and invalidates the comparison.

## Methods compared

Per `docs/evaluation.md` and `evaluation/baselines/README.md`:

1. **Stack-trace + lexical** — the shipped baseline (`baseline-stacktrace-lexical`).
2. **Stack-trace only** — rank purely by stack-frame depth (ablation of the
   lexical signal).
3. **Lexical search (BM25)** — planned.
4. **Off-the-shelf code embeddings** — planned; subject to each model's license
   (some are non-commercial — never redistribute such weights).
5. **FailureLens trained model** — **once one exists**; today none is implied.

## Metrics

**Primary KPIs** (decide Go/No-Go), reported **where function labels exist**:

- **Function-level Top-1 accuracy.**
- **Function-level MRR** (mean reciprocal rank).

**Secondary measurements** (always reported, never the headline on their own):

- File-level Top-1 / Top-5 (and Top-k) accuracy.
- Recall@k.
- Failure cases (kept and categorized).
- Latency, memory, and model size (require a model backend to compare).

### Missing-label handling

- Function-level metrics are computed **only** for cases that provide function
  labels. **Never** substitute a file label for a missing function label.
- When a label is missing, the corresponding metric is **N/A** for that case —
  not zero-filled and not back-filled from a coarser label.

## Reproducible reporting

Every report (committed under `evaluation/reports/`, produced by the harness —
never hand-written) records:

- **Sample counts** — number of cases evaluated, and how many had function labels.
- **Revisions** — data/benchmark version, code commit, and model id + pinned
  revision (for model backends).
- **Commands** — the exact command(s) used, re-runnable under the same conditions.
- **Resource conditions** — `top_k`, hardware regime, time/memory limits.
- **Uncertainty** — report dispersion across cases (the exact treatment — e.g.
  per-case breakdown and/or confidence intervals — is frozen at the decision
  gate). Do not report a single number without its basis.

## Failure and bias analysis

- **Keep failures.** Do not cherry-pick successes; list and categorize the misses.
- **Slice the results** by repository and error type to expose uneven behaviour
  (e.g. the known bias toward the exception site over the root cause).
- **Leakage check.** Confirm the evaluation repositories/cases are held out from
  any training split (`docs/data-contract.md`).
- **No unrun experiment counts as a success**, and training/evaluation data are
  never mixed.

## Claim governance

A quality or superiority claim may be published **only** when:

1. An **independent report** reproduces the numbers from committed data.
2. The comparison met **candidate parity** and recorded all revisions/commands.
3. The frozen acceptance criteria (`docs/decision-gate.md`) were set **before**
   the trained model's held-out results were inspected.

Until then, keep accuracy/superiority claims out of the README and marketing copy
(the only numbers that may appear are those the harness measures on committed
data — see `docs/evaluation.md`).

## Related

- `docs/evaluation.md`, `evaluation/README.md`, `evaluation/metrics/README.md` —
  what runs today and the metric definitions.
- `docs/requirements.md`, `docs/data-contract.md`, `docs/decision-gate.md` — the
  rest of the design gate.
