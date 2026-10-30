# Shipped-baseline common pool bridge (Issue #28)

This is evaluator-owned, opt-in code. It does **not** run Gate C, publish a model,
change the frozen thresholds, or replace the shipped analyzer. The comparison is
**v3 reranking shipped-baseline retrieval**. It is not a claim about v3
end-to-end retrieval. Keep all held-out inputs, full trees, mutation recipes,
weights, raw traces, and per-case outputs outside Git, CI, and PR artifacts.

## Frozen provenance and method

- Merge base: `6c9c1c030fb5f472a9a9e4882b62050c9cc1bfca` (merged PR #27).
- The v3 bridge pins artifact SHA-256
  `506492b034fadb3d7ee6e7288d04b66dfa477c25b1de41d9d70166353cd343d4`.
  The artifact and audited base-model cache are supplied locally at runtime;
  neither is included here.
- `load_product_cases(...)` first enforces the existing exact-record digest,
  schema/quality gates, candidate-snapshot checks, rights assertion, 30-case /
  three-repository minimum, and train/eval repository disjointness. The
  evaluator must independently authenticate the rights and actual training
  corpus manifests; their assertions alone are not proof.
- A separate local tree manifest must cover **every** record, with a pinned
  source commit, an absolute failing-tree root and base Git repository, the
  SHA-256 of **every Python file the default product indexer would enumerate**,
  and a deterministic single-function text replacement. The verifier compares
  the exact indexed path set against the commit tree and failing root. All
  unchanged files must have the committed Git blob bytes; the changed file
  must be the exact result of the mutation recipe applied to the last matching
  top-level function in that commit, as in the approved synthetic generator.
  Missing/extra indexed files, stale hashes, indexed symlinks, oversized or
  unreadable file bytes, mutation mismatches, and unexpected index warnings
  fail closed. A committed non-UTF-8 Python file is still byte-attested but is
  skipped by the shipped indexer; its exact expected warning is allowed. This
  is product behavior, not a missing-tree fallback. Non-indexed files are
  outside the comparator's code pool and are not attested. Source archives
  with external non-Python symlinks may omit those links from the local copy;
  this limitation must be disclosed in the private provenance review.
- Manifest shape (placeholder strings only):

```json
{
  "cases": {
    "synthetic-case": {
      "base_repository": "/local/approved/source-clone",
      "root": "/local/complete-failing-tree",
      "commit": "<40 lowercase hex>",
      "indexed_files": {"module.py": "<64 lowercase hex>"},
      "mutation": {
        "file": "module.py", "function": "cause",
        "old": "synthetic old text", "new": "synthetic new text"
      }
    }
  }
}
```

The evaluator constructs and safeguards this manifest and trees locally; the
code does not reconstruct or download them automatically. File labels may be
source-root-relative aliases: each must map to **exactly one** full-tree indexed
path by exact path or suffix. Ambiguous/missing mappings, missing/ambiguous
function spans, duplicate output IDs, or a changed pool reject the whole run.
The original candidate-snapshot manifest remains a second independent check
on label source. A generic redacted exception never includes record content.

`FailureLens.from_baseline().diagnose` receives each **unmodified `error_log`**
and the verified complete failing tree. The evaluator builds the normal product
index; `top_k` is raised only to at least the parsed frame count to collect
all retrieved candidates without truncation. It stores the actual product
order in a label-blind `ProductBaseline`. Each output is mapped to the unique
indexed function span (or file fallback) and SHA-pinned source text. The exact
same immutable `RankRequest` candidate tuple, IDs, source text/digests, and
query framing go to both rankers. The v3 bridge uses its frozen Stage 2
sanitized `error_log` feature, **not** the product's raw log+trace parser; the
product baseline consumes the raw log. No labels or rationale enter either
ranker. The v3 output must be a permutation of this retrieved pool.

A genuine empty product retrieval remains one case on the original denominator:
both methods record a zero function/file miss where a label exists. The model
is not called for an empty pool. Positive causes outside the retrieved pool
score zero, not N/A. Only a genuinely missing function label is function-level
N/A. Aggregate counts include abstentions and positive-file/function absence;
retrieved-pool recall is reported separately from ranking accuracy. The
existing three seeds, paired bootstrap, Top-1/MRR/deltas, latency/resource
budgets, and frozen thresholds are unchanged. The baseline factory must be
passed explicitly to `run_independent`; its old default supplied-pool lexical
adapter remains for historical tests and is **not** an eligible shipped-baseline
comparator.

## Gate C hold

Public synthetic tests and `make check` validate the bridge only. The
independent evaluator must review complete failing-tree provenance, rights,
training/eval disjointness against the *actual* v3 training corpus, all-case
mapping and pool parity, and a public synthetic model smoke before one bounded
three-seed held-out run. Coordinator review is required before that run. Any
incomparable case blocks the whole score: no filtered subset or KPI claim.
No output here is an independent evaluation or adoption/publication decision.
