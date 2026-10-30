# Private evaluation input and parity boundary (Issue #21)

`evaluation/private_adapter.py` is an **opt-in library boundary**, not a Gate C
runner. It performs no work on import and is not invoked by the ordinary demo
harness or CI with private inputs. The public tests create synthetic Git
repositories and records at test time. No held-out record, label, generator, or
path is committed.

## Runtime contract

Call `load_private_cases(records_path, expected_sha256, source_manifest_path,
train_manifest_path, top_k=K)` from evaluator-owned code. All paths and the
expected lowercase SHA-256 of the *exact record-set file bytes* are supplied at
runtime. The JSON record set is either an array of canonical records or an
object with a `records` array. It is checked against
`datasets/schemas/record.schema.json` and `failurelens.quality.validate_record_set`
with redistribution screening enabled. **Any** rejected record rejects the
whole set; there is no filtering. Production mode (the default) requires at
least 30 cases and three distinct rights-cleared repositories.

The source manifest is evaluator-owned local JSON. It pins each repository and
**every candidate file's failing-state snapshot for each record**. Each snapshot
is an absolute path to UTF-8 source bytes and a lowercase SHA-256 of the entire
file, not a clean `git show` export. The adapter refuses absent, extra, unreadable,
or digest-mismatched snapshots. The location range is sliced from those verified
failing bytes. The manifest and source files must remain private.

```json
{
  "repositories": {
    "example/repo": {"commit": "<40 lowercase hex>", "rights_cleared": true}
  },
  "snapshots": {
    "example-case": {
      "module.py": {"path": "/private/failing-source.py", "sha256": "<64 lowercase hex>"}
    }
  }
}
```

Every record's `source_repository` must have a manifest entry and its
`source_commit` must equal the pinned commit. The evaluator must independently
establish that the snapshot was captured or deterministically reconstructed from
the failing state and that its digest belongs to the intended record. A file
snapshot hash proves byte integrity after pinning; it does not itself prove the
failure occurred. A candidate with a function name requires a valid start/end
line range. A missing/empty positive source location fails. Rights clearance is
an evaluator-provided assertion, **not** inferred from SPDX or Git metadata;
the evaluator must separately verify it against the approved rights ledger.
The source commit is provenance metadata, not the candidate's source revision.
This library has no clean-source fallback and rejects old source manifests.

`error_log` and optional `stack_trace` are JSON-framed into a single UTF-8
`RankRequest.query` with separate named fields. Missing or null trace is
represented as JSON `null`; an empty string remains empty. This preserves exact
log/trace text without ambiguous separators. Both rankers receive the same
request object and query bytes, bound into the request digest.

The approved train manifest is local JSON:

```json
{"approval_id":"<nonempty approval reference>","training_repositories":["example/train-repo"]}
```

The caller must supply the complete repository set of the **actual approved
training corpus**. An empty list describes no training corpus and cannot prove
non-overlap for a later model. Any exact train/eval repository intersection
rejects the evaluation. The adapter cannot independently authenticate the
approval reference or discover repositories omitted from the train manifest;
those are upstream data-provenance obligations before Gate C.

## Ranker contract

`load_private_cases` returns `PrivateCase` objects for the evaluator. Only
`case.request` is passed to a ranker: a frozen `RankRequest(query, candidates,
top_k, digest)`. Each candidate has a stable `file::function:start-end` ID,
file/function IDs, pinned failing-state source text, and its SHA-256. Labels and rationale
stay in `PrivateCase`, not in `RankRequest` or `Candidate`. A ranker returns
`RankResult(ordered_ids, request_digest, top_k)`; `ordered_ids` must be a tuple
containing **every** candidate exactly once. `compare_rankers` passes the *same*
request object to baseline and model and rejects changed content/query/pool,
missing/extra/duplicate IDs, a mismatched digest or `top_k`, or an invalid
response. Reordering is allowed. The request digest binds the full source text,
not just candidate IDs.

`positive_function_ids is None` means the function label is missing and its
metric is N/A, never zero or a substituted file label. This issue does not
compute any metric or run a model on held-out cases. Three-seed reporting,
confidence intervals, p95/memory/parameter measurements, and Gate C decisions
remain separate tasks under `docs/decision-gate.md`.

Failures raise `PrivateInputError` with a generic reason; no private case ID,
label, source path, source text, ranker exception message, or record detail is
included. Keep caller tracebacks and any future report output private. The
existing regex/entropy secret and PII screens are heuristic and do not replace
manual review.

## Public verification

```bash
python3 -m venv /tmp/failurelens-check
/tmp/failurelens-check/bin/python -m pip install -e 'packages/python[dev,train]'
PATH=/tmp/failurelens-check/bin:$PATH make check PY=/tmp/failurelens-check/bin/python
```

`make check` runs Ruff lint/format, strict mypy on the package, and pytest. The
new contract tests exercise synthetic records only. The older
`evaluation/run_baseline_eval.py` remains a separate one-case public demo and
is not a valid Gate C comparison.
