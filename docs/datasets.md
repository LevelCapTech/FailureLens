# Datasets

This document specifies how FailureLens builds fault-localization data. It
reflects the project charter (§10). **Status: design + schemas.** No large data
and no fabricated labels ship in this repository.

The approved **data contract** — the source/rights matrix, the rights-cleared
pilot, and the **semantic quality gates** that JSON Schema alone cannot enforce —
is in [`docs/data-contract.md`](data-contract.md) (a design-gate deliverable).

## Principles

- **Label reliability over quantity.** A bug *injected* at a location is not
  automatically the *root cause* of the observed failure. Every positive label
  carries an explicit `label_rationale`.
- **Reproducibility.** Each record pins the source repository + commit and records
  the exact test command and pre/post execution results.
- **No forced downloads.** Data generation is a separate, opt-in process; it never
  runs in CI and never downloads during normal use of the library.

## Record schema

The canonical record format is a JSON Schema at
`datasets/schemas/record.schema.json` (draft 2020-12). Required fields include:

`record_id, dataset_id, source_repository, source_commit, source_license,
language, test_command, pre_change_result, post_change_result, error_log,
candidates, label_rationale, generated_at, generator_version` — plus optional
`injection_method, injection_location, has_fix_patch, stack_trace`.

`candidates[]` carry a `location` and a `label` of `positive` (confirmed cause)
or `negative` (non-cause). A conforming example is in
`datasets/fixtures/sample_record.json`, kept in sync by
`tests/regression/test_dataset_schema.py`.

### Validity rule for generated cases

A generated case is valid only when:
`pre_change_result.status == "passed"` **and**
`post_change_result.status == "failed"`, and the failure is attributable to the
change — **not** to install/network/environment errors.

## Generation pipeline (planned, §10.2)

1. Fetch target repo; pin the commit.
2. Run target test(s) on the clean state — must pass.
3. Inject a change at a recorded location.
4. Re-run — must now fail.
5. Collect the failure log; extract the stack trace.
6. Record injection location + candidate code units.
7. Produce positive/negative candidates with rationale.
8. Validate quality; detect duplicates; split by repository.
9. Save metadata (tool version, timestamps, licenses).

Where possible, reuse the library's tested components (`failurelens.parsers`,
`failurelens.indexing`) rather than re-implementing parsing/indexing.

## Candidate sources (verify terms first)

- **SWE-smith** — synthetic fix tasks from real repos
  ([repo](https://github.com/SWE-bench/SWE-smith),
  [dataset](https://huggingface.co/datasets/SWE-bench/SWE-smith)). Not every
  instance has a failing test or obtainable stack trace.
- **SWE-bench** — real issues/patches/history
  ([repo](https://github.com/SWE-bench/SWE-bench)). A fix patch is not a complete
  root-cause label, and many cases lack real logs.
- Open-source bug-fix histories, test-failure/fix commit pairs, mutation data,
  and license-cleared public bug datasets.

## Data-leak avoidance (§10.4)

Split by **repository** and by **bug-fix case** so the same project/fix never
appears in both train and eval. Variants of one bug stay on the same side. Never
tune on the evaluation set.

## Publishing (§10.5)

Only publish data whose source license permits redistribution. Verify there are
no secrets, credentials, or personal data. Publishing only the *collection and
transformation code* (not the data) is an acceptable option when terms are
unclear.

See `datasets/README.md` and the subdirectory READMEs for the directory-by-
directory plan.
