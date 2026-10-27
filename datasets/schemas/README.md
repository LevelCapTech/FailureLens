# Schemas

Canonical, machine-readable schemas for dataset records.

- `record.schema.json` - JSON Schema (draft 2020-12) for a single
  fault-localization record. It encodes the data-quality fields required by the
  project issue (§10.3): source repository/commit/license, language, injection
  method/location, test command, pre/post execution results, error log, stack
  trace, positive/negative candidates, label rationale, and generation metadata.

The example record in `datasets/fixtures/sample_record.json` conforms to this
schema, and `tests/regression/test_dataset_schema.py` guards against drift.

Key invariants:

- An **injected-bug location is not assumed to be the root cause**; the positive
  label requires a `label_rationale`.
- A valid generated case has `pre_change_result.status == "passed"` and
  `post_change_result.status == "failed"`.
