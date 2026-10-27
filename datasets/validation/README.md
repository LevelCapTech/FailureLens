# Validation

Quality gates for generated/collected records:

- Completeness of required fields (see `datasets/schemas/record.schema.json`).
- **Data-leak** checks: the same repository or bug-fix case must not appear in
  both train and eval splits; variants of one bug stay on the same side.
- Duplicate detection (including near-duplicate code).
- Exclusion of incomplete records **with a recorded reason**.

> Status: the **semantic quality gates** (beyond JSON Schema) are **implemented**
> in [`failurelens.quality`](../../packages/python/src/failurelens/quality.py):
> `validate_record` / `validate_record_set` enforce the license, candidate,
> rationale, validity, environment, secrets/PII, and duplicate gates from
> `docs/data-contract.md` and record a `GateRejectionReason` for every dropped
> record. The checked-in example record is validated structurally by
> `tests/regression/test_dataset_schema.py` and against the semantic gates by
> `tests/regression/test_dataset_quality_gates.py`. The **data-leak split**
> tooling (non-leaking train/eval splits) remains **planned**.
