# Validation

Quality gates for generated/collected records:

- Completeness of required fields (see `datasets/schemas/record.schema.json`).
- **Data-leak** checks: the same repository or bug-fix case must not appear in
  both train and eval splits; variants of one bug stay on the same side.
  Enforced by `check_splits.py`; the record-id check is per exact id.
- Duplicate detection (including near-duplicate code).
- Exclusion of incomplete records **with a recorded reason**.

> Status: the **split-leakage check** is implemented in
> [`failurelens.split_leakage`](../../packages/python/src/failurelens/split_leakage.py)
> and run offline with `python datasets/validation/check_splits.py --split train=DIR --split eval=DIR`.
> It detects repositories and record ids shared across splits; it does not assign splits.
>
> The **semantic quality gates** (beyond JSON Schema) are **implemented**
> in [`failurelens.quality`](../../packages/python/src/failurelens/quality.py):
> `validate_record` / `validate_record_set` enforce the license, candidate,
> rationale, validity, environment, secrets/PII, and duplicate gates from
> `docs/data-contract.md` and record a `GateRejectionReason` for every dropped
> record. The checked-in example record is validated structurally by
> `tests/regression/test_dataset_schema.py` and against the semantic gates by
> `tests/regression/test_dataset_quality_gates.py`. The **data-leak split**
> tooling (non-leaking train/eval splits) remains **planned**.
