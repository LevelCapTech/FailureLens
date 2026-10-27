# Validation (planned)

Quality gates for generated/collected records:

- Completeness of required fields (see `datasets/schemas/record.schema.json`).
- **Data-leak** checks: the same repository or bug-fix case must not appear in
  both train and eval splits; variants of one bug stay on the same side.
- Duplicate detection (including near-duplicate code).
- Exclusion of incomplete records **with a recorded reason**.

> Status: **planned.** The checked-in example record is validated structurally by
> `tests/regression/test_dataset_schema.py`.
