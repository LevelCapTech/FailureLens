# Benchmarks

Labelled fault-localization cases used by the evaluation harness
(`evaluation/run_baseline_eval.py`).

## File format

A benchmark is a JSON file:

```json
{
  "schema_version": "0.1.0",
  "cases": [
    {
      "case_id": "unique-id",
      "repository": "path/relative/to/repo/root",
      "error_log": "path/relative/to/repo/root",
      "relevant_files": ["pkg/mod.py"],
      "relevant_functions": ["pkg/mod.py::function_name"],
      "label_rationale": "Why these locations are the ground-truth root cause."
    }
  ]
}
```

- `relevant_files` / `relevant_functions` are the **ground-truth** locations. A
  prediction is correct if it matches one of them.
- Omit `relevant_functions` when the function-level root cause is unknown. The
  harness then skips function-level scoring for that case instead of reusing the
  file label.
- `label_rationale` is required for every case: state *why* the label is the
  root cause, not merely where the exception surfaced.

## Current contents

- `sample_cases.json` - a single reproducible demonstration case built from
  `examples/fixtures/sample_project`. This is **not** a performance claim; it
  exists to exercise the harness end to end and to document the expected output
  format.

## Adding real cases

When adding cases derived from real repositories:

1. Verify the source repository's license permits redistribution of any quoted
   code or logs (see `datasets/README.md`).
2. Keep a case's repository out of any training split (data-leak avoidance).
3. Record the exact commit and test command used to produce the failure.
4. Prefer fewer, well-labelled cases over many weakly-labelled ones.
