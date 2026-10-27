# Gate A data pilot finding (2026-10-27)

Scope: one in-repository, Apache-2.0 fixture at source commit `61af5a5c04a3eb94b55471c3208a3a5452c18aa2`. This is a manual feasibility check, not an evaluation dataset, performance claim, publication approval, or Gate B threshold.

## Execution and evidence

The existing fixture has a latent empty-input bug. In a temporary copy of `examples/fixtures/sample_project`, a clean state was made by inserting `if not values: return 0.0` immediately before `total = sum(values)` in `calculator/core.py:average`. The manual change removed exactly that guard, restoring the pinned fixture code. Neither source fixture nor other repository code was changed.

From the temporary fixture directory, `python3 app.py` exited 0 and printed `Average score: 0.0` before the change; after it, the same command exited 1 with `ZeroDivisionError: division by zero`. The captured traceback had five frames: `app.py` main and `report_scores`, `calculator/core.py` average and divide, with divide the exception site. `datasets/generated/sample_project_empty_average_gatea_0001.json` stores the observed failure log with only the ephemeral `/tmp/.../sample_project/` prefix normalized to the fixture path. `stack_trace` was manually extracted from those five observed frames. The removal of the guard is the attributable change; the positive cause is `average` and the negative surfacing site is `divide`. The rationale is in the record.

The canonical schema has no `license`, `source_type`, or `provenance` fields. Those are recorded in `datasets/generated/pilot_manifest.json` (`CC-BY-4.0`, `synthetic`, fixture/commit provenance); the canonical record retains `source_license: Apache-2.0`. This separation is required to keep the record schema-conformant. `CC-BY-4.0` describes the new annotation/record metadata, not a relicensing of fixture source code. No external data was fetched.

## Verification (observed)

Run from repository root: `python3 datasets/generated/validate_pilot.py`.

- Structural schema check: **1 passed, 0 rejected, 0 errors**. The standalone stdlib checker checks required/unknown fields, nested references, types, enum, and minimum constraints for this record. It is not a full JSON Schema implementation.
- Existing `failurelens.quality.validate_record_set([record])`: **1 passed, 0 rejected, no rejection reasons**. The module was loaded directly from source because the core package dependency was unavailable on this execution node; the existing validation function itself ran unchanged. Secret/PII gates are heuristic screeners, not proof of absence.
- Repository split: one source repository assigned to `train`; **train 1 repository, eval 0, overlap 0**. No evaluation set can be formed from one fixture without repository leakage. The manifest does not authorize training.
- Rights ledger: the in-repo Apache-2.0 fixture remains the only **Usable** source in `docs/data-contract.md`; SWE-smith, SWE-bench, OSS bug-fix histories, and rights-cleared-repo mutation remain **Pending**. `datasets/README.md` now links the pilot artifact while preserving this boundary.

`tests/regression/test_dataset_schema.py` was **not run** because `pytest` was not installed on this node; the equivalent targeted structural check above was run. Full `make check` was not run. The traceback parser in `failurelens.parsers` was not invoked because package imports require unavailable core dependencies; extraction was manual from the observed log.

## Constraints for next gate

This is one synthetic case from one repository, not a held-out sample and not evidence of generalizable label quality or model performance. It cannot estimate duplicate behavior across repositories, independent split feasibility, or realistic source/license diversity. Existing semantic screening cannot establish that a rationale is causally correct or guarantee secret/PII absence; the recorded label was manually reviewed. There is no new collection pipeline, automated injection, scale-up, model training, or Gate B numeric freeze in this pilot. A future independent evaluation set requires additional rights-cleared repositories and separate evaluation-label custody.
