# Datasets

Tooling and schemas for building **fault-localization training/evaluation data**.
This is a core research area for FailureLens.

> **Status: design + schemas.** This directory defines *how* data is collected,
> generated, validated and recorded. It intentionally ships **no** large data
> and **no** fabricated labels. The data-generation pipeline is built up
> incrementally; nothing here downloads data or runs heavy jobs automatically.

> **Label reliability over quantity.** Prefer fewer, trustworthy labels. A bug
> *injected* at one location is not necessarily the *root cause* of the observed
> failure - label semantics must be explicit (see §10.4 of the project issue).

## Layout

| Path | Purpose |
|------|---------|
| `sources/` | Notes on candidate data sources and their license/redistribution terms. |
| `collection/` | Scripts/notes to collect real bug-fix data (commits, issues, tests). |
| `generation/` | Pipeline to generate failures (e.g. mutation / bug injection) and capture logs. |
| `preprocessing/` | Normalisation, tokenisation, candidate extraction, de-duplication. |
| `validation/` | Quality gates: completeness, leakage checks, duplicate detection. |
| `schemas/` | The canonical record schema (`record.schema.json`) and field docs. |
| `fixtures/` | Tiny checked-in examples that conform to the schema (for tests/docs). |

## Data generation pipeline (intended)

Per the issue §10.2, a generated case is only valid when the target test
**passes before** and **fails after** the injected change, and the failure is
attributable to that change (not to install/network/environment errors):

1. Fetch target repository, pin the commit.
2. Run the target test(s) on the clean state - must pass.
3. Inject a change (e.g. mutation) at a recorded location.
4. Re-run the target test(s) - must now fail.
5. Collect the failure log and extract the stack trace.
6. Record the injection location and the candidate code units.
7. Produce positive/negative candidates with an explicit label rationale.
8. Validate quality, detect duplicates, and split by repository.
9. Save metadata (tool version, timestamps, licenses).

## Data-leak avoidance

Split by **repository** (and by bug-fix case) so the same project or fix never
appears in both train and eval. Variants generated from the same bug must stay on
the same side of the split. Never tune on the evaluation set.

## Publishing

Only publish data whose source license permits redistribution. Verify there are
no secrets, credentials, or personal data. Publishing just the *collection and
transformation code* (not the data itself) is an acceptable option.

See `schemas/README.md` for the record format and `docs/datasets.md` for the full
specification.
