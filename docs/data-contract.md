# Data contract — sources, rights matrix, and pilot

A **design-gate deliverable** for the AI-model work (issue #3). It specifies the
data the project may use, the rights each source carries, a small rights-cleared
**pilot**, and the **semantic quality gates** that JSON Schema validation alone
cannot enforce. It extends `docs/datasets.md` (the full dataset specification).

> **Status: proposed design, awaiting human approval.** No large data and no
> fabricated labels ship in this repository, and **no unapproved data is
> used**. The generation/collection pipeline is planned (`datasets/`
> subdirectory READMEs). The **semantic quality gates** below are now
> implemented in code (`failurelens.quality`, issue #5); the remaining
> collection/pilot work stays in separately scoped `[IMPLEMENT]` issues.

## Principles

- **Rights first.** Inventory each source's license/terms, training permission,
  redistribution permission, and log rights **before** any use.
- **Label reliability over quantity.** An injected-bug location is not
  automatically the root cause; every positive label carries a rationale.
- **Reproducibility.** Pin the source commit and the exact test command; record
  pre/post execution results and tool/source versions.
- **No forced downloads.** Data work is opt-in and never runs in CI.

## Data source / rights matrix

Candidate sources and the rights that must be confirmed before use. SWE-smith and
SWE-bench are **candidates, not pre-cleared datasets**. "Pending" means the right
has not yet been verified; it must be resolved to an explicit value (with a
citation) before that use. Records with `source_license: "unknown"` may be
quarantined for rights investigation only; they are not usable pilot, baseline,
evaluation, or training data.

| Source | Repo / version | License / terms | Training use | Redistribution | Log rights | Status |
|--------|----------------|-----------------|--------------|----------------|-----------|--------|
| In-repo fixtures (`examples/fixtures/sample_project`) | this repo, pinned | Apache-2.0 | Yes | Yes | N/A (our logs) | **Usable** |
| SWE-smith | pin before use | Verify (per source) | Pending | Pending | Pending | **Pending** |
| SWE-bench | pin before use | Verify (per source) | Pending | Pending | Pending | **Pending** |
| OSS bug-fix histories / test-fix commit pairs | per upstream repo | Per upstream license | Pending | Pending | Pending | **Pending** |
| Mutation / bug-injection on rights-cleared repos | pin before use | Inherit source repo | Depends | Depends | Depends | **Pending** |

A fix patch is **not** a complete root-cause label, and many upstream cases lack
real failing logs — see `datasets/sources/README.md`. Per-source redistribution
terms differ even when training use is allowed.

## Rights-cleared pilot (feasibility first)

Before any scale-up, run a **small pilot** using only sources whose status is
**Usable** (rights confirmed). The pilot proves the end-to-end pipeline and label
discipline on a handful of cases; it does **not** set acceptance thresholds and is
not an authorization to publish data.

1. Select rights-cleared repositories; pin each commit and its test command.
2. Run the target test(s) on the clean state — must **pass**.
3. Introduce/locate a change; re-run — must **fail**, attributable to the change
   (not to install/network/environment errors).
4. Capture the failure log; extract the stack trace (reuse
   `failurelens.parsers` / `failurelens.indexing`).
5. Record positive/negative candidates with an explicit `label_rationale`
   (root cause vs surfacing site).
6. Apply the **semantic quality gates** below; drop rejected records with a
   recorded reason.
7. Split by repository; hold the evaluation labels apart from model development.

Pilot **size and repository counts are intentionally not fixed here** — they are
proposed after feasibility and frozen with human approval at the decision gate
(`docs/decision-gate.md`). Do not invent them.

## Record requirements

Every record conforms to `datasets/schemas/record.schema.json` and pins:
`source_repository`, `source_commit`, `source_license`, `test_command`,
`pre_change_result`, `post_change_result`, `error_log`, `candidates`,
`label_rationale`, `generated_at`, `generator_version`. A conforming example is
`datasets/fixtures/sample_record.json`.

## Semantic quality gates (beyond JSON Schema)

JSON Schema validation is **necessary but not sufficient**: the current schema
accepts `source_license: "unknown"` and an **empty** `candidates` array, and does
not require a non-empty rationale. The following semantic gates must therefore be
enforced in addition to structural validation. (Specified here; now **implemented**
in `failurelens.quality` (issue #5) via `validate_record` / `validate_record_set`,
alongside the structural check in `tests/regression/test_dataset_schema.py`.)

| Gate | Rule | Rejection reason recorded |
|------|------|---------------------------|
| License | `source_license` must be a verified, known SPDX id (not `unknown`) for pilot, baseline, evaluation, or training use; for any **redistributed** data it must also be on a redistribution allowlist. `unknown` records may only be quarantined for rights investigation and are rejected as `license-unknown`. | `license-unknown` / `license-not-redistributable` |
| Candidates | At least one `positive` **and** at least one `negative` candidate; the array is non-empty. | `no-positive` / `no-negative` / `empty-candidates` |
| Rationale | `label_rationale` is non-empty and explains the **root cause** vs the surfacing site. | `empty-rationale` |
| Validity | `pre_change_result.status == "passed"` **and** `post_change_result.status == "failed"`, with failure attributable to the change. | `not-pass-before-fail-after` / `env-failure` |
| Environment | Reject install/network/environment failures (not caused by the change). | `env-failure` |
| Secrets / PII | Screen `error_log`, `stack_trace`, and any quoted source for secrets, credentials, tokens, and personal data; reject on any hit. | `secret-detected` / `pii-detected` |
| Duplicates | Reject near-duplicate records (including near-duplicate code). | `duplicate` |

## Non-leaking splits

Split by **repository** and by **bug-fix case**; variants derived from one bug
stay on the **same** side. Hold the **evaluation labels apart** from model
development and never tune on the evaluation set. See `datasets/validation/`.

## Secrets and personal data

Never commit secrets, credentials, tokens, or personal data in code, fixtures,
datasets, or logs. The secrets/PII gate above screens candidate records; records
that fail are rejected, not scrubbed-and-kept unless the scrub is verified.

## Versioning

Record the source repository + pinned commit, the test command, the generator
tool version, `dataset_id`/version, and generation timestamps on every record, so
any case is reproducible and attributable.

## Publishing

Only publish data whose source license **permits redistribution** and that passes
the secrets/PII gate. When terms are unclear, publish the **collection/
transformation code** rather than the data. This gate authorizes **no** dataset
release (`docs/decision-gate.md`).

## Related

- `docs/datasets.md`, `datasets/README.md`, and the `datasets/` subdirectory
  READMEs — the full dataset specification and directory plan.
- `docs/requirements.md`, `docs/evaluation-protocol.md`, `docs/decision-gate.md` —
  the rest of the design gate.
