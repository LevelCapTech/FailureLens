# Security (technical guidance)

This page covers security considerations for *using and extending* FailureLens.
For **reporting a vulnerability**, see the root `SECURITY.md`.

## Threat model in brief

FailureLens reads (never writes) a user's source code and failure logs, and
optionally loads ML models. The main risks are: executing untrusted model code,
pulling vulnerable dependencies, and leaking secrets that appear in logs/data.

## Model loading

- **Pin revisions.** Always load external base models at a pinned `revision`
  (commit SHA), never a floating tag. The API accepts `revision=...` and configs
  must record it.
- **No `trust_remote_code=True`** unless the exact custom code has been reviewed
  at a pinned revision. Loading a model that executes arbitrary repository code is
  a remote-code-execution risk. Default to `false`.
- **Nothing is bundled or auto-downloaded.** The baseline needs no model. Model
  backends download only when the user explicitly requests one, via the `ml`
  extra.

## Dependencies & supply chain

- Core runtime deps are minimal (**pydantic**, **typer**) to shrink attack
  surface.
- The optional `ml` extra enforces **security floors** — keep these minimums or
  raise them, never lower them:
  - `torch>=2.6.0`
  - `transformers>=5.10.0`
  - `sentence-transformers>=5.6.0`
- GitHub Actions are pinned to maintained major versions (`checkout@v4`,
  `setup-python@v5`, `upload-artifact@v4`).
- Before adding or bumping a dependency, check it against the GitHub Advisory
  Database.

## Input handling

- Treat logs and repository contents as untrusted text. The baseline parses them
  with the standard library and does not execute repository code.
- The CLI distinguishes *inconclusive analysis* from *execution failure* via exit
  codes, and never emits a raw stack trace for ordinary bad input.
- Be mindful of memory on very large logs/repositories; indexing skips oversized
  files (`IndexConfig.max_file_bytes`) and excluded directories by default.

## Secrets & personal data

- **Never** commit secrets, credentials, tokens, or personal data — in code,
  fixtures, datasets, or logs.
- Before publishing any dataset or report, verify it contains no secrets or PII
  (see `docs/datasets.md`). Prefer publishing collection/transformation code over
  redistributing uncertain data.

## Licensing (a safety issue too)

- Apache-2.0 covers the **source**. A trained model's license depends on its
  **base model and training data** — determine it before publishing weights.
- Some comparison models (e.g. SweRankEmbed-Small) are **non-commercial**; respect
  their terms and do not redistribute their weights from this repository.

## Checklist for contributors

- [ ] External models loaded at a pinned revision, `trust_remote_code=False`.
- [ ] No new heavy deps in the core package; extras keep security floors.
- [ ] New deps checked against the advisory database.
- [ ] No secrets/PII added anywhere.
- [ ] Licenses verified for any model or data referenced.
