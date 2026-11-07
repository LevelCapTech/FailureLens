# CLAUDE.md

@AGENTS.md

`AGENTS.md` is the operational contract for this repository. The notes below
only add Claude Code specifics and never override it.

## Working agreements

- Prefer the smallest correct change; keep the offline CPU baseline runnable.
- Run `make check` (ruff, ruff format --check, mypy strict, pytest) before
  proposing a commit. Use `make demo` to smoke-test the CLI on the fixture.
- Do not fabricate models, weights, or metrics. Report only numbers produced by
  the evaluation harness on committed data, and state skipped or failing steps
  plainly.
- Keep held-out labels, full trees, raw traces, weights, and per-case outputs
  out of Git, CI, and PR artifacts.
- Do not publish, release, upload models or datasets, start training runs, or
  fetch new data without explicit approval from a maintainer. Approval for one
  of these does not extend to another.
- Commit or push only when asked. Do not use `git add -A`; stage named files so
  local-only files are never included.
- Before editing, check `git status`: the working tree may contain in-progress
  changes that are not yours.

## Where to look

- Architecture and specs: `docs/architecture.md`; gates and approvals:
  `docs/decision-gate.md`; model status: `docs/model-card.md`.
- Output contract: `packages/python/src/failurelens/schemas.py` and
  `docs/api.md`; CLI exit codes: `docs/cli.md`.
- Tracking (optional MLflow): `docs/experiment-tracking.md`.

## Style reminders

- `from __future__ import annotations` in library modules except `cli.py`.
- ASCII only in source and docs that ruff checks; `X | None`, `StrEnum`.
- Never import torch/transformers at module top level in core paths.
