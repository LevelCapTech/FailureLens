# Model architectures (planned)

Python modules defining model architectures - typically a thin head/wrapper
around a pretrained base encoder loaded via `transformers` / `sentence-transformers`.

Guidelines when adding code here:

- Keep inference and training concerns separate (inference lives in the library
  at `failurelens.inference`; training utilities live under `training/`).
- Depend only on the optional `ml` extra, never on the library's core
  dependencies, so the baseline stays lightweight.
- Pin the base-model revision and avoid `trust_remote_code=True` unless a
  reviewed, pinned revision justifies it.

> Status: **planned.** No architecture code exists yet.
