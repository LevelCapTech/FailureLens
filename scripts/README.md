# Scripts

Convenience entry points for working with this repository.

For everyday tasks prefer the root `Makefile` (`make help` lists targets) and the
runnable examples under `examples/`. The baseline evaluation harness lives at
`evaluation/run_baseline_eval.py`.

`hf_jobs_release.py` explicitly prepares and submits a bounded Hugging Face Job
that rebuilds and publishes the experimental encoder without uploading a local
checkpoint. See [`docs/hf-jobs-publication.md`](../docs/hf-jobs-publication.md)
for data-transfer approval, costs, provenance and verification.

`export_hf_model.py` is a local-only exporter pinned to the historical local v3
artifact. Its draft card is not the published cloud model card.
