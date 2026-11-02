# Hugging Face Jobs publication

`scripts/hf_jobs_release.py` prepares a bounded cloud rebuild and publication of
the existing experimental Stage 2 recipe to `LevelCapTech/failurelens`.
It does not upload a local model checkpoint. The source ISP connection sends
only the build script and 6,405 bytes of compressed, sanitized training features.
The base weights, rebuilt weights, checkpoints and Hub upload remain on Hugging
Face infrastructure.

## Current state

The release script and public synthetic tests are implemented. An offline CPU
export smoke test passed with the existing local v3 artifact. The tracked project
plus the release changes passed `make check` (192 passed, 29 optional ML tests
skipped). This is build verification, not an accuracy evaluation.

The owner explicitly approved sending the 6.4 KB of private derived training
features and saved authentication credential as encrypted Hugging Face Job
secrets, the bounded paid T4 Job, and publication of the generated model files.
The release completed on 2026-11-02. The final
[Job](https://huggingface.co/jobs/LevelCapTech/6ae7d7a8c78863f9c62ff0e8) ended in
`COMPLETED` and logged `hub_reload_verified: true`. An independent anonymous
check confirmed public access, the exact 18-file inventory, every file in the
model manifest and the Hub LFS weight SHA-256. The runnable example's pinned
revision resolves to the same weight checksum. No release Jobs remain active.

| Artifact | Verified value |
|----------|----------------|
| Public model | [LevelCapTech/failurelens](https://huggingface.co/LevelCapTech/failurelens) |
| Final revision | `2426e838cf922a26e22836a4f93b3f135b4f2005` |
| Weights revision used by the example | `ec8ffde17b5cf5a5bf6f206864fdbe985f373440` |
| Weights SHA-256 | `8e8fb971f8c1dce22ae29ea4f682cb8387790ad608e06d3800b62d3413a0a9e4` |
| Weights size | 546,938,144 bytes |

The Git tag `model-v0.1.0-experimental` identifies the publication tooling and
documentation for this Hub revision. The training source remains pinned to
`2985bfcfbf5bf79bc3b3b0589284f6a4ca481970`.

The final Job used a 1088-second limit after conservatively deducting all 712
seconds since the first Job's creation from the approved 1800-second budget,
including scheduling and local repair time. All four Jobs were observed terminal
15 minutes after the first was created. Actual billing remains the provider's
account record; this document records limits and observed states, not an invoice.

Earlier attempts ended in `ERROR`: `6ae7d4e00e0338d1bb7db13e` encountered
OS-managed Python; `6ae7d620c78863f9c62ff035` identified a missing `jsonschema`
dependency; `6ae7d68e0e0338d1bb7db18a` trained successfully but failed the card's
language-code validation before upload. The final script uses an isolated
virtual environment, includes the complete training-package dependencies and
validates the card against the Hub **before** starting a paid Job. Python is a
tag; the natural-language field contains the valid `en` code.

Independent held-out evaluation (Gate C) remains pending. This is an experimental
release; it does not establish model superiority,
production readiness or autonomous agent adoption.

## Cloud build

- Hardware: `t4-small`, one attempt, a 30-minute execution timeout. Live hardware
  pricing was $0.006667/minute when checked; maximum execution cost is about
  $0.20 for 30 minutes, excluding any separate account charges. Retries must
  deduct the already-used budget and confirm previous Jobs are terminal.
- Runtime: PyTorch 2.11.0 / CUDA 12.8 image pinned by registry digest.
- Source: `2985bfcfbf5bf79bc3b3b0589284f6a4ca481970`, with the downloaded archive's
  SHA-256 checked before extraction.
- Base: `nomic-ai/CodeRankEmbed` at
  `3c4b60807d71f79b43f3c4363786d9493691f8b1`, safetensors only.
- Features: the existing `load_stage2_examples` loader verifies the private
  record digest and source commits, reconstructs function text and strips
  generator markers and private traceback paths. Its 6/6 repository split is
  validated again before submission and training. These features are sent only
  in the encrypted `FAILURELENS_TRAIN_PAYLOAD` Job secret.
- Authentication: the saved Hub credential is sent in the encrypted `HF_TOKEN`
  Job secret. Neither secret belongs in command arguments, ordinary environment
  parameters, repository files, model files or logs.
- Training: the existing final-encoder-block-only recipe, seed 42, three epochs.
  This produces a new artifact; different GPU hardware may produce different
  bytes from the existing local v3. Its own checksum and history are recorded.
- Export: original CLS pooling and model configuration, the exact trained
  safetensors, and audited base code with a fail-closed transformers-5 attention
  mask compatibility patch. The training function inherits an 8192-token limit;
  the legacy config's `max_length=512` is not enforced. The effective limit and
  observed input lengths are recorded instead of asserting a 512-token run.

## Run after the transfer is approved

Use a Python environment with the installed core FailureLens package and
`jsonschema==4.26.0` for preparation, and `huggingface_hub==2.4.0` for submission.
The commands may use
different environments; neither local step needs PyTorch or downloads weights.

```bash
python3 scripts/hf_jobs_release.py \
  --prepare /tmp/failurelens-hf-release/training-features.json.gz

python3 scripts/hf_jobs_release.py \
  --submit /tmp/failurelens-hf-release/training-features.json.gz \
  --state /tmp/failurelens-hf-release/job.json
```

The feature file is created exclusively with mode 600. Preparation refuses to
overwrite it. Submission refuses an existing Job state file to prevent a
continuation from accidentally starting another paid Job. Read the recorded
Job ID and inspect that specific Job before deciding whether a retry is needed.
A logging timeout does not mean the remote Job stopped.

## Completion evidence

The worker checks that the assembled model loads offline on CPU, contains every
trained tensor exactly, differs from the base only in the final encoder block,
and produces finite normalized 768-dimensional embeddings. It then uploads the
weights from the cloud, pins the model-card example to the immutable weights
commit, reloads the final Hub revision and compares its embeddings with the
verified local export. It also checks public visibility and the Hub's weight
SHA-256 against the artifact.

Confirm the Job's terminal status and its final verification log. Independently
check the public model revision, weight size and LFS SHA-256, model configuration,
model card, provenance and license notices. Do not call the release complete on
Job submission or upload alone.

Only the model, tokenizer, configs, custom inference code, model card, aggregate
provenance and notices are uploaded. Private training examples, source probes,
raw records, held-out records/labels and optimizer checkpoints are excluded.
The model card documents the standalone SentenceTransformers interface; it does
not claim that the generic FailureLens `from_pretrained` path already supports
the trained repository's custom code or training-specific query transform.
