# Issue #19 Stage 2 — Production Results

**Date**: 2026-10-30
**Node**: rtx5060ti (`fd91961...`)
**Branch**: `feature/issue19-stage2-coderankeembed-finetune`
**git SHA**: `49d6a89294ceb6fdce01acdea6fec46b2824e899` (origin/main as of PR #20)

## Summary

Stage 2 production implementation: reproducible CodeRankEmbed fine-tuning
experiment using actual source code from mutated failing snapshots.
Checkpoint/resume verified exactly deterministic. Private safetensors artifact
saved locally; no weights pushed to Git or HF Hub.

## Data Reconstruction Proof

Source code reconstructed from isolated copies of private source repos at
audited commits. Positive texts contain the injected mutation (`# issue17
injected fault` marker); negative texts are clean functions. Verified by
integration test (`test_positive_contains_mutation_marker`,
`test_positive_differs_from_negative`, `test_no_label_rationale_in_features`).

See `docs/issue19-stage2-method.md` for full reconstruction method.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `nomic-ai/CodeRankEmbed` @ `3c4b608...` |
| Dataset | `failurelens-train-issue17-v2` (SHA256: `74909f51...`) |
| Dataset split | repo-level: click → train (6), attrs → val (6) |
| Trainable params | 9,440,256 / 136,731,648 (last NomicBertBlock) |
| Epochs | 3 |
| Batch size | 2 |
| Learning rate | 2e-5 (AdamW + CosineAnnealingLR) |
| Weight decay | 0.01 |
| InfoNCE temperature | 0.07 |
| Seed | 42 |

## Training Results (2026-10-30, git SHA `49d6a89`)

| Epoch | Train loss | Val loss |
|-------|-----------|---------|
| 1 | 0.4644 | 0.5560 |
| 2 | 0.3242 | 0.5439 |
| 3 | 0.3675 | 0.5400 |

- Elapsed: 3.5 s
- Peak GPU memory: 894 MiB / 16,311 MiB available (rtx5060ti 16 GB)

## Checkpoint/Resume Verification

Resumed from `checkpoint_epoch_2` → ran only epoch 3.

| Metric | Full run | Resumed |
|--------|---------|---------|
| Epoch 3 train loss | 0.3675250709056854 | 0.3675250709056854 |
| Epoch 3 val loss | 0.5400200486183167 | 0.5400200486183167 |
| Final artifact SHA256 | `7520d7aa...` | `7520d7aa...` |

**Result**: exact match at float64 precision. The checkpoint saves/restores model
weights (safetensors), optimizer state, scheduler state, and RNG states
(torch CPU + CUDA).

## Private Artifact

Saved to `~/.local/share/failurelens/experiment-issue19-stage2/model_final.safetensors`

| Item | Value |
|------|-------|
| SHA256 | `7520d7aa182cf00bae8e9d2bae948590386c73c5b451fedddd744e2ba9f2ae8b` |
| Format | safetensors |
| Not pushed to Git | ✓ (gitignored: `*.safetensors`) |
| Not pushed to HF Hub | ✓ |
| Eval gate B not accessed | ✓ |

## make check Results

```
ruff check .                                          0 errors
ruff format --check .                                 0 would be reformatted
mypy (strict, packages/python/src/failurelens)        Success: 29 source files
pytest (full suite, no --run-ml-integration)          130 passed, 22 skipped
pytest tests/training/test_stage2_finetune.py \
    --run-ml-integration                             14 passed
```

## Caveats and Limitations

- **Tiny synthetic dataset**: 6 training examples from one repository (pallets/click)
  and 6 validation examples from another (python-attrs/attrs). Metrics reflect
  fitting this synthetic set; generalisation to real-world repos is unverified.
- **Val loss does not decrease**: slight val decrease (0.5560 → 0.5400) is
  expected with a 6-example val set; not statistically meaningful.
- **Repo-level leakage avoidance**: the repo-level split prevents within-repo
  leakage but not cross-repo distribution shift (click vs attrs).
- Gate C quality claims are NOT made. This is a reproducible private experiment;
  a larger held-out evaluation is out of scope for this PR.

## Security Floor Maintained

| Check | Result |
|-------|--------|
| Model pin | `3c4b60807d71f79b43f3c4363786d9493691f8b1` (unchanged) |
| trust_remote_code allowlist | (model_id, revision) pair only; local paths rejected |
| safetensors-only loader | `.bin` files rejected at load time |
| safetensors-only checkpoints | model weights saved as `.safetensors` |
| Eval gate B | Not accessed |
| Weights in Git | Not committed |
| Weights on HF Hub | Not uploaded |
