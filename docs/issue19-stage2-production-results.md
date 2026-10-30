# Issue #19 Stage 2 — Production Results (v2: marker-leakage fix)

**Date**: 2026-10-30
**Node**: rtx5060ti (`fd91961...`)
**Branch**: `feature/issue19-stage2-coderankeembed-finetune`
**git SHA**: to be recorded at commit (see section below)

## Summary

Stage 2 v2: marker-leakage fix applied.  Coordinator review (2026-10-30) found
that the `# issue17 injected fault` generator marker was present in both positive
code and error-log queries, making the task trivially solvable from the marker
alone.  This release removes the marker from all three training features (query,
positive, negative) and re-runs the full experiment + checkpoint/resume proof.

Old artifact (`experiment-issue19-stage2`, SHA256 `7520d7aa...`) is preserved at
its original path and must NOT be used in Gate C evaluation.

## Leakage Fix Summary

| Location | Before | After |
|----------|--------|-------|
| Positive code (`_extract_mutated_function_text`) | Injected line ended with `  # issue17 injected fault` | Injected line has no generator comment |
| Query (`error_log`) | Raw error_log including marker in traceback | `_strip_generator_marker()` applied |
| Negative code | Already clean | `_strip_generator_marker()` applied (defensive) |
| `_reconstruct_injection_statement` | Used marker to locate raise stmt; returned clean stmt | Same: marker used internally only, stripped before returning |

## Security Fix: `weights_only=True`

`_load_checkpoint` in `finetune.py` now loads `training_state.pt` with
`weights_only=True` (PyTorch 2.11 confirmed compatible for optimizer/scheduler/
RNG state dicts).  Cross-run resume is rejected by validating model_revision,
dataset_sha256, and config before any state is restored.

## Data Reconstruction Proof

Source code reconstructed from isolated copies of private source repos.
Positive texts contain the injected mutation (marker stripped);
negative texts are clean functions with marker stripped.

Verified by integration tests:
- `test_no_generator_marker_in_positive` (12 cases)
- `test_no_generator_marker_in_query` (12 cases)
- `test_no_generator_marker_in_negative` (12 cases)
- `test_positive_reconstruction_digest` (byte-identical independent reconstruction)
- `test_positive_differs_from_negative`
- `test_no_label_rationale_in_features`

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

## Training Results (2026-10-30, v2 — no marker leakage)

| Epoch | Train loss | Val loss |
|-------|-----------|---------|
| 1 | 0.4799 | 0.5963 |
| 2 | 0.3721 | 0.5872 |
| 3 | 0.4192 | 0.5843 |

- Elapsed: 3.5 s
- Peak GPU memory: 891 MiB / 16,311 MiB available (rtx5060ti 16 GB)
- Git SHA: `1c1516ee9f10575fedd9d1b04135ff5018ad9757` (pre-commit; updated after push)

Note: loss values differ from v1 because the marker shortcut is no longer
available to the model.  The model must learn from actual code similarity.

## Checkpoint/Resume Verification

Resumed from `checkpoint_epoch_2` in v2 run → ran only epoch 3.

| Metric | Full run | Resumed |
|--------|---------|---------|
| Epoch 3 train loss | 0.4191608130931854 | 0.4191608130931854 |
| Epoch 3 val loss | 0.5842865606149038 | 0.5842865606149038 |
| Final artifact SHA256 | `e91955b4...` | `e91955b4...` |

**Result**: exact match at float64 precision.

## Private Artifact (v2)

Saved to `~/.local/share/failurelens/experiment-issue19-stage2-v2/model_final.safetensors`

| Item | Value |
|------|-------|
| SHA256 | `e91955b4563e0439d1d4a27826f49b268df7912f5e142bbdf970a2e77bcac8e9` |
| Format | safetensors |
| Not pushed to Git | ✓ (gitignored: `*.safetensors`) |
| Not pushed to HF Hub | ✓ |
| Eval gate B not accessed | ✓ |

Old v1 artifact (`experiment-issue19-stage2/model_final.safetensors`,
SHA256 `7520d7aa...`) preserved at original path; not to be used in Gate C.

## make check Results (v2)

```
ruff check .                                          0 errors
ruff format --check .                                 0 would be reformatted
mypy (strict, packages/python/src/failurelens)        Success: 29 source files
pytest (full suite, no --run-ml-integration)          130 passed, 25 skipped
pytest tests/training/test_stage2_finetune.py \
    --run-ml-integration                             17 passed
```

## Caveats and Limitations

- **Tiny synthetic dataset**: 6 training examples from one repository (pallets/click)
  and 6 validation examples from another (python-attrs/attrs). Metrics reflect
  fitting this synthetic set; generalisation to real-world repos is unverified.
- **Val loss does not decrease meaningfully**: slight val decrease is expected with
  a 6-example val set; not statistically meaningful.
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
| `weights_only=True` on resume | ✓ (PyTorch 2.11 compatible; cross-run rejected) |
| Generator marker in features | ✗ (stripped from query/positive/negative) |
| Eval gate B | Not accessed |
| Weights in Git | Not committed |
| Weights on HF Hub | Not uploaded |
