# Issue #19 Stage 2 — Production Results

## v3: full generator signal sanitization (Gate C candidate)

**Date**: 2026-10-30
**Node**: rtx5060ti (`fd91961...`)
**Code git SHA**: `795b0a1c6f94198d93ebd36e45d3d61e8574549e`
**Branch**: `feature/issue19-stage2-coderankeembed-finetune`

### What changed (v2 → v3)

Independent steward audit (PR#25 issuecomment-6513753530) found remaining
generator-specific signals in the 4 `raised exception` records:

| Signal type | Affected features | Count |
|-------------|-------------------|-------|
| Private absolute path in `File "..."` traceback line | query | 4 |
| Generator-specific exception message `"injected X fault"` | query + positive | 4 each |
| `_find_function_node` matched by name only (not line range) | (correctness) | — |

Fixes in `data_stage2.py`:
- `_strip_generator_marker` extended to also (a) strip private absolute paths
  from traceback `File "..."` lines (keeps source-relative path), (b) replace
  `"injected X fault"` exception messages with `""` preserving exception type,
  (c) strip `ExceptionType: injected X fault` from traceback summary lines.
- `_find_function_node` now takes `start_line + end_line`, disambiguates
  duplicate function names by line, and raises on ambiguity or mismatch.
- `_verify_source_commits()` added: checks each record's `source_commit`
  against the pinned source repo HEAD before reconstruction.

Fixes in tests:
- 6 new `TestFindFunctionNodeUnit` tests (always run, no ML gate).
- 6 new fail-closed 12-case ML integration tests for private paths,
  injected exception messages, and source commit verification.
- Total ML integration tests: 29 (was 17).

### Aggregate signal counts (v3, all 12 examples × 3 features = 36 texts)

| Signal | Count in features |
|--------|-------------------|
| `# issue17 injected fault` marker | 0 |
| Private path `/failurelens/train-issue17-v2/sources/` | 0 |
| `injected ... fault` exception message | 0 |

### Training results (v3, seed=42, rtx5060ti, code SHA `795b0a1`)

| Epoch | Train loss | Val loss |
|-------|-----------|---------|
| 1 | 0.7298 | 0.7346 |
| 2 | 0.4976 | 0.7241 |
| 3 | 0.5162 | 0.7203 |

- Elapsed: 3.5 s
- Peak GPU memory: 885 MiB / 15,847 MiB available (RTX 5060 Ti 16 GB)

### Checkpoint/resume verification (v3)

| Metric | Full run | Resumed from epoch-2 ckpt |
|--------|---------|--------------------------|
| Epoch 3 train loss | 0.5161840518315634 | 0.5161840518315634 |
| Epoch 3 val loss | 0.7203208208084106 | 0.7203208208084106 |
| Final artifact SHA256 | `506492b0...` | `506492b0...` |

**Result**: exact match at float64 precision.

### Private artifact (v3)

Location: `~/.local/share/failurelens/experiment-issue19-stage2-v3/model_final.safetensors`

| Item | Value |
|------|-------|
| SHA256 | `506492b034fadb3d7ee6e7288d04b66dfa477c25b1de41d9d70166353cd343d4` |
| Code git SHA | `795b0a1c6f94198d93ebd36e45d3d61e8574549e` |
| Dataset SHA256 | `74909f51b588c311de798bee29b748cfc1ead65fd2948053c8f3a1c729e465c1` |
| Model pin | `3c4b60807d71f79b43f3c4363786d9493691f8b1` |
| Format | safetensors |
| Not pushed to Git | ✓ |
| Not pushed to HF Hub | ✓ |
| Eval gate B not accessed | ✓ |

### make check results (v3)

```
ruff check .                                              0 errors
ruff format --check .                                     0 would be reformatted
mypy (strict, packages/python/src/failurelens)            Success: 29 source files
pytest (full suite, no --run-ml-integration)              136 passed, 31 skipped
pytest tests/training/test_stage2_finetune.py \
    --run-ml-integration                                 29 passed
```

---

## v2: marker-leakage fix (preserved; NOT for Gate C)

**Date**: 2026-10-30
**Code git SHA**: `1c1516ee9f10575fedd9d1b04135ff5018ad9757`

> **⚠ v2 artifact (`e91955b4...`) is invalid for Gate C.**
> Independent steward found remaining private paths and injected exception messages.
> Use v3 artifact (`506492b0...`) from code SHA `795b0a1` for Gate C.

### Summary

Stage 2 v2: stripped the `# issue17 injected fault` generator marker from all
three features.  Coordinator review (issuecomment-6513355104) had found the marker
made the task trivially solvable.  v2 fixed that but the independent steward
subsequently found additional signals (private paths + injected messages) remaining
in 4 raised-exception records.  v3 completes the sanitization.

### Training results (v2, seed=42, rtx5060ti, code SHA `1c1516e`)

| Epoch | Train loss | Val loss |
|-------|-----------|---------|
| 1 | 0.4799 | 0.5963 |
| 2 | 0.3721 | 0.5872 |
| 3 | 0.4192 | 0.5843 |

Checkpoint/resume: epoch 3 loss exact match (0.4191608130931854).

### Private artifact (v2 — preserved, not for Gate C)

SHA256: `e91955b4563e0439d1d4a27826f49b268df7912f5e142bbdf970a2e77bcac8e9`
Path: `~/.local/share/failurelens/experiment-issue19-stage2-v2/`

---

## v1: original (tainted; NOT for Gate C)

**Code git SHA**: `49d6a89294ceb6fdce01acdea6fec46b2824e899`

SHA256: `7520d7aa182cf00bae8e9d2bae948590386c73c5b451fedddd744e2ba9f2ae8b`
Path: `~/.local/share/failurelens/experiment-issue19-stage2/`

Generator marker `# issue17 injected fault` present in positive and query features.
Do not use for Gate C.

---

## Experiment configuration (all versions)

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

## Caveats and limitations

- **Tiny synthetic dataset**: 6 training / 6 validation examples from one
  repository each.  Metrics reflect fitting this synthetic set; generalisation
  to real-world repos is unverified.
- **Val loss**: slight decrease; not statistically meaningful at this scale.
- Gate C quality claims are NOT made.  A larger held-out evaluation is out of
  scope for this PR.

## Security floor (v3)

| Check | Result |
|-------|--------|
| Model pin | `3c4b60807d71f79b43f3c4363786d9493691f8b1` (unchanged) |
| safetensors-only loader | `.bin` files rejected at load time |
| safetensors-only checkpoints | model weights saved as `.safetensors` |
| `weights_only=True` on resume | ✓ (PyTorch 2.11 compatible; cross-run rejected) |
| Generator marker in features | ✗ (stripped: 0 hits across 36 texts) |
| Private paths in features | ✗ (stripped: 0 hits across 36 texts) |
| Injected exception messages | ✗ (stripped: 0 hits across 36 texts) |
| Eval gate B | Not accessed |
| Weights in Git | Not committed |
| Weights on HF Hub | Not uploaded |
