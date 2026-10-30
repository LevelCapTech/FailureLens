# Issue #19 Stage 1 — Production Implementation Results

**Date**: 2026-10-30  
**Node**: rtx5060ti (`fd91961...`)  
**Branch**: `feature/issue19-coderankeembed-smoke`  
**git SHA**: `b9c564289ba699fe0eca48ff0f50beb861576d19`  
**PR**: #20

## Summary

Stage 1 production implementation: `get_extended_attention_mask` compatibility shim for
`nomic-ai/CodeRankEmbed` under `transformers>=5.x`, security boundary enforcement, and ML test
gating. Also includes a reproducible toy training pipeline (ToyEncoder, not CodeRankEmbed)
as infrastructure demonstration. Issue #19 checkpoint/resume requirement for actual
CodeRankEmbed experiment is deferred to a future PR.

## Changes (this PR)

### `packages/python/src/failurelens/inference/embedding.py`

- `_nomic_bert_get_extended_attention_mask`: instance-level shim for the removed `ModelUtilsMixin.get_extended_attention_mask`, scoped to NomicBert only.
- `_apply_nomic_bert_compat`: applies shim only when method is absent and `"NomicBert"` is in the type name.
- `load_sentence_transformer`: accepts `trust_remote_code`; enforces allowlist; rejects `.bin` weights; auto-applies shim.
- `_NOMIC_CODERANKEEMBED_AUDITED_REVISION`: pinned audited commit.
- `_TRUST_REMOTE_CODE_ALLOWLIST`: frozen set of permitted `(model_id, revision)` pairs.
- `_check_trust_remote_code_allowlist`: raises `ModelError` for any combination outside the allowlist; local paths are always rejected (name alone cannot verify audited contents).

### `packages/python/pyproject.toml`

- Added `einops>=0.7.0` to `[ml]` extras.

### `tests/inference/` (new)

- `test_nomic_bert_shim.py` — 14 unit tests: shim math, padding invariance, guard logic. Skipped entirely if `torch` is absent (`pytest.importorskip`).
- `test_nomic_coderankeembed_integration.py` — 5 integration tests: skipped without `--run-ml-integration`.

### `packages/python/tests/test_embedding.py` (extended)

- 12 new security tests: allowlist accepts exact pair; rejects unpinned/wrong revision/unaudited repo, local path with correct name (spoofed dir), local path with wrong dir name; `load_sentence_transformer` rejects untrusted models and `.bin` weights in local dir; allowlist constant correctness.

### `conftest.py` (root)

- `--run-ml-integration` pytest option.

### `training/` (new)

Reproducible toy training pipeline (CPU, pure-Python ToyEncoder, fixture data):

- `training/smoke.py` — `run_toy_training()`: full run with per-epoch checkpoints + artifact JSON.
- `training/config.py` — `TrainConfig` (pydantic schema, validated fields).
- `training/data.py` — `load_training_examples`: schema validation, quality gates, excluded-repo filter.
- `training/objective.py` — `InfoNCEObjective`, `ToyEncoder` (deterministic, seeded).
- `training/scripts/smoke_train.py` — CLI entry point.
- `training/configs/toy_config.json` — pinned config.

## Security / Safety

| Check | Result |
|-------|--------|
| Model pin | `3c4b60807d71f79b43f3c4363786d9493691f8b1` (unchanged) |
| trust_remote_code allowlist | Exact `(model_id, revision)` HF Hub pair only; local paths rejected (name alone cannot verify contents) |
| safetensors-only (load path) | `.bin` files in local model dir raise `ModelError` before SentenceTransformer is called |
| safetensors-only (cache) | Verified by integration test: `_MODEL_CACHE_ROOT.rglob("*.bin") == []` |
| Non-nomic guard | `"NomicBert" in type(inner).__name__` |
| No global monkey-patch | Instance-level `types.MethodType` only |
| Model cache in git | Blocked (`.gitignore`: `models/cache/`, `*.safetensors`, `*.bin`) |

Rejection tests: unpinned revision, wrong revision, unaudited repo, wrong local dir, `.bin` presence.

## Test Results (with dev+train extras, `make check` equivalent)

```
ruff check .                                     0 errors
ruff format --check .                            0 would be reformatted
mypy (strict)                                    Success: 29 source files
pytest (full suite, no --run-ml-integration)     130 passed, 8 skipped
pytest tests/inference/ --run-ml-integration     19 passed
```

Skips in the full suite:
- `test_nomic_bert_shim.py` skips entirely when `torch` is absent (CI without ML deps).
- `test_nomic_coderankeembed_integration.py` skips without `--run-ml-integration`.
- Two `MissingDependencyError` guard tests skip when ml extra is installed.

## Reproducible Toy Training Experiment

**Purpose**: Demonstrate reproducible training infrastructure with checkpoint/resume, git SHA,
seed, and data digest recorded. This is the CPU toy-encoder path; the CodeRankEmbed GPU
experiment below is verification-only.

**Config** (`training/configs/toy_config.json`):

| Parameter | Value |
|-----------|-------|
| base_model | toy-bi-encoder |
| revision | train-toy-0.1.0 |
| dataset | datasets/fixtures/sample_record.json |
| dataset_version | fixture-v1 |
| seed | 7 |
| optimizer | adamw |
| learning_rate | 0.05 |
| batch_size | 1 |
| epochs | 3 |
| objective | infonce |

**Data digest**:

| File | SHA256 |
|------|--------|
| `datasets/fixtures/sample_record.json` | `26b1fd0a82e8281e50ebf8e49d2ec5164b9ce05bb0708be3e0b8ffc802ac9e0b` |

**Results** (run 2026-10-30, git SHA `b9c5642`, seed=7):

| Epoch | Train loss |
|-------|-----------|
| 1 | 0.476593 |
| 2 | 0.103510 |
| 3 | 0.057152 |

**Checkpoint/resume** (verified deterministic):
- Per-epoch checkpoints saved to `training/experiments/toy-smoke/` (git-ignored).
- Resume from `checkpoint_epoch_2.json` → epoch-3 loss = **0.057152** (matches full run exactly).
- Checkpoints are private, local-only; no weights pushed to Git or HF.

**Reproduce** (from repo root, with venv activated):

```bash
python -m training.scripts.smoke_train \
    --config training/configs/toy_config.json \
    --dataset datasets/fixtures/sample_record.json \
    --output-dir /tmp/fl-toy-run

# Resume from epoch-2 checkpoint:
python -m training.scripts.smoke_train \
    --config training/configs/toy_config.json \
    --dataset datasets/fixtures/sample_record.json \
    --resume-from /tmp/fl-toy-run/checkpoint_epoch_2.json \
    --output-dir /tmp/fl-toy-resume
```

## CodeRankEmbed Shim Verification (one-off smoke, no checkpoint/resume — Stage 1 scope only)

| Parameter | Value |
|-----------|-------|
| Dataset | `failurelens-train-issue17-v2` (SHA256: `74909f51...`) |
| Records | 10 train / 2 val, seed=42 |
| Model | `nomic-ai/CodeRankEmbed@3c4b608` |
| Trainable params | Last encoder layer (9,440,256 / ~109M total) |
| Epochs | 3 |
| Batch size | 2 |
| Learning rate | 2e-5 (AdamW) |
| Objective | InfoNCE (temperature=0.07) |

| Metric | Value |
|--------|-------|
| Starting train loss | 0.7444 |
| Ending train loss | 0.4907 |
| Starting val loss | 0.4495 |
| Ending val loss | 0.4706 |
| Runtime | 1.7 s |
| Peak GPU memory | 878 MB / 4490 MB free |

Caveats:
- Val set is 2 records — not statistically meaningful.
- Slight val increase is expected at 2-record val set.
- `label_rationale` NOT used. Gate B NOT accessed. Weights NOT saved or released.
- This run verifies the shim enables gradient flow (Stage 1 scope). Checkpoint/resume for
  actual CodeRankEmbed is **not** in scope for this PR; it will be addressed in a follow-up.

## Appendix: Shim Math

```
Input (2D, B×S):  1 (attended) → 0.0    0 (padding) → torch.finfo(float32).min
Output (4D, B×1×1×S): additive attention mask for NomicBert attention scores
```
