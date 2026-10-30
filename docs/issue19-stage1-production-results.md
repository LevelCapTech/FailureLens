# Issue #19 Stage 1 — Production Implementation Results

**Date**: 2026-10-30  
**Node**: rtx5060ti (`fd91961...`)  
**Branch**: `feature/issue19-coderankeembed-smoke`  
**HEAD**: (see git log)  
**PR#18 integrated**: b8212dd merged into this branch

## Summary

Production implementation of the `get_extended_attention_mask` compatibility shim for
`nomic-ai/CodeRankEmbed` under `transformers>=5.x`. The throwaway spike at
`/tmp/spike-issue19-compat/` has been rewritten as auditable production code.

## Changes

### `packages/python/src/failurelens/inference/embedding.py`

- Added `_nomic_bert_get_extended_attention_mask` — drop-in replacement for the removed
  `ModelUtilsMixin.get_extended_attention_mask`, scoped to instance-level binding only.
- Added `_apply_nomic_bert_compat` — applies the shim only when both conditions hold:
  1. `get_extended_attention_mask` is absent on the inner model
  2. The inner model type name contains `NomicBert` (non-nomic guard)
- Updated `load_sentence_transformer` to accept `trust_remote_code` and call
  `_apply_nomic_bert_compat` automatically after construction.
- Added `_NOMIC_CODERANKEEMBED_AUDITED_REVISION` constant — the exact pinned commit
  for which `trust_remote_code=True` was audited.

### `packages/python/pyproject.toml`

- Added `einops>=0.7.0` to `[project.optional-dependencies].ml` — required by
  `modeling_hf_nomic_bert.py` but previously absent from the declared extras.

### `tests/inference/` (new)

- `test_nomic_bert_shim.py` — 14 unit tests:
  - Shim math: 2-D shape, 1→0.0 / 0→finfo.min, mixed mask, 4-D pass-through,
    invalid-dim raises, None pass-through
  - Padding invariance: float32 output dtype, padding positions equal finfo.min
  - Guard logic: NomicBert gets shim, existing method unchanged, non-nomic skipped,
    bad ST structure graceful, bound-method identity
- `test_nomic_coderankeembed_integration.py` — 5 integration tests (skipped without
  `--run-ml-integration`): shape `(3, 768)`, all-finite, L2-normalized, padding
  invariance, safetensors-only cache

### `conftest.py` (root)

- Adds `--run-ml-integration` option used by the integration test.

## Security / Safety

| Check | Result |
|-------|--------|
| Model pin | `3c4b60807d71f79b43f3c4363786d9493691f8b1` (unchanged) |
| safetensors-only load | ✓ (no `.bin` files) |
| `trust_remote_code=True` scope | Only at exact audited pin |
| Non-nomic guard | `"NomicBert" in type(inner).__name__` |
| No global monkey-patch | Instance-level `types.MethodType` only |
| Model weights in git | Blocked by `.gitignore` (`models/cache/`, `*.safetensors`, `*.bin`) |

## Test Results

```
tests/inference/test_nomic_bert_shim.py          14 passed
tests/inference/test_nomic_coderankeembed_integration.py  5 passed (--run-ml-integration)
packages/python/tests/ (full suite, --ignore=tests/training)  113 passed, 8 skipped
```

(Note: `tests/training/test_toy_pipeline.py` requires `[train]` extras including `jsonschema`
which is not in the dev venv. This is a pre-existing issue unrelated to this PR.)

## Bounded Local Experiment (private, not in repo)

**Purpose**: Verify the shim enables gradient flow for fine-tuning; measure loss trajectory.

| Parameter | Value |
|-----------|-------|
| Dataset | `failurelens-train-issue17-v2` (SHA256: `74909f51...`) |
| Dataset records | 12 total (10 train, 2 val, seeded split seed=42) |
| Source code | Pinned click@`2247b35` / attrs@`c8bb9bf` |
| Model | `nomic-ai/CodeRankEmbed@3c4b608` |
| Trainable params | Last encoder layer (9,440,256 / ~109M total) |
| Epochs | 3 |
| Batch size | 2 |
| Learning rate | 2e-5 (AdamW) |
| Objective | InfoNCE (temperature=0.07) |

**Results**:

| Metric | Value |
|--------|-------|
| Starting train loss | 0.7444 |
| Ending train loss | 0.4907 |
| Starting val loss | 0.4495 |
| Ending val loss | 0.4706 |
| Runtime | 1.7 s |
| Peak GPU memory | 878 MB |
| GPU free before load | 4490 MB / 15848 MB |

**Caveats**:
- Val set is 2 records — not statistically meaningful; reported for completeness only.
- No accuracy claims.
- Slight val loss increase is expected with a 2-record val set.
- Weights not saved; records, generators, and weights remain local and out of Git.
- `label_rationale` NOT used as model input.
- Gate B heldout data NOT accessed.

## Appendix: Shim Math Summary

```
Input (2D, B×S):  1 (attended) → 0.0    0 (padding) → torch.finfo(float32).min
Output (4D, B×1×1×S): additive attention mask for NomicBert attention scores
```
