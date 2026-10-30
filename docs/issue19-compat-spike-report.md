# Issue #19 Compatibility Spike — `get_extended_attention_mask` Fix

**Date**: 2026-10-30  
**Node**: rtx5060ti (`fd91961...`)  
**Branch**: `feature/issue19-coderankeembed-smoke`  
**HEAD**: 795bc01  
**Prototype location**: `/tmp/spike-issue19-compat/test_shim.py` (outside repo, throwaway)

## Environment Verified

| Item | Value |
|------|-------|
| Node hostname | rtx5060ti ✓ |
| Repo origin | git@github.com:LevelCapTech/FailureLens.git ✓ |
| Remote main | 611bc0e ✓ |
| Worktree branch HEAD | 795bc01 ✓ |
| Other worktrees | intact ✓ |
| transformers | 5.22.0 |
| torch | 2.11.0+cu128 |
| CUDA | True (RTX 5060 Ti) |
| safetensors | 0.8.0 |
| einops | 0.8.2 |
| sentence-transformers | 6.2.0 |

## Root Cause Confirmed

`get_extended_attention_mask` was a `ModuleUtilsMixin` method removed entirely in transformers 5.x.  
The pinned model custom code at `3c4b60807d71f79b43f3c4363786d9493691f8b1` calls it at line 1067 of `modeling_hf_nomic_bert.py`:

```python
attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
```

It converts 2D `(B, S)` binary masks to 4D `(B, 1, 1, S)` additive float masks used directly in attention score addition.

## Fix: Project-Local Compatibility Shim

**Strategy**: monkey-patch the method onto the loaded `NomicBertModel` instance after `SentenceTransformer(...)` is constructed. No downgrade required. No model revision change required. No modification to pinned model files.

**Shim function** (tested in prototype):

```python
import types
import torch


def _nomic_bert_attention_mask_shim(self, attention_mask, input_shape, device=None, dtype=None):
    """Replaces removed transformers 4.x get_extended_attention_mask for NomicBert."""
    if attention_mask is None:
        return None
    if attention_mask.dim() == 2:
        extended = attention_mask[:, None, None, :]
        extended = extended.to(dtype=torch.float32)
        extended = (1.0 - extended) * torch.finfo(torch.float32).min
        return extended
    elif attention_mask.dim() == 4:
        return attention_mask
    else:
        raise ValueError(f"Unexpected attention_mask dim {attention_mask.dim()}")
```

**Application point**: `load_sentence_transformer` in `packages/python/src/failurelens/inference/embedding.py`, after `SentenceTransformer(...)` is constructed:

```python
# Apply compat shim for NomicBert models under transformers>=5.x
inner = model[0].auto_model
if not hasattr(inner, "get_extended_attention_mask"):
    inner.get_extended_attention_mask = types.MethodType(_nomic_bert_attention_mask_shim, inner)
```

The guard `not hasattr(...)` ensures it is a no-op if a future transformers version restores the method or a different model is loaded.

## Prototype Test Results

**Tested command**:
```
VENV=<path-to-ml-venv>  # set to the venv with ML extras installed
cd /tmp && $VENV/bin/python -I /tmp/spike-issue19-compat/test_shim.py
```

**All checks passed**:

| Check | Result |
|-------|--------|
| `get_extended_attention_mask` missing from `PreTrainedModel` | True ✓ |
| `AttributeError` confirmed before shim | ✓ |
| Model type | `NomicBertModel` ✓ |
| Output shape | `(3, 768)` ✓ |
| All finite | True ✓ |
| L2 norms (unnormalized) | 24.477 / 24.433 / 24.547 ✓ |
| Padding invariance (cosine_sim single vs batched) | 1.000000 ✓ |
| GPU allocated | 530.3 MB |
| GPU reserved | 596.0 MB |
| Encode time (3 strings, GPU) | 1.056s |
| Model load time | 1.22s |
| Heldout data untouched | ✓ |
| `.bin` files absent | ✓ |

## Dependency Amendment Required

`einops` is required by `modeling_hf_nomic_bert.py` but absent from `[project.optional-dependencies].ml` in `packages/python/pyproject.toml`.

**Change**: add `"einops>=0.7.0"` to the `ml` extras list.

(Current `einops` in dev venv: 0.8.2 — already satisfies `>=0.7.0`.)

## Model Cache gitignore

Added to `.gitignore` on `feature/issue19-coderankeembed-smoke`:
```
models/cache/
*.safetensors
*.bin
```

This ensures the 547 MB `model.safetensors` (sha256 `827529...`) cannot be accidentally committed.

## Tests Needed (for production PR)

1. **Unit — shim math**: verify shim produces `(B, 1, 1, S)` shape from 2D input; verify `1→0.0`, `0→torch.finfo(float32).min`.
2. **Unit — padding invariance**: shim output is identical for padded vs non-padded positions.
3. **Integration — load + encode**: `load_sentence_transformer("nomic-ai/CodeRankEmbed", revision="3c4b60807d71f79b43f3c4363786d9493691f8b1", trust_remote_code=True)` succeeds; output shape `(N, 768)` and all finite.
4. **Regression — guard**: shim is skipped if `get_extended_attention_mask` is present on the model.

## Spike Status

| Item | Result |
|------|--------|
| Blocker root cause confirmed | ✓ |
| Fix validated on prototype | ✓ |
| No downgrade needed | ✓ |
| Security floor maintained | ✓ (transformers 5.22 used) |
| GPU/CPU memory acceptable | ✓ (530MB GPU, <1MB extra for shim) |
| Model cache gitignored | ✓ |
| Production code not touched | ✓ (throwaway prototype only) |
| PR not opened | ✓ (awaiting coordinator decision) |

**Ready for production PR.** Coordinator to decide scope: shim-only fix in `embedding.py` + `einops` dep amendment + `.gitignore` update.
