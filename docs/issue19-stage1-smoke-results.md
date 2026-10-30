# Issue #19 Stage 1 — CodeRankEmbed Smoke Test Results

> **Note (added after resolution)**: The BLOCKER recorded in this log (`get_extended_attention_mask` missing in transformers 5.x) was resolved in the same PR by the project-local compatibility shim in `packages/python/src/failurelens/inference/embedding.py`. This document is a historical work-log of the original smoke-test session. See `docs/issue19-stage1-production-results.md` for the current production state.

**Date**: 2026-10-30  
**Node**: rtx5060ti (`fd91961048d37ea9b44ec661350933aad4928b402d3b4a1c8901c9e6e16df7ad`)  
**Branch**: `feature/issue19-coderankeembed-smoke`  
**HEAD**: bf18ed68 (= remote main)  
**Worktree**: `.worktrees/issue19-stage1`  

## Verification

| Check | Result |
|-------|--------|
| Node hostname | rtx5060ti ✓ |
| Repo origin | git@github.com:LevelCapTech/FailureLens.git ✓ |
| Remote main SHA | bf18ed6 (≥ required bf18ed68) ✓ |
| Branch base | origin/main (bf18ed6) ✓ |
| Existing worktrees preserved | 4 worktrees intact ✓ |
| VRAM | RTX 5060 Ti, 16311 MiB total, ~4623 MiB free |
| Disk | 417 GB available at /home |

## Dependencies

| Package | Installed | Security Floor | Status |
|---------|-----------|----------------|--------|
| torch | 2.11.0+cu128 | ≥2.6.0 | ✓ |
| transformers | 5.22.0 | ≥5.10.0 | ✓ |
| sentence-transformers | 6.2.0 | ≥5.6.0 | ✓ |
| safetensors | 0.8.0 | ≥0.4.3 | ✓ |
| numpy | 2.5.4 | ≥1.26.0 | ✓ |
| huggingface-hub | 2.3.0 | ≥0.24.0 | ✓ |
| einops | 0.8.2 | **not in manifest** | ⚠️ required by model |

## Model Download

- **Model**: `nomic-ai/CodeRankEmbed`  
- **Pinned revision**: `3c4b60807d71f79b43f3c4363786d9493691f8b1`  
- **Isolated cache**: `.worktrees/issue19-stage1/models/cache/nomic-coderankeembed-smoke/`  
- **Files downloaded**: safetensors + config + tokenizer only (no .bin, per safetensors-only policy)  
- **model.safetensors size**: 546,938,168 bytes  
- **model.safetensors SHA256**: `827529bcd58aef0d9082e66eeff7e7d53a02f62bd005f841a26b3d3e2fb17ebe`  
- **No pytorch_model.bin present**: VERIFIED  
- **License**: MIT (per model card) + Apache-2.0 (repo code)  

## Security Audit of Custom Code

Files requiring `trust_remote_code=True`:
- `configuration_hf_nomic_bert.py`: NomicBertConfig extends GPT2Config; benign, only sets config attributes.
- `modeling_hf_nomic_bert.py`: NomicBertModel extends PreTrainedModel; reviewed for dangerous patterns.

| Risk Pattern | Present | Notes |
|-------------|---------|-------|
| subprocess / os.system | No | — |
| eval() / exec() | No | — |
| torch.load (pickle) | Yes (lines 78, 332) | Blocked: no .bin in local snapshot, code falls to safe_load_file |
| Network calls in forward | No | — |

**torch.load reachability with safetensors-only download:**
- Line 78: `load_safe=True` branch taken when `model.safetensors` is present → `safe_load_file` used, `torch.load` not reached.
- Line 332: Local path check `os.path.exists(model_name)/pytorch_model.bin` → `.bin` absent, code falls through to `safe_load_file(model.safetensors)`.

**Conclusion**: `trust_remote_code=True` is safe at this pinned revision when safetensors-only.

## Model Loading Result

Model loaded successfully via `SentenceTransformer(local_snapshot_path, trust_remote_code=True)` (1.71s, CUDA:0).

## BLOCKER — Inference Failure

**Error**: `AttributeError: 'NomicBertModel' object has no attribute 'get_extended_attention_mask'`  
**Location**: `modeling_hf_nomic_bert.py` line 1067 (custom forward method)  
**Cause**: `get_extended_attention_mask` was a `ModuleUtilsMixin` method in transformers 4.x that was **fully removed** in transformers 5.x. Confirmed absent from transformers 5.22.0.

**Conflict**:
- Project security floor: `transformers>=5.10.0`  
- Custom code revision `3c4b60807d71`: written for transformers 4.x API

**Downgrade is not an option**: `transformers<5.0.0` violates the security floor.

## Missing Dependency

`einops` is required by `modeling_hf_nomic_bert.py` but is **not listed** in the project's `ml` or `train` extras in `packages/python/pyproject.toml`.

Action needed: add `einops>=0.7.0` to `[project.optional-dependencies].ml`.

## Suggested Fix Paths (for owner decision)

1. **Patch local adapter** (preferred): Create a project-local shim that overrides `NomicBertModel.forward` to replace the `get_extended_attention_mask` call with the transformers 5.x equivalent. The function converts 2D attention masks (B×S) to 4D (B×1×1×S) additive masks; this can be implemented inline without any 4.x dependency.

2. **Upgrade pinned revision**: Check if nomic-ai released a newer revision of CodeRankEmbed with transformers 5.x compatibility. Requires new owner-approved pinned revision.

3. **Alternative base model**: Use a standard BERT/RoBERTa bi-encoder that does not require `trust_remote_code` or custom 4.x code.

## Stage 1 Status

| Item | Status |
|------|--------|
| Node/repo/branch verified | ✓ PASS |
| safetensors-only download | ✓ PASS |
| Security audit (custom code) | ✓ PASS |
| Dependency versions (security floors) | ✓ PASS |
| Model loads | ✓ PASS |
| Inference smoke | ❌ BLOCKED |
| Heldout data untouched | ✓ CONFIRMED |
| Issue #17 data untouched | ✓ CONFIRMED |

**Stage 1 result: BLOCKER** — inference incompatible with transformers 5.x security floor.  
Training cannot proceed until the `get_extended_attention_mask` incompatibility is resolved.
