# Issue #19 Stage 2 — Source Code Reconstruction Method

**Date**: 2026-10-30
**Branch**: `feature/issue19-stage2-coderankeembed-finetune`

## Purpose

This document describes, in aggregate, how Stage 2 derives positive/negative
training text from the private `failurelens-train-issue17-v2` dataset without
exposing individual labels, the generator, or the held-out eval boundary.

## Why Not Location Metadata (Stage 1 Baseline)

Stage 1 used `"repo :: file :: function=... :: lines=..."` as positive/negative
text.  This is file-location metadata, not actual source code.  A model trained
on this cannot generalise to new repositories with different naming conventions;
it learns repository-specific identifiers rather than code semantics.

## Reconstruction Approach

### 1. Source Authority

The private `train-issue17-v2/sources/` directory contains the source
repositories (`pallets/click`, `python-attrs/attrs`) checked out at the audited
commits recorded in each record (`source_commit`).  These are the same commits
used by the generator; they are not modified.

### 2. Isolated Copies

For each record, Stage 2 creates an **isolated per-record copy** of the target
source file in a temporary directory.  All mutation operations are performed on
the copy; the authoritative source in `sources/` is never written to.  The copy
is discarded after text extraction.

### 3. Mutation Reconstruction

The injected statement is reconstructed deterministically from record fields:

| `injection_method` | Reconstructed statement |
|--------------------|------------------------|
| `wrong-value return` | `return False` |
| `early-return None` | `return None` |
| `raised exception` | Parsed from `error_log`: the injected `raise` statement appears verbatim in the stack trace with the `# issue17 injected fault` comment |

No hardcoded mapping of function names to statements is used.  Only
`injection_method` and (for raise statements) `error_log` are read.

### 4. Positive Text: Mutated Failing Snapshot

Python's `ast` module locates the function at `injection_location.function`
in the isolated copy.  The injection statement is inserted as the first body
statement (matching the generator's logic).  After insertion, the function
spans `[start_line, end_line + 1]` (1-indexed inclusive); these lines are
extracted as the positive text.

**Result**: The positive text is the actual Python source code of the
fault-injected function — the code that causes the test failure — not metadata
or clean code.  The mutation marker `# issue17 injected fault` is present and
its presence is verified by the integration test.

### 5. Negative Text: Clean Function at Candidate Location

The negative candidate location points to a different function in the same
source file.  This function is extracted from the **unmodified** (clean) source
file using the same `ast`-based line extraction.

**Result**: The negative text is real source code of a function that was
confirmed *not* called during the failure, verified clean by the profiled test
run in the generator.

### 6. Features Used (and Explicitly Excluded)

| Feature | Used? | Notes |
|---------|-------|-------|
| `error_log` | Yes (query text) | Actual traceback from the failing test |
| Function source code at positive loc | Yes | Mutated snapshot |
| Function source code at negative loc | Yes | Clean snapshot |
| `injection_method` + `error_log` for raise stmt | Yes | Deterministic reconstruction only |
| `label_rationale` | **No** | Contains human-readable ground-truth explanation |
| Function or variable names as direct features | **No** | Source code text is used, not extracted identifiers |
| Gate B heldout labels or generator | **No** | Boundary preserved |

## Split Policy

With 12 records across two repositories, a random split risks within-repository
leakage (training on 5 click functions, validating on 1 click function).  Stage
2 uses a **repo-level 6/6 split**:

- **Train (6)**: all `pallets/click` records
- **Val (6)**: all `python-attrs/attrs` records

Limitation: with only 6 training examples from one synthetic repository, the
quantitative metrics reflect the ability to fit this tiny synthetic set, not
real-world generalisation.  This is documented as a known limitation.

## Checkpoint/Resume Protocol

After each training epoch:
1. `checkpoint_epoch_N/model.safetensors` — NomicBertModel backbone weights
2. `checkpoint_epoch_N/training_state.pt` — AdamW optimizer state, cosine LR
   scheduler state, PyTorch CPU and CUDA RNG states
3. `checkpoint_epoch_N/metadata.json` — epoch, loss history, config, git SHA,
   dataset digest

Resume loads checkpoint N, restores all state (including RNG), and continues
from epoch N+1.  The integration test verifies epoch 3 loss from a resumed run
(starting at epoch 2 checkpoint) matches epoch 3 loss from the full run within
1e-5 absolute tolerance.

## Security Constraints Maintained

- Model loaded via `load_sentence_transformer` with the audited pin
  `3c4b60807d71f79b43f3c4363786d9493691f8b1`; local path spoofing rejected.
- `use_safetensors=True` enforced at load time; `.bin` files in cache rejected.
- Model weights saved as safetensors (never pickle).
- Final artifact saved to a private local path outside the repository; not
  pushed to Git or HF Hub.
- Gate B `heldout-gateb-v1` dataset not accessed at any stage.
