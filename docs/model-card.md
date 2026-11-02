# Model card — FailureLens

> **Status: an experimental cloud rebuild is published; Gate C evaluation remains
> pending.** The owner explicitly approved the bounded Hugging Face Job and
> publication of its rebuilt model. The Job completed and the published model
> passed Hub reload verification. This does not establish superiority,
> production readiness, or adoption.
> Do not cite quality numbers that are not produced by the evaluation harness
> on committed data. The local v3 artifact below and the cloud rebuild are
> distinct artifacts with their own checksums.

## Summary

| Field | Value |
|-------|-------|
| Name | FailureLens CodeRankEmbed Stage 2 v3 |
| Task | Fault localization — rank candidate code locations for a failure |
| Base model | `nomic-ai/CodeRankEmbed` @ `3c4b60807d71f79b43f3c4363786d9493691f8b1` (MIT) |
| Fine-tuning method | Last NomicBertBlock encoder layer only (partial fine-tuning) |
| Trainable params | 9,440,256 / 136,731,648 total (~6.9%) |
| Artifact SHA-256 | `506492b034fadb3d7ee6e7288d04b66dfa477c25b1de41d9d70166353cd343d4` |
| Artifact path (private) | `~/.local/share/failurelens/experiment-issue19-stage2-v3/model_final.safetensors` |
| Format | safetensors only (`.bin` rejected at load time) |
| Training hardware | NVIDIA GeForce RTX 5060 Ti (16 GB) |
| Runtime | CPU or CUDA; offline after model cache |
| Code git SHA | `795b0a1c6f94198d93ebd36e45d3d61e8574549e` |
| Trained weights in Git | No |
| Trained weights on HF Hub | A distinct cloud rebuild is published; see below and the [Job runbook](hf-jobs-publication.md) |
| Gate C evaluation | Not yet run |
| Gate D adoption/publication | Experimental publication approved; adoption not established |
| Shipped baseline (for comparison) | `analysis_method = baseline-stacktrace-lexical` |

## Published cloud rebuild (2026-11-02)

The public [LevelCapTech/failurelens](https://huggingface.co/LevelCapTech/failurelens)
model was rebuilt on Hugging Face Jobs using the same 6/6 training recipe.
It is distinct from the historical local v3 artifact listed above.

| Field | Verified value |
|-------|----------------|
| Final Hub revision | `2426e838cf922a26e22836a4f93b3f135b4f2005` |
| Weights revision pinned in the runnable example | `ec8ffde17b5cf5a5bf6f206864fdbe985f373440` |
| Weights SHA-256 | `8e8fb971f8c1dce22ae29ea4f682cb8387790ad608e06d3800b62d3413a0a9e4` |
| Weights size | 546,938,144 bytes, safetensors |
| Training source | `2985bfcfbf5bf79bc3b3b0589284f6a4ca481970` |
| Completed Job | [6ae7d7a8c78863f9c62ff0e8](https://huggingface.co/jobs/LevelCapTech/6ae7d7a8c78863f9c62ff0e8) |
| Hardware | Tesla T4, 15,095 MiB detected VRAM |
| Training duration | 7.6 seconds (excludes setup, downloads, export and verification) |
| Peak allocated GPU memory | 885 MiB |
| Effective model sequence limit | 8192 tokens; longest observed training feature: 991 tokens |
| Verification | Exact trained tensors, offline CPU embeddings, Hub reload output parity, anonymous access and all manifest digests |

The Hub model card contains the supported standalone SentenceTransformers
example with an immutable revision. The generic FailureLens `from_pretrained`
path does not yet allow this trained repository's custom code and uses a
different query transform. The offline baseline remains the default working
library interface. No independent accuracy or superiority result is implied.

## Intended use

- Help developers, CI systems, and AI coding agents **narrow down** where a
  failure likely originates, with auditable evidence.
- Input: an error log / stack trace + a source repository. Output: ranked
  candidate locations (file / function / line span) with evidence.

### Out of scope

- Not an LLM API wrapper, generic code search, or error-message summariser.
- Does **not** guarantee a fix is correct — it suggests candidates.
- Not a multi-language tool yet (Python traceback support first).

## Architecture

Fine-tuned bi-encoder over `nomic-ai/CodeRankEmbed` (NomicBERT, 137M params,
8192-token context). Only the last encoder layer is trainable. Trained on
InfoNCE loss with positive/negative code-function pairs reconstructed from
private synthetic failure records (see `docs/training.md`).

At inference, the v3 bridge (`evaluation/v3_bridge.py`) applies the exact
Stage 2 query transform (sanitized `error_log.strip()`); the model scores
each candidate code fragment against the query in a shared retrieval pool.

## Training data

- Dataset: `failurelens-train-issue17-v2`
  (SHA-256: `74909f51b588c311de798bee29b748cfc1ead65fd2948053c8f3a1c729e465c1`)
- 12 synthetic failure records, 2 rights-cleared repositories:
  `pallets/click` (train, 6 records) and `python-attrs/attrs` (val, 6 records)
- Repo-level split to avoid cross-split leakage
- Positive text: mutated source function (actual failing code), sanitized of
  all generator markers, private paths, and injected exception messages
- Negative text: clean source function confirmed absent from the failure trace

## Training configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR scheduler | CosineAnnealingLR |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| InfoNCE temperature | 0.07 |
| Batch size | 2 |
| Epochs | 3 |
| Seed | 42 |
| Effective model sequence limit | 8192 (legacy config's `max_length=512` is not applied by the training function) |

## Training results (v3, seed=42)

| Epoch | Train loss | Val loss |
|-------|-----------|---------|
| 1 | 0.7298 | 0.7346 |
| 2 | 0.4976 | 0.7241 |
| 3 | 0.5162 | 0.7203 |

Elapsed: 3.5 s. Peak GPU: 885 MiB / 15,847 MiB. Checkpoint/resume verified
(epoch 3 loss bit-exact between full run and resume from epoch-2 checkpoint).

## Limitations and caveats

- **Tiny synthetic dataset**: 6 training / 6 validation examples from one
  repository each. Metrics above reflect fitting this synthetic set; no
  claim about generalisation to real-world repositories.
- **No independent quality result yet**: Gate C (held-out evaluation on ≥30
  cases across ≥3 rights-cleared repositories) has not run. The frozen Gate B
  thresholds (function Top-1 ≥ 0.4, MRR ≥ 0.6) remain unverified.
- **Comparator limitation**: the `supplied-pool-stack-lexical` baseline adapter
  is not the shipped `baseline-stacktrace-lexical` analyzer; a valid
  shipped-baseline Gate C delta requires the common-pool protocol from
  `docs/shipped-baseline-common-pool.md`.
- Only Python tracebacks parsed today; other formats return no candidates.
- Lexical matching is shallow; the embedding reranker adds semantic matching
  but has not been independently evaluated.

## Baseline (for comparison while waiting for Gate C)

The shipped deterministic baseline (`analysis_method = baseline-stacktrace-lexical`):
`score = 0.6·depth + 0.2·exception_site + 0.4·lexical`

Reproduced with:

```bash
python evaluation/run_baseline_eval.py
```

Gate A measured results (n=1, single fixture case, not generalisable):
- file: Top-1/5/10 = 1.0, MRR = 1.0
- function: Top-1 = 0.0, Top-5/10 = 1.0, MRR = 0.5

## Gate status

| Gate | Status |
|------|--------|
| Gate A — Pilot + baseline measurement | Done (2026-10-27) |
| Gate B — Acceptance criteria frozen | Done (2026-10-29, human-owner approved) |
| Implementation Go | Done (2026-10-29, human-owner approved) |
| Gate C — Independent held-out evaluation | **Not yet run** (blocked by comparator protocol review) |
| Gate D — Adoption / publication | Owner approved experimental cloud publication; adoption decision remains separate |

## Licensing

- Repository source: Apache-2.0
- Base model (`nomic-ai/CodeRankEmbed`): MIT
- Cloud-release fine-tuned weights: MIT, following the base model's declared
  license. Upstream model-code headers and license notices are retained,
  including the BSD-3-Clause notice for the FlashAttention-derived code.
  Apache-2.0 covers the repository source and build script.
- Training data: private synthetic records; rights reviewed per `docs/data-contract.md`.

## Provenance

- Code: `LevelCapTech/FailureLens` at `795b0a1c6f94198d93ebd36e45d3d61e8574549e`
- Training: `training/finetune.py` (InfoNCE, safetensors, `weights_only=True`)
- Sanitization audit: PR #25 issuecomment-6513753530 (independent steward)
- Artifact: `experiment-issue19-stage2-v3/model_final.safetensors`
  (SHA-256: `506492b034fadb3d7ee6e7288d04b66dfa477c25b1de41d9d70166353cd343d4`)
- Tainted artifacts (do not use for Gate C): v1 (`7520d7aa...`), v2 (`e91955b4...`)
