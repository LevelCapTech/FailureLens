---
language:
- en
- python
license: mit
base_model:
- nomic-ai/CodeRankEmbed
tags:
- fault-localization
- code-retrieval
- bi-encoder
- sentence-transformers
- safetensors
pipeline_tag: feature-extraction
---

<!--
DRAFT — NOT PUBLISHED.

This is a ready-to-publish model card for the HuggingFace model repository.
It must NOT be published before Gate D (adoption/publication) is explicitly
approved by the human owner. Gate C (independent held-out evaluation) must
complete first. Remove this comment block before publication.

Before publishing:
1. Replace all PLACEHOLDER values marked with [PLACEHOLDER: ...]
2. Verify Gate C results are final and match the evaluation report
3. Confirm license with human owner (MIT from base model is likely appropriate)
4. Verify CodeRankEmbed license has not changed at time of publication
5. Confirm HuggingFace repo name with coordinator
6. Remove this entire comment block
-->

# FailureLens — Fine-tuned Fault Localization Reranker

FailureLens is a fine-tuned bi-encoder for **fault localization**: given a Python
error log or stack trace and a source repository, it ranks candidate code
functions by their likelihood of being the fault root cause.

> Built on [nomic-ai/CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed)
> (MIT), fine-tuned with an InfoNCE objective on synthetic failure records from
> rights-cleared open-source Python repositories.

## Quick start

```python
from failurelens import FailureLens

fl = FailureLens.from_pretrained("LevelCapTech/failurelens-coderankeembed-stage2")
result = fl.diagnose(error_log=YOUR_TRACEBACK, repo_path=PATH_TO_REPO)
for candidate in result.candidates[:5]:
    print(candidate.location, candidate.score)
```

Requires the `failurelens` package with the `ml` extra:

```bash
pip install "failurelens[ml]"
```

## Model details

| Field | Value |
|-------|-------|
| Base model | `nomic-ai/CodeRankEmbed` @ `3c4b608` (MIT) |
| Architecture | NomicBERT bi-encoder, 137M params total |
| Fine-tuned layers | Last encoder block (~9.4M / 136.7M trainable params) |
| Task | Code function ranking for fault localization |
| Objective | InfoNCE contrastive loss |
| Language | Python (other languages: no parser yet) |
| Context length | 512 tokens (training), 8192 (base model capacity) |
| Format | safetensors |

## Training

Fine-tuned on the private `failurelens-train-issue17-v2` dataset:
- 12 synthetic failure records from 2 rights-cleared open-source Python repos
- Repo-level train/val split (`pallets/click` / `python-attrs/attrs`)
- Positive: mutated function body causing the injected failure
- Negative: clean function from the same file confirmed absent from the trace
- 3 epochs, AdamW + CosineAnnealingLR, batch size 2, LR 2e-5, temperature 0.07
- Trained on NVIDIA RTX 5060 Ti (16 GB), ~3.5 s

### Known limitations

- **Tiny training set** (6 train / 6 val examples). Quality on real-world
  repositories has not been independently verified.
- **Python only** — other language tracebacks return no candidates.
- **Reranker, not end-to-end retriever** — the model reranks candidates
  retrieved by the shipped baseline analyzer; retrieval quality matters too.

## Evaluation

[PLACEHOLDER: Insert Gate C evaluation results here once available.
Required fields: function Top-1, function MRR, file Top-1, p95 latency,
peak memory, number of evaluation cases, number of repositories,
three-seed min/median/max, 95% bootstrap confidence intervals.]

Gate B acceptance criteria (frozen 2026-10-29):
- Function-level Top-1 ≥ 0.4 (baseline: 0.0)
- Function-level MRR ≥ 0.6 (baseline: 0.5)
- Absolute improvement: Top-1 ≥ +0.4, MRR ≥ +0.1
- Evaluation: ≥ 30 cases, ≥ 3 rights-cleared repositories
- p95 latency ≤ 5 s; process peak ≤ 8 GiB

## Artifact provenance

| Item | Value |
|------|-------|
| Code repository | `LevelCapTech/FailureLens` |
| Training code SHA | `795b0a1c6f94198d93ebd36e45d3d61e8574549e` |
| Artifact SHA-256 | `506492b034fadb3d7ee6e7288d04b66dfa477c25b1de41d9d70166353cd343d4` |
| Dataset SHA-256 | `74909f51b588c311de798bee29b748cfc1ead65fd2948053c8f3a1c729e465c1` |
| Base model revision | `3c4b60807d71f79b43f3c4363786d9493691f8b1` |

## License

This model is released under the [MIT License](LICENSE).

The base model `nomic-ai/CodeRankEmbed` is MIT licensed. The fine-tuned weights
are a derivative work and are released under the same MIT license.

## Citation

```bibtex
[PLACEHOLDER: add citation once a paper or report is available]
```

## Related resources

- Source code and evaluation harness: [LevelCapTech/FailureLens](https://github.com/LevelCapTech/FailureLens)
- Base model: [nomic-ai/CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed)
