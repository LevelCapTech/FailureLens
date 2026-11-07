"""Build and publish the experimental model entirely on Hugging Face Jobs.

Preparing and submitting are separate explicit actions. Only compressed,
sanitized training features and this script leave the submitting machine.
Training features and the Hub token are passed as encrypted Job secrets.
"""

from __future__ import annotations

import argparse
import base64
import gc
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

MODEL_REPO = "LevelCapTech/failurelens"
BASE_MODEL = "nomic-ai/CodeRankEmbed"
BASE_REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"
SOURCE_REVISION = "2985bfcfbf5bf79bc3b3b0589284f6a4ca481970"
SOURCE_ARCHIVE_SHA256 = "c632b33ca539cc6185a4e20bbb68af78cb80a42b3ac9b98c2510efa4316bfc5d"
DATASET_SHA256 = "74909f51b588c311de798bee29b748cfc1ead65fd2948053c8f3a1c729e465c1"
MODELING_SHA256 = "502ccfb9c2d5dac976109ac2f04dc3125d8540329e5d4af92bc587d6dc65edcd"
IMAGE = "pytorch/pytorch@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be"
FLAVOR = "t4-small"
TIMEOUT = "30m"
CLOUD_BOOTSTRAP = """\
import os
import sys
import tempfile
import venv

root = tempfile.mkdtemp(prefix="failurelens-cloud-venv-")
# Reuse the image's PyTorch and pip without modifying OS-managed Python.
venv.EnvBuilder(system_site_packages=True, with_pip=False).create(root)
python = os.path.join(root, "bin", "python")
os.execv(python, [python, "-u", "-c", sys.argv[1], "--worker"])
"""
MAX_PAYLOAD_BYTES = 64_000
BASE_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "configuration_hf_nomic_bert.py",
    "modeling_hf_nomic_bert.py",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json",
)
DEPENDENCIES = (
    "transformers==5.22.0",
    "sentence-transformers==6.2.0",
    "huggingface_hub==2.4.0",
    "safetensors==0.8.0",
    "einops==0.8.1",
    "pydantic==2.14.2",
    "jsonschema==4.26.0",
    "typer>=0.12",
)
MODEL_CARD_METADATA = """---
language:
- en
license: mit
library_name: sentence-transformers
base_model: nomic-ai/CodeRankEmbed
pipeline_tag: feature-extraction
tags:
- python
- fault-localization
- code-retrieval
- experimental
- safetensors
---
"""
MASK_ORIGINAL = (
    "        attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)"
)
MASK_REPLACEMENT = """\
        # FailureLens: transformers 5 removed get_extended_attention_mask.
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float32)
                attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
            elif attention_mask.dim() != 4:
                raise ValueError("Expected a 2-D or 4-D attention mask")
"""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_payload(path: Path) -> None:
    """Run the existing digest-checked loader; do not export private records."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.data_stage2 import load_stage2_examples

    train, val = load_stage2_examples()
    payload = {
        "dataset_sha256": DATASET_SHA256,
        "train": [asdict(example) for example in train],
        "val": [asdict(example) for example in val],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    compressed = gzip.compress(raw, mtime=0)
    if len(compressed) > MAX_PAYLOAD_BYTES:
        raise ValueError("Training payload exceeds the small-upload limit")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(compressed)
    print(json.dumps({"payload_bytes": len(compressed), "train": len(train), "val": len(val)}))


def submit(payload_path: Path, state_path: Path, *, timeout_seconds: int | None = None) -> None:
    if state_path.exists():
        raise FileExistsError("Job state already exists; inspect the existing Job before retrying")
    if timeout_seconds is not None and not 0 < timeout_seconds <= 1800:
        raise ValueError("Job timeout must be between 1 and 1800 seconds")
    timeout = TIMEOUT if timeout_seconds is None else f"{timeout_seconds}s"
    from huggingface_hub import HfApi, get_token

    compressed = payload_path.read_bytes()
    if len(compressed) > MAX_PAYLOAD_BYTES:
        raise ValueError("Training payload exceeds the small-upload limit")
    payload = json.loads(gzip.decompress(compressed))
    validate_payload(payload)
    token = get_token()
    if not token:
        raise RuntimeError("Run hf auth login first")
    api = HfApi()
    if api.whoami().get("name") != MODEL_REPO.split("/")[0]:
        raise ValueError("Authenticated account does not match the publication owner")
    api._validate_yaml(MODEL_CARD_METADATA, repo_type="model")
    current = api.model_info(MODEL_REPO)
    script = Path(__file__).read_text(encoding="utf-8")
    job = api.run_job(
        image=IMAGE,
        command=["python", "-u", "-c", CLOUD_BOOTSTRAP, script],
        secrets={
            "HF_TOKEN": token,
            "FAILURELENS_TRAIN_PAYLOAD": base64.b64encode(compressed).decode("ascii"),
        },
        env={
            "FAILURELENS_PARENT_COMMIT": current.sha,
            "FAILURELENS_SCRIPT_SHA256": hashlib.sha256(script.encode()).hexdigest(),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
        },
        flavor=FLAVOR,
        timeout=timeout,
        attempts=1,
        name="failurelens-cloud-build-publish",
        labels={"project": "failurelens", "purpose": "model-publication"},
        namespace=MODEL_REPO.split("/")[0],
    )
    state = {"id": job.id, "url": job.url, "flavor": FLAVOR, "timeout": timeout}
    state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2))


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError("Unexpected training data provenance")
    for split, repo in [("train", "pallets/click"), ("val", "python-attrs/attrs")]:
        examples = payload.get(split)
        if not isinstance(examples, list) or len(examples) != 6:
            raise ValueError("Unexpected training split")
        for example in examples:
            if example.get("source_repository") != repo or example.get("split") != split:
                raise ValueError("Training repository split mismatch")
            for key in ("query", "positive", "negative"):
                text = example.get(key)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Empty training feature")
                if any(marker in text for marker in ("# issue17", "/home/", "injected ")):
                    raise ValueError("Unsanitized training feature")


def fetch_source(root: Path) -> Path:
    url = f"https://codeload.github.com/LevelCapTech/FailureLens/tar.gz/{SOURCE_REVISION}"
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != SOURCE_ARCHIVE_SHA256:
        raise ValueError("Pinned source archive digest mismatch")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        source.extractall(root, filter="data")
    return root / f"FailureLens-{SOURCE_REVISION}"


def patch_modeling(content: str) -> str:
    if hashlib.sha256(content.encode()).hexdigest() != MODELING_SHA256:
        raise ValueError("Audited model code digest mismatch")
    if content.count(MASK_ORIGINAL) != 1:
        raise ValueError("Unexpected attention-mask implementation")
    patched = content.replace(MASK_ORIGINAL, MASK_REPLACEMENT, 1)
    compile(patched, "modeling_hf_nomic_bert.py", "exec")
    return patched


def model_card(provenance: dict[str, Any], revision: str | None = None) -> str:
    revision_code = (
        f'revision = "{revision}"'
        if revision
        else 'revision = HfApi().model_info("LevelCapTech/failurelens").sha'
    )
    return f"""{MODEL_CARD_METADATA}
# FailureLens: experimental fault-localization encoder

This is a **trained experimental model**, rebuilt on Hugging Face Jobs from
the FailureLens Stage 2 recipe. Given a Python traceback and candidate function
source texts, it supplies embeddings for cosine-similarity ranking.

**Independent held-out evaluation (Gate C) is pending. No improvement over the
shipped baseline or the base encoder has been established. This release is for
experimentation, not evidence of production readiness or agent adoption.**

## Use

Install PyTorch 2.11.0, transformers 5.22.0, sentence-transformers 6.2.0,
safetensors 0.8.0 and einops 0.8.1. The model contains NomicBERT custom Python
code from the pinned base, with a documented transformers-5 attention-mask fix.
Review that code before enabling `trust_remote_code`. This example pins the
model revision and uses only safetensors weights.

```python
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

{revision_code}
model = SentenceTransformer(
    "LevelCapTech/failurelens",
    revision=revision,
    trust_remote_code=True,
    device="cpu",
    model_kwargs={{"use_safetensors": True}},
)
error_log = "ZeroDivisionError: division by zero"
candidates = [
    "def average(values): return sum(values) / len(values)",
    "def greet(name): return 'Hello ' + name",
]
# Stage 2 uses the stripped traceback directly, without a query prefix.
query = model.encode([error_log.strip()], normalize_embeddings=True)
code = model.encode(candidates, normalize_embeddings=True)
scores = (query @ code.T)[0]
ranked_indices = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
print([(i, float(scores[i])) for i in ranked_indices])
```

Scores are similarities, not probabilities. Callers supply the candidate pool;
this encoder does not locate files, invent line numbers, or generate fixes.
The current FailureLens library's generic `from_pretrained` path has a different
query transform and a restricted custom-code allowlist. The standalone example
above is the supported interface for this experimental release.

## Training and limitations

- Base: `{BASE_MODEL}` at `{BASE_REVISION}` (MIT).
- 136,731,648 parameters; only the final encoder block (9,440,256 parameters)
  is fine-tuned, with per-triplet InfoNCE and cosine similarity.
- 12 private, sanitized synthetic examples: 6 train from `pallets/click` and
  6 validation from `python-attrs/attrs`; source repositories are disjoint.
- Seed 42, 3 epochs, batch size 2, AdamW, learning rate 2e-5, weight decay 0.01,
  temperature 0.07, cosine learning-rate schedule. CLS pooling is retained.
- Effective model sequence limit: {provenance["effective_max_seq_length"]} tokens.
  The legacy config's `max_length=512` is not applied by the training function;
  the effective limit and observed feature lengths are recorded in provenance.
- Built on {provenance["training"]["run_metadata"]["device_name"]}. This cloud
  rebuild has its own checksum; it is not claimed bit-identical to local v3.
- Validation losses only reflect this tiny synthetic set. No natural-bug,
  multilingual, latency-budget, or independent accuracy claim is made.
- Baseline retrieval can omit the actual cause; reranking cannot recover an
  omitted function. Python traceback support in the library remains limited.

## Reproducibility and distribution

The Job fetched pinned public source and base weights, trained, assembled and
tested the model, and uploaded the weights directly to the Hub. The submitting
connection sent only this build script and approximately 6 KB of compressed
training features as an encrypted Job secret. Private records, features,
optimizer checkpoints and held-out labels are not distributed here.

- Job: https://huggingface.co/jobs/LevelCapTech/{provenance["job_id"]}
- Training source: https://github.com/LevelCapTech/FailureLens/tree/{SOURCE_REVISION}
- Weights SHA-256: `{provenance["artifact_sha256"]}`
- Data SHA-256: `{DATASET_SHA256}`
- Exact dependencies, training history, code hashes and smoke checks:
  [failurelens_provenance.json](failurelens_provenance.json).

The weights are distributed under MIT, as declared by the base model.
Upstream code retains its copyright headers and license notices; see
[LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
FailureLens's repository source and build script are Apache-2.0.
"""


def worker() -> None:
    """Run only inside the bounded, explicitly submitted cloud Job."""
    encoded = os.environ.pop("FAILURELENS_TRAIN_PAYLOAD")
    compressed = base64.b64decode(encoded, validate=True)
    if len(compressed) > MAX_PAYLOAD_BYTES:
        raise ValueError("Oversized training payload")
    raw = gzip.decompress(compressed)
    payload = json.loads(raw)
    validate_payload(payload)
    print("Installing pinned build dependencies", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", *DEPENDENCIES], check=True
    )

    import importlib.metadata

    import numpy as np
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from safetensors.torch import load_file
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available() or torch.__version__.split("+")[0] != "2.11.0":
        raise RuntimeError("Expected the pinned CUDA-enabled PyTorch 2.11.0 image")
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix="failurelens-cloud-") as temp:
        root = Path(temp)
        source = fetch_source(root)
        sys.path[:0] = [str(source), str(source / "packages/python/src")]
        import training.finetune as finetune
        from training.data_stage2 import Stage2Example

        # The downloaded archive has no .git; its bytes and commit are pinned above.
        finetune._git_sha = lambda: SOURCE_REVISION
        cache = root / "base-cache"
        snapshot = Path(
            snapshot_download(
                BASE_MODEL,
                revision=BASE_REVISION,
                cache_dir=cache,
                allow_patterns=[*BASE_FILES, "model.safetensors"],
            )
        )
        modeling = (snapshot / "modeling_hf_nomic_bert.py").read_text()
        patched = patch_modeling(modeling)
        config = json.loads((source / "training/configs/stage2_config.json").read_text())
        print("Training the existing 6/6 Stage 2 split on the cloud GPU", flush=True)
        result = finetune.run_stage2_finetune(
            config,
            train_examples=[Stage2Example(**e) for e in payload["train"]],
            val_examples=[Stage2Example(**e) for e in payload["val"]],
            output_dir=root / "training",
            model_cache_dir=str(cache),
        )
        gc.collect()
        torch.cuda.empty_cache()
        output = root / "release"
        output.mkdir()
        for name in BASE_FILES:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot / name, target)
        (output / "modeling_hf_nomic_bert.py").write_text(patched)
        shutil.copyfile(result["artifact_path"], output / "model.safetensors")

        # Validate the self-contained export with no network or pickle fallback.
        model = SentenceTransformer(
            str(output),
            trust_remote_code=True,
            device="cpu",
            local_files_only=True,
            model_kwargs={"use_safetensors": True},
        )
        state = load_file(output / "model.safetensors")
        exported = model[0].model.state_dict()
        if set(state) != set(exported) or any(
            not torch.equal(state[k], exported[k]) for k in state
        ):
            raise ValueError("Exported model does not contain the exact trained tensors")
        base_state = load_file(snapshot / "model.safetensors")
        changed = [key for key in state if not torch.equal(state[key], base_state[key])]
        if not changed or any(not key.startswith("encoder.layers.11.") for key in changed):
            raise ValueError("Unexpected trainable-weight changes")
        texts = ["ZeroDivisionError: division by zero", "def average(xs): return sum(xs) / len(xs)"]
        vectors = model.encode(texts, normalize_embeddings=True)
        if vectors.shape != (2, 768) or not np.isfinite(vectors).all():
            raise ValueError("Invalid exported embeddings")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
            raise ValueError("Embedding normalization check failed")
        effective_max_length = model.max_seq_length
        feature_lengths = [
            len(model.tokenizer.encode(e[k]))
            for split in ("train", "val")
            for e in payload[split]
            for k in ("query", "positive", "negative")
        ]
        training = {k: v for k, v in result.items() if k != "artifact_path"}
        provenance = {
            "build_kind": "hf-jobs-stage2-rebuild",
            "job_id": os.environ["JOB_ID"],
            "base_model": BASE_MODEL,
            "base_model_revision": BASE_REVISION,
            "source_revision": SOURCE_REVISION,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "build_script_sha256": os.environ["FAILURELENS_SCRIPT_SHA256"],
            "artifact_sha256": digest(output / "model.safetensors"),
            "dataset_sha256": DATASET_SHA256,
            "training_features_sha256": hashlib.sha256(raw).hexdigest(),
            "effective_max_seq_length": effective_max_length,
            "max_observed_training_feature_tokens": max(feature_lengths),
            "training": training,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in [
                    "torch",
                    "transformers",
                    "sentence-transformers",
                    "huggingface_hub",
                    "safetensors",
                    "einops",
                    "pydantic",
                    "jsonschema",
                    "typer",
                ]
            },
            "verification": {
                "exact_trained_tensors": True,
                "changed_tensors": len(changed),
                "changed_layers": [11],
                "offline_cpu_embeddings": True,
            },
            "independent_evaluation": "pending; no accuracy or superiority claim",
            "files": {
                str(p.relative_to(output)): digest(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        }
        (output / "failurelens_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (output / "README.md").write_text(model_card(provenance))
        license_text = """MIT License

Copyright (c) 2026 LevelCapTech (fine-tuning modifications)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        (output / "LICENSE").write_text(license_text)
        notices = [
            f"# Third-party notices\n\nBase model: {BASE_MODEL} @ {BASE_REVISION}; the upstream model card declares MIT.\nOriginal model-code copyright headers are preserved.\n\n"
        ]
        for repo, revision, filename in [
            ("pallets/click", "2247b35ea1c47c727d7a06e51fa280e12a863ff6", "LICENSE.txt"),
            ("python-attrs/attrs", "c8bb9bfaa2843e7844b5bb13bed4370b51feec4f", "LICENSE"),
            ("Dao-AILab/flash-attention", "main", "LICENSE"),
        ]:
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/{filename}"
            with urllib.request.urlopen(url, timeout=30) as response:
                notice = response.read().decode()
            notices.append(
                f"## Upstream notice: {repo}\n\nSource: {url}\n\n```text\n{notice}\n```\n"
            )
        (output / "THIRD_PARTY_NOTICES.md").write_text("\n".join(notices))
        shutil.copyfile(source / "LICENSE", output / "LICENSE-APACHE-2.0")

        api = HfApi()
        print("Export verified; uploading trained weights directly from Hugging Face", flush=True)
        commit = api.upload_folder(
            repo_id=MODEL_REPO,
            folder_path=output,
            commit_message="Publish experimental Stage 2 model built on Hugging Face Jobs",
            parent_commit=os.environ["FAILURELENS_PARENT_COMMIT"],
        )
        # Pin the runnable example to the immutable weights commit.
        card = model_card(provenance, commit.oid)
        final = api.upload_file(
            repo_id=MODEL_REPO,
            path_in_repo="README.md",
            path_or_fileobj=card.encode(),
            commit_message="Pin usage example to the verified model commit",
            parent_commit=commit.oid,
        )
        del model, state, base_state, exported
        gc.collect()
        print("Reloading the published immutable revision from the Hub", flush=True)
        published = SentenceTransformer(
            MODEL_REPO,
            revision=final.oid,
            trust_remote_code=True,
            device="cpu",
            model_kwargs={"use_safetensors": True},
            cache_folder=str(root / "published-cache"),
        )
        hub_vectors = published.encode(texts, normalize_embeddings=True)
        if not np.allclose(vectors, hub_vectors, rtol=1e-5, atol=1e-6):
            raise ValueError("Published model embeddings differ from the verified export")
        info = api.model_info(MODEL_REPO, revision=final.oid, files_metadata=True)
        weight = next(f for f in info.siblings if f.rfilename == "model.safetensors")
        if info.private or weight.lfs.sha256 != provenance["artifact_sha256"]:
            raise ValueError("Public weight metadata mismatch")
        print(
            json.dumps(
                {
                    "published": MODEL_REPO,
                    "revision": final.oid,
                    "artifact_sha256": provenance["artifact_sha256"],
                    "hub_reload_verified": True,
                }
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path, metavar="PRIVATE_PAYLOAD_PATH")
    mode.add_argument("--submit", type=Path, metavar="PRIVATE_PAYLOAD_PATH")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--state", type=Path, default=Path("/tmp/failurelens-hf-release/job.json"))
    parser.add_argument(
        "--timeout-seconds", type=int, help="Shorter remaining budget, at most 1800"
    )
    args = parser.parse_args()
    if args.prepare:
        prepare_payload(args.prepare)
    elif args.submit:
        submit(args.submit, args.state, timeout_seconds=args.timeout_seconds)
    else:
        worker()


if __name__ == "__main__":
    main()
