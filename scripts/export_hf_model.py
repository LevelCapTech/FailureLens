"""Assemble a HuggingFace-ready model directory from the Stage 2 fine-tuned artifact.

Usage (dry-run verify only — no upload):
    python scripts/export_hf_model.py --artifact PATH_TO_model_final.safetensors \\
        --cache PATH_TO_nomic_coderankeembed_cache --output /tmp/failurelens-hf-export

Gate D prerequisite: this script prepares the local directory only.
Upload to HuggingFace Hub is a separate step requiring explicit Gate D approval.
Do NOT run the upload section without human-owner authorization.

The assembled directory:
  model.safetensors          — fine-tuned NomicBERT backbone (v3)
  config.json                — NomicBERT architecture config (from base model)
  tokenizer.json             — tokenizer (from base model)
  tokenizer_config.json      — tokenizer config (from base model)
  special_tokens_map.json    — special tokens (from base model)
  vocab.txt                  — vocabulary (from base model)
  modules.json               — SentenceTransformers module list
  sentence_bert_config.json  — max_seq_length etc.
  1_Pooling/config.json      — pooling config (mean pooling)
  modeling_hf_nomic_bert.py  — custom model code, patched for transformers 5.x
  configuration_hf_nomic_bert.py — custom config code (from base model)
  README.md                  — model card (from docs/hf-model-card-draft.md)
  failurelens_provenance.json — SHA-256 digests for artifact + base model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ARTIFACT_SHA256 = "506492b034fadb3d7ee6e7288d04b66dfa477c25b1de41d9d70166353cd343d4"
BASE_MODEL_REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"

# Files copied verbatim from the base model snapshot
BASE_MODEL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "modules.json",
    "sentence_bert_config.json",
    "configuration_hf_nomic_bert.py",
]
POOLING_FILES = ["1_Pooling/config.json"]

# Replacement for the transformers 5.x incompatible get_extended_attention_mask call.
# The original line in NomicBertModel.forward() reads:
#   attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
# transformers>=5.0 removed ModelUtilsMixin.get_extended_attention_mask.
_ORIG_MASK_LINE = (
    "        attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)"
)
_PATCHED_MASK_LINES = """\
        # Inline transformers-5.x compatible attention mask expansion (get_extended_attention_mask
        # was removed in transformers>=5.0; this replicates its behavior for 2-D masks).
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
        elif attention_mask is not None and attention_mask.dim() == 4:
            pass  # already extended
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_snapshot(cache_root: Path) -> Path:
    """Locate the exact audited revision inside a HF cache directory."""
    snapshot = cache_root / "models--nomic-ai--CodeRankEmbed" / "snapshots" / BASE_MODEL_REVISION
    if snapshot.is_dir():
        return snapshot
    # Also accept bare snapshot (cache_root IS the snapshot)
    if (cache_root / "config.json").exists():
        cfg = json.loads((cache_root / "config.json").read_text(encoding="utf-8"))
        if cfg.get("model_type") == "nomic_bert":
            return cache_root
    raise FileNotFoundError(
        f"Base model snapshot not found at expected path: {snapshot}\n"
        f"Pass the directory containing config.json for revision {BASE_MODEL_REVISION}."
    )


def patch_modeling_file(src: Path, dst: Path) -> bool:
    """Copy modeling_hf_nomic_bert.py, patching the transformers 5.x incompatibility.

    Returns True if the patch was applied, False if the pattern was not found
    (meaning the base model may have already been updated or the file changed).
    """
    content = src.read_text(encoding="utf-8")
    if _ORIG_MASK_LINE not in content:
        # Pattern not found — copy verbatim and warn
        shutil.copy2(src, dst)
        return False
    patched = content.replace(_ORIG_MASK_LINE, _PATCHED_MASK_LINES, 1)
    dst.write_text(patched, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Export FailureLens v3 to HF format.")
    parser.add_argument("--artifact", required=True, type=Path, help="model_final.safetensors")
    parser.add_argument("--cache", required=True, type=Path, help="Base model cache root")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path(__file__).parent.parent / "docs/hf-model-card-draft.md",
        help="README.md source (default: docs/hf-model-card-draft.md)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite output directory if it exists"
    )
    args = parser.parse_args()

    artifact: Path = args.artifact
    cache: Path = args.cache
    output: Path = args.output
    readme: Path = args.readme

    # --- verify artifact ---
    if not artifact.is_file():
        print(f"ERROR: artifact not found: {artifact}", file=sys.stderr)
        return 1
    digest = sha256_file(artifact)
    if digest != ARTIFACT_SHA256:
        print(
            f"ERROR: artifact SHA-256 mismatch.\n  Expected: {ARTIFACT_SHA256}\n  Got:      {digest}",
            file=sys.stderr,
        )
        return 1
    print(f"Artifact verified: {artifact.name} ({digest[:16]}...)")

    # --- locate base model snapshot ---
    try:
        snapshot = find_snapshot(cache)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Base model snapshot: {snapshot}")

    # --- prepare output directory ---
    if output.exists():
        if not args.force:
            print(
                f"ERROR: output directory exists: {output}  (use --force to overwrite)",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # --- copy fine-tuned weights ---
    dst_weights = output / "model.safetensors"
    shutil.copy2(artifact, dst_weights)
    print(f"Copied weights -> {dst_weights.name}")

    # --- copy base model files (verbatim) ---
    for rel in BASE_MODEL_FILES:
        src = snapshot / rel
        dst = output / rel
        if not src.exists():
            print(f"WARNING: base model file missing: {rel}", file=sys.stderr)
            continue
        shutil.copy2(src, dst)
    print(f"Copied {len(BASE_MODEL_FILES)} base model files.")

    # --- patch modeling_hf_nomic_bert.py for transformers 5.x ---
    modeling_src = snapshot / "modeling_hf_nomic_bert.py"
    modeling_dst = output / "modeling_hf_nomic_bert.py"
    if modeling_src.exists():
        applied = patch_modeling_file(modeling_src, modeling_dst)
        if applied:
            print("Patched modeling_hf_nomic_bert.py for transformers 5.x compatibility.")
        else:
            print(
                "WARNING: transformers 5.x patch not applied — pattern not found. "
                "Verify compatibility manually.",
                file=sys.stderr,
            )
    else:
        print("WARNING: modeling_hf_nomic_bert.py not found in snapshot", file=sys.stderr)

    # --- copy pooling config ---
    for rel in POOLING_FILES:
        src = snapshot / rel
        dst = output / rel
        if not src.exists():
            print(f"WARNING: pooling file missing: {rel}", file=sys.stderr)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print("Copied 1_Pooling/config.json")

    # --- README.md ---
    if readme.is_file():
        shutil.copy2(readme, output / "README.md")
        print(f"Copied README.md from {readme.name}")
    else:
        print(f"WARNING: README source not found: {readme}", file=sys.stderr)

    # --- write provenance sidecar ---
    provenance = {
        "artifact_sha256": ARTIFACT_SHA256,
        "base_model": "nomic-ai/CodeRankEmbed",
        "base_model_revision": BASE_MODEL_REVISION,
        "training_code_sha": "795b0a1c6f94198d93ebd36e45d3d61e8574549e",
        "dataset_sha256": "74909f51b588c311de798bee29b748cfc1ead65fd2948053c8f3a1c729e465c1",
    }
    (output / "failurelens_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("Wrote failurelens_provenance.json")

    print(f"\nExport complete: {output}")
    print("Contents:")
    for p in sorted(output.rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size // 1024
            print(f"  {p.relative_to(output)}  ({size_kb} KB)")

    print("\nNOTE: Gate D (adoption/publication) approval required before uploading.")
    print("      Replace PLACEHOLDER sections in README.md with Gate C results first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
