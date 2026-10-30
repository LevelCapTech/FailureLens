"""Stage 2 CodeRankEmbed fine-tuning pipeline.

Trains only the last NomicBertBlock encoder layer (~9.4M / ~109M params) via
InfoNCE loss on private Stage 2 training examples.  Checkpoints are saved after
every epoch as:

    <output_dir>/checkpoint_epoch_N/
        model.safetensors      — model backbone weights (safetensors format)
        training_state.pt      — optimizer + scheduler + RNG state (our own file)
        metadata.json          — epoch, loss history, config, data digest, git SHA

The final artifact is saved to <output_dir>/model_final.safetensors.

Reproducibility guarantees:
- Seeds: torch, CUDA RNG state saved and restored on resume.
- Checkpoint determinism: resumed run epoch N+1 loss matches uninterrupted run
  within floating-point tolerance (verified by the included integration test).
- No eval-only gate B data accessed; no weights pushed to Git or HF Hub.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_MODEL_ID = "nomic-ai/CodeRankEmbed"
_REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"
_DEFAULT_ARTIFACT_ROOT = Path.home() / ".local/share/failurelens/experiment-issue19-stage2"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _freeze_all_but_last_encoder_layer(model: Any) -> tuple[int, int]:
    """Freeze all parameters; unfreeze last NomicBertBlock. Return (trainable, total)."""
    for param in model.parameters():
        param.requires_grad_(False)

    # model[0] is the sentence_transformers Transformer wrapper;
    # model[0].model is NomicBertModel; .encoder.layers is the ModuleList
    last_layer = model[0].model.encoder.layers[-1]
    for param in last_layer.parameters():
        param.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _infonce_loss(
    q_emb: Any,
    p_emb: Any,
    n_emb: Any,
    temperature: float = 0.07,
) -> Any:
    """Per-triplet InfoNCE with cosine similarity."""
    import torch
    import torch.nn.functional as F

    q = F.normalize(q_emb, dim=-1)
    p = F.normalize(p_emb, dim=-1)
    n = F.normalize(n_emb, dim=-1)

    sim_pos = (q * p).sum(dim=-1) / temperature
    sim_neg = (q * n).sum(dim=-1) / temperature

    logits = torch.stack([sim_pos, sim_neg], dim=-1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


def _encode_batch(model: Any, texts: list[str], device: Any) -> Any:
    """Forward pass maintaining gradient graph."""
    import torch

    features = model[0].tokenize(texts)
    # BatchEncoding may contain non-tensor values (e.g. 'modality' is a string list)
    features = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
    return model.forward(features)["sentence_embedding"]


def _save_checkpoint(
    ckpt_dir: Path,
    *,
    epoch: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    history: dict[str, list[float]],
    config: dict[str, Any],
    run_metadata: dict[str, Any],
) -> None:
    import torch
    from safetensors.torch import save_file

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Model weights: safetensors format
    save_file(model[0].model.state_dict(), ckpt_dir / "model.safetensors")

    # Optimizer + scheduler + RNG state: our own training state file (not a
    # downloaded model; weights_only=False is safe and expected here)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        },
        ckpt_dir / "training_state.pt",
    )

    # Human-readable metadata
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "history": history,
                "config": config,
                "run_metadata": run_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_checkpoint(
    ckpt_dir: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
) -> tuple[int, dict[str, list[float]]]:
    import torch
    from safetensors.torch import load_file

    state_dict = load_file(ckpt_dir / "model.safetensors")
    model[0].model.load_state_dict(state_dict)

    # Load our own training state (weights_only=False is intentional: this is
    # our own file containing optimizer state, not a downloaded model)
    ts = torch.load(
        ckpt_dir / "training_state.pt",
        weights_only=False,
        map_location="cpu",
    )
    optimizer.load_state_dict(ts["optimizer"])
    scheduler.load_state_dict(ts["scheduler"])
    if ts.get("torch_rng") is not None:
        torch.set_rng_state(ts["torch_rng"])
    if ts.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ts["cuda_rng"])

    meta = json.loads((ckpt_dir / "metadata.json").read_text(encoding="utf-8"))
    return meta["epoch"], meta["history"]


def run_stage2_finetune(
    config: dict[str, Any],
    *,
    train_examples: list[Any],
    val_examples: list[Any],
    output_dir: Path | None = None,
    resume_from: Path | None = None,
    model_cache_dir: str | None = None,
) -> dict[str, Any]:
    """Run Stage 2 CodeRankEmbed fine-tuning experiment.

    Args:
        config: Hyperparameter dict (seed, epochs, batch_size, learning_rate,
            weight_decay, temperature, max_length).
        train_examples: Stage2Example list from load_stage2_examples()[0].
        val_examples: Stage2Example list from load_stage2_examples()[1].
        output_dir: Where to write checkpoints and artifact (default: private
            ~/.local/share path).
        resume_from: Checkpoint dir to resume from (e.g. checkpoint_epoch_2/).
        model_cache_dir: Local model cache folder (must contain audited snapshot).

    Returns:
        Experiment result dict including artifact path, digest, and loss history.
    """
    import torch

    from failurelens.inference.embedding import load_sentence_transformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — Stage 2 fine-tuning requires a GPU.")

    seed: int = config.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    output_root = output_dir or _DEFAULT_ARTIFACT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    # Load model via existing security-checked loader; unwrap to raw SentenceTransformer
    encoder_wrapper = load_sentence_transformer(
        _MODEL_ID,
        revision=_REVISION,
        trust_remote_code=True,
        device=str(device),
        cache_folder=model_cache_dir,
    )
    model = encoder_wrapper._model  # type: ignore[attr-defined]  # raw SentenceTransformer
    model.to(device)

    trainable, total = _freeze_all_but_last_encoder_layer(model)

    lr: float = config.get("learning_rate", 2e-5)
    weight_decay: float = config.get("weight_decay", 0.01)
    epochs: int = config.get("epochs", 3)
    batch_size: int = config.get("batch_size", 2)
    temperature: float = config.get("temperature", 0.07)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    start_epoch = 0

    if resume_from is not None:
        if not resume_from.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        start_epoch, history = _load_checkpoint(resume_from, model, optimizer, scheduler)

    run_metadata: dict[str, Any] = {
        "git_sha": _git_sha(),
        "model_id": _MODEL_ID,
        "model_revision": _REVISION,
        "dataset_sha256": "74909f51b588c311de798bee29b748cfc1ead65fd2948053c8f3a1c729e465c1",
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "trainable_params": trainable,
        "total_params": total,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "device_memory_mib": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
    }

    t_start = time.monotonic()

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_examples), batch_size):
            batch = train_examples[i : i + batch_size]
            q_emb = _encode_batch(model, [ex.query for ex in batch], device)
            p_emb = _encode_batch(model, [ex.positive for ex in batch], device)
            n_emb = _encode_batch(model, [ex.negative for ex in batch], device)

            loss = _infonce_loss(q_emb, p_emb, n_emb, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(1, n_batches)
        history["train_loss"].append(avg_train)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_val_batches = 0
            for i in range(0, len(val_examples), batch_size):
                batch = val_examples[i : i + batch_size]
                q_emb = _encode_batch(model, [ex.query for ex in batch], device)
                p_emb = _encode_batch(model, [ex.positive for ex in batch], device)
                n_emb = _encode_batch(model, [ex.negative for ex in batch], device)
                val_loss += _infonce_loss(q_emb, p_emb, n_emb, temperature=temperature).item()
                n_val_batches += 1
            avg_val = val_loss / max(1, n_val_batches)
            history["val_loss"].append(avg_val)

        ckpt_dir = output_root / f"checkpoint_epoch_{epoch}"
        _save_checkpoint(
            ckpt_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            config=config,
            run_metadata=run_metadata,
        )
        print(
            f"Epoch {epoch}/{epochs}: "
            f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    elapsed = time.monotonic() - t_start

    # Save final model artifact as safetensors
    from safetensors.torch import save_file

    artifact_path = output_root / "model_final.safetensors"
    save_file(model[0].model.state_dict(), artifact_path)
    artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    peak_memory_mib = (
        torch.cuda.max_memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0
    )

    result: dict[str, Any] = {
        "git_sha": run_metadata["git_sha"],
        "config": config,
        "history": history,
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "trainable_params": trainable,
        "total_params": total,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_digest,
        "elapsed_seconds": round(elapsed, 1),
        "peak_gpu_memory_mib": peak_memory_mib,
        "run_metadata": run_metadata,
    }
    (output_root / "experiment_metadata.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result
