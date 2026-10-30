"""Opt-in, evaluator-owned v3 scorer for canonical unlabeled RankRequests.

No model or private record is loaded on import. No held-out input is logged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evaluation.independent_runner import MethodSpec
from evaluation.private_adapter import PrivateInputError, RankRequest, RankResult
from training.data_stage2 import _strip_generator_marker

MODEL_ID = "nomic-ai/CodeRankEmbed"
MODEL_REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"
ARTIFACT_SHA256 = "506492b034fadb3d7ee6e7288d04b66dfa477c25b1de41d9d70166353cd343d4"
TRAIN_SHA256 = "74909f51b588c311de798bee29b748cfc1ead65fd2948053c8f3a1c729e465c1"


def _reject(reason: str) -> None:
    raise PrivateInputError(f"v3 bridge rejected: {reason}") from None


def training_query(error_log: str) -> str:
    """The exact Stage 2 query transform, without JSON frame or stack_trace."""
    return _strip_generator_marker(error_log.strip())


def parse_query(query: str) -> str:
    """Accept only canonical JSON framing and already-sanitized training feature."""
    try:
        framed = json.loads(query)
    except (TypeError, ValueError):
        _reject("invalid query framing")
    if (
        not isinstance(framed, dict)
        or set(framed) != {"error_log", "stack_trace"}
        or not isinstance(framed["error_log"], str)
        or not framed["error_log"].strip()
        or (framed["stack_trace"] is not None and not isinstance(framed["stack_trace"], str))
        or json.dumps(framed, sort_keys=True, separators=(",", ":"), ensure_ascii=False) != query
    ):
        _reject("invalid query framing")
    log = framed["error_log"]
    normalized = training_query(log)
    if normalized != log.strip():
        _reject("query sanitation mismatch")
    return normalized


def _verify_artifact(path: Path) -> tuple[int, dict[str, Any]]:
    if path.name != "model_final.safetensors" or not path.is_file():
        _reject("unavailable model artifact")
    try:
        with path.open("rb") as artifact:
            digest = hashlib.file_digest(artifact, "sha256").hexdigest()
        size = path.stat().st_size
        meta = json.loads((path.parent / "experiment_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        _reject("unavailable model provenance")
    run = meta.get("run_metadata") if isinstance(meta, dict) else None
    if (
        digest != ARTIFACT_SHA256
        or not isinstance(run, dict)
        or run.get("model_id") != MODEL_ID
        or run.get("model_revision") != MODEL_REVISION
        or run.get("dataset_sha256") != TRAIN_SHA256
        or meta.get("artifact_sha256") != ARTIFACT_SHA256
    ):
        _reject("model provenance mismatch")
    return size, run


def model_spec(artifact_path: Path) -> MethodSpec:
    size, run = _verify_artifact(artifact_path)
    count = run.get("total_params")
    if type(count) is not int or count < 1 or count > 1_000_000_000:
        _reject("invalid model parameter metadata")
    return MethodSpec("v3-coderankeembed-frozen", ARTIFACT_SHA256, count, size)


class V3Ranker:
    def __init__(self, encoder: Any) -> None:
        self._encoder = encoder

    def __call__(self, request: RankRequest) -> RankResult:
        query = parse_query(request.query)
        if not request.candidates or len({c.id for c in request.candidates}) != len(
            request.candidates
        ):
            _reject("invalid candidate pool")
        if type(request.top_k) is not int or request.top_k < 1:
            _reject("invalid top_k")
        for candidate in request.candidates:
            if (
                hashlib.sha256(candidate.source_text.encode()).hexdigest()
                != candidate.source_sha256
            ):
                _reject("candidate digest mismatch")
            if not candidate.source_text.strip():
                _reject("empty candidate source")
            if _strip_generator_marker(candidate.source_text) != candidate.source_text:
                _reject("candidate sanitation mismatch")
        try:
            # Single-text calls cap transient activation memory regardless of pool size.
            q = self._encoder.encode([query], normalize_embeddings=True, batch_size=1)[0]
            scores = []
            for candidate in request.candidates:
                c = self._encoder.encode(
                    [candidate.source_text], normalize_embeddings=True, batch_size=1
                )[0]
                similarity = float(sum(float(a) * float(b) for a, b in zip(q, c, strict=True)))
                if not math.isfinite(similarity):
                    _reject("nonfinite model score")
                scores.append((candidate.id, similarity))
        except PrivateInputError:
            raise
        except Exception:
            _reject("model inference failed")
        # Full permutation; stable ID tie break independent of supplied order.
        ordered = tuple(item for item, _ in sorted(scores, key=lambda row: (-row[1], row[0])))
        return RankResult(ordered, request.digest, request.top_k)


def make_model_factory(
    artifact_path: Path, *, cache_folder: str | None = None, device: str = "cpu"
) -> Callable[[int], V3Ranker]:
    """Create the independent_runner model_factory(seed); load only on invocation."""
    model_spec(artifact_path)
    if device not in {"cpu", "cuda"}:
        _reject("invalid device")
    if cache_folder is None:
        _reject("unavailable audited model cache")
    snapshot = Path(cache_folder) / "models--nomic-ai--CodeRankEmbed" / "snapshots" / MODEL_REVISION
    if not snapshot.is_dir() or any(Path(cache_folder).rglob("*.bin")):
        _reject("unavailable audited model cache")

    def model_factory(seed: int) -> V3Ranker:
        if type(seed) is not int:
            _reject("invalid seed")
        _verify_artifact(artifact_path)
        try:
            import torch
            from safetensors.torch import load_file

            from failurelens.inference.embedding import load_sentence_transformer

            torch.manual_seed(seed)
            if device == "cuda":
                if not torch.cuda.is_available():
                    _reject("requested device unavailable")
                torch.cuda.manual_seed_all(seed)
            encoder = load_sentence_transformer(
                MODEL_ID,
                revision=MODEL_REVISION,
                trust_remote_code=True,
                device=device,
                cache_folder=cache_folder,
            )
            model = encoder._model  # raw SentenceTransformer
            state = load_file(artifact_path, device="cpu")
            model[0].model.load_state_dict(state, strict=True)
            model.eval()
            return V3Ranker(model)
        except PrivateInputError:
            raise
        except Exception:
            _reject("model unavailable or incompatible")

    return model_factory


def model_factory(seed: int) -> V3Ranker:
    """Environment-configured entry point for a private isolated invocation."""
    artifact = os.environ.get("FAILURELENS_V3_ARTIFACT")
    if not artifact:
        _reject("unavailable model artifact")
    return make_model_factory(
        Path(artifact),
        cache_folder=os.environ.get("FAILURELENS_V3_CACHE"),
        device=os.environ.get("FAILURELENS_V3_DEVICE", "cpu"),
    )(seed)
