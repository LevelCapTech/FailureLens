"""Optional embedding-based reranker (requires the ``ml`` extra).

This backend reranks the baseline's retrieved candidates by the cosine
similarity between an embedding of the failure and an embedding of each
candidate's source. It is **experimental** and uses an *externally hosted* base
model; FailureLens does not yet ship its own fine-tuned weights.

The reranking math is isolated from the model loading so it can be unit-tested
with a stub encoder (no heavy dependencies, no downloads).

Compat note
-----------
``transformers>=5.0`` removed ``ModelUtilsMixin.get_extended_attention_mask``.
The nomic-ai/CodeRankEmbed custom modelling code at revision
``3c4b60807d71f79b43f3c4363786d9493691f8b1`` calls that method.  The
:func:`_apply_nomic_bert_compat` adapter patches the method back onto the loaded
instance at construction time so that no downgrade or fork is required.  The
patch is strictly scoped: it is only applied when the inner model is missing the
method AND is identified as a NomicBert variant, never globally.
"""

from __future__ import annotations

import math
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from failurelens.config import DiagnoseConfig
from failurelens.errors import MissingDependencyError, ModelError
from failurelens.indexing.base import RepositoryIndex
from failurelens.inference.baseline import _make_source_reader
from failurelens.ranking.baseline import SourceReader
from failurelens.retrieval.baseline import RawCandidate, resolve_frames, retrieve_candidates
from failurelens.schemas import (
    AnalysisStatus,
    Candidate,
    DiagnosisResult,
    Evidence,
    EvidenceKind,
    ParsedLog,
    ScoreType,
)

# Exact audited commit for nomic-ai/CodeRankEmbed.
# trust_remote_code=True is only safe at this pin; changing the revision
# requires a fresh security audit.
_NOMIC_CODERANK_EMBED_AUDITED_REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"

# Allowlist of (model_id, revision) pairs permitted to use trust_remote_code=True.
# Expanding this list requires a separate security audit of the new revision's custom code.
_TRUST_REMOTE_CODE_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {("nomic-ai/CodeRankEmbed", _NOMIC_CODERANK_EMBED_AUDITED_REVISION)}
)


def _check_trust_remote_code_allowlist(model_id: str, revision: str | None) -> None:
    """Raise :class:`ModelError` when *model_id*/*revision* is not in the allowlist.

    trust_remote_code=True executes arbitrary Python bundled with the model.
    We only permit the single audited (model_id, revision) pair; the model_id
    must be an exact HuggingFace Hub repository ID, never a local path.
    Local paths cannot be verified by name alone and are therefore rejected.
    """
    if (model_id, revision) in _TRUST_REMOTE_CODE_ALLOWLIST:
        return
    raise ModelError(
        f"trust_remote_code=True is only permitted for the audited "
        f"(model_id, revision) pair.  Allowlist: {sorted(_TRUST_REMOTE_CODE_ALLOWLIST)}.  "
        f"Got model_id={model_id!r}, revision={revision!r}."
    )


@runtime_checkable
class EmbeddingEncoder(Protocol):
    """Minimal text encoder interface (a subset of SentenceTransformer)."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector per input text."""
        ...


def _nomic_bert_get_extended_attention_mask(
    self: object,
    attention_mask: object,
    input_shape: object,
    device: object = None,
    dtype: object = None,
) -> object:
    """Replaces ``transformers`` 4.x ``get_extended_attention_mask`` for NomicBert.

    Converts a 2-D ``(B, S)`` binary mask into the 4-D ``(B, 1, 1, S)`` additive
    float mask expected by the NomicBert attention computation.  Values of 1
    (attended) become 0.0 and values of 0 (padding) become ``-inf`` (represented
    as ``torch.finfo(float32).min`` for numerical stability).

    A 4-D mask is passed through unchanged.  Any other dimensionality raises
    ``ValueError`` so errors surface early instead of producing silent NaN output.
    """
    import torch

    if attention_mask is None:
        return None
    if attention_mask.dim() == 2:  # type: ignore[attr-defined]
        extended = attention_mask[:, None, None, :]  # type: ignore[index]
        extended = extended.to(dtype=torch.float32)
        extended = (1.0 - extended) * torch.finfo(torch.float32).min
        return extended
    if attention_mask.dim() == 4:  # type: ignore[attr-defined]
        return attention_mask
    raise ValueError(
        f"Unexpected attention_mask dim {attention_mask.dim()}; expected 2 or 4."  # type: ignore[attr-defined]
    )


def _apply_nomic_bert_compat(model: object) -> None:
    """Apply the ``get_extended_attention_mask`` shim to *model* if needed.

    Guards:
    * Only applied when the method is absent (no-op on future transformers versions
      that restore it, or models that already provide it).
    * Only applied when the inner model type name contains ``NomicBert`` — the shim
      is specific to NomicBert's attention math and must not be injected into
      unrelated models.

    ``model`` is expected to be a ``SentenceTransformer`` instance; ``model[0]``
    is the first transformer module and ``model[0].auto_model`` is the underlying
    HuggingFace model.
    """
    try:
        inner = model[0].auto_model  # type: ignore[index]
    except (TypeError, KeyError, AttributeError):
        return

    if hasattr(inner, "get_extended_attention_mask"):
        return  # already present; nothing to do

    inner_type = type(inner).__name__
    if "NomicBert" not in inner_type:
        return  # non-nomic guard: do not patch unrelated models

    inner.get_extended_attention_mask = types.MethodType(
        _nomic_bert_get_extended_attention_mask, inner
    )


def load_sentence_transformer(
    model_id: str,
    *,
    revision: str | None = None,
    device: str | None = None,
    trust_remote_code: bool = False,
    cache_folder: str | None = None,
) -> EmbeddingEncoder:
    """Load a Sentence-Transformers model as an :class:`EmbeddingEncoder`.

    For ``nomic-ai/CodeRankEmbed`` pass the exact audited revision and set
    ``trust_remote_code=True``; the caller is responsible for pinning::

        load_sentence_transformer(
            "nomic-ai/CodeRankEmbed",
            revision="3c4b60807d71f79b43f3c4363786d9493691f8b1",
            trust_remote_code=True,
        )

    The NomicBert compat shim is applied automatically when needed.

    Raises :class:`MissingDependencyError` if the ``ml`` extra is not installed.
    Raises :class:`ModelError` on load failure.
    """
    if trust_remote_code:
        _check_trust_remote_code_allowlist(model_id, revision)

    # Refuse to load if any .bin (pickle) weight files are present in a local path.
    # safetensors-only is required; .bin files must never silently fall back to pickle.
    local_path = Path(model_id)
    if local_path.is_dir():
        bin_files = list(local_path.rglob("*.bin"))
        if bin_files:
            raise ModelError(
                f"Refused to load {model_id!r}: found unsafe .bin weight files "
                f"that would be loaded via pickle: {[str(b) for b in bin_files]}. "
                "Only safetensors weights are permitted."
            )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised via the guard test
        raise MissingDependencyError(
            feature="The pretrained embedding backend",
            extra="ml",
            packages="sentence-transformers, torch",
        ) from exc

    try:
        model = SentenceTransformer(
            model_id,
            revision=revision,
            device=device,
            trust_remote_code=trust_remote_code,
            cache_folder=cache_folder,
            model_kwargs={"torch_dtype": "auto", "use_safetensors": True},
        )
    except Exception as exc:  # pragma: no cover - requires network/model
        raise ModelError(f"Could not load embedding model {model_id!r}: {exc}") from exc

    _apply_nomic_bert_compat(model)

    return _SentenceTransformerEncoder(model)


class _SentenceTransformerEncoder:
    """Adapts a SentenceTransformer to the :class:`EmbeddingEncoder` protocol."""

    def __init__(self, model: object) -> None:
        self._model = model

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        # normalize_embeddings keeps cosine similarity numerically stable.
        return self._model.encode(  # type: ignore[attr-defined,no-any-return]
            list(texts), normalize_embeddings=True
        ).tolist()


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_query(parsed_log: ParsedLog) -> str:
    parts: list[str] = []
    if parsed_log.error_type:
        parts.append(parsed_log.error_type)
    if parsed_log.error_message:
        parts.append(parsed_log.error_message)
    functions = [f.function for f in parsed_log.frames if f.function]
    if functions:
        parts.append("call path: " + " -> ".join(functions))
    return "\n".join(parts) if parts else "unknown failure"


class EmbeddingAnalyzer:
    """Reranks baseline candidates by embedding similarity (experimental)."""

    analysis_method = "embedding-rerank"

    def __init__(
        self,
        *,
        encoder: EmbeddingEncoder,
        model_id: str,
        model_revision: str | None = None,
    ) -> None:
        self._encoder = encoder
        self.model_id: str | None = model_id
        self.model_revision = model_revision

    def _result(
        self,
        *,
        repository: str,
        parsed_log: ParsedLog,
        status: AnalysisStatus,
        candidates: list[Candidate],
        warnings: list[str],
    ) -> DiagnosisResult:
        return DiagnosisResult(
            status=status,
            repository=repository,
            error_type=parsed_log.error_type,
            error_message=parsed_log.error_message,
            candidates=candidates,
            analysis_method=self.analysis_method,
            model_id=self.model_id,
            model_revision=self.model_revision,
            warnings=warnings,
        )

    def analyze(
        self,
        *,
        repository: str,
        parsed_log: ParsedLog,
        index: RepositoryIndex,
        config: DiagnoseConfig,
    ) -> DiagnosisResult:
        warnings: list[str] = list(index.warnings)

        if not parsed_log.frames:
            warnings.append("The log parsed but contained no stack frames to localize.")
            return self._result(
                repository=repository,
                parsed_log=parsed_log,
                status=AnalysisStatus.INSUFFICIENT_EVIDENCE,
                candidates=[],
                warnings=warnings,
            )

        frames = resolve_frames(parsed_log.frames, index)
        raw_candidates = retrieve_candidates(frames, index)
        if not raw_candidates:
            warnings.append(
                "Stack frames were found but none resolved to files in the target repository."
            )
            return self._result(
                repository=repository,
                parsed_log=parsed_log,
                status=AnalysisStatus.NO_CANDIDATES,
                candidates=[],
                warnings=warnings,
            )

        reader = _make_source_reader(Path(index.root))
        query = _build_query(parsed_log)
        candidate_texts = [self._candidate_text(candidate, reader) for candidate in raw_candidates]

        vectors = self._encoder.encode([query, *candidate_texts])
        query_vector = vectors[0]
        similarities = [
            _cosine_similarity(query_vector, vectors[i + 1]) for i in range(len(raw_candidates))
        ]

        order = sorted(
            range(len(raw_candidates)),
            key=lambda i: (similarities[i], raw_candidates[i].max_frame_order),
            reverse=True,
        )

        candidates: list[Candidate] = []
        for rank, idx in enumerate(order[: config.top_k], start=1):
            candidate = raw_candidates[idx]
            similarity = max(0.0, min(1.0, (similarities[idx] + 1.0) / 2.0))
            evidence = [
                *candidate.evidence,
                Evidence(
                    kind=EvidenceKind.SIMILARITY,
                    description=f"Embedding similarity to the failure: {similarity:.3f}.",
                ),
            ]
            candidates.append(
                Candidate(
                    rank=rank,
                    file=candidate.file,
                    function=candidate.function,
                    start_line=candidate.start_line,
                    end_line=candidate.end_line,
                    score=round(similarity, 6),
                    score_type=ScoreType.SIMILARITY,
                    evidence=evidence,
                )
            )

        return self._result(
            repository=repository,
            parsed_log=parsed_log,
            status=AnalysisStatus.OK,
            candidates=candidates,
            warnings=warnings,
        )

    @staticmethod
    def _candidate_text(candidate: RawCandidate, reader: SourceReader) -> str:
        source = reader(candidate.file)
        if source and candidate.start_line is not None:
            lines = source.splitlines()
            start = max(candidate.start_line - 1, 0)
            end = candidate.end_line if candidate.end_line is not None else candidate.start_line
            span = "\n".join(lines[start:end]).strip()
            if span:
                return span
        return candidate.qualified_name or candidate.file


__all__ = [
    "_NOMIC_CODERANK_EMBED_AUDITED_REVISION",
    "_TRUST_REMOTE_CODE_ALLOWLIST",
    "EmbeddingAnalyzer",
    "EmbeddingEncoder",
    "_check_trust_remote_code_allowlist",
    "load_sentence_transformer",
]
