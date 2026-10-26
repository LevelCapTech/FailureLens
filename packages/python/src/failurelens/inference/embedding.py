"""Optional embedding-based reranker (requires the ``ml`` extra).

This backend reranks the baseline's retrieved candidates by the cosine
similarity between an embedding of the failure and an embedding of each
candidate's source. It is **experimental** and uses an *externally hosted* base
model; FailureLens does not yet ship its own fine-tuned weights.

The reranking math is isolated from the model loading so it can be unit-tested
with a stub encoder (no heavy dependencies, no downloads).
"""

from __future__ import annotations

import math
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


@runtime_checkable
class EmbeddingEncoder(Protocol):
    """Minimal text encoder interface (a subset of SentenceTransformer)."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector per input text."""
        ...


def load_sentence_transformer(
    model_id: str, *, revision: str | None = None, device: str | None = None
) -> EmbeddingEncoder:
    """Load a Sentence-Transformers model as an :class:`EmbeddingEncoder`.

    Raises :class:`MissingDependencyError` if the ``ml`` extra is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised via the guard test
        raise MissingDependencyError(
            feature="The pretrained embedding backend",
            extra="ml",
            packages="sentence-transformers, torch",
        ) from exc

    try:
        model = SentenceTransformer(model_id, revision=revision, device=device)
    except Exception as exc:  # pragma: no cover - requires network/model
        raise ModelError(f"Could not load embedding model {model_id!r}: {exc}") from exc

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
                    kind=EvidenceKind.LEXICAL_MATCH,
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
    "EmbeddingAnalyzer",
    "EmbeddingEncoder",
    "load_sentence_transformer",
]
