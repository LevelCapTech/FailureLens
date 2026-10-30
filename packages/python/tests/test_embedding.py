"""Tests for the optional embedding reranker.

These tests use a tiny deterministic stub encoder so the reranking logic can be
verified without installing heavy dependencies or downloading any model.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from failurelens import FailureLens
from failurelens.config import DiagnoseConfig
from failurelens.errors import MissingDependencyError, ModelError
from failurelens.indexing.repository import build_index
from failurelens.inference.embedding import (
    _NOMIC_CODERANKEEMBED_AUDITED_REVISION,
    _TRUST_REMOTE_CODE_ALLOWLIST,
    EmbeddingAnalyzer,
    EmbeddingEncoder,
    _check_trust_remote_code_allowlist,
    _cosine_similarity,
    load_sentence_transformer,
)
from failurelens.parsers.python_traceback import PythonTracebackParser
from failurelens.schemas import AnalysisStatus, ScoreType


class _StubEncoder:
    """Returns [1, 0] for the query and for texts containing ``target``, else [0, 1]."""

    def __init__(self, target: str) -> None:
        self.target = target

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors: list[list[float]] = []
        for position, text in enumerate(texts):
            if position == 0 or self.target in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def test_stub_encoder_satisfies_protocol() -> None:
    assert isinstance(_StubEncoder("x"), EmbeddingEncoder)


def test_embedding_rerank_reorders_candidates(sample_repo: Path, sample_traceback: str) -> None:
    index = build_index(sample_repo)
    parsed = PythonTracebackParser().parse(sample_traceback)
    assert parsed is not None

    # "values" appears only in average()'s source, so the stub makes it the best match.
    analyzer = EmbeddingAnalyzer(encoder=_StubEncoder("values"), model_id="stub/model")
    result = analyzer.analyze(
        repository=str(sample_repo),
        parsed_log=parsed,
        index=index,
        config=DiagnoseConfig(),
    )
    assert result.status is AnalysisStatus.OK
    assert result.analysis_method == "embedding-rerank"
    assert result.model_id == "stub/model"
    assert result.candidates[0].function == "average"
    assert result.candidates[0].score_type is ScoreType.SIMILARITY


def test_cosine_similarity() -> None:
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    # Degenerate inputs are handled without raising.
    assert _cosine_similarity([], [1.0]) == 0.0
    assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_load_sentence_transformer_missing_dependency() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(MissingDependencyError):
            load_sentence_transformer("any/model")
    else:  # pragma: no cover - only when the ml extra is installed
        pytest.skip("ml extra installed; dependency guard not exercised")


def test_from_pretrained_without_ml_extra_raises() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(MissingDependencyError):
            FailureLens.from_pretrained("any/model")
    else:  # pragma: no cover - only when the ml extra is installed
        pytest.skip("ml extra installed; dependency guard not exercised")


# ---------------------------------------------------------------------------
# Trust-remote-code allowlist enforcement (no ML deps required)
# ---------------------------------------------------------------------------


def test_allowlist_accepts_exact_pair() -> None:
    _check_trust_remote_code_allowlist(
        "nomic-ai/CodeRankEmbed", _NOMIC_CODERANKEEMBED_AUDITED_REVISION
    )


def test_allowlist_rejects_unpinned_revision() -> None:
    with pytest.raises(ModelError, match="only permitted"):
        _check_trust_remote_code_allowlist("nomic-ai/CodeRankEmbed", None)


def test_allowlist_rejects_wrong_revision() -> None:
    with pytest.raises(ModelError, match="only permitted"):
        _check_trust_remote_code_allowlist("nomic-ai/CodeRankEmbed", "deadbeef" * 5)


def test_allowlist_rejects_unaudited_repo() -> None:
    with pytest.raises(ModelError, match="only permitted"):
        _check_trust_remote_code_allowlist(
            "some-other/Model", _NOMIC_CODERANKEEMBED_AUDITED_REVISION
        )


def test_allowlist_rejects_unpinned_other_repo() -> None:
    with pytest.raises(ModelError, match="only permitted"):
        _check_trust_remote_code_allowlist("some-other/Model", None)


def test_allowlist_accepts_local_snapshot_path(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / _NOMIC_CODERANKEEMBED_AUDITED_REVISION
    snapshot_dir.mkdir()
    _check_trust_remote_code_allowlist(str(snapshot_dir), None)


def test_allowlist_rejects_local_path_with_wrong_dir_name(tmp_path: Path) -> None:
    wrong_dir = tmp_path / "deadbeef"
    wrong_dir.mkdir()
    with pytest.raises(ModelError, match="only permitted"):
        _check_trust_remote_code_allowlist(str(wrong_dir), None)


def test_load_rejects_trust_remote_code_for_unaudited_model() -> None:
    with pytest.raises(ModelError, match="only permitted"):
        load_sentence_transformer("evil-repo/bad-model", trust_remote_code=True)


def test_load_rejects_trust_remote_code_unpinned() -> None:
    with pytest.raises(ModelError, match="only permitted"):
        load_sentence_transformer("nomic-ai/CodeRankEmbed", trust_remote_code=True, revision=None)


def test_load_rejects_bin_files_in_local_dir(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "pytorch_model.bin").write_bytes(b"")
    with pytest.raises(ModelError, match=r"\.bin"):
        load_sentence_transformer(str(tmp_path), trust_remote_code=False)


def test_allowlist_constant_contains_audited_pair() -> None:
    assert (
        "nomic-ai/CodeRankEmbed",
        _NOMIC_CODERANKEEMBED_AUDITED_REVISION,
    ) in _TRUST_REMOTE_CODE_ALLOWLIST
