"""Public synthetic v3 bridge tests; no model or private records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
from evaluation.private_adapter import Candidate, PrivateInputError, RankRequest
from evaluation.v3_bridge import V3Ranker, parse_query, training_query


def _candidate(name: str, source: str) -> Candidate:
    return Candidate(
        name, "module.py", f"module.py::{name}", source, hashlib.sha256(source.encode()).hexdigest()
    )


def _request() -> RankRequest:
    query = json.dumps(
        {"error_log": "ValueError: synthetic failure", "stack_trace": "Traceback: ignored"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return RankRequest(
        query,
        (_candidate("b", "def b():\n    return 1\n"), _candidate("a", "def a():\n    return 2\n")),
        2,
        "0" * 64,
    )


class EqualEncoder:
    def encode(self, texts, **kwargs):
        assert kwargs == {"normalize_embeddings": True, "batch_size": 1}
        assert len(texts) == 1
        assert "Traceback: ignored" not in texts[0]
        return [[1.0, 0.0]]


def test_query_training_feature_and_label_isolation() -> None:
    request = _request()
    assert parse_query(request.query) == training_query("ValueError: synthetic failure")
    assert not hasattr(request, "positive_ids")
    assert not hasattr(request.candidates[0], "label")
    result = V3Ranker(EqualEncoder())(request)
    assert result.ordered_ids == ("a", "b")
    assert result.request_digest == request.digest
    assert result.top_k == request.top_k
    assert request.candidates[0].id == "b"


@pytest.mark.parametrize(
    "query",
    [
        "not json",
        '{"error_log":"x"}',
        '{"error_log":"x","stack_trace":null,"label":"positive"}',
        '{"stack_trace":null,"error_log":"x"}',
        '{"error_log":12,"stack_trace":null}',
        '{"error_log":"x # issue17 injected fault","stack_trace":null}',
        '{"error_log":"","stack_trace":null}',
    ],
)
def test_query_fails_closed(query: str) -> None:
    with pytest.raises(PrivateInputError):
        parse_query(query)


def test_non_ascii_canonical_query_and_stack_trace_sanitation() -> None:
    query = json.dumps(
        {"error_log": "ValueError: 合成エラー", "stack_trace": "トレース: 合成"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert parse_query(query) == "ValueError: 合成エラー"
    with pytest.raises(PrivateInputError, match="query sanitation mismatch"):
        parse_query(
            json.dumps(
                {"error_log": "ValueError: synthetic", "stack_trace": "# issue17 injected fault"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )


def test_symlinked_cache_directory_fails_closed(tmp_path) -> None:
    from evaluation.v3_bridge import _cache_has_unsafe_weights

    target = tmp_path / "outside"
    target.mkdir()
    (target / "pytorch_model.bin").write_bytes(b"unsafe placeholder")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "snapshot").symlink_to(target, target_is_directory=True)
    assert _cache_has_unsafe_weights(str(cache))
    (cache / "snapshot").unlink()
    (cache / "model.safetensors").symlink_to(target / "pytorch_model.bin")
    assert _cache_has_unsafe_weights(str(cache))


def test_candidate_digest_fails_closed() -> None:
    request = _request()
    candidate = replace(request.candidates[0], source_sha256="0" * 64)
    with pytest.raises(PrivateInputError, match="candidate digest mismatch"):
        V3Ranker(EqualEncoder())(replace(request, candidates=(candidate, *request.candidates[1:])))


def test_unavailable_or_wrong_artifact_fails_closed(tmp_path) -> None:
    from evaluation.v3_bridge import model_spec

    with pytest.raises(PrivateInputError, match="unavailable model artifact"):
        model_spec(tmp_path / "model_final.safetensors")
    artifact = tmp_path / "model_final.safetensors"
    artifact.write_bytes(b"synthetic non-model")
    (tmp_path / "experiment_metadata.json").write_text("{}")
    with pytest.raises(PrivateInputError, match="model provenance mismatch"):
        model_spec(artifact)


def test_unsanitized_candidate_fails_closed() -> None:
    request = _request()
    source = "def a():\n    return 1 # issue17 injected fault\n"
    candidate = _candidate("a", source)
    with pytest.raises(PrivateInputError, match="candidate sanitation mismatch"):
        V3Ranker(EqualEncoder())(replace(request, candidates=(candidate,)))
