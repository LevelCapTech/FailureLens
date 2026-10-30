"""Integration smoke test for nomic-ai/CodeRankEmbed under transformers>=5.x.

Requires the ``ml`` extra AND the model cache at the audited revision.
Skipped unless ``--run-ml-integration`` is passed.

Run locally:
    pytest tests/inference/test_nomic_coderankeembed_integration.py \
        --run-ml-integration -v
"""

from __future__ import annotations

import os
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_MODEL_CACHE_ROOT = _REPO_ROOT / "models" / "cache" / "nomic-coderankeembed-smoke"
_REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"
_SNAPSHOT_PATH = (
    _MODEL_CACHE_ROOT
    / "models--nomic-ai--CodeRankEmbed"
    / "snapshots"
    / _REVISION
)


def _skip_unless_ml_integration(request):
    if not request.config.getoption("--run-ml-integration", default=False):
        pytest.skip("Pass --run-ml-integration to run ML integration tests")


@pytest.fixture(scope="module")
def encoder(request):
    _skip_unless_ml_integration(request)
    try:
        import torch
    except ImportError:
        pytest.skip("ml extra not installed (torch missing)")

    from failurelens.inference.embedding import (
        _NOMIC_CODERANKEEMBED_AUDITED_REVISION,
        load_sentence_transformer,
    )

    assert _REVISION == _NOMIC_CODERANKEEMBED_AUDITED_REVISION, (
        f"Test pin {_REVISION!r} != audited pin {_NOMIC_CODERANKEEMBED_AUDITED_REVISION!r}"
    )

    if not _SNAPSHOT_PATH.exists():
        pytest.skip(f"Model snapshot not found at {_SNAPSHOT_PATH}")

    # Load from local snapshot path to work offline without HF hub network calls.
    return load_sentence_transformer(
        str(_SNAPSHOT_PATH),
        trust_remote_code=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


SYNTHETIC_STRINGS = [
    "AttributeError: 'NoneType' object has no attribute 'split'\n  File app.py line 42",
    "def process_input(raw):\n    return raw.split(',')",
    "import json\ndef load_config(path):\n    with open(path) as f:\n        return json.load(f)",
]


class TestCodeRankEmbedSmoke:
    def test_output_shape(self, encoder):
        import numpy as np

        vecs = encoder.encode(SYNTHETIC_STRINGS)
        arr = np.array(vecs)
        assert arr.shape == (3, 768), f"Expected (3, 768), got {arr.shape}"

    def test_all_finite(self, encoder):
        import numpy as np

        vecs = encoder.encode(SYNTHETIC_STRINGS)
        arr = np.array(vecs)
        assert np.all(np.isfinite(arr)), "Embeddings contain non-finite values"

    def test_normalized(self, encoder):
        import numpy as np

        vecs = encoder.encode(SYNTHETIC_STRINGS)
        arr = np.array(vecs)
        norms = np.linalg.norm(arr, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5, err_msg="Embeddings are not L2-normalized")

    def test_padding_invariance(self, encoder):
        """Single-item encode and batch encode must agree within tolerance."""
        import numpy as np

        single = np.array(encoder.encode([SYNTHETIC_STRINGS[0]])[0])
        batch = np.array(encoder.encode(SYNTHETIC_STRINGS)[0])
        cos_sim = float(np.dot(single, batch))
        assert abs(cos_sim - 1.0) < 1e-4, (
            f"Cosine similarity {cos_sim:.6f} != 1.0 (padding not invariant)"
        )

    def test_safetensors_only(self, request):
        _skip_unless_ml_integration(request)
        bin_files = list(_MODEL_CACHE_ROOT.rglob("*.bin"))
        assert bin_files == [], f"Found .bin files in cache: {bin_files}"
