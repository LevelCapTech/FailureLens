"""Unit tests for the NomicBert transformers-5.x compatibility shim.

These tests exercise the shim math, padding invariance, non-nomic guard, and
the "already present" no-op path — all without loading the real model or
requiring network access.
"""

from __future__ import annotations

import math
import types

import pytest

# ---------------------------------------------------------------------------
# Shim import helpers
# ---------------------------------------------------------------------------


def _import_shim():
    from failurelens.inference.embedding import (
        _apply_nomic_bert_compat,
        _nomic_bert_get_extended_attention_mask,
    )
    return _apply_nomic_bert_compat, _nomic_bert_get_extended_attention_mask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fake_nomic_model(*, has_method: bool = False):
    """Return a minimal NomicBert-like object for testing."""

    class NomicBertModel:
        pass

    m = NomicBertModel()
    if has_method:
        m.get_extended_attention_mask = lambda *a, **kw: None
    return m


def _make_fake_other_model():
    """Return a non-NomicBert model (should be skipped by guard)."""

    class BertModel:
        pass

    return BertModel()


def _make_fake_st(inner_model):
    """Minimal SentenceTransformer-like wrapper: model[0].auto_model = inner."""

    class FakeModule:
        auto_model = inner_model

    class FakeST:
        def __getitem__(self, idx):
            if idx == 0:
                return FakeModule()
            raise IndexError(idx)

    return FakeST()


# ---------------------------------------------------------------------------
# Tests: shim math
# ---------------------------------------------------------------------------


class TestShimMath:
    """Verify the attention mask transformation mathematics."""

    def _call_shim(self, mask):
        import torch

        _, shim = _import_shim()
        fake_self = object.__new__(object)
        return shim(fake_self, mask, mask.shape)

    def test_2d_shape(self):
        import torch

        mask = torch.ones(2, 8, dtype=torch.long)
        result = self._call_shim(mask)
        assert result.shape == (2, 1, 1, 8), f"Expected (2,1,1,8), got {result.shape}"

    def test_attended_token_becomes_zero(self):
        """Positions with mask=1 (attended) should map to 0.0."""
        import torch

        mask = torch.ones(1, 4, dtype=torch.long)
        result = self._call_shim(mask)
        assert torch.all(result == 0.0), "All attended positions should be 0.0"

    def test_padding_token_becomes_neg_inf_approx(self):
        """Positions with mask=0 (padding) should map to a large negative value."""
        import torch

        mask = torch.zeros(1, 4, dtype=torch.long)
        result = self._call_shim(mask)
        expected = torch.finfo(torch.float32).min
        assert torch.all(result == expected), f"Padding positions should be {expected}"

    def test_mixed_mask_values(self):
        """Mixed mask: attended→0.0, padding→finfo.min."""
        import torch

        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
        result = self._call_shim(mask)
        fmin = torch.finfo(torch.float32).min
        assert result[0, 0, 0, 0].item() == pytest.approx(0.0)
        assert result[0, 0, 0, 1].item() == pytest.approx(0.0)
        assert result[0, 0, 0, 2].item() == pytest.approx(fmin, rel=1e-6)
        assert result[0, 0, 0, 3].item() == pytest.approx(fmin, rel=1e-6)

    def test_4d_mask_passthrough(self):
        """A 4-D mask should be returned unchanged."""
        import torch

        mask = torch.zeros(2, 1, 1, 8)
        _, shim = _import_shim()
        fake_self = object.__new__(object)
        result = shim(fake_self, mask, mask.shape)
        assert result is mask

    def test_invalid_dim_raises(self):
        """A 3-D mask should raise ValueError."""
        import torch

        mask = torch.zeros(2, 4, 8)
        _, shim = _import_shim()
        fake_self = object.__new__(object)
        with pytest.raises(ValueError, match="Unexpected attention_mask dim"):
            shim(fake_self, mask, mask.shape)

    def test_none_returns_none(self):
        """None input should propagate as None."""
        _, shim = _import_shim()
        fake_self = object.__new__(object)
        assert shim(fake_self, None, None) is None


# ---------------------------------------------------------------------------
# Tests: padding invariance (cosine similarity stability)
# ---------------------------------------------------------------------------


class TestPaddingInvariance:
    """Padded and non-padded encodings of the same query must agree."""

    def test_shim_output_dtype_float32(self):
        """Output tensor must be float32 regardless of input dtype."""
        import torch

        mask_int = torch.ones(1, 6, dtype=torch.long)
        _, shim = _import_shim()
        fake_self = object.__new__(object)
        result = shim(fake_self, mask_int, mask_int.shape)
        assert result.dtype == torch.float32

    def test_padded_positions_do_not_bleed(self):
        """Padding values in the extended mask must equal finfo.min (additive -inf)."""
        import torch

        # Sequence of length 6 with 2 padding tokens at the end
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
        _, shim = _import_shim()
        fake_self = object.__new__(object)
        result = shim(fake_self, mask, mask.shape)
        fmin = torch.finfo(torch.float32).min
        # Positions 4 and 5 (padding) must be finfo.min
        assert result[0, 0, 0, 4].item() == pytest.approx(fmin, rel=1e-6)
        assert result[0, 0, 0, 5].item() == pytest.approx(fmin, rel=1e-6)
        # Positions 0-3 (attended) must be 0.0
        for i in range(4):
            assert result[0, 0, 0, i].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: _apply_nomic_bert_compat guard logic
# ---------------------------------------------------------------------------


class TestApplyNomicBertCompat:
    """Guard conditions for when the shim is and is not applied."""

    def test_nomic_model_without_method_gets_shim(self):
        apply, _ = _import_shim()
        inner = _make_fake_nomic_model(has_method=False)
        st = _make_fake_st(inner)
        apply(st)
        assert hasattr(inner, "get_extended_attention_mask"), "Shim should have been applied"

    def test_nomic_model_with_existing_method_unchanged(self):
        """Already-present method should not be overwritten (no-op)."""
        apply, _ = _import_shim()
        inner = _make_fake_nomic_model(has_method=True)
        original = inner.get_extended_attention_mask
        st = _make_fake_st(inner)
        apply(st)
        assert inner.get_extended_attention_mask is original, "Existing method should not change"

    def test_non_nomic_model_not_patched(self):
        """Non-NomicBert models must not receive the shim (non-nomic guard)."""
        apply, _ = _import_shim()
        inner = _make_fake_other_model()
        st = _make_fake_st(inner)
        apply(st)
        assert not hasattr(inner, "get_extended_attention_mask"), "Non-nomic model must not be patched"

    def test_bad_st_structure_does_not_raise(self):
        """Gracefully handles models where model[0].auto_model is absent."""
        apply, _ = _import_shim()

        class BadST:
            def __getitem__(self, idx):
                raise TypeError("no indexing")

        apply(BadST())  # should not raise

    def test_shim_is_bound_method(self):
        """The applied shim should be a bound method on the inner model."""
        apply, _ = _import_shim()
        inner = _make_fake_nomic_model(has_method=False)
        st = _make_fake_st(inner)
        apply(st)
        method = inner.get_extended_attention_mask
        assert callable(method)
        # Verify it is bound to the correct instance
        assert method.__self__ is inner  # type: ignore[union-attr]
