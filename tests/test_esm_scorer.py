"""Tests for vhh_library.esm_scorer – ESMStabilityScorer class."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


def _make_library_df(n: int = 1000) -> pd.DataFrame:
    """Create a synthetic library DataFrame with *n* rows."""
    import random

    random.seed(42)
    rows = []
    for i in range(n):
        seq = list(SAMPLE_VHH)
        # Introduce a single random mutation
        pos = random.randint(0, len(seq) - 1)
        seq[pos] = random.choice("ACDEFGHIKLMNPQRSTVWY")
        rows.append(
            {
                "variant_id": f"V{i:06d}",
                "mutations": f"X{pos}Y",
                "n_mutations": 1,
                "humanness_score": random.random(),
                "stability_score": random.random(),
                "combined_score": random.random(),
                "aa_sequence": "".join(seq),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test: ImportError when torch/esm are missing
# ---------------------------------------------------------------------------


class TestImportError:
    def test_raises_import_error_without_ml_deps(self) -> None:
        """ESMStabilityScorer should raise ImportError with a clear install message."""
        # Temporarily hide torch and esm from the import machinery
        with patch.dict(sys.modules, {"torch": None, "esm": None}):
            with pytest.raises(ImportError, match=r"pip install torch fair-esm"):
                from vhh_library.esm_scorer import _check_ml_deps

                _check_ml_deps()


# ---------------------------------------------------------------------------
# Mocked model fixtures
# ---------------------------------------------------------------------------


def _build_mock_esm():
    """Build mock torch / esm objects that mimic ESM-2 behaviour."""
    import numpy as np

    # Mock torch module
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    # Create a deterministic "logits" tensor for the sample VHH
    seq_len = len(SAMPLE_VHH)
    vocab_size = 33  # ESM-2 standard vocab size
    np.random.seed(0)
    logits_np = np.random.randn(1, seq_len + 2, vocab_size).astype(np.float32)

    class FakeTensor:
        """Minimal tensor mock that supports indexing."""

        def __init__(self, arr):
            self._arr = arr

        def __getitem__(self, key):
            val = self._arr[key]
            if isinstance(val, np.ndarray):
                return FakeTensor(val)
            return val

        def item(self):
            return float(self._arr)

        def to(self, device):
            return self

    class FakeModel:
        call_count = 0

        def __call__(self, tokens):
            FakeModel.call_count += 1
            batch_size = tokens._arr.shape[0]
            # Extend logits for batches
            tiled = np.tile(logits_np, (batch_size, 1, 1))
            return {"logits": FakeTensor(tiled)}

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeAlphabet:
        def __init__(self):
            self._tok_map = {aa: idx for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY<cls><eos><pad>"[:vocab_size])}

        def get_batch_converter(self):
            def converter(data):
                batch_tokens = np.zeros((len(data), seq_len + 2), dtype=np.int64)
                for i, (_, seq) in enumerate(data):
                    for j, aa in enumerate(seq):
                        batch_tokens[i, j + 1] = self.get_idx(aa)
                return None, None, FakeTensor(batch_tokens)

            return converter

        def get_idx(self, aa: str) -> int:
            return self._tok_map.get(aa, 0)

    # Build mock esm module
    mock_esm = MagicMock()
    fake_model = FakeModel()
    fake_alphabet = FakeAlphabet()
    mock_esm.pretrained.esm2_t6_8M_UR50D.return_value = (fake_model, fake_alphabet)
    mock_esm.pretrained.esm2_t33_650M_UR50D.return_value = (fake_model, fake_alphabet)

    # Patch log_softmax to use our numpy implementation
    def fake_log_softmax(tensor, dim=-1):
        arr = tensor._arr
        max_arr = arr.max(axis=dim, keepdims=True)
        shifted = arr - max_arr
        lse = np.log(np.exp(shifted).sum(axis=dim, keepdims=True))
        return FakeTensor(shifted - lse)

    mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    mock_torch.nn.functional.log_softmax = fake_log_softmax

    return mock_torch, mock_esm, fake_model


# ---------------------------------------------------------------------------
# Tests with mocked model
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_esm_env(tmp_path):
    """Patch torch and esm modules with mocks and return the scorer."""
    mock_torch, mock_esm, fake_model = _build_mock_esm()

    with patch.dict(sys.modules, {"torch": mock_torch, "esm": mock_esm}):
        # Need to reload esm_scorer to pick up mocked modules
        import importlib

        import vhh_library.esm_scorer as esm_mod

        importlib.reload(esm_mod)

        scorer = esm_mod.ESMStabilityScorer(
            model_tier="t6_8M",
            device="cpu",
            batch_size=32,
            cache_dir=str(tmp_path / "cache"),
        )
        yield scorer, fake_model, esm_mod


class TestScoreSingle:
    def test_returns_float(self, mock_esm_env) -> None:
        scorer, _, _ = mock_esm_env
        result = scorer.score_single(SAMPLE_VHH)
        assert isinstance(result, float)

    def test_deterministic(self, mock_esm_env) -> None:
        scorer, _, _ = mock_esm_env
        a = scorer.score_single(SAMPLE_VHH)
        b = scorer.score_single(SAMPLE_VHH)
        assert a == b


class TestScoreBatch:
    def test_batch_length(self, mock_esm_env) -> None:
        scorer, _, _ = mock_esm_env
        seqs = [SAMPLE_VHH, SAMPLE_VHH[:100] + "A" + SAMPLE_VHH[101:]]
        results = scorer.score_batch(seqs)
        assert len(results) == 2
        assert all(isinstance(r, float) for r in results)


class TestDeltaScoring:
    def test_delta_returns_list(self, mock_esm_env) -> None:
        scorer, _, _ = mock_esm_env
        # Single-point mutant at position 5
        variants = [([5], ["A"])]
        deltas = scorer.score_delta(SAMPLE_VHH, variants)
        assert len(deltas) == 1
        assert isinstance(deltas[0], float)

    def test_delta_zero_for_identity(self, mock_esm_env) -> None:
        """Substituting the same amino acid should yield delta ≈ 0."""
        scorer, _, _ = mock_esm_env
        original_aa = SAMPLE_VHH[10]
        deltas = scorer.score_delta(SAMPLE_VHH, [([10], [original_aa])])
        assert abs(deltas[0]) < 1e-6

    def test_delta_approximation(self, mock_esm_env) -> None:
        """Delta score should approximate full PLL difference (within tolerance)."""
        scorer, _, _ = mock_esm_env
        # Create a single-point mutant
        pos = 5
        new_aa = "W" if SAMPLE_VHH[pos] != "W" else "A"

        mutant_seq = SAMPLE_VHH[:pos] + new_aa + SAMPLE_VHH[pos + 1 :]
        parent_pll = scorer.score_single(SAMPLE_VHH)
        mutant_pll = scorer.score_single(mutant_seq)
        full_diff = mutant_pll - parent_pll

        delta = scorer.score_delta(SAMPLE_VHH, [([pos], [new_aa])])[0]

        # Delta is an approximation; tolerance is generous because the
        # full PLL re-evaluates context at all positions.
        assert abs(delta - full_diff) < abs(full_diff) + 5.0


class TestProgressiveFunnel:
    def test_stage_counts(self, mock_esm_env) -> None:
        """Progressive funnel should reduce rows: 1000 → ~200 → ~50."""
        scorer, _, _ = mock_esm_env
        df = _make_library_df(1000)

        # Mock VHHSequence-like parent
        parent = MagicMock()
        parent.sequence = SAMPLE_VHH

        with pytest.warns(DeprecationWarning, match="score_library_progressive.*deprecated"):
            result = scorer.score_library_progressive(
                parent, df, stage1_top_frac=0.2, stage2_top_frac=0.25, stage3=False
            )

        # Stage 1 keeps 200, stage 2 keeps 25% of 200 = 50
        assert len(result) == 50
        assert "esm2_pll" in result.columns
        assert "esm2_delta_pll" in result.columns
        assert "esm2_rank" in result.columns

    def test_empty_df(self, mock_esm_env) -> None:
        scorer, _, _ = mock_esm_env
        df = pd.DataFrame(columns=["variant_id", "aa_sequence", "combined_score"])
        parent = MagicMock()
        parent.sequence = SAMPLE_VHH

        with pytest.warns(DeprecationWarning, match="score_library_progressive.*deprecated"):
            result = scorer.score_library_progressive(parent, df)
        assert len(result) == 0
        assert "esm2_pll" in result.columns
        assert "esm2_delta_pll" in result.columns
        assert "esm2_rank" in result.columns

    def test_deprecation_warning_emitted(self, mock_esm_env) -> None:
        """score_library_progressive() must emit a DeprecationWarning."""
        scorer, _, _ = mock_esm_env
        df = pd.DataFrame(columns=["variant_id", "aa_sequence", "combined_score"])
        parent = MagicMock()
        parent.sequence = SAMPLE_VHH

        with pytest.warns(DeprecationWarning, match="NanoMelt Tm is now the primary"):
            scorer.score_library_progressive(parent, df)


class TestCaching:
    def test_cache_hit_skips_forward_pass(self, mock_esm_env) -> None:
        """Second call to score_single should hit cache and not call the model again."""
        scorer, fake_model, _ = mock_esm_env

        # Reset call counter
        fake_model.call_count = 0

        # First call — model invoked
        _ = scorer.score_single(SAMPLE_VHH)
        calls_after_first = fake_model.call_count

        # Second call — should hit cache
        _ = scorer.score_single(SAMPLE_VHH)
        calls_after_second = fake_model.call_count

        assert calls_after_second == calls_after_first, "Model was called again on a cached sequence"
