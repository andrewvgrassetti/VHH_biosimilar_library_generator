"""Tests for vhh_library.predictors.nanomelt — NanoMelt thermal-stability predictor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vhh_library.predictors.base import Predictor
from vhh_library.predictors.nanomelt import (
    NANOMELT_AVAILABLE,
    NanoMeltPredictor,
    _sigmoid_normalize_tm,
)
from vhh_library.sequence import VHHSequence

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture(scope="module")
def vhh() -> VHHSequence:
    return VHHSequence(SAMPLE_VHH)


# ---------------------------------------------------------------------------
# Sigmoid normalisation helper
# ---------------------------------------------------------------------------


class TestSigmoidNormalizeTm:
    """Verify the Tm → [0, 1] normalisation."""

    def test_midpoint(self) -> None:
        """Tm at the centre of ideal window ≈ 0.5."""
        midpoint = (55.0 + 80.0) / 2.0  # 67.5
        val = _sigmoid_normalize_tm(midpoint)
        assert val == pytest.approx(0.5, abs=0.01)

    def test_high_tm(self) -> None:
        """Very high Tm should map to near 1.0."""
        val = _sigmoid_normalize_tm(120.0)
        assert val > 0.99

    def test_low_tm(self) -> None:
        """Very low Tm should map to near 0.0."""
        val = _sigmoid_normalize_tm(0.0)
        assert val < 0.01

    def test_moderate_tm(self) -> None:
        """A moderate Tm should be somewhere in (0, 1)."""
        val = _sigmoid_normalize_tm(65.0)
        assert 0.0 < val < 1.0

    def test_output_always_in_unit_interval(self) -> None:
        """For a range of Tms, result is always in [0, 1]."""
        for tm in range(-20, 120):
            val = _sigmoid_normalize_tm(float(tm))
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# NanoMeltPredictor — unavailable package
# ---------------------------------------------------------------------------


class TestNanoMeltUnavailable:
    """When nanomelt is not installed, construction should fail gracefully."""

    def test_import_error_when_unavailable(self) -> None:
        """Constructing NanoMeltPredictor raises ImportError when nanomelt is absent."""
        with patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", False):
            with pytest.raises(ImportError, match="nanomelt"):
                NanoMeltPredictor()

    def test_nanomelt_available_flag_is_bool(self) -> None:
        """NANOMELT_AVAILABLE should be a boolean."""
        assert isinstance(NANOMELT_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# NanoMeltPredictor — with mocked backend
# ---------------------------------------------------------------------------

# We need to bypass the NANOMELT_AVAILABLE guard while still mocking the backend.


def _make_predictor_with_mock_backend(
    mock_backend: MagicMock,
    device: str = "cpu",
    batch_size: int | None = None,
) -> NanoMeltPredictor:
    """Create a NanoMeltPredictor with a pre-injected mock backend."""
    with patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", True):
        pred = NanoMeltPredictor(device=device, batch_size=batch_size)
    # Inject mock backend directly (bypassing lazy loading)
    pred._backend = mock_backend
    pred._resolved_device = device
    return pred


class TestNanoMeltPredictor:
    """Tests for NanoMeltPredictor using a mocked NanoMelt backend."""

    def test_name(self) -> None:
        mock_backend = MagicMock()
        pred = _make_predictor_with_mock_backend(mock_backend)
        assert pred.name == "nanomelt"

    def test_is_predictor_subclass(self) -> None:
        mock_backend = MagicMock()
        pred = _make_predictor_with_mock_backend(mock_backend)
        assert isinstance(pred, Predictor)

    def test_score_sequence_returns_expected_keys(self, vhh: VHHSequence) -> None:
        mock_backend = MagicMock()
        mock_backend.predict_tm.return_value = [68.5]
        pred = _make_predictor_with_mock_backend(mock_backend)

        result = pred.score_sequence(vhh)
        assert "composite_score" in result
        assert "nanomelt_tm" in result
        assert result["nanomelt_tm"] == pytest.approx(68.5)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_score_sequence_calls_backend(self, vhh: VHHSequence) -> None:
        mock_backend = MagicMock()
        mock_backend.predict_tm.return_value = [70.0]
        pred = _make_predictor_with_mock_backend(mock_backend)

        pred.score_sequence(vhh)
        mock_backend.predict_tm.assert_called_once_with([vhh.sequence], device="cpu")

    def test_nanomelt_tm_pred(self, vhh: VHHSequence) -> None:
        mock_backend = MagicMock()
        mock_backend.predict_tm.return_value = [72.3]
        pred = _make_predictor_with_mock_backend(mock_backend)

        tm = pred.nanomelt_tm_pred(vhh)
        assert tm == pytest.approx(72.3)

    def test_delta_nanomelt_tm(self, vhh: VHHSequence) -> None:
        mock_backend = MagicMock()
        # Wild-type Tm = 70.0, mutant Tm = 73.5 → delta = +3.5
        mock_backend.predict_tm.return_value = [70.0, 73.5]
        pred = _make_predictor_with_mock_backend(mock_backend)

        delta = pred.delta_nanomelt_tm(vhh, vhh)
        assert delta == pytest.approx(3.5)

    def test_delta_nanomelt_tm_destabilising(self, vhh: VHHSequence) -> None:
        mock_backend = MagicMock()
        # Wild-type Tm = 70.0, mutant Tm = 65.0 → delta = -5.0
        mock_backend.predict_tm.return_value = [70.0, 65.0]
        pred = _make_predictor_with_mock_backend(mock_backend)

        delta = pred.delta_nanomelt_tm(vhh, vhh)
        assert delta == pytest.approx(-5.0)

    def test_score_batch(self, vhh: VHHSequence) -> None:
        mock_backend = MagicMock()
        mock_backend.predict_tm.return_value = [68.0, 72.0, 65.0]
        pred = _make_predictor_with_mock_backend(mock_backend)

        results = pred.score_batch([vhh, vhh, vhh])
        assert len(results) == 3
        for r in results:
            assert "composite_score" in r
            assert "nanomelt_tm" in r
            assert 0.0 <= r["composite_score"] <= 1.0

        assert results[0]["nanomelt_tm"] == pytest.approx(68.0)
        assert results[1]["nanomelt_tm"] == pytest.approx(72.0)
        assert results[2]["nanomelt_tm"] == pytest.approx(65.0)

    def test_score_batch_empty(self) -> None:
        mock_backend = MagicMock()
        pred = _make_predictor_with_mock_backend(mock_backend)
        assert pred.score_batch([]) == []
        mock_backend.predict_tm.assert_not_called()

    def test_score_batch_forwards_batch_size(self, vhh: VHHSequence) -> None:
        mock_backend = MagicMock()
        mock_backend.predict_tm.return_value = [70.0]
        pred = _make_predictor_with_mock_backend(mock_backend, batch_size=32)

        pred.score_batch([vhh])
        call_kwargs = mock_backend.predict_tm.call_args
        assert call_kwargs.kwargs.get("batch_size") == 32

    def test_score_batch_no_batch_size_when_none(self, vhh: VHHSequence) -> None:
        mock_backend = MagicMock()
        mock_backend.predict_tm.return_value = [70.0]
        pred = _make_predictor_with_mock_backend(mock_backend, batch_size=None)

        pred.score_batch([vhh])
        call_kwargs = mock_backend.predict_tm.call_args
        assert "batch_size" not in call_kwargs.kwargs

    def test_composite_score_in_range(self, vhh: VHHSequence) -> None:
        mock_backend = MagicMock()
        mock_backend.predict_tm.return_value = [67.5]
        pred = _make_predictor_with_mock_backend(mock_backend)

        result = pred.score_sequence(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_composite_score_high_tm(self, vhh: VHHSequence) -> None:
        """Very high Tm should yield composite near 1.0."""
        mock_backend = MagicMock()
        mock_backend.predict_tm.return_value = [95.0]
        pred = _make_predictor_with_mock_backend(mock_backend)

        result = pred.score_sequence(vhh)
        assert result["composite_score"] > 0.95

    def test_composite_score_low_tm(self, vhh: VHHSequence) -> None:
        """Very low Tm should yield composite near 0.0."""
        mock_backend = MagicMock()
        mock_backend.predict_tm.return_value = [20.0]
        pred = _make_predictor_with_mock_backend(mock_backend)

        result = pred.score_sequence(vhh)
        assert result["composite_score"] < 0.05


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------


class TestLazyLoading:
    """Verify backend is not created during __init__."""

    def test_backend_is_none_on_init(self) -> None:
        with patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", True):
            pred = NanoMeltPredictor(device="cpu")
        assert pred._backend is None

    def test_backend_created_on_first_score(self, vhh: VHHSequence) -> None:
        mock_backend_instance = MagicMock()
        mock_backend_instance.predict_tm.return_value = [70.0]

        with (
            patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", True),
            patch(
                "vhh_library.predictors.nanomelt._NanoMeltBackend",
                return_value=mock_backend_instance,
            ) as mock_cls,
        ):
            pred = NanoMeltPredictor(device="cpu")
            assert pred._backend is None
            pred.score_sequence(vhh)
            mock_cls.assert_called_once()
            assert pred._backend is mock_backend_instance


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


class TestDeviceHandling:
    """Verify device parameter is resolved and forwarded correctly."""

    def test_device_stored(self) -> None:
        with patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", True):
            pred = NanoMeltPredictor(device="cuda")
        assert pred._device == "cuda"

    def test_auto_device_resolves(self, vhh: VHHSequence) -> None:
        mock_backend_instance = MagicMock()
        mock_backend_instance.predict_tm.return_value = [70.0]

        with (
            patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", True),
            patch(
                "vhh_library.predictors.nanomelt._NanoMeltBackend",
                return_value=mock_backend_instance,
            ),
            patch(
                "vhh_library.predictors.nanomelt.resolve_device",
                return_value="cpu",
            ) as mock_resolve,
        ):
            pred = NanoMeltPredictor(device="auto")
            pred.score_sequence(vhh)
            mock_resolve.assert_called_once_with("auto")
            assert pred._resolved_device == "cpu"


# ---------------------------------------------------------------------------
# Integration test — only runs when nanomelt is installed
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not NANOMELT_AVAILABLE, reason="nanomelt package not installed")
class TestNanoMeltIntegration:
    """Integration tests that use the real NanoMelt backend.

    Skipped in CI unless nanomelt is installed.
    """

    def test_real_score_sequence(self, vhh: VHHSequence) -> None:
        pred = NanoMeltPredictor(device="cpu")
        result = pred.score_sequence(vhh)
        assert "composite_score" in result
        assert "nanomelt_tm" in result
        assert 0.0 <= result["composite_score"] <= 1.0
        # Tm should be a reasonable temperature (sanity check)
        assert 20.0 < result["nanomelt_tm"] < 100.0

    def test_real_nanomelt_tm_pred(self, vhh: VHHSequence) -> None:
        pred = NanoMeltPredictor(device="cpu")
        tm = pred.nanomelt_tm_pred(vhh)
        assert isinstance(tm, float)
        assert 20.0 < tm < 100.0

    def test_real_delta_tm(self, vhh: VHHSequence) -> None:
        pred = NanoMeltPredictor(device="cpu")
        # Same sequence → delta should be 0
        delta = pred.delta_nanomelt_tm(vhh, vhh)
        assert delta == pytest.approx(0.0, abs=0.01)

    def test_real_score_batch(self, vhh: VHHSequence) -> None:
        pred = NanoMeltPredictor(device="cpu")
        results = pred.score_batch([vhh, vhh])
        assert len(results) == 2
        for r in results:
            assert "composite_score" in r
            assert "nanomelt_tm" in r
