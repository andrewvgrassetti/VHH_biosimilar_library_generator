"""Tests for vhh_library.predictors.nanomelt — NanoMelt thermal-stability predictor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
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
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_backend(tm_values: list[float]) -> MagicMock:
    """Create a mock ``NanoMeltPredPipe`` callable returning a DataFrame."""
    df = pd.DataFrame({"NanoMelt Tm (C)": tm_values})
    mock_fn = MagicMock(return_value=df)
    return mock_fn


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
    return pred


# ---------------------------------------------------------------------------
# NanoMeltPredictor — with mocked backend
# ---------------------------------------------------------------------------


class TestNanoMeltPredictor:
    """Tests for NanoMeltPredictor using a mocked NanoMelt backend."""

    def test_name(self) -> None:
        mock_backend = _make_mock_backend([70.0])
        pred = _make_predictor_with_mock_backend(mock_backend)
        assert pred.name == "nanomelt"

    def test_is_predictor_subclass(self) -> None:
        mock_backend = _make_mock_backend([70.0])
        pred = _make_predictor_with_mock_backend(mock_backend)
        assert isinstance(pred, Predictor)

    def test_score_sequence_returns_expected_keys(self, vhh: VHHSequence) -> None:
        mock_backend = _make_mock_backend([68.5])
        pred = _make_predictor_with_mock_backend(mock_backend)

        result = pred.score_sequence(vhh)
        assert "composite_score" in result
        assert "nanomelt_tm" in result
        assert result["nanomelt_tm"] == pytest.approx(68.5)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_score_sequence_calls_backend(self, vhh: VHHSequence) -> None:
        mock_backend = _make_mock_backend([70.0])
        pred = _make_predictor_with_mock_backend(mock_backend)

        pred.score_sequence(vhh)
        mock_backend.assert_called_once()
        call_kwargs = mock_backend.call_args.kwargs
        assert call_kwargs["do_align"] is True
        assert call_kwargs["ncpus"] == 1
        assert len(call_kwargs["seq_records"]) == 1

    def test_nanomelt_tm_pred(self, vhh: VHHSequence) -> None:
        mock_backend = _make_mock_backend([72.3])
        pred = _make_predictor_with_mock_backend(mock_backend)

        tm = pred.nanomelt_tm_pred(vhh)
        assert tm == pytest.approx(72.3)

    def test_delta_nanomelt_tm(self, vhh: VHHSequence) -> None:
        mock_backend = _make_mock_backend([70.0, 73.5])
        pred = _make_predictor_with_mock_backend(mock_backend)

        delta = pred.delta_nanomelt_tm(vhh, vhh)
        assert delta == pytest.approx(3.5)

    def test_delta_nanomelt_tm_destabilising(self, vhh: VHHSequence) -> None:
        mock_backend = _make_mock_backend([70.0, 65.0])
        pred = _make_predictor_with_mock_backend(mock_backend)

        delta = pred.delta_nanomelt_tm(vhh, vhh)
        assert delta == pytest.approx(-5.0)

    def test_score_batch(self, vhh: VHHSequence) -> None:
        mock_backend = _make_mock_backend([68.0, 72.0, 65.0])
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
        mock_backend = _make_mock_backend([])
        pred = _make_predictor_with_mock_backend(mock_backend)
        assert pred.score_batch([]) == []
        mock_backend.assert_not_called()

    def test_score_batch_forwards_batch_size(self, vhh: VHHSequence) -> None:
        mock_backend = _make_mock_backend([70.0])
        pred = _make_predictor_with_mock_backend(mock_backend, batch_size=32)

        pred.score_batch([vhh])
        call_kwargs = mock_backend.call_args.kwargs
        assert call_kwargs.get("batch_size") == 32

    def test_score_batch_no_batch_size_when_none(self, vhh: VHHSequence) -> None:
        mock_backend = _make_mock_backend([70.0])
        pred = _make_predictor_with_mock_backend(mock_backend, batch_size=None)

        pred.score_batch([vhh])
        call_kwargs = mock_backend.call_args.kwargs
        assert "batch_size" not in call_kwargs

    def test_composite_score_in_range(self, vhh: VHHSequence) -> None:
        mock_backend = _make_mock_backend([67.5])
        pred = _make_predictor_with_mock_backend(mock_backend)

        result = pred.score_sequence(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_composite_score_high_tm(self, vhh: VHHSequence) -> None:
        """Very high Tm should yield composite near 1.0."""
        mock_backend = _make_mock_backend([95.0])
        pred = _make_predictor_with_mock_backend(mock_backend)

        result = pred.score_sequence(vhh)
        assert result["composite_score"] > 0.95

    def test_composite_score_low_tm(self, vhh: VHHSequence) -> None:
        """Very low Tm should yield composite near 0.0."""
        mock_backend = _make_mock_backend([20.0])
        pred = _make_predictor_with_mock_backend(mock_backend)

        result = pred.score_sequence(vhh)
        assert result["composite_score"] < 0.05

    def test_seqrecord_conversion(self, vhh: VHHSequence) -> None:
        """Verify that VHH sequences are converted to SeqRecord objects."""
        mock_backend = _make_mock_backend([70.0])
        pred = _make_predictor_with_mock_backend(mock_backend)

        pred.score_sequence(vhh)
        call_kwargs = mock_backend.call_args.kwargs
        records = call_kwargs["seq_records"]
        assert len(records) == 1
        assert str(records[0].seq) == vhh.sequence

    def test_backend_stdout_suppressed(self, vhh: VHHSequence, capsys: pytest.CaptureFixture[str]) -> None:
        """NanoMelt backend stdout (e.g. 'Loading ESM data') must be suppressed."""

        def _noisy_backend(**kwargs):
            # Simulate NanoMelt printing to stdout
            print("Loading ESM data")
            return pd.DataFrame({"NanoMelt Tm (C)": [70.0]})

        pred = _make_predictor_with_mock_backend(MagicMock(side_effect=_noisy_backend))
        pred.score_sequence(vhh)
        captured = capsys.readouterr()
        assert "Loading ESM data" not in captured.out


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
        mock_pred_pipe = MagicMock(return_value=pd.DataFrame({"NanoMelt Tm (C)": [70.0]}))

        with (
            patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", True),
            patch.dict(
                "sys.modules",
                {
                    "nanomelt": MagicMock(),
                    "nanomelt.predict": MagicMock(NanoMeltPredPipe=mock_pred_pipe),
                },
            ),
        ):
            pred = NanoMeltPredictor(device="cpu")
            assert pred._backend is None
            pred.score_sequence(vhh)
            assert pred._backend is mock_pred_pipe


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


class TestWarmUp:
    """Verify warm_up() pre-loads the backend and runs a dummy prediction."""

    def test_warm_up_loads_backend(self) -> None:
        """warm_up() should trigger _ensure_backend and run a prediction."""
        mock_backend = _make_mock_backend([65.0])
        pred = _make_predictor_with_mock_backend(mock_backend)
        pred.warm_up()
        # The mock backend should have been called once with the warm-up sequence.
        mock_backend.assert_called_once()
        call_kwargs = mock_backend.call_args.kwargs
        assert len(call_kwargs["seq_records"]) == 1
        assert call_kwargs["seq_records"][0].id == "warmup"

    def test_warm_up_is_non_fatal(self) -> None:
        """If the backend raises during warm-up, it should not propagate."""

        def _failing_backend(**kwargs):
            raise RuntimeError("GPU unavailable")

        mock_backend = MagicMock(side_effect=_failing_backend)
        pred = _make_predictor_with_mock_backend(mock_backend)
        # Must not raise — warm_up is non-fatal.
        pred.warm_up()

    def test_warm_up_before_score_batch(self, vhh: VHHSequence) -> None:
        """After warm_up, score_batch still works normally."""
        call_count = 0

        def _counting_backend(**kwargs):
            nonlocal call_count
            n = len(kwargs["seq_records"])
            call_count += 1
            return pd.DataFrame({"NanoMelt Tm (C)": [70.0] * n})

        mock_backend = MagicMock(side_effect=_counting_backend)
        pred = _make_predictor_with_mock_backend(mock_backend)
        pred.warm_up()
        assert call_count == 1  # warm-up call
        results = pred.score_batch([vhh, vhh])
        assert call_count == 2  # score_batch call
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------


class TestDeviceHandling:
    """Verify device parameter warning and storage."""

    def test_device_stored(self) -> None:
        with patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", True):
            pred = NanoMeltPredictor(device="cpu")
        assert pred._device == "cpu"

    def test_auto_device_no_warning(self) -> None:
        with patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", True):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                NanoMeltPredictor(device="auto")

    def test_non_standard_device_warns(self) -> None:
        with patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", True):
            with pytest.warns(UserWarning, match="device.*ignored"):
                NanoMeltPredictor(device="cuda")


# ---------------------------------------------------------------------------
# Constructor parameters
# ---------------------------------------------------------------------------


class TestConstructorParams:
    """Verify new constructor parameters are stored and forwarded."""

    def test_do_align_default(self) -> None:
        with patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", True):
            pred = NanoMeltPredictor()
        assert pred._do_align is True

    def test_ncpus_default(self) -> None:
        with patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", True):
            pred = NanoMeltPredictor()
        assert pred._ncpus == 1

    def test_custom_do_align_and_ncpus(self, vhh: VHHSequence) -> None:
        mock_backend = _make_mock_backend([70.0])
        with patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", True):
            pred = NanoMeltPredictor(do_align=False, ncpus=4)
        pred._backend = mock_backend

        pred.score_sequence(vhh)
        call_kwargs = mock_backend.call_args.kwargs
        assert call_kwargs["do_align"] is False
        assert call_kwargs["ncpus"] == 4


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
