"""Tests for vhh_library.benchmark — validation and benchmarking framework."""

from __future__ import annotations

import json
import math
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from vhh_library.benchmark import (
    BenchmarkReport,
    CorrelationMetrics,
    CrossValidationResult,
    LOOPrediction,
    compare_scoring_methods,
    compute_correlation_metrics,
    cross_validate_calibration,
    leave_one_out_predictions,
    load_benchmark_dataset,
    plot_correlation_scatter,
    plot_residuals,
    plot_scoring_comparison,
    run_benchmark,
    validate_library_predictions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """Generate synthetic predicted/experimental data with known correlation."""
    rng = np.random.default_rng(42)
    n = 30
    experimental = rng.uniform(45, 85, n).tolist()
    # predicted ≈ experimental + noise
    predicted = [e + rng.normal(0, 3) for e in experimental]
    return predicted, experimental


@pytest.fixture
def perfect_data():
    """Perfectly correlated data."""
    experimental = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]
    predicted = experimental[:]
    return predicted, experimental


@pytest.fixture
def constant_data():
    """All values the same."""
    vals = [65.0] * 5
    return vals[:], vals[:]


@pytest.fixture
def small_data():
    """Minimal 2-sample data."""
    return [60.0, 70.0], [62.0, 68.0]


@pytest.fixture
def benchmark_json(tmp_path):
    """Create a temporary benchmark JSON file."""
    data = {
        "version": 1,
        "description": "Test benchmark data",
        "benchmark_vhhs": [
            {
                "name": f"VHH_{i}",
                "sequence": "QVQLVES" + "A" * (100 + i),
                "experimental_tm": 50.0 + i * 3.0,
                "source": "test",
            }
            for i in range(10)
        ],
    }
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# load_benchmark_dataset
# ---------------------------------------------------------------------------


class TestLoadBenchmarkDataset:
    def test_loads_builtin_dataset(self):
        vhhs = load_benchmark_dataset()
        assert len(vhhs) >= 5
        for v in vhhs:
            assert "name" in v
            assert "sequence" in v
            assert "experimental_tm" in v

    def test_loads_custom_file(self, benchmark_json):
        vhhs = load_benchmark_dataset(benchmark_json)
        assert len(vhhs) == 10
        assert vhhs[0]["name"] == "VHH_0"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_benchmark_dataset(tmp_path / "nonexistent.json")

    def test_raises_on_empty_entries(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"benchmark_vhhs": []}))
        with pytest.raises(ValueError, match="no VHH entries"):
            load_benchmark_dataset(path)


# ---------------------------------------------------------------------------
# compute_correlation_metrics
# ---------------------------------------------------------------------------


class TestComputeCorrelationMetrics:
    def test_perfect_correlation(self, perfect_data):
        pred, exp = perfect_data
        m = compute_correlation_metrics(pred, exp)
        assert m.spearman_rho == pytest.approx(1.0, abs=1e-6)
        assert m.pearson_r == pytest.approx(1.0, abs=1e-6)
        assert m.mae == pytest.approx(0.0, abs=1e-6)
        assert m.rmse == pytest.approx(0.0, abs=1e-6)
        assert m.ranking_accuracy == pytest.approx(1.0, abs=1e-6)
        assert m.n_samples == 7

    def test_synthetic_data(self, synthetic_data):
        pred, exp = synthetic_data
        m = compute_correlation_metrics(pred, exp)
        assert m.n_samples == 30
        assert m.spearman_rho > 0.8  # should be highly correlated
        assert m.pearson_r > 0.8
        assert m.mae > 0  # not perfect
        assert m.rmse > 0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            compute_correlation_metrics([1.0, 2.0], [1.0])

    def test_single_sample(self):
        m = compute_correlation_metrics([65.0], [65.0])
        assert m.n_samples == 1
        assert math.isnan(m.spearman_rho)
        assert math.isnan(m.pearson_r)

    def test_all_same_values(self, constant_data):
        pred, exp = constant_data
        m = compute_correlation_metrics(pred, exp)
        assert m.n_samples == 5
        assert m.mae == pytest.approx(0.0, abs=1e-6)
        # Spearman/Pearson may be NaN for constant data
        # Just ensure no crash

    def test_handles_nan_values(self):
        pred = [1.0, float("nan"), 3.0, 4.0]
        exp = [1.1, 2.0, float("nan"), 3.9]
        m = compute_correlation_metrics(pred, exp)
        assert m.n_samples == 2  # only (1.0, 1.1) and (4.0, 3.9) survive

    def test_all_nan(self):
        m = compute_correlation_metrics([float("nan"), float("nan")], [1.0, 2.0])
        assert m.n_samples == 0
        assert math.isnan(m.spearman_rho)

    def test_negative_correlation(self):
        pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        exp = [5.0, 4.0, 3.0, 2.0, 1.0]
        m = compute_correlation_metrics(pred, exp)
        assert m.spearman_rho == pytest.approx(-1.0, abs=1e-6)
        assert m.ranking_accuracy == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# cross_validate_calibration
# ---------------------------------------------------------------------------


class TestCrossValidateCalibration:
    def test_basic_cv(self):
        rng = np.random.default_rng(0)
        plls = rng.uniform(-2, -0.5, 20).tolist()
        tms = [60.0 + p * 10 for p in plls]  # linear relationship
        result = cross_validate_calibration(plls, tms, k=5)
        assert isinstance(result, CrossValidationResult)
        assert result.k == 5
        assert len(result.per_fold_r2) == 5
        assert len(result.per_fold_mae) == 5
        assert result.mean_r2 > 0.5  # should be decent for linear data

    def test_cv_caps_folds_at_n(self):
        plls = [-1.0, -0.8, -0.6, -0.4]
        tms = [50.0, 55.0, 60.0, 65.0]
        result = cross_validate_calibration(plls, tms, k=10)
        # k is capped to n=4
        assert result.k == 4

    def test_cv_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            cross_validate_calibration([1.0, 2.0], [3.0])

    def test_cv_too_few_samples(self):
        with pytest.raises(ValueError, match="At least 2 samples"):
            cross_validate_calibration([1.0], [2.0])

    def test_cv_reproducible(self):
        plls = [-1.0, -0.8, -0.6, -0.4, -0.3]
        tms = [50.0, 55.0, 60.0, 65.0, 67.0]
        r1 = cross_validate_calibration(plls, tms, k=3, seed=42)
        r2 = cross_validate_calibration(plls, tms, k=3, seed=42)
        assert len(r1.per_fold_r2) == len(r2.per_fold_r2)
        for a, b in zip(r1.per_fold_r2, r2.per_fold_r2):
            if math.isnan(a):
                assert math.isnan(b)
            else:
                assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# leave_one_out_predictions
# ---------------------------------------------------------------------------


class TestLeaveOneOutPredictions:
    def test_basic_loo(self):
        names = [f"V{i}" for i in range(5)]
        plls = [-1.0, -0.8, -0.6, -0.4, -0.2]
        tms = [50.0, 55.0, 60.0, 65.0, 70.0]
        results = leave_one_out_predictions(names, plls, tms)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, LOOPrediction)
            assert isinstance(r.predicted_tm, float)
            assert r.residual == pytest.approx(r.predicted_tm - r.experimental_tm)

    def test_loo_too_few_samples(self):
        with pytest.raises(ValueError, match="At least 3"):
            leave_one_out_predictions(["A", "B"], [-1.0, -0.5], [50.0, 60.0])

    def test_loo_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            leave_one_out_predictions(["A", "B", "C"], [-1.0, -0.5], [50.0, 60.0, 70.0])

    def test_loo_residuals_reasonable(self):
        """For a perfect linear relationship, LOO residuals should be small."""
        names = [f"V{i}" for i in range(10)]
        plls = [float(i) * 0.1 for i in range(10)]
        tms = [50.0 + 20 * p for p in plls]  # perfect linear
        results = leave_one_out_predictions(names, plls, tms)
        for r in results:
            assert abs(r.residual) < 1.0  # near-zero for perfect data


# ---------------------------------------------------------------------------
# validate_library_predictions
# ---------------------------------------------------------------------------


class TestValidateLibraryPredictions:
    def test_basic_validation(self):
        library_df = pd.DataFrame({
            "variant_id": ["v1", "v2", "v3", "v4", "v5"],
            "composite_score": [0.9, 0.8, 0.7, 0.6, 0.5],
        })
        csv_data = "variant_id,experimental_tm\nv1,75\nv2,72\nv3,68\nv4,63\nv5,58\n"
        m = validate_library_predictions(library_df, StringIO(csv_data))
        assert m.n_samples == 5
        assert m.spearman_rho > 0.9

    def test_partial_overlap(self):
        library_df = pd.DataFrame({
            "variant_id": ["v1", "v2", "v3"],
            "composite_score": [0.9, 0.8, 0.7],
        })
        csv_data = "variant_id,experimental_tm\nv1,75\nv2,72\nv99,50\n"
        m = validate_library_predictions(library_df, StringIO(csv_data))
        assert m.n_samples == 2  # only v1, v2 matched

    def test_no_overlap_raises(self):
        library_df = pd.DataFrame({
            "variant_id": ["v1"],
            "composite_score": [0.9],
        })
        csv_data = "variant_id,experimental_tm\nv99,50\n"
        with pytest.raises(ValueError, match="No matching"):
            validate_library_predictions(library_df, StringIO(csv_data))

    def test_auto_detects_tm_column(self):
        library_df = pd.DataFrame({
            "variant_id": ["v1", "v2"],
            "composite_score": [0.9, 0.6],
        })
        csv_data = "variant_id,Tm\nv1,75\nv2,60\n"
        m = validate_library_predictions(library_df, StringIO(csv_data))
        assert m.n_samples == 2

    def test_auto_detects_ranking_column(self):
        library_df = pd.DataFrame({
            "variant_id": ["v1", "v2", "v3"],
            "composite_score": [0.9, 0.8, 0.7],
        })
        csv_data = "variant_id,ranking\nv1,1\nv2,2\nv3,3\n"
        m = validate_library_predictions(library_df, StringIO(csv_data))
        assert m.n_samples == 3

    def test_missing_experimental_column_raises(self):
        library_df = pd.DataFrame({
            "variant_id": ["v1"],
            "composite_score": [0.9],
        })
        csv_data = "variant_id,some_col\nv1,foo\n"
        with pytest.raises(ValueError, match="Could not find"):
            validate_library_predictions(library_df, StringIO(csv_data))


# ---------------------------------------------------------------------------
# compare_scoring_methods
# ---------------------------------------------------------------------------


class TestCompareScoringMethods:
    def test_compare_multiple_methods(self):
        exp_tms = [50.0, 55.0, 60.0, 65.0, 70.0]
        results = compare_scoring_methods(
            exp_tms,
            {
                "method_good": [51.0, 54.0, 61.0, 64.0, 69.0],
                "method_random": [70.0, 50.0, 65.0, 55.0, 60.0],
            },
        )
        assert "method_good" in results
        assert "method_random" in results
        assert results["method_good"].spearman_rho > results["method_random"].spearman_rho

    def test_skips_mismatched_lengths(self):
        results = compare_scoring_methods(
            [50.0, 60.0, 70.0],
            {"bad_method": [1.0, 2.0]},  # wrong length
        )
        assert "bad_method" not in results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class TestPlotting:
    def test_scatter_plot_creates_figure(self, synthetic_data):
        pred, exp = synthetic_data
        metrics = compute_correlation_metrics(pred, exp)
        fig = plot_correlation_scatter(pred, exp, metrics=metrics)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_scatter_plot_without_metrics(self, synthetic_data):
        pred, exp = synthetic_data
        fig = plot_correlation_scatter(pred, exp)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_residual_plot(self, synthetic_data):
        pred, exp = synthetic_data
        fig = plot_residuals(pred, exp)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_scoring_comparison_plot(self):
        comparison = {
            "A": CorrelationMetrics(0.9, 0.01, 0.85, 0.01, 2.0, 3.0, 0.85, 10),
            "B": CorrelationMetrics(0.5, 0.1, 0.45, 0.1, 5.0, 7.0, 0.6, 10),
        }
        fig = plot_scoring_comparison(comparison)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# run_benchmark (high-level)
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def test_with_composite_scores(self, benchmark_json):
        vhhs = load_benchmark_dataset(benchmark_json)
        scores = [v["experimental_tm"] + 1.0 for v in vhhs]  # close to truth
        report = run_benchmark(benchmark_vhhs=vhhs, composite_scores=scores)
        assert isinstance(report, BenchmarkReport)
        assert report.correlation.n_samples == 10
        assert report.correlation.spearman_rho > 0.9

    def test_with_plls(self, benchmark_json):
        vhhs = load_benchmark_dataset(benchmark_json)
        # Fake PLLs that are linearly related to Tm
        plls = [v["experimental_tm"] * -0.02 for v in vhhs]
        report = run_benchmark(benchmark_vhhs=vhhs, per_residue_plls=plls, cv_folds=3)
        assert report.cross_validation is not None
        assert report.cross_validation.k == 3
        assert len(report.loo_predictions) == 10

    def test_with_no_scores(self, benchmark_json):
        vhhs = load_benchmark_dataset(benchmark_json)
        report = run_benchmark(benchmark_vhhs=vhhs)
        assert math.isnan(report.correlation.spearman_rho)

    def test_with_builtin_dataset(self):
        """Ensure the built-in dataset can be loaded and benchmarked."""
        vhhs = load_benchmark_dataset()
        plls = [float(i) * -0.1 for i in range(len(vhhs))]
        report = run_benchmark(benchmark_vhhs=vhhs, per_residue_plls=plls)
        assert isinstance(report, BenchmarkReport)
        assert report.correlation.n_samples == len(vhhs)
