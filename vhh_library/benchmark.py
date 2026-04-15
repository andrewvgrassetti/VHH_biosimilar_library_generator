"""Validation and benchmarking framework for VHH stability predictions.

Provides correlation analysis, cross-validation, leave-one-out prediction,
and library-level validation of predicted vs. experimental stability data.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    import matplotlib.figure

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_BENCHMARK_FILE = _DATA_DIR / "benchmark_vhhs.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CorrelationMetrics:
    """Statistical correlation between predicted and experimental values."""

    spearman_rho: float
    spearman_pvalue: float
    pearson_r: float
    pearson_pvalue: float
    mae: float
    rmse: float
    ranking_accuracy: float
    n_samples: int


@dataclass
class CrossValidationResult:
    """Summary of k-fold cross-validation."""

    k: int
    per_fold_r2: list[float]
    per_fold_mae: list[float]
    mean_r2: float
    std_r2: float
    mean_mae: float
    std_mae: float


@dataclass
class LOOPrediction:
    """Leave-one-out prediction result for a single VHH."""

    name: str
    experimental_tm: float
    predicted_tm: float
    residual: float
    calibration_r2: float


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    correlation: CorrelationMetrics
    cross_validation: CrossValidationResult | None = None
    loo_predictions: list[LOOPrediction] = field(default_factory=list)
    scoring_comparison: dict[str, CorrelationMetrics] | None = None


# ---------------------------------------------------------------------------
# Benchmark dataset loading
# ---------------------------------------------------------------------------


def load_benchmark_dataset(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the built-in benchmark VHH dataset.

    Parameters
    ----------
    path:
        Override path to a benchmark JSON file.  When *None* the shipped
        ``data/benchmark_vhhs.json`` is used.

    Returns
    -------
    list[dict]
        Each dict has keys: ``name``, ``sequence``, ``experimental_tm``,
        ``source``.
    """
    bench_path = Path(path) if path is not None else _BENCHMARK_FILE
    if not bench_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {bench_path}")

    with open(bench_path) as fh:
        data = json.load(fh)

    vhhs = data.get("benchmark_vhhs", [])
    if not vhhs:
        raise ValueError("Benchmark file contains no VHH entries")
    return vhhs


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def compute_correlation_metrics(
    predicted: list[float],
    experimental: list[float],
) -> CorrelationMetrics:
    """Compute correlation statistics between predicted and experimental values.

    Parameters
    ----------
    predicted:
        Model-predicted values (e.g. predicted Tm or composite score).
    experimental:
        Ground-truth experimental values.

    Returns
    -------
    CorrelationMetrics
    """
    if len(predicted) != len(experimental):
        raise ValueError("predicted and experimental must have the same length")

    # Filter out NaN pairs
    pairs = [
        (p, e)
        for p, e in zip(predicted, experimental)
        if not (math.isnan(p) or math.isnan(e))
    ]
    n = len(pairs)
    if n < 2:
        return CorrelationMetrics(
            spearman_rho=float("nan"),
            spearman_pvalue=float("nan"),
            pearson_r=float("nan"),
            pearson_pvalue=float("nan"),
            mae=float("nan"),
            rmse=float("nan"),
            ranking_accuracy=float("nan"),
            n_samples=n,
        )

    pred_arr = np.array([p for p, _ in pairs])
    exp_arr = np.array([e for _, e in pairs])

    # Spearman & Pearson
    sp_rho, sp_p = stats.spearmanr(pred_arr, exp_arr)
    pe_r, pe_p = stats.pearsonr(pred_arr, exp_arr)

    # MAE & RMSE
    residuals = pred_arr - exp_arr
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # Ranking accuracy: fraction of correctly-ordered pairs
    ranking_accuracy = _compute_ranking_accuracy(pred_arr, exp_arr)

    return CorrelationMetrics(
        spearman_rho=float(sp_rho),
        spearman_pvalue=float(sp_p),
        pearson_r=float(pe_r),
        pearson_pvalue=float(pe_p),
        mae=mae,
        rmse=rmse,
        ranking_accuracy=ranking_accuracy,
        n_samples=n,
    )


def _compute_ranking_accuracy(predicted: np.ndarray, experimental: np.ndarray) -> float:
    """Fraction of pairwise comparisons where predicted ordering matches experimental."""
    n = len(predicted)
    if n < 2:
        return float("nan")

    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff_pred = predicted[i] - predicted[j]
            diff_exp = experimental[i] - experimental[j]
            # Skip ties in experimental values
            if abs(diff_exp) < 1e-12:
                continue
            total += 1
            if diff_pred * diff_exp > 0:
                concordant += 1

    return concordant / total if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_calibration(
    per_residue_plls: list[float],
    experimental_tms: list[float],
    k: int = 5,
    seed: int = 42,
) -> CrossValidationResult:
    """K-fold cross-validation of the PLL → Tm linear calibration.

    Parameters
    ----------
    per_residue_plls:
        Per-residue pseudo-log-likelihood scores.
    experimental_tms:
        Known melting temperatures.
    k:
        Number of folds.
    seed:
        Random seed for reproducible fold assignment.

    Returns
    -------
    CrossValidationResult
    """
    n = len(per_residue_plls)
    if n != len(experimental_tms):
        raise ValueError("per_residue_plls and experimental_tms must have the same length")
    if n < 2:
        raise ValueError("At least 2 samples required for cross-validation")
    k = min(k, n)  # cap folds at sample count
    if k < 2:
        raise ValueError("At least 2 folds required for cross-validation")

    pll_arr = np.array(per_residue_plls)
    tm_arr = np.array(experimental_tms)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    fold_r2: list[float] = []
    fold_mae: list[float] = []

    fold_sizes = np.array_split(indices, k)
    for fold_idx in fold_sizes:
        mask = np.ones(n, dtype=bool)
        mask[fold_idx] = False
        train_x, train_y = pll_arr[mask], tm_arr[mask]
        test_x, test_y = pll_arr[fold_idx], tm_arr[fold_idx]

        if len(train_x) < 2:
            continue

        coeffs = np.polyfit(train_x, train_y, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        pred_y = slope * test_x + intercept

        # R²
        ss_res = float(np.sum((test_y - pred_y) ** 2))
        ss_tot = float(np.sum((test_y - np.mean(test_y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
        fold_r2.append(r2)

        # MAE
        mae = float(np.mean(np.abs(test_y - pred_y)))
        fold_mae.append(mae)

    mean_r2 = float(np.mean(fold_r2)) if fold_r2 else float("nan")
    std_r2 = float(np.std(fold_r2)) if fold_r2 else float("nan")
    mean_mae = float(np.mean(fold_mae)) if fold_mae else float("nan")
    std_mae = float(np.std(fold_mae)) if fold_mae else float("nan")

    return CrossValidationResult(
        k=k,
        per_fold_r2=fold_r2,
        per_fold_mae=fold_mae,
        mean_r2=mean_r2,
        std_r2=std_r2,
        mean_mae=mean_mae,
        std_mae=std_mae,
    )


# ---------------------------------------------------------------------------
# Leave-one-out prediction
# ---------------------------------------------------------------------------


def leave_one_out_predictions(
    names: list[str],
    per_residue_plls: list[float],
    experimental_tms: list[float],
) -> list[LOOPrediction]:
    """Leave-one-out calibration: for each VHH, fit on the rest and predict.

    Parameters
    ----------
    names:
        Identifier for each VHH.
    per_residue_plls:
        Per-residue PLL scores.
    experimental_tms:
        Known Tm values.

    Returns
    -------
    list[LOOPrediction]
    """
    n = len(names)
    if n != len(per_residue_plls) or n != len(experimental_tms):
        raise ValueError("All input lists must have the same length")
    if n < 3:
        raise ValueError("At least 3 samples required for leave-one-out")

    pll_arr = np.array(per_residue_plls)
    tm_arr = np.array(experimental_tms)
    results: list[LOOPrediction] = []

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        train_x, train_y = pll_arr[mask], tm_arr[mask]
        coeffs = np.polyfit(train_x, train_y, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])

        pred_tm = slope * pll_arr[i] + intercept

        # R² of the training fit
        train_pred = slope * train_x + intercept
        ss_res = float(np.sum((train_y - train_pred) ** 2))
        ss_tot = float(np.sum((train_y - np.mean(train_y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

        results.append(
            LOOPrediction(
                name=names[i],
                experimental_tm=float(tm_arr[i]),
                predicted_tm=float(pred_tm),
                residual=float(pred_tm - tm_arr[i]),
                calibration_r2=r2,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Library-level validation
# ---------------------------------------------------------------------------


def validate_library_predictions(
    library_df: pd.DataFrame,
    experimental_csv: str | Path | StringIO,
    predicted_col: str = "composite_score",
    variant_id_col: str = "variant_id",
    exp_tm_col: str | None = None,
    exp_rank_col: str | None = None,
) -> CorrelationMetrics:
    """Validate library predictions against experimental results.

    Parameters
    ----------
    library_df:
        Library DataFrame with at least ``variant_id_col`` and ``predicted_col``.
    experimental_csv:
        Path or file-like to CSV with at least ``variant_id_col`` and one of
        ``experimental_tm`` / ``experimental_ranking``.
    predicted_col:
        Column name in *library_df* to use as the predicted value.
    variant_id_col:
        Column to join on.
    exp_tm_col:
        Name of the Tm column in experimental CSV.  Auto-detected if *None*.
    exp_rank_col:
        Name of the ranking column in experimental CSV.  Auto-detected if *None*.

    Returns
    -------
    CorrelationMetrics
    """
    exp_df = pd.read_csv(experimental_csv)

    # Auto-detect experimental column
    if exp_tm_col is None and exp_rank_col is None:
        for candidate in ("experimental_tm", "Tm", "tm", "melting_temperature"):
            if candidate in exp_df.columns:
                exp_tm_col = candidate
                break
        if exp_tm_col is None:
            for candidate in ("experimental_ranking", "ranking", "rank"):
                if candidate in exp_df.columns:
                    exp_rank_col = candidate
                    break

    if exp_tm_col is None and exp_rank_col is None:
        raise ValueError(
            "Could not find an experimental Tm or ranking column in the CSV. "
            "Expected one of: experimental_tm, Tm, tm, melting_temperature, "
            "experimental_ranking, ranking, rank"
        )

    # Merge
    merged = library_df.merge(exp_df, on=variant_id_col, how="inner")
    if merged.empty:
        raise ValueError(f"No matching variants found on column '{variant_id_col}'")

    if predicted_col not in merged.columns:
        raise ValueError(f"Predicted column '{predicted_col}' not found in library DataFrame")

    exp_col = exp_tm_col or exp_rank_col
    if exp_col not in merged.columns:
        raise ValueError(f"Experimental column '{exp_col}' not found after merge")

    predicted = merged[predicted_col].astype(float).tolist()
    experimental = merged[exp_col].astype(float).tolist()

    return compute_correlation_metrics(predicted, experimental)


# ---------------------------------------------------------------------------
# Scoring comparison
# ---------------------------------------------------------------------------


def compare_scoring_methods(
    experimental_tms: list[float],
    scoring_results: dict[str, list[float]],
) -> dict[str, CorrelationMetrics]:
    """Compare multiple scoring approaches against experimental Tm.

    Parameters
    ----------
    experimental_tms:
        Known Tm values.
    scoring_results:
        Dict mapping scoring method name → list of predicted values.

    Returns
    -------
    dict[str, CorrelationMetrics]
    """
    comparison: dict[str, CorrelationMetrics] = {}
    for method_name, predicted in scoring_results.items():
        if len(predicted) != len(experimental_tms):
            logger.warning(
                "Skipping method '%s': %d predictions vs %d experimental values",
                method_name,
                len(predicted),
                len(experimental_tms),
            )
            continue
        comparison[method_name] = compute_correlation_metrics(predicted, experimental_tms)
    return comparison


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_correlation_scatter(
    predicted: list[float],
    experimental: list[float],
    metrics: CorrelationMetrics | None = None,
    title: str = "Predicted vs Experimental Tm",
    xlabel: str = "Predicted",
    ylabel: str = "Experimental",
) -> "matplotlib.figure.Figure":
    """Create a scatter plot with regression line and confidence interval.

    Returns a Matplotlib *Figure* (caller should close after display).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))

    pred_arr = np.array(predicted)
    exp_arr = np.array(experimental)

    ax.scatter(pred_arr, exp_arr, alpha=0.6, s=30, edgecolors="white", linewidth=0.5)

    # Regression line with 95 % CI
    if len(pred_arr) >= 2:
        coeffs = np.polyfit(pred_arr, exp_arr, 1)
        x_line = np.linspace(pred_arr.min(), pred_arr.max(), 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, "r-", linewidth=1.5, label="Regression")

        # Confidence interval (simple approach via residual SE)
        y_pred_all = np.polyval(coeffs, pred_arr)
        residuals = exp_arr - y_pred_all
        se = np.std(residuals)
        ax.fill_between(
            x_line, y_line - 1.96 * se, y_line + 1.96 * se,
            alpha=0.15, color="red", label="95% CI",
        )

    if metrics is not None:
        text = (
            f"Spearman ρ = {metrics.spearman_rho:.3f}\n"
            f"Pearson r = {metrics.pearson_r:.3f}\n"
            f"MAE = {metrics.mae:.2f}\n"
            f"RMSE = {metrics.rmse:.2f}"
        )
        ax.text(
            0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_residuals(
    predicted: list[float],
    experimental: list[float],
    title: str = "Residual Plot",
) -> "matplotlib.figure.Figure":
    """Create a residual plot (predicted on x-axis, residual on y-axis)."""
    import matplotlib.pyplot as plt

    pred_arr = np.array(predicted)
    exp_arr = np.array(experimental)
    residuals = pred_arr - exp_arr

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pred_arr, residuals, alpha=0.6, s=30, edgecolors="white", linewidth=0.5)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Predicted − Experimental)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scoring_comparison(
    comparison: dict[str, CorrelationMetrics],
) -> "matplotlib.figure.Figure":
    """Bar chart comparing Spearman ρ across scoring methods."""
    import matplotlib.pyplot as plt

    methods = list(comparison.keys())
    rhos = [comparison[m].spearman_rho for m in methods]

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.5), 4))
    bars = ax.bar(range(len(methods)), rhos, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Scoring Method Comparison")
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    for bar, rho in zip(bars, rhos):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
            f"{rho:.3f}", ha="center", fontsize=8,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# High-level benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    benchmark_vhhs: list[dict[str, Any]] | None = None,
    per_residue_plls: list[float] | None = None,
    composite_scores: list[float] | None = None,
    cv_folds: int = 5,
) -> BenchmarkReport:
    """Run the full benchmark suite on a set of VHHs with known Tm values.

    Parameters
    ----------
    benchmark_vhhs:
        List of dicts with ``name``, ``sequence``, ``experimental_tm``.
        If *None*, the built-in dataset is loaded.
    per_residue_plls:
        Pre-computed per-residue PLL scores (one per VHH).
        If *None*, PLL-dependent analyses (cross-validation, LOO) are skipped.
    composite_scores:
        Pre-computed composite stability scores (one per VHH).
        If *None*, composite correlation is skipped.
    cv_folds:
        Number of cross-validation folds.

    Returns
    -------
    BenchmarkReport
    """
    if benchmark_vhhs is None:
        benchmark_vhhs = load_benchmark_dataset()

    names = [v["name"] for v in benchmark_vhhs]
    exp_tms = [float(v["experimental_tm"]) for v in benchmark_vhhs]
    n = len(benchmark_vhhs)

    # Correlation with composite scores
    correlation: CorrelationMetrics | None = None
    if composite_scores is not None and len(composite_scores) == n:
        correlation = compute_correlation_metrics(composite_scores, exp_tms)

    # PLL-based analyses
    cv_result: CrossValidationResult | None = None
    loo_results: list[LOOPrediction] = []

    if per_residue_plls is not None and len(per_residue_plls) == n:
        # Predicted Tm from PLLs (using a quick fit for correlation)
        pll_arr = np.array(per_residue_plls)
        tm_arr = np.array(exp_tms)
        coeffs = np.polyfit(pll_arr, tm_arr, 1)
        pred_tms = [float(np.polyval(coeffs, p)) for p in per_residue_plls]

        if correlation is None:
            correlation = compute_correlation_metrics(pred_tms, exp_tms)

        # Cross-validation
        if n >= max(cv_folds, 3):
            cv_result = cross_validate_calibration(per_residue_plls, exp_tms, k=cv_folds)

        # Leave-one-out
        if n >= 3:
            loo_results = leave_one_out_predictions(names, per_residue_plls, exp_tms)

    # Fallback if no scores were provided at all
    if correlation is None:
        correlation = CorrelationMetrics(
            spearman_rho=float("nan"),
            spearman_pvalue=float("nan"),
            pearson_r=float("nan"),
            pearson_pvalue=float("nan"),
            mae=float("nan"),
            rmse=float("nan"),
            ranking_accuracy=float("nan"),
            n_samples=n,
        )

    return BenchmarkReport(
        correlation=correlation,
        cross_validation=cv_result,
        loo_predictions=loo_results,
    )
