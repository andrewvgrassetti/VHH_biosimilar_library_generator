"""Persistent calibration system for the VHH stability scorer.

Provides functions to load, run, and reset PLL→Tm calibration parameters
that are saved to ``data/stability_calibration.json``.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CALIBRATION_FILE = _DATA_DIR / "stability_calibration.json"

_DEFAULT_TEMPLATE: dict = {
    "version": 1,
    "created_at": None,
    "description": "Stability calibration derived from reference VHHs with known Tm values",
    "calibration_vhhs": [],
    "parameters": {
        "pll_to_tm_slope": 12.5,
        "pll_to_tm_intercept": 95.0,
        "tm_ideal_min": 55.0,
        "tm_ideal_max": 80.0,
        "penalty_disulfide": 0.20,
        "penalty_aggregation": 0.10,
        "penalty_charge": 0.05,
        "hallmark_bonus_weight": 0.10,
        "legacy_weights": {
            "disulfide": 0.25,
            "hallmark": 0.20,
            "aggregation": 0.25,
            "charge": 0.15,
            "hydrophobic": 0.15,
        },
    },
}


@dataclass
class CalibrationResult:
    """Result of a calibration run."""

    pll_to_tm_slope: float
    pll_to_tm_intercept: float
    tm_ideal_min: float
    tm_ideal_max: float
    r_squared: float
    n_samples: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_calibration(path: str | Path | None = None) -> dict | None:
    """Load saved calibration parameters from the JSON file.

    Returns ``None`` if the file does not exist or if ``calibration_vhhs``
    is empty (i.e., only defaults are present and no real calibration has
    been run).
    """
    cal_path = Path(path) if path is not None else _CALIBRATION_FILE
    if not cal_path.exists():
        return None

    try:
        with open(cal_path) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read calibration file %s", cal_path)
        return None

    # Only return calibration if it has been populated with real VHH data
    if not data.get("calibration_vhhs"):
        return None

    return data


def run_calibration(
    sequences: list[str],
    known_tms: list[float],
    names: list[str] | None = None,
    esm_model_tier: str = "auto",
    calibration_path: str | Path | None = None,
) -> CalibrationResult:
    """Calibrate PLL→Tm regression from VHH sequences with known Tm values.

    Parameters
    ----------
    sequences:
        VHH amino-acid sequences.
    known_tms:
        Experimentally measured melting temperatures (°C), one per sequence.
    names:
        Optional human-readable names for each VHH.
    esm_model_tier:
        ESM-2 model tier for PLL computation (default ``"auto"``).
    calibration_path:
        Override the default calibration file location.

    Returns
    -------
    CalibrationResult
        Summary of the fitted calibration.
    """
    if len(sequences) != len(known_tms):
        raise ValueError("sequences and known_tms must have the same length")
    if len(sequences) < 2:
        raise ValueError("At least 2 calibration VHHs are required")

    from vhh_library.esm_scorer import ESMStabilityScorer

    scorer = ESMStabilityScorer(model_tier=esm_model_tier, device="auto")
    plls = scorer.score_batch(sequences)

    # Normalise to per-residue PLL
    per_residue_plls = [pll / max(len(seq), 1) for pll, seq in zip(plls, sequences)]

    # Linear regression: per_residue_pll → Tm
    try:
        import numpy as np

        coeffs = np.polyfit(per_residue_plls, known_tms, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
    except ImportError:
        slope, intercept = _least_squares_fit(per_residue_plls, known_tms)

    # R² calculation
    r_squared = _compute_r_squared(per_residue_plls, known_tms, slope, intercept)

    # Ideal Tm range from calibration data (10th–90th percentile)
    sorted_tms = sorted(known_tms)
    n = len(sorted_tms)
    idx_10 = max(0, int(n * 0.1))
    idx_90 = min(n - 1, int(n * 0.9))
    tm_ideal_min = sorted_tms[idx_10]
    tm_ideal_max = sorted_tms[idx_90]

    # Build calibration VHH records
    cal_vhhs = []
    for i, (seq, tm) in enumerate(zip(sequences, known_tms)):
        name = names[i] if names and i < len(names) else f"VHH_{i + 1}"
        cal_vhhs.append({
            "name": name,
            "sequence": seq,
            "experimental_tm": tm,
            "per_residue_pll": per_residue_plls[i],
        })

    # Save to disk
    cal_path = Path(calibration_path) if calibration_path is not None else _CALIBRATION_FILE
    data = copy.deepcopy(_DEFAULT_TEMPLATE)
    data["created_at"] = datetime.now(timezone.utc).isoformat()
    data["calibration_vhhs"] = cal_vhhs
    data["parameters"]["pll_to_tm_slope"] = slope
    data["parameters"]["pll_to_tm_intercept"] = intercept
    data["parameters"]["tm_ideal_min"] = tm_ideal_min
    data["parameters"]["tm_ideal_max"] = tm_ideal_max

    cal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_path, "w") as fh:
        json.dump(data, fh, indent=2)

    return CalibrationResult(
        pll_to_tm_slope=slope,
        pll_to_tm_intercept=intercept,
        tm_ideal_min=tm_ideal_min,
        tm_ideal_max=tm_ideal_max,
        r_squared=r_squared,
        n_samples=len(sequences),
    )


def reset_calibration(path: str | Path | None = None) -> None:
    """Reset the calibration file back to the default template."""
    cal_path = Path(path) if path is not None else _CALIBRATION_FILE
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_path, "w") as fh:
        json.dump(_DEFAULT_TEMPLATE, fh, indent=2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _least_squares_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Simple ordinary least-squares for a linear fit (no numpy needed)."""
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0, sum_y / max(n, 1)
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def _compute_r_squared(
    xs: list[float], ys: list[float], slope: float, intercept: float
) -> float:
    """Compute R² for a linear fit."""
    y_mean = sum(ys) / max(len(ys), 1)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    if ss_tot < 1e-12:
        return 1.0
    return 1.0 - ss_res / ss_tot
