"""
Shared helpers for the visualization modules: parameter-label/unit lookup,
filename slug-ification, mean/SEM computation, and a small wrapper around
matplotlib that uses the Agg backend (no display required, safe in headless
runs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameter pretty labels and units (used for axis labels and filenames)
# ---------------------------------------------------------------------------
# Display labels and filename slugs for the 15 publication parameters,
# byte-for-byte matching ``old_code/analyze_data.py:get_publication_parameter_mapping``
# and the ``Timeseries_*`` filenames in old_results/Publication_Plots/.
#
# parameter_csv_column -> (display_label, filename_slug)
_PARAM_DISPLAY: Dict[str, Tuple[str, str]] = {
    "mean_ttot_ms":              ("Ttot (ms)",            "Ttot_ms"),
    "mean_frequency_bpm":        ("Frequency (bpm)",      "Frequency_bpm"),
    "mean_ti_ms":                ("Ti (ms)",              "Ti_ms"),
    "mean_te_ms":                ("Te (ms)",              "Te_ms"),
    "mean_pif_centered_ml_s":    ("PIF (mL/s)",           "PIF_mL_s"),
    "mean_pef_centered_ml_s":    ("PEF (mL/s)",           "PEF_mL_s"),
    "mean_pif_to_pef_ml_s":      ("PIF-to-PEF (mL/s)",    "PIF_to_PEF_mL_s"),
    "mean_tv_ml":                ("Tv (mL)",              "Tv_mL"),
    "sigh_rate_per_min":         ("Sigh rate (/min)",     "Sigh_rate__min"),
    "mean_sigh_duration_ms":     ("Sigh duration (ms)",   "Sigh_duration_ms"),
    "cov_instant_freq":          ("CoV (instantaneous)",  "CoV_instantaneous"),
    "alternate_cov":             ("CoV",                  "CoV_alternate"),
    "pif_to_pef_cov":            ("CoV (PIF-to-PEF)",     "CoV_PIF_to_PEF"),
    "apnea_rate_per_min":        ("Apnea rate (/min)",    "Apnea_rate__min"),
    "apnea_mean_ms":             ("Apnea duration (ms)",  "Apnea_duration_ms"),
}


def display_label(parameter: str) -> str:
    """Old-code-compatible y-axis label for ``parameter``."""
    label, _ = _PARAM_DISPLAY.get(parameter, (parameter, parameter))
    return label


def filename_slug(parameter: str) -> str:
    """Old-code-compatible filename token for ``parameter``."""
    return _PARAM_DISPLAY.get(parameter, (parameter, parameter))[1]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def mean_sem(series: pd.Series) -> Tuple[float, float, int]:
    """Return ``(mean, sem, n)`` for a numeric series, dropping NaN. SEM is
    ``std / sqrt(n)`` with ddof=1; if n<2 the SEM is NaN."""
    s = series.dropna()
    n = len(s)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    mean = float(s.mean())
    sem = float(s.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return (mean, sem, n)


def global_ylim(
    data: pd.DataFrame,
    parameter: str,
    periods,
    *,
    condition_col: str = "risk_clean",
) -> Optional[Tuple[float, float]]:
    """Compute one (y_min, y_max) range across all four periods for a parameter,
    so within/across/developmental plots of the same parameter share an axis.

    Mirrors ``old_code/analyze_data.py:calculate_global_ylim_all_periods``. Pulls
    values from both the P22 (across) and high-risk/treated (within) subsets
    of every period, applies the apnea-duration grey-zero rule, and adds 2%
    padding. Returns ``None`` if the parameter has no values anywhere.
    """
    if parameter not in data.columns:
        return None
    high_value = "Vehicle" if condition_col == "treatment_clean" else "high_risk"
    high_values = ("FFA", "Vehicle") if condition_col == "treatment_clean" else ("high_risk",)

    has_nan = False
    all_values = []
    for period in periods:
        sub = data[data["period"] == period]
        if sub.empty:
            continue
        p22 = sub[sub["age_clean"] == 22]
        if not p22.empty:
            all_values.extend(p22[parameter].dropna().tolist())
            if parameter == "apnea_mean_ms" and p22[parameter].isna().any():
                has_nan = True
        hi = sub[sub[condition_col].astype(str).isin(high_values)]
        if not hi.empty:
            all_values.extend(hi[parameter].dropna().tolist())
            if parameter == "apnea_mean_ms" and hi[parameter].isna().any():
                has_nan = True

    if not all_values:
        return None
    if parameter == "apnea_mean_ms" and has_nan:
        y_min, y_max = 0.0, float(max(all_values))
    else:
        y_min, y_max = float(min(all_values)), float(max(all_values))
    rng = y_max - y_min
    if rng > 0:
        y_min -= 0.02 * rng
        y_max += 0.02 * rng
    else:
        pad = 0.02 * abs(y_min) if y_min != 0 else 0.1
        y_min -= pad
        y_max += pad
    return (y_min, y_max)


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------
def make_axes(figsize: Tuple[float, float] = (5.0, 4.0)) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = 200) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
