"""
Publication plots for experiment 4 (P19 het: survivors vs eventual SUDEP).

Two plot families, mirroring the exp1 / exp2 publication layout but with
``sudep_status`` (survivor vs sudep) as the categorical splitter:

  - ``Within each time period/<DisplayPeriod>_<param>_survival.png``
    Strip plot, two cells (Survivor green / SUDEP red), per (period, parameter).

  - ``Across time periods/Timeseries_<param>_survival.png``
    Connected-line timeseries with two traces, per parameter.

All subjects are P19 het, so markers are ``^`` (the package-wide P19 marker).
Colors mirror the bar plot used in older drafts: emerald for survivors, red for
SUDEP.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from ._common import display_label, filename_slug, make_axes, save_figure
from .colors import italicize_scn1a


_PERIODS_TO_PLOT = ("Baseline", "Ictal", "Immediate Postictal", "Recovery")
_PERIOD_DISPLAY = {
    "Baseline": "Baseline",
    "Ictal": "Ictal",
    "Immediate Postictal": "Postictal",
    "Recovery": "Recovery",
}
_PERIOD_LABELS_TS = ("Baseline", "Ictal", "Postictal", "Recovery")

# All subjects are Scn1a+/- het, so both groups stay in the red family
# (matching colors.py's HR Scn1a #FF0000 and LR Scn1a #FFA07A); pale red is the
# lower-risk outcome (survivor) and full red is the higher-risk outcome (SUDEP),
# mirroring the LR/HR convention used elsewhere in the package.
_SURVIVOR_COLOR = "#FFA07A"
_SUDEP_COLOR = "#FF0000"
_MARKER = "^"     # all subjects are P19

_PARAMETERS = (
    "mean_ttot_ms", "mean_frequency_bpm", "mean_ti_ms", "mean_te_ms",
    "mean_pif_centered_ml_s", "mean_pef_centered_ml_s", "mean_pif_to_pef_ml_s",
    "mean_tv_ml", "sigh_rate_per_min", "mean_sigh_duration_ms",
    "cov_instant_freq", "alternate_cov", "pif_to_pef_cov",
    "apnea_rate_per_min", "apnea_mean_ms",
)

_FIG_SIZE_STRIP = (7, 10)
_FIG_SIZE_TS = (12, 10)
_JITTER = 0.15
_MARKER_SIZE = 150
_POINT_SIZE_TS = 60
_MARKER_ALPHA = 0.7
_MEAN_CAPSIZE = 10
_MEAN_MARKERSIZE = 50
_MEAN_EDGEWIDTH = 6
_MEAN_LINEWIDTH = 3
_ERROR_LINEWIDTH = 2
_ERROR_CAPSIZE = 5
_YLABEL_FONTSIZE = 40
_TICK_FONTSIZE = 32
_LEGEND_FONTSIZE = 24


def plot_survivor_publication(
    merged: pd.DataFrame,
    output_dir: Path,
    *,
    parameters: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Produce the experiment-4 publication plot bundle.

    ``merged`` must already be filtered to the survivor/SUDEP cohort (P19 het,
    ``sudep_status`` in {"survivor", "sudep"}); see ``experiment4.run`` which
    applies that filter before calling this function.
    """
    output_dir = Path(output_dir)
    if parameters is None:
        parameters = [p for p in _PARAMETERS if p in merged.columns]

    saved: List[Path] = []
    within_dir = output_dir / "Within each time period"
    across_dir = output_dir / "Across time periods"

    for param in parameters:
        ylim = _ylim_across_periods(merged, param)
        for period in _PERIODS_TO_PLOT:
            path = _draw_within_period(
                merged, param, period,
                output_path=within_dir / f"{_PERIOD_DISPLAY[period]}_{filename_slug(param)}_survival.png",
                ylim=ylim,
            )
            if path is not None:
                saved.append(path)
        path = _draw_across_periods(
            merged, param,
            output_path=across_dir / f"Timeseries_{filename_slug(param)}_survival.png",
        )
        if path is not None:
            saved.append(path)
    return saved


def _ylim_across_periods(data: pd.DataFrame, parameter: str) -> Optional[Tuple[float, float]]:
    if parameter not in data.columns:
        return None
    has_nan = False
    values: List[float] = []
    for period in _PERIODS_TO_PLOT:
        col = data.loc[data["period"] == period, parameter]
        values.extend(col.dropna().tolist())
        if parameter == "apnea_mean_ms" and col.isna().any():
            has_nan = True
    if not values:
        return None
    if parameter == "apnea_mean_ms" and has_nan:
        y_min, y_max = 0.0, float(max(values))
    else:
        y_min, y_max = float(min(values)), float(max(values))
    rng = y_max - y_min
    if rng > 0:
        y_min -= 0.02 * rng
        y_max += 0.02 * rng
    else:
        pad = 0.02 * abs(y_min) if y_min != 0 else 0.1
        y_min -= pad
        y_max += pad
    return (y_min, y_max)


def _draw_within_period(
    data: pd.DataFrame,
    parameter: str,
    period: str,
    *,
    output_path: Path,
    ylim: Optional[Tuple[float, float]],
) -> Optional[Path]:
    sub = data[data["period"] == period]
    if sub.empty:
        return None
    surv = sub.loc[sub["sudep_status"] == "survivor", parameter]
    sudep = sub.loc[sub["sudep_status"] == "sudep", parameter]
    if surv.dropna().empty and sudep.dropna().empty:
        return None

    fig, ax = make_axes(figsize=_FIG_SIZE_STRIP)
    rng = np.random.default_rng(5)
    means, sems, x_positions = [], [], []
    cells = (
        ("Survivor", _SURVIVOR_COLOR, surv),
        ("SUDEP", _SUDEP_COLOR, sudep),
    )

    for i, (_label, color, raw) in enumerate(cells):
        if parameter == "apnea_mean_ms":
            valid = raw.dropna()
            nan_count = int(raw.isna().sum())
            if not valid.empty:
                xs = i + rng.uniform(-_JITTER, _JITTER, size=len(valid))
                ax.scatter(xs, valid, color=color, alpha=_MARKER_ALPHA,
                           s=_MARKER_SIZE, marker=_MARKER,
                           edgecolors="black", linewidth=0.5)
            if nan_count > 0:
                xs = i + rng.uniform(-_JITTER, _JITTER, size=nan_count)
                ax.scatter(xs, np.zeros(nan_count), color="grey",
                           alpha=_MARKER_ALPHA, s=_MARKER_SIZE, marker=_MARKER,
                           edgecolors="black", linewidth=0.5)
            if not valid.empty:
                means.append(float(valid.mean()))
                sems.append(_sem(valid))
                x_positions.append(i)
        else:
            valid = raw.dropna()
            if valid.empty:
                continue
            xs = i + rng.uniform(-_JITTER, _JITTER, size=len(valid))
            ax.scatter(xs, valid, color=color, alpha=_MARKER_ALPHA,
                       s=_MARKER_SIZE, marker=_MARKER,
                       edgecolors="black", linewidth=0.5)
            means.append(float(valid.mean()))
            sems.append(_sem(valid))
            x_positions.append(i)

    if not means:
        import matplotlib.pyplot as plt
        plt.close(fig)
        return None

    ax.errorbar(
        x_positions, means, yerr=sems, fmt="_", color="black",
        capsize=_MEAN_CAPSIZE, markersize=_MEAN_MARKERSIZE,
        markeredgewidth=_MEAN_EDGEWIDTH, zorder=10,
    )
    ax.set_ylabel(f"{_PERIOD_DISPLAY[period]}\n{display_label(parameter)}", fontsize=_YLABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
    ax.set_xticks([0, 1])
    ax.set_xlim(-0.6, 1.4)
    ax.set_xticklabels(
        [italicize_scn1a("Survivor\nScn1a+/- P19"),
         italicize_scn1a("SUDEP\nScn1a+/- P19")],
        fontsize=_TICK_FONTSIZE,
    )
    ax.xaxis.set_tick_params(rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    save_figure(fig, output_path)
    return output_path


def _draw_across_periods(
    data: pd.DataFrame,
    parameter: str,
    *,
    output_path: Path,
) -> Optional[Path]:
    fig, ax = make_axes(figsize=_FIG_SIZE_TS)
    rng = np.random.default_rng(6)
    drew_anything = False
    groups = (
        ("Survivor", _SURVIVOR_COLOR, "survivor"),
        ("SUDEP", _SUDEP_COLOR, "sudep"),
    )

    for _label, color, status in groups:
        means, sems, x_positions = [], [], []
        for i, period in enumerate(_PERIODS_TO_PLOT):
            mask = (data["period"] == period) & (data["sudep_status"] == status)
            valid = data.loc[mask, parameter].dropna()
            if valid.empty:
                continue
            xs = i + rng.uniform(-_JITTER, _JITTER, size=len(valid))
            ax.scatter(xs, valid, color=color, alpha=1.0, s=_POINT_SIZE_TS,
                       marker=_MARKER, edgecolors="black", linewidth=0.5)
            means.append(float(valid.mean()))
            sems.append(_sem(valid))
            x_positions.append(i)
            drew_anything = True
        if means:
            ax.plot(x_positions, means, color=color, linewidth=_MEAN_LINEWIDTH,
                    alpha=1.0, zorder=10)
            ax.errorbar(x_positions, means, yerr=sems, fmt="none", ecolor=color,
                        elinewidth=_ERROR_LINEWIDTH, capsize=_ERROR_CAPSIZE,
                        alpha=0.8, zorder=10)

    if not drew_anything:
        import matplotlib.pyplot as plt
        plt.close(fig)
        return None

    ax.set_ylabel(display_label(parameter), fontsize=_YLABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
    ax.set_xticks(range(len(_PERIODS_TO_PLOT)))
    ax.set_xticklabels(list(_PERIOD_LABELS_TS), fontsize=_TICK_FONTSIZE,
                       rotation=45, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color=_SURVIVOR_COLOR, marker=_MARKER, linewidth=_MEAN_LINEWIDTH,
               markersize=10, markeredgecolor="black", markeredgewidth=0.5),
        Line2D([0], [0], color=_SUDEP_COLOR, marker=_MARKER, linewidth=_MEAN_LINEWIDTH,
               markersize=10, markeredgecolor="black", markeredgewidth=0.5),
    ]
    labels = [
        italicize_scn1a("Survivor Scn1a+/- P19"),
        italicize_scn1a("SUDEP Scn1a+/- P19"),
    ]
    ax.legend(handles, labels, fontsize=_LEGEND_FONTSIZE, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2)
    save_figure(fig, output_path)
    return output_path


def _sem(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))
