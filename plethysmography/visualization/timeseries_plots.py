"""
Connected-line plots showing one parameter across the four named periods
(Baseline -> Ictal -> Immediate Postictal -> Recovery), one trace per group.

Mirrors ``old_code/analyze_data.py``:

  * ``create_timeseries_across_plot`` : P22-only, four (genotype x condition)
    cells, all circle markers, pale vs full colors for LR vs HR.
  * ``create_timeseries_within_plot`` : HR-only, four (genotype x age) cells,
    triangle (P19) vs circle (P22) markers, faded P19 colors.

Each trace draws individual mouse points jittered horizontally, then the
connecting line and error bars on top. Filenames mirror
``Timeseries_<slug>_across.png`` and ``Timeseries_<slug>_within.png``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from ._common import display_label, filename_slug, make_axes, save_figure
from .colors import (
    DEFAULT_PALETTE,
    HR_TIMESERIES_PALETTE,
    MARKERS_BY_AGE,
    TREATMENT_PALETTE,
    italicize_scn1a,
)


_PERIODS_ORDER = ("Baseline", "Ictal", "Immediate Postictal", "Recovery")
_PERIOD_LABELS = ("Baseline", "Ictal", "Postictal", "Recovery")
_FIG_SIZE = (12, 10)
_JITTER = 0.15
_POINT_SIZE = 60
_MEAN_LINEWIDTH = 3
_ERROR_LINEWIDTH = 2
_ERROR_CAPSIZE = 5
_YLABEL_FONTSIZE = 40
_TICK_FONTSIZE = 32
_LEGEND_FONTSIZE = 24


def plot_across_periods(
    data: pd.DataFrame,
    parameter: str,
    output_dir: Path,
    *,
    condition_col: str = "risk_clean",
) -> Optional[Path]:
    """Generate ``_across`` and ``_within`` timeseries plots for one parameter.
    Returns the path to ``_across`` (or ``_within`` if P22 cohort is empty;
    ``None`` if both are empty)."""
    output_dir = Path(output_dir)
    out_path: Optional[Path] = None

    # ---- across (P22 only) -------------------------------------------------
    p22 = data[data["age_clean"] == 22]
    if not p22.empty:
        out_path = _draw_across(
            p22, parameter, condition_col=condition_col,
            output_path=output_dir / f"Timeseries_{filename_slug(parameter)}_across.png",
        )

    # ---- within (high-risk / FFA-cohort, P19 vs P22) ------------------------
    high = _high_values(condition_col)
    hr = data[data[condition_col].astype(str).isin(high)]
    if not hr.empty:
        within_path = _draw_within(
            hr, parameter, condition_col=condition_col,
            output_path=output_dir / f"Timeseries_{filename_slug(parameter)}_within.png",
        )
        if out_path is None:
            out_path = within_path
    return out_path


def _draw_across(
    p22_data: pd.DataFrame,
    parameter: str,
    *,
    condition_col: str,
    output_path: Path,
) -> Optional[Path]:
    if condition_col == "treatment_clean":
        groups = [
            ("WT Vehicle",            "WT",  "Vehicle", TREATMENT_PALETTE[("WT",  "Vehicle")], "o"),
            ("WT FFA",                "WT",  "FFA",     TREATMENT_PALETTE[("WT",  "FFA")],     "o"),
            ("Scn1a+/- Vehicle",      "het", "Vehicle", TREATMENT_PALETTE[("het", "Vehicle")], "o"),
            ("Scn1a+/- FFA",          "het", "FFA",     TREATMENT_PALETTE[("het", "FFA")],     "o"),
        ]
    else:
        groups = [
            ("LR WT P22",            "WT",  "low_risk",  DEFAULT_PALETTE[("WT",  "low_risk")],  "o"),
            ("HR WT P22",            "WT",  "high_risk", DEFAULT_PALETTE[("WT",  "high_risk")], "o"),
            ("LR Scn1a+/- P22",      "het", "low_risk",  DEFAULT_PALETTE[("het", "low_risk")],  "o"),
            ("HR Scn1a+/- P22",      "het", "high_risk", DEFAULT_PALETTE[("het", "high_risk")], "o"),
        ]
    return _draw_traces(
        p22_data, parameter, groups,
        match_cols=("genotype_clean", condition_col),
        output_path=output_path,
    )


def _draw_within(
    hr_data: pd.DataFrame,
    parameter: str,
    *,
    condition_col: str,
    output_path: Path,
) -> Optional[Path]:
    if condition_col == "treatment_clean":
        # FFA cohort within plot: same four (genotype, age) cells as exp 1.
        groups = [
            ("WT P19",          "WT",  19, HR_TIMESERIES_PALETTE[("WT",  19)], MARKERS_BY_AGE[19]),
            ("WT P22",          "WT",  22, HR_TIMESERIES_PALETTE[("WT",  22)], MARKERS_BY_AGE[22]),
            ("Scn1a+/- P19",    "het", 19, HR_TIMESERIES_PALETTE[("het", 19)], MARKERS_BY_AGE[19]),
            ("Scn1a+/- P22",    "het", 22, HR_TIMESERIES_PALETTE[("het", 22)], MARKERS_BY_AGE[22]),
        ]
    else:
        groups = [
            ("HR WT P19",       "WT",  19, HR_TIMESERIES_PALETTE[("WT",  19)], MARKERS_BY_AGE[19]),
            ("HR WT P22",       "WT",  22, HR_TIMESERIES_PALETTE[("WT",  22)], MARKERS_BY_AGE[22]),
            ("HR Scn1a+/- P19", "het", 19, HR_TIMESERIES_PALETTE[("het", 19)], MARKERS_BY_AGE[19]),
            ("HR Scn1a+/- P22", "het", 22, HR_TIMESERIES_PALETTE[("het", 22)], MARKERS_BY_AGE[22]),
        ]
    return _draw_traces(
        hr_data, parameter, groups,
        match_cols=("genotype_clean", "age_clean"),
        output_path=output_path,
    )


def _draw_traces(
    data: pd.DataFrame,
    parameter: str,
    groups: Sequence[Tuple[str, str, object, str, str]],
    *,
    match_cols: Tuple[str, str],
    output_path: Path,
) -> Optional[Path]:
    fig, ax = make_axes(figsize=_FIG_SIZE)
    rng = np.random.default_rng(0)
    drew_anything = False

    for label, gen_value, second_value, color, marker in groups:
        means, sems, x_positions = [], [], []
        for i, period in enumerate(_PERIODS_ORDER):
            mask = (data["period"] == period) & (data[match_cols[0]].astype(str) == gen_value)
            second_col = match_cols[1]
            if second_col in {"age_clean"}:
                mask &= data[second_col] == second_value
            else:
                mask &= data[second_col].astype(str) == str(second_value)
            valid = data.loc[mask, parameter].dropna()
            if valid.empty:
                continue
            xs = i + rng.uniform(-_JITTER, _JITTER, size=len(valid))
            ax.scatter(xs, valid, color=color, alpha=1.0, s=_POINT_SIZE,
                       marker=marker, edgecolors="black", linewidth=0.5)
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

    _format_axes(ax, parameter, groups)
    save_figure(fig, output_path)
    return output_path


def _format_axes(ax: Axes, parameter: str, groups: Sequence[Tuple[str, str, object, str, str]]) -> None:
    ax.set_ylabel(display_label(parameter), fontsize=_YLABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
    ax.set_xticks(range(len(_PERIODS_ORDER)))
    ax.set_xticklabels(list(_PERIOD_LABELS), fontsize=_TICK_FONTSIZE,
                       rotation=45, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = []
    labels = []
    for label, _gen, _second, color, marker in groups:
        h = Line2D([0], [0], color=color, marker=marker, linewidth=_MEAN_LINEWIDTH,
                   markersize=10, markeredgecolor="black", markeredgewidth=0.5)
        handles.append(h)
        labels.append(italicize_scn1a(label))
    ax.legend(
        handles, labels,
        fontsize=_LEGEND_FONTSIZE, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.35), ncol=2,
    )


def _sem(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _high_values(condition_col: str):
    if condition_col == "treatment_clean":
        return ("FFA", "Vehicle")
    return ("high_risk",)
