"""
Strip plots for one parameter at one period: jittered scatter of every mouse
plus a black mean +/- SEM marker.

Mirrors ``old_code/analyze_data.py``:

  * :func:`plot_within_period` -> ``create_within_plot`` (high-risk only,
    P19 vs P22, triangle vs circle markers, blue WT / red Scn1a, with paler
    P19 colors when condition_col == 'risk_clean').
  * :func:`plot_across_period` -> ``create_across_plot`` (P22 only, LR vs
    HR, all circles, pale vs full blue/red).

Output filenames mirror old_results/Publication_Plots/Within each time period/
exactly: ``<Period>_<param_slug>_within.png`` and ``..._across.png``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from ._common import display_label, filename_slug, make_axes, save_figure
from .colors import (
    DEFAULT_PALETTE,
    HR_BAR_PALETTE,
    MARKERS_BY_AGE,
    TREATMENT_PALETTE,
    italicize_scn1a,
)


_JITTER = 0.15
_MARKER_SIZE = 150
_MARKER_ALPHA = 0.7
_MARKER_EDGE_LINEWIDTH = 0.5
_MEAN_CAPSIZE = 10
_MEAN_MARKERSIZE = 50
_MEAN_EDGEWIDTH = 6
_FIG_SIZE = (10, 10)
_YLABEL_FONTSIZE = 40
_TICK_FONTSIZE = 32


def plot_within_period(
    data: pd.DataFrame,
    parameter: str,
    period: str,
    output_dir: Path,
    *,
    condition_col: str = "risk_clean",
    display_period: Optional[str] = None,
    ylim: Optional[tuple] = None,
) -> Optional[Path]:
    """Generate the standard pair of plots for one (period, parameter):

      - ``<display_period>_<slug>_across.png`` : P22 mice, four (genotype x
        condition) cells.
      - ``<display_period>_<slug>_within.png`` : within the high-risk /
        treated cohort, four (genotype x age) cells.

    ``display_period`` is the token used in the filename and the y-label first
    line. Defaults to ``period`` itself; old code maps "Immediate Postictal"
    -> "Postictal" so we accept that mapping from the caller.

    ``ylim`` is the shared (y_min, y_max) computed across all 4 periods (use
    :func:`._common.global_ylim` to compute it once).

    Returns the path to the ``_across`` plot (or the ``_within`` plot, if
    P22 data is empty). Returns ``None`` if both subsets are empty.
    """
    output_dir = Path(output_dir)
    out_path: Optional[Path] = None
    display_period = display_period or period

    # ---- across (P22 only) -------------------------------------------------
    p22 = data[(data["age_clean"] == 22) & (data["period"] == period)]
    if not p22.empty:
        out_path = _draw_across(
            p22, parameter, display_period,
            output_path=output_dir / f"{display_period}_{filename_slug(parameter)}_across.png",
            condition_col=condition_col,
            ylim=ylim,
        )

    # ---- within (HR / FFA-cohort, P19 vs P22) ------------------------------
    high_values = _high_values(condition_col)
    hr = data[
        (data[condition_col].astype(str).isin(high_values))
        & (data["period"] == period)
    ]
    if not hr.empty:
        within_path = _draw_within(
            hr, parameter, display_period,
            output_path=output_dir / f"{display_period}_{filename_slug(parameter)}_within.png",
            condition_col=condition_col,
            ylim=ylim,
        )
        if out_path is None:
            out_path = within_path
    return out_path


# ---------------------------------------------------------------------------
# Internal: across
# ---------------------------------------------------------------------------
def _draw_across(
    p22_data: pd.DataFrame,
    parameter: str,
    period: str,
    *,
    output_path: Path,
    condition_col: str,
    ylim: Optional[tuple] = None,
) -> Optional[Path]:
    """``create_across_plot``-equivalent. P22 mice, four cells along the
    genotype x condition axis. All markers are circles."""
    if condition_col == "treatment_clean":
        categories = [
            ("WT", "Vehicle"),
            ("WT", "FFA"),
            ("het", "Vehicle"),
            ("het", "FFA"),
        ]
        palette = TREATMENT_PALETTE
        label_for = _treatment_label
    else:
        categories = [
            ("WT", "low_risk"),
            ("WT", "high_risk"),
            ("het", "low_risk"),
            ("het", "high_risk"),
        ]
        palette = DEFAULT_PALETTE
        label_for = _risk_label

    fig, ax = make_axes(figsize=_FIG_SIZE)
    means, sems, x_positions = [], [], []
    rng = np.random.default_rng(0)

    for i, combo in enumerate(categories):
        gen, cond = combo
        mask = (
            (p22_data["genotype_clean"].astype(str) == gen)
            & (p22_data[condition_col].astype(str) == cond)
        )
        sub = p22_data.loc[mask, parameter]
        if parameter == "apnea_mean_ms":
            valid = sub.dropna()
            nan_count = sub.isna().sum()
            color = palette[combo]
            if not valid.empty:
                xs = i + rng.uniform(-_JITTER, _JITTER, size=len(valid))
                ax.scatter(xs, valid, color=color, alpha=_MARKER_ALPHA,
                           s=_MARKER_SIZE, marker="o",
                           edgecolors="black", linewidth=_MARKER_EDGE_LINEWIDTH)
            if nan_count > 0:
                xs = i + rng.uniform(-_JITTER, _JITTER, size=int(nan_count))
                ax.scatter(xs, np.zeros(int(nan_count)), color="grey",
                           alpha=_MARKER_ALPHA, s=_MARKER_SIZE, marker="o",
                           edgecolors="black", linewidth=_MARKER_EDGE_LINEWIDTH)
            if not valid.empty:
                means.append(float(valid.mean()))
                sems.append(_sem(valid))
                x_positions.append(i)
        else:
            valid = sub.dropna()
            if valid.empty:
                continue
            color = palette[combo]
            xs = i + rng.uniform(-_JITTER, _JITTER, size=len(valid))
            ax.scatter(xs, valid, color=color, alpha=_MARKER_ALPHA,
                       s=_MARKER_SIZE, marker="o",
                       edgecolors="black", linewidth=_MARKER_EDGE_LINEWIDTH)
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
    _format_axes(
        ax,
        ylabel=f"{period}\n{display_label(parameter)}",
        xtick_labels=[label_for(c) for c in categories],
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Internal: within
# ---------------------------------------------------------------------------
def _draw_within(
    hr_data: pd.DataFrame,
    parameter: str,
    period: str,
    *,
    output_path: Path,
    condition_col: str,
    ylim: Optional[tuple] = None,
) -> Optional[Path]:
    """``create_within_plot``-equivalent. High-risk (or FFA-cohort) only,
    four cells along the genotype x age axis. P19 = triangle marker, P22 =
    circle. P19 cells use pale colors; P22 cells use full saturation."""
    fig, ax = make_axes(figsize=_FIG_SIZE)
    categories = [("WT", 19), ("WT", 22), ("het", 19), ("het", 22)]
    means, sems, x_positions = [], [], []
    rng = np.random.default_rng(1)

    for i, (gen, age) in enumerate(categories):
        mask = (
            (hr_data["genotype_clean"].astype(str) == gen)
            & (hr_data["age_clean"] == age)
        )
        sub = hr_data.loc[mask, parameter]
        marker = MARKERS_BY_AGE.get(int(age), "o")
        color = HR_BAR_PALETTE.get((gen, int(age)), "#6b7280")

        if parameter == "apnea_mean_ms":
            valid = sub.dropna()
            nan_count = int(sub.isna().sum())
            if not valid.empty:
                xs = i + rng.uniform(-_JITTER, _JITTER, size=len(valid))
                ax.scatter(xs, valid, color=color, alpha=_MARKER_ALPHA,
                           s=_MARKER_SIZE, marker=marker,
                           edgecolors="black", linewidth=_MARKER_EDGE_LINEWIDTH)
            if nan_count > 0:
                xs = i + rng.uniform(-_JITTER, _JITTER, size=nan_count)
                ax.scatter(xs, np.zeros(nan_count), color="grey",
                           alpha=_MARKER_ALPHA, s=_MARKER_SIZE, marker=marker,
                           edgecolors="black", linewidth=_MARKER_EDGE_LINEWIDTH)
            if not valid.empty:
                means.append(float(valid.mean()))
                sems.append(_sem(valid))
                x_positions.append(i)
        else:
            valid = sub.dropna()
            if valid.empty:
                continue
            xs = i + rng.uniform(-_JITTER, _JITTER, size=len(valid))
            ax.scatter(xs, valid, color=color, alpha=_MARKER_ALPHA,
                       s=_MARKER_SIZE, marker=marker,
                       edgecolors="black", linewidth=_MARKER_EDGE_LINEWIDTH)
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
    label_for = _within_label_for(condition_col)
    _format_axes(
        ax,
        ylabel=f"{period}\n{display_label(parameter)}",
        xtick_labels=[label_for(g, a) for g, a in categories],
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _sem(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _high_values(condition_col: str):
    if condition_col == "treatment_clean":
        return ("FFA", "Vehicle")
    return ("high_risk",)


def _format_axes(ax: Axes, *, ylabel: str, xtick_labels) -> None:
    """Apply old-code formatting: 40pt y-label, 32pt ticks, 45deg rotation,
    italic Scn1a, hidden top/right spines."""
    ax.set_ylabel(ylabel, fontsize=_YLABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
    ax.set_xticks(range(len(xtick_labels)))
    formatted = [italicize_scn1a(lbl) for lbl in xtick_labels]
    ax.set_xticklabels(formatted, fontsize=_TICK_FONTSIZE)
    ax.xaxis.set_tick_params(rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _risk_label(combo) -> str:
    gen, cond = combo
    risk = "LR" if cond == "low_risk" else "HR"
    geno = "Scn1a+/-" if gen == "het" else "WT"
    return f"{risk} {geno}\nP22"


def _treatment_label(combo) -> str:
    gen, treat = combo
    geno = "Scn1a+/-" if gen == "het" else "WT"
    return f"{geno}\n{treat}"


def _within_label_for(condition_col: str):
    if condition_col == "treatment_clean":
        def label(gen, age):
            geno = "Scn1a+/-" if gen == "het" else "WT"
            return f"{geno}\nP{age}"
        return label

    def label(gen, age):
        geno = "Scn1a+/-" if gen == "het" else "WT"
        return f"HR {geno}\nP{age}"
    return label
