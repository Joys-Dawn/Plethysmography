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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from ._common import (
    APNEA_DURATION_PARAMS,
    across_style_params,
    add_apnea_duration_reference_line,
    display_label,
    filename_slug,
    group_label,
    make_axes,
    save_figure,
    treatment_word,
)
from .colors import (
    DEFAULT_PALETTE,
    DS_PALE_ORANGE,
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
    palette: Optional[Dict[Tuple[str, str], str]] = None,
    do_timeseries_across: bool = True,
    do_timeseries_within: bool = True,
) -> Optional[Path]:
    """Generate ``_across`` and ``_within`` timeseries plots for one parameter.
    Returns the path to ``_across`` (or ``_within`` if across is skipped or
    empty; ``None`` if both are empty).

    ``palette`` is only consulted by the ``condition_col == "treatment_clean"``
    branch of ``_draw_across`` (acute vs chronic visual differentiation);
    risk-cohort and within plots ignore it.
    """
    output_dir = Path(output_dir)
    out_path: Optional[Path] = None

    # ---- across (P22 only) -------------------------------------------------
    if do_timeseries_across:
        p22 = data[data["age_clean"] == 22]
        if not p22.empty:
            out_path = _draw_across(
                p22, parameter, condition_col=condition_col,
                output_path=output_dir / f"Timeseries_{filename_slug(parameter)}_across.png",
                palette=palette,
            )

    # ---- within (high-risk / FFA-cohort, P19 vs P22) ------------------------
    if do_timeseries_within:
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


# ---------------------------------------------------------------------------
# Experiment 1b (Item G): 2-trace developmental across-periods timeseries.
# The exp1 _draw_across is P22-/condition-hardwired (4 cells, P22-only across),
# so the developmental pair gets its own net-new driver rather than a
# degenerate filtered reuse. Two traces: HR Scn1a+/- P19 (full red, ^) and
# LR Scn1a+/- P22 (pale red, o), connected across the four named periods.
# ---------------------------------------------------------------------------
_DEVELOPMENTAL_TS_GROUPS: Tuple[Tuple[str, str, str, str, str], ...] = (
    (group_label("Scn1a+/-", 19, "HR"), "het", "high_risk", DS_PALE_ORANGE, MARKERS_BY_AGE[19]),
    (group_label("Scn1a+/-", 22, "LR"), "het", "low_risk",  DS_PALE_ORANGE, MARKERS_BY_AGE[22]),
)

# Mirrors publication_plots._detect_parameters (kept inline to avoid a
# publication_plots -> timeseries_plots import cycle).
_DEVELOPMENTAL_TS_PARAMS: Tuple[str, ...] = (
    "mean_ttot_ms_no_apnea", "mean_frequency_bpm_no_apnea",
    "mean_ti_ms_no_apnea", "mean_te_ms_no_apnea",
    "mean_pif_centered_ml_s", "mean_pef_centered_ml_s", "mean_pif_to_pef_ml_s",
    "mean_tv_ml", "sigh_rate_per_min", "mean_sigh_duration_ms",
    "cov", "pif_to_pef_cov",
    "apnea_rate_per_min", "apnea_mean_ms_imputed",
    "apnea_burden_s_per_min",
)


def draw_developmental_timeseries(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    parameters: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Connected-line timeseries for the experiment-1b developmental pair
    (HR Scn1a+/- P19 vs LR Scn1a+/- P22), one file per parameter.

    ``data`` must already carry ``genotype_clean`` / ``risk_clean`` /
    ``age_clean`` (from ``stats.helpers.prepare_breathing_data``) and should
    be the exp1b 2-group cohort — within that cohort ``risk_clean`` uniquely
    identifies each trace, so traces are matched on
    ``(genotype_clean, risk_clean)``. Apnea-duration uses the real V1
    ``apnea_mean_ms`` (via :func:`across_style_params`), same as every other
    across-periods timeseries. Returns the saved paths.
    """
    output_dir = Path(output_dir)
    if parameters is None:
        parameters = [p for p in _DEVELOPMENTAL_TS_PARAMS if p in data.columns]

    saved: List[Path] = []
    for parameter in across_style_params(parameters):
        if parameter not in data.columns:
            continue
        out = _draw_traces(
            data, parameter, _DEVELOPMENTAL_TS_GROUPS,
            match_cols=("genotype_clean", "risk_clean"),
            output_path=output_dir
            / f"Timeseries_{filename_slug(parameter)}_developmental.png",
        )
        if out is not None:
            saved.append(out)
    return saved


def _draw_across(
    p22_data: pd.DataFrame,
    parameter: str,
    *,
    condition_col: str,
    output_path: Path,
    palette: Optional[Dict[Tuple[str, str], str]] = None,
) -> Optional[Path]:
    if condition_col == "treatment_clean":
        # across is P22-only; full 3-feature label genotype -> P22 -> drug.
        # exp3 (acute) injects ACUTE_FFA_PALETTE via the palette kwarg;
        # exp2 (chronic) leaves palette=None and gets TREATMENT_PALETTE.
        # Markers carry the treatment dimension: Vehicle -> o, FFA -> ^.
        eff_palette = palette or TREATMENT_PALETTE
        groups = [
            (group_label("WT", 22, treatment_word("Vehicle")),       "WT",  "Vehicle", eff_palette[("WT",  "Vehicle")], MARKERS_BY_AGE[22]),
            (group_label("WT", 22, treatment_word("FFA")),           "WT",  "FFA",     eff_palette[("WT",  "FFA")],     MARKERS_BY_AGE[22]),
            (group_label("Scn1a+/-", 22, treatment_word("Vehicle")), "het", "Vehicle", eff_palette[("het", "Vehicle")], MARKERS_BY_AGE[22]),
            (group_label("Scn1a+/-", 22, treatment_word("FFA")),     "het", "FFA",     eff_palette[("het", "FFA")],     MARKERS_BY_AGE[22]),
        ]
    else:
        groups = [
            (group_label("WT", 22, "LR"),       "WT",  "low_risk",  DEFAULT_PALETTE[("WT",  "low_risk")],  "o"),
            (group_label("WT", 22, "HR"),       "WT",  "high_risk", DEFAULT_PALETTE[("WT",  "high_risk")], "o"),
            (group_label("Scn1a+/-", 22, "LR"), "het", "low_risk",  DEFAULT_PALETTE[("het", "low_risk")],  "o"),
            (group_label("Scn1a+/-", 22, "HR"), "het", "high_risk", DEFAULT_PALETTE[("het", "high_risk")], "o"),
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
        # FFA cohort within plot: same four (genotype, age) cells as exp 1;
        # cells pool FFA + Vehicle so no treatment word applies (Item C).
        groups = [
            (group_label("WT", 19, None),       "WT",  19, HR_TIMESERIES_PALETTE[("WT",  19)], MARKERS_BY_AGE[19]),
            (group_label("WT", 22, None),       "WT",  22, HR_TIMESERIES_PALETTE[("WT",  22)], MARKERS_BY_AGE[22]),
            (group_label("Scn1a+/-", 19, None), "het", 19, HR_TIMESERIES_PALETTE[("het", 19)], MARKERS_BY_AGE[19]),
            (group_label("Scn1a+/-", 22, None), "het", 22, HR_TIMESERIES_PALETTE[("het", 22)], MARKERS_BY_AGE[22]),
        ]
    else:
        groups = [
            (group_label("WT", 19, "HR"),       "WT",  19, HR_TIMESERIES_PALETTE[("WT",  19)], MARKERS_BY_AGE[19]),
            (group_label("WT", 22, "HR"),       "WT",  22, HR_TIMESERIES_PALETTE[("WT",  22)], MARKERS_BY_AGE[22]),
            (group_label("Scn1a+/-", 19, "HR"), "het", 19, HR_TIMESERIES_PALETTE[("het", 19)], MARKERS_BY_AGE[19]),
            (group_label("Scn1a+/-", 22, "HR"), "het", 22, HR_TIMESERIES_PALETTE[("het", 22)], MARKERS_BY_AGE[22]),
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
    if parameter in APNEA_DURATION_PARAMS:
        add_apnea_duration_reference_line(ax)
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
