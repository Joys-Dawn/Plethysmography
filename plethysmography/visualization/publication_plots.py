"""
Top-level driver that produces the publication plot bundle for one experiment.

Layout mirrors ``old_results/Publication_Plots/`` exactly:

  - ``Within each time period/`` -- bar (strip) plots per (period, parameter)
  - ``Across time periods/``     -- timeseries per parameter
  - ``HR P19 vs LR P22/``        -- developmental comparison plots (exp 1 only)
  - ``FFA/{By_age, By_drug, By_genotype}/`` -- subgroup timeseries (exp 2)
  - ``Postictal_Binned/``        -- postictal 30 s line plots (optional)
  - ``Ictal_Binned/``            -- ictal 1 s line plots (optional)

Bin plots are optional because they need raw preprocessed-period signals;
the bar / timeseries plots only need the breathing CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from ._common import display_label, filename_slug, global_ylim, make_axes, save_figure
from .bar_plots import plot_within_period
from .binned_plots import plot_ictal_binned, plot_postictal_binned
from .colors import (
    HR_TIMESERIES_PALETTE,
    MARKERS_BY_AGE,
    TREATMENT_PALETTE,
    italicize_scn1a,
)
from .timeseries_plots import plot_across_periods


_PERIODS_TO_PLOT = ("Baseline", "Ictal", "Immediate Postictal", "Recovery")
_PERIOD_DISPLAY = {
    "Baseline": "Baseline",
    "Ictal": "Ictal",
    "Immediate Postictal": "Postictal",
    "Recovery": "Recovery",
}
_PERIOD_DURATION_PARAM = "period_duration_s"
_FIG_SIZE_DEVELOPMENTAL = (7, 10)
_PERIODS_ORDER_TS = _PERIODS_TO_PLOT
_PERIOD_LABELS_TS = ("Baseline", "Ictal", "Postictal", "Recovery")


def generate_publication_plots(
    breathing_df: pd.DataFrame,
    output_dir: Path,
    *,
    parameters: Optional[Sequence[str]] = None,
    condition_col: str = "risk_clean",
    postictal_period_data: Optional[Sequence[Tuple[str, np.ndarray, np.ndarray, float]]] = None,
    ictal_period_data: Optional[Sequence[Tuple[str, np.ndarray, np.ndarray, float]]] = None,
    metadata_for_bins: Optional[Dict[str, Dict[str, str]]] = None,
    baseline_median_ttot_ms: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Path]]:
    """Generate the standard publication plot bundle. ``breathing_df`` should
    already have the cleaned grouping columns (``genotype_clean``,
    ``risk_clean`` / ``treatment_clean``, ``age_clean``); use
    :func:`plethysmography.stats.helpers.prepare_breathing_data` to produce
    them. Returns ``{plot_kind: [path, ...]}``.
    """
    output_dir = Path(output_dir)
    if parameters is None:
        parameters = _detect_parameters(breathing_df)

    saved: Dict[str, List[Path]] = {"within": [], "timeseries": [], "postictal": [], "ictal": []}

    within_dir = output_dir / "Within each time period"
    for param in parameters:
        ylim = global_ylim(
            breathing_df, param, _PERIODS_TO_PLOT, condition_col=condition_col,
        )
        for period in _PERIODS_TO_PLOT:
            path = plot_within_period(
                breathing_df, param, period, within_dir,
                condition_col=condition_col,
                display_period=_PERIOD_DISPLAY[period],
                ylim=ylim,
            )
            if path is not None:
                saved["within"].append(path)

    # Period duration: Ictal-only special-case plots (seizure duration scatter
    # for LR Scn1a P22 vs HR Scn1a P22, and HR Scn1a P19 vs HR Scn1a P22).
    # Mirrors old_code/analyze_data.py:create_period_duration_plots. Skipped
    # for experiment 2 (treatment cohort doesn't use the LR/HR split).
    if (
        _PERIOD_DURATION_PARAM in breathing_df.columns
        and condition_col == "risk_clean"
    ):
        for path in _draw_period_duration_plots(breathing_df, within_dir):
            saved["within"].append(path)

    across_dir = output_dir / "Across time periods"
    for param in parameters:
        path = plot_across_periods(
            breathing_df, param, across_dir,
            condition_col=condition_col,
        )
        if path is not None:
            saved["timeseries"].append(path)

    if postictal_period_data is not None and metadata_for_bins is not None:
        saved["postictal"] = plot_postictal_binned(
            postictal_period_data, metadata_for_bins,
            output_dir / "Postictal_Binned",
            condition_col=condition_col,
            baseline_median_ttot_ms=baseline_median_ttot_ms,
        )
    if ictal_period_data is not None and metadata_for_bins is not None:
        saved["ictal"] = plot_ictal_binned(
            ictal_period_data, metadata_for_bins,
            output_dir / "Ictal_Binned",
            condition_col=condition_col,
            baseline_median_ttot_ms=baseline_median_ttot_ms,
        )
    return saved


# ---------------------------------------------------------------------------
# Developmental comparison: HR Scn1a P19 vs LR Scn1a P22
# Mirrors old_code/analyze_data.py:create_developmental_plot.
# ---------------------------------------------------------------------------
def plot_developmental_comparison(
    breathing_df: pd.DataFrame,
    output_dir: Path,
    *,
    parameters: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Bar (strip) plots comparing HR Scn1a+/- P19 against LR Scn1a+/- P22 for
    each (parameter, period). Mirrors
    ``old_results/Publication_Plots/HR P19 vs LR P22/`` exactly.

    ``breathing_df`` must already carry ``genotype_clean``, ``risk_clean``,
    ``age_clean`` from
    :func:`plethysmography.stats.helpers.prepare_breathing_data`.
    """
    output_dir = Path(output_dir)
    if parameters is None:
        parameters = _detect_parameters(breathing_df)
    df = breathing_df[breathing_df["genotype_clean"].astype(str).str.lower() == "het"]
    if df.empty:
        return []

    saved: List[Path] = []
    for param in parameters:
        if param not in df.columns:
            continue
        ylim = global_ylim(breathing_df, param, _PERIODS_TO_PLOT)
        for period in _PERIODS_TO_PLOT:
            sub = df[df["period"] == period]
            hr_p19 = sub[(sub["risk_clean"] == "high_risk") & (sub["age_clean"] == 19)][param]
            lr_p22 = sub[(sub["risk_clean"] == "low_risk")  & (sub["age_clean"] == 22)][param]
            if hr_p19.dropna().empty or lr_p22.dropna().empty:
                continue
            display_period = _PERIOD_DISPLAY[period]
            path = _draw_developmental(
                hr_p19, lr_p22,
                title_period=display_period,
                parameter=param,
                output_path=output_dir / f"{display_period}_{filename_slug(param)}_developmental.png",
                ylim=ylim,
            )
            if path is not None:
                saved.append(path)
    return saved


def _draw_period_duration_plots(
    breathing_df: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Two seizure-duration plots, Ictal-only:

      - ``Period_Duration_Ictal_across.png`` : Scn1a P22, LR vs HR (circles).
      - ``Period_Duration_Ictal_within.png`` : HR Scn1a, P19 vs P22 (^ vs o,
        red).

    Mirrors ``old_code/analyze_data.py:create_period_duration_plots``.
    """
    out: List[Path] = []
    ictal = breathing_df[breathing_df["period"] == "Ictal"]
    if ictal.empty:
        return out

    # ---- across: Scn1a P22, LR vs HR -----------------------------------
    p22_scn = ictal[
        (ictal["genotype_clean"].astype(str) == "het")
        & (ictal["age_clean"] == 22)
    ]
    if not p22_scn.empty:
        path = _draw_two_category_period_duration(
            [
                ("LR Scn1a+/-\nP22", "#FFA07A", "o",
                 p22_scn[p22_scn["risk_clean"] == "low_risk"]["period_duration_s"]),
                ("HR Scn1a+/-\nP22", "#FF0000", "o",
                 p22_scn[p22_scn["risk_clean"] == "high_risk"]["period_duration_s"]),
            ],
            output_path=output_dir / "Period_Duration_Ictal_across.png",
        )
        if path is not None:
            out.append(path)

    # ---- within: HR Scn1a, P19 vs P22 ----------------------------------
    hr_scn = ictal[
        (ictal["genotype_clean"].astype(str) == "het")
        & (ictal["risk_clean"] == "high_risk")
    ]
    if not hr_scn.empty:
        path = _draw_two_category_period_duration(
            [
                ("HR Scn1a+/-\nP19", "#FF0000", "^",
                 hr_scn[hr_scn["age_clean"] == 19]["period_duration_s"]),
                ("HR Scn1a+/-\nP22", "#FF0000", "o",
                 hr_scn[hr_scn["age_clean"] == 22]["period_duration_s"]),
            ],
            output_path=output_dir / "Period_Duration_Ictal_within.png",
        )
        if path is not None:
            out.append(path)
    return out


def _draw_two_category_period_duration(
    cells,
    *,
    output_path: Path,
) -> Optional[Path]:
    fig, ax = make_axes(figsize=_FIG_SIZE_DEVELOPMENTAL)
    rng = np.random.default_rng(4)
    means, sems, x_positions = [], [], []
    for i, (label, color, marker, raw) in enumerate(cells):
        valid = raw.dropna()
        if valid.empty:
            continue
        xs = i + rng.uniform(-0.15, 0.15, size=len(valid))
        ax.scatter(xs, valid, color=color, alpha=0.7, s=150, marker=marker,
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
        capsize=10, markersize=50, markeredgewidth=6, zorder=10,
    )
    ax.set_ylabel("Ictal Period\nDuration (s)", fontsize=40)
    ax.tick_params(axis="both", labelsize=32)
    ax.set_xticks([0, 1])
    ax.set_xlim(-0.6, 1.4)
    ax.set_xticklabels(
        [italicize_scn1a(c[0]) for c in cells], fontsize=32,
    )
    ax.xaxis.set_tick_params(rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, output_path)
    return output_path


def _draw_developmental(
    hr_p19: pd.Series,
    lr_p22: pd.Series,
    *,
    title_period: str,
    parameter: str,
    output_path: Path,
    ylim: Optional[Tuple[float, float]] = None,
) -> Optional[Path]:
    """One panel: two cells (HR Scn1a+/- P19, LR Scn1a+/- P22) with red and
    pale-red colors and ^ vs o markers, black mean +/- SEM marker."""
    categories = [
        ("HR Scn1a+/- P19", "#FF0000", "^", hr_p19),
        ("LR Scn1a+/- P22", "#FFA07A", "o", lr_p22),
    ]
    fig, ax = make_axes(figsize=_FIG_SIZE_DEVELOPMENTAL)
    rng = np.random.default_rng(2)
    means, sems, x_positions = [], [], []
    for i, (label, color, marker, raw) in enumerate(categories):
        if parameter == "apnea_mean_ms":
            valid = raw.dropna()
            nan_count = int(raw.isna().sum())
            if not valid.empty:
                xs = i + rng.uniform(-0.15, 0.15, size=len(valid))
                ax.scatter(xs, valid, color=color, alpha=0.7, s=150, marker=marker,
                           edgecolors="black", linewidth=0.5)
            if nan_count > 0:
                xs = i + rng.uniform(-0.15, 0.15, size=nan_count)
                ax.scatter(xs, np.zeros(nan_count), color="grey", alpha=0.7,
                           s=150, marker=marker, edgecolors="black", linewidth=0.5)
            if not valid.empty:
                means.append(float(valid.mean()))
                sems.append(_sem(valid))
                x_positions.append(i)
        else:
            valid = raw.dropna()
            if valid.empty:
                continue
            xs = i + rng.uniform(-0.15, 0.15, size=len(valid))
            ax.scatter(xs, valid, color=color, alpha=0.7, s=150, marker=marker,
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
        capsize=10, markersize=50, markeredgewidth=6, zorder=10,
    )
    ax.set_ylabel(f"{title_period}\n{display_label(parameter)}", fontsize=40)
    ax.tick_params(axis="both", labelsize=32)
    ax.set_xticks([0, 1])
    ax.set_xlim(-0.6, 1.4)
    ax.set_xticklabels(
        [italicize_scn1a("HR Scn1a+/-\nP19"), italicize_scn1a("LR Scn1a+/-\nP22")],
        fontsize=32,
    )
    ax.xaxis.set_tick_params(rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# FFA subgroup timeseries (By_age / By_drug / By_genotype)
# Mirrors old_code/fenfluramine_plots.py:create_one_fenfluramine_timeseries.
# ---------------------------------------------------------------------------
def plot_ffa_subgroups(
    breathing_df: pd.DataFrame,
    output_dir: Path,
    *,
    parameters: Optional[Sequence[str]] = None,
) -> Dict[str, List[Path]]:
    """Generate the three FFA subgroup folders:

      - ``By_age/Timeseries_<slug>_P19.png`` and ``..._P22.png``
      - ``By_drug/Timeseries_<slug>_FFA.png`` and ``..._Vehicle.png``
      - ``By_genotype/Timeseries_<slug>_Scn1a.png`` and ``..._WT.png``

    Each file is a connected-line timeseries with 4 traces.
    """
    output_dir = Path(output_dir)
    if parameters is None:
        parameters = _detect_parameters(breathing_df)
    saved: Dict[str, List[Path]] = {"by_age": [], "by_drug": [], "by_genotype": []}

    # By_age: fix age, vary (genotype, treatment). P19 plots use circle markers,
    # P22 plots use square markers (mirrors old_code/fenfluramine_plots.py:190-200).
    for age_value, age_label, marker in ((19, "P19", "o"), (22, "P22", "s")):
        for param in parameters:
            if param not in breathing_df.columns:
                continue
            specs = [
                {"label": "WT Vehicle",       "genotype_clean": "WT",  "treatment_clean": "Vehicle", "color": TREATMENT_PALETTE[("WT",  "Vehicle")], "marker": marker},
                {"label": "WT FFA",            "genotype_clean": "WT",  "treatment_clean": "FFA",     "color": TREATMENT_PALETTE[("WT",  "FFA")],     "marker": marker},
                {"label": "Scn1a+/- Vehicle", "genotype_clean": "het", "treatment_clean": "Vehicle", "color": TREATMENT_PALETTE[("het", "Vehicle")], "marker": marker},
                {"label": "Scn1a+/- FFA",     "genotype_clean": "het", "treatment_clean": "FFA",     "color": TREATMENT_PALETTE[("het", "FFA")],     "marker": marker},
            ]
            sub = breathing_df[breathing_df["age_clean"] == age_value]
            path = _draw_ffa_timeseries(
                sub, param, specs,
                output_path=output_dir / "By_age" / f"Timeseries_{filename_slug(param)}_{age_label}.png",
            )
            if path is not None:
                saved["by_age"].append(path)

    # By_drug: fix treatment, vary (genotype, age)
    for treatment in ("FFA", "Vehicle"):
        for param in parameters:
            if param not in breathing_df.columns:
                continue
            specs = [
                {"label": "WT P19",        "genotype_clean": "WT",  "treatment_clean": treatment, "age_clean": 19, "color": TREATMENT_PALETTE[("WT",  treatment)], "marker": "^"},
                {"label": "WT P22",        "genotype_clean": "WT",  "treatment_clean": treatment, "age_clean": 22, "color": TREATMENT_PALETTE[("WT",  treatment)], "marker": "o"},
                {"label": "Scn1a+/- P19",  "genotype_clean": "het", "treatment_clean": treatment, "age_clean": 19, "color": TREATMENT_PALETTE[("het", treatment)], "marker": "^"},
                {"label": "Scn1a+/- P22",  "genotype_clean": "het", "treatment_clean": treatment, "age_clean": 22, "color": TREATMENT_PALETTE[("het", treatment)], "marker": "o"},
            ]
            sub = breathing_df[breathing_df["treatment_clean"].astype(str) == treatment]
            path = _draw_ffa_timeseries(
                sub, param, specs,
                output_path=output_dir / "By_drug" / f"Timeseries_{filename_slug(param)}_{treatment}.png",
            )
            if path is not None:
                saved["by_drug"].append(path)

    # By_genotype: fix genotype, vary (treatment, age)
    for geno_value, geno_label in (("het", "Scn1a"), ("WT", "WT")):
        for param in parameters:
            if param not in breathing_df.columns:
                continue
            specs = [
                {"label": "P19 Vehicle", "genotype_clean": geno_value, "treatment_clean": "Vehicle", "age_clean": 19, "color": TREATMENT_PALETTE[(geno_value, "Vehicle")], "marker": "^"},
                {"label": "P22 Vehicle", "genotype_clean": geno_value, "treatment_clean": "Vehicle", "age_clean": 22, "color": TREATMENT_PALETTE[(geno_value, "Vehicle")], "marker": "o"},
                {"label": "P19 FFA",     "genotype_clean": geno_value, "treatment_clean": "FFA",     "age_clean": 19, "color": TREATMENT_PALETTE[(geno_value, "FFA")],     "marker": "^"},
                {"label": "P22 FFA",     "genotype_clean": geno_value, "treatment_clean": "FFA",     "age_clean": 22, "color": TREATMENT_PALETTE[(geno_value, "FFA")],     "marker": "o"},
            ]
            sub = breathing_df[breathing_df["genotype_clean"].astype(str) == geno_value]
            path = _draw_ffa_timeseries(
                sub, param, specs,
                output_path=output_dir / "By_genotype" / f"Timeseries_{filename_slug(param)}_{geno_label}.png",
            )
            if path is not None:
                saved["by_genotype"].append(path)

    return saved


def _draw_ffa_timeseries(
    data: pd.DataFrame,
    parameter: str,
    specs,
    *,
    output_path: Path,
) -> Optional[Path]:
    """One FFA-style timeseries panel with 4 traces. Mirrors
    ``create_one_fenfluramine_timeseries`` exactly: jittered scatter, mean
    line, error bars, legend below plot."""
    if data.empty:
        return None
    fig, ax = make_axes(figsize=(12, 10))
    rng = np.random.default_rng(3)
    drew = False

    for spec in specs:
        mask = (
            (data["genotype_clean"].astype(str) == spec["genotype_clean"])
            & (data["treatment_clean"].astype(str) == spec["treatment_clean"])
        )
        if "age_clean" in spec:
            mask &= data["age_clean"] == spec["age_clean"]
        sub = data.loc[mask]
        means, sems, x_positions = [], [], []
        for i, period in enumerate(_PERIODS_ORDER_TS):
            valid = sub.loc[sub["period"] == period, parameter].dropna()
            if valid.empty:
                continue
            xs = i + rng.uniform(-0.15, 0.15, size=len(valid))
            ax.scatter(xs, valid, color=spec["color"], alpha=1.0, s=60,
                       marker=spec["marker"], edgecolors="black", linewidth=0.5)
            means.append(float(valid.mean()))
            sems.append(_sem(valid))
            x_positions.append(i)
            drew = True
        if means:
            ax.plot(x_positions, means, color=spec["color"], linewidth=3, alpha=1.0, zorder=10)
            ax.errorbar(x_positions, means, yerr=sems, fmt="none", ecolor=spec["color"],
                        elinewidth=2, capsize=5, alpha=0.8, zorder=10)

    if not drew:
        import matplotlib.pyplot as plt
        plt.close(fig)
        return None

    _format_axes_ffa(ax, parameter, specs)
    save_figure(fig, output_path)
    return output_path


def _format_axes_ffa(ax: Axes, parameter: str, specs) -> None:
    ax.set_ylabel(display_label(parameter), fontsize=40)
    ax.tick_params(axis="both", labelsize=32)
    ax.set_xticks(range(len(_PERIODS_ORDER_TS)))
    ax.set_xticklabels(list(_PERIOD_LABELS_TS), fontsize=32, rotation=45, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = [], []
    for spec in specs:
        h = Line2D([0], [0], color=spec["color"], marker=spec["marker"],
                   linewidth=3, markersize=10, markeredgecolor="black",
                   markeredgewidth=0.5)
        handles.append(h)
        labels.append(italicize_scn1a(spec["label"]))
    ax.legend(handles, labels, fontsize=24, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.35), ncol=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sem(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _detect_parameters(breathing_df: pd.DataFrame) -> List[str]:
    candidates = [
        "mean_ttot_ms", "mean_frequency_bpm", "mean_ti_ms", "mean_te_ms",
        "mean_pif_centered_ml_s", "mean_pef_centered_ml_s", "mean_pif_to_pef_ml_s",
        "mean_tv_ml", "sigh_rate_per_min", "mean_sigh_duration_ms",
        "cov_instant_freq", "alternate_cov", "pif_to_pef_cov",
        "apnea_rate_per_min", "apnea_mean_ms",
    ]
    return [c for c in candidates if c in breathing_df.columns]
