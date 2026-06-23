"""
Top-level driver that produces the publication plot bundle for one experiment.

Section 1 layout (current):

  - ``Time_period_<period>/``    -- bar (strip) plots per parameter, one
                                    folder per period (baseline / ictal /
                                    postictal / recovery)
  - ``Time_periods_all/``        -- across-period timeseries per parameter
  - ``Binned_postictal/``        -- postictal 30 s line plots (optional)
  - ``Binned_ictal/``            -- ictal 1 s line plots (optional)

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

from ._common import (
    APNEA_DURATION_PARAMS,
    across_style_params,
    add_apnea_duration_reference_line,
    display_label,
    draw_mouse_age_pair_lines,
    filename_slug,
    period_ylim,
    group_label,
    make_axes,
    save_figure,
    treatment_word,
    two_line_label,
    within_style_params,
)
from .bar_plots import plot_within_period
from .binned_plots import plot_ictal_binned, plot_postictal_binned
from .colors import (
    DS_BRIGHT_RED,
    DS_PALE_ORANGE,
    MARKERS_BY_AGE,
    ffa_cell_style,
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
    palette: Optional[Dict[Tuple[str, str], str]] = None,
    do_across_strips: bool = True,
    do_within: bool = True,
    do_timeseries_across: bool = True,
    do_timeseries_within: bool = True,
) -> Dict[str, List[Path]]:
    """Generate the standard publication plot bundle. ``breathing_df`` should
    already have the cleaned grouping columns (``genotype_clean``,
    ``risk_clean`` / ``treatment_clean``, ``age_clean``); use
    :func:`plethysmography.stats.helpers.prepare_breathing_data` to produce
    them. Returns ``{plot_kind: [path, ...]}``.

    ``palette`` is forwarded to the four ``treatment_clean``-aware drivers
    (``plot_within_period``, ``plot_across_periods``,
    ``plot_postictal_binned``, ``plot_ictal_binned``). It is only consulted
    inside their ``condition_col == "treatment_clean"`` branches; with
    ``palette=None`` the chronic ``TREATMENT_PALETTE`` default is preserved
    byte-for-byte (exp2). Exp3 (acute) passes ``palette=ACUTE_FFA_PALETTE``.

    ``do_across_strips=False`` skips the P22 ``_*_across.png`` per-period
    strips; ``do_within=False`` skips the ``_*_within.png`` strips (genotype
    x age). Section 1.3 exp3 uses ``do_within=False`` because the acute
    cohort is P22-only. Exp2 sets ``do_across_strips=False`` and
    ``do_timeseries_across=False`` because those panels duplicate the
    ``Time_period_*_by_age/Strip_*_P22`` and
    ``Time_periods_all_by_age/Timeseries_*_P22`` facet outputs.
    ``do_timeseries_within=False`` skips ``Timeseries_*_within.png`` in
    ``Time_periods_all/``.
    """
    output_dir = Path(output_dir)
    if parameters is None:
        parameters = _detect_parameters(breathing_df)

    saved: Dict[str, List[Path]] = {"within": [], "timeseries": [], "postictal": [], "ictal": []}

    # Per-period strip plots: each period's PNGs go into their own
    # ``Time_period_<lowercase-display>/`` folder directly under
    # ``output_dir`` (Section 1 layout).
    if do_across_strips or do_within:
        for param in within_style_params(parameters):
            for period in _PERIODS_TO_PLOT:
                ylim = period_ylim(
                    breathing_df, param, period, condition_col=condition_col,
                )
                period_dir = output_dir / _period_folder(period)
                path = plot_within_period(
                    breathing_df, param, period, period_dir,
                    condition_col=condition_col,
                    display_period=_PERIOD_DISPLAY[period],
                    ylim=ylim,
                    palette=palette,
                    do_across=do_across_strips,
                    do_within=do_within,
                )
                if path is not None:
                    saved["within"].append(path)

    if do_within:
        # Period duration: Ictal-only special-case plots (seizure duration
        # scatter for LR Scn1a P22 vs HR Scn1a P22, and HR Scn1a P19 vs HR
        # Scn1a P22). Mirrors old_code/analyze_data.py:create_period_duration_plots.
        # Skipped for experiment 2 (treatment cohort doesn't use the LR/HR split).
        if (
            _PERIOD_DURATION_PARAM in breathing_df.columns
            and condition_col == "risk_clean"
        ):
            ictal_dir = output_dir / _period_folder("Ictal")
            for path in _draw_period_duration_plots(breathing_df, ictal_dir):
                saved["within"].append(path)

    if do_timeseries_across or do_timeseries_within:
        across_dir = output_dir / "Time_periods_all"
        for param in across_style_params(parameters):
            path = plot_across_periods(
                breathing_df, param, across_dir,
                condition_col=condition_col,
                palette=palette,
                do_timeseries_across=do_timeseries_across,
                do_timeseries_within=do_timeseries_within,
            )
            if path is not None:
                saved["timeseries"].append(path)

    if postictal_period_data is not None and metadata_for_bins is not None:
        saved["postictal"] = plot_postictal_binned(
            postictal_period_data, metadata_for_bins,
            output_dir / "Binned_postictal",
            condition_col=condition_col,
            baseline_median_ttot_ms=baseline_median_ttot_ms,
            palette=palette,
        )
    if ictal_period_data is not None and metadata_for_bins is not None:
        saved["ictal"] = plot_ictal_binned(
            ictal_period_data, metadata_for_bins,
            output_dir / "Binned_ictal",
            condition_col=condition_col,
            baseline_median_ttot_ms=baseline_median_ttot_ms,
            palette=palette,
        )
    return saved


def _period_folder(period: str) -> str:
    """Lowercase-display token used for per-period plot folder names
    (e.g. ``"Immediate Postictal"`` → ``"Time_period_postictal"``)."""
    return f"Time_period_{_PERIOD_DISPLAY[period].lower()}"


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
    """Bar (strip) plots comparing HR Scn1a+/- P19 against LR Scn1a+/- P22
    for each (parameter, period). ``output_dir`` is the experiment's
    publication root; each PNG is written into the period-scoped
    ``Time_period_<period>/`` subfolder under it.

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
    # Developmental strips are within-style: emit V1 (real) + V2 (imputed).
    for param in within_style_params(parameters):
        if param not in df.columns:
            continue
        for period in _PERIODS_TO_PLOT:
            ylim = period_ylim(breathing_df, param, period)
            sub = df[df["period"] == period]
            hr_p19 = sub[(sub["risk_clean"] == "high_risk") & (sub["age_clean"] == 19)][param]
            lr_p22 = sub[(sub["risk_clean"] == "low_risk")  & (sub["age_clean"] == 22)][param]
            if hr_p19.dropna().empty or lr_p22.dropna().empty:
                continue
            display_period = _PERIOD_DISPLAY[period]
            period_dir = output_dir / _period_folder(period)
            path = _draw_developmental(
                hr_p19, lr_p22,
                title_period=display_period,
                parameter=param,
                output_path=period_dir / f"{display_period}_{filename_slug(param)}_developmental.png",
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
                (group_label("Scn1a+/-", 22, "LR"), DS_PALE_ORANGE, "o",
                 p22_scn[p22_scn["risk_clean"] == "low_risk"]["period_duration_s"]),
                (group_label("Scn1a+/-", 22, "HR"), DS_BRIGHT_RED, "o",
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
                (group_label("Scn1a+/-", 19, "HR"), DS_PALE_ORANGE, "^",
                 hr_scn[hr_scn["age_clean"] == 19]["period_duration_s"]),
                (group_label("Scn1a+/-", 22, "HR"), DS_BRIGHT_RED, "o",
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
        [italicize_scn1a(two_line_label(c[0])) for c in cells],
        fontsize=32, ha="center",
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
        (group_label("Scn1a+/-", 19, "HR"), DS_PALE_ORANGE, "^", hr_p19),
        (group_label("Scn1a+/-", 22, "LR"), DS_PALE_ORANGE, "o", lr_p22),
    ]
    fig, ax = make_axes(figsize=_FIG_SIZE_DEVELOPMENTAL)
    rng = np.random.default_rng(2)
    means, sems, x_positions = [], [], []
    for i, (label, color, marker, raw) in enumerate(categories):
        # Grey marker-at-0 special-case for apnea_mean_ms removed; V1 plots
        # >=1-apnea traces only, V2 (imputed) has no NaNs (Item B).
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
        [italicize_scn1a(two_line_label(group_label("Scn1a+/-", 19, "HR"))),
         italicize_scn1a(two_line_label(group_label("Scn1a+/-", 22, "LR")))],
        fontsize=32, ha="center",
    )
    ax.xaxis.set_tick_params(rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    if parameter in APNEA_DURATION_PARAMS:
        add_apnea_duration_reference_line(ax)
    save_figure(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# FFA subgroup timeseries (Time_periods_all_by_{age,drug,genotype}/)
# Mirrors old_code/fenfluramine_plots.py:create_one_fenfluramine_timeseries.
# ---------------------------------------------------------------------------
def _ffa_trace_spec(
    geno_display: str,
    geno_key: str,
    age: int,
    treatment: str,
) -> dict:
    color, marker = ffa_cell_style(geno_key, age, treatment)
    spec = {
        "label": group_label(geno_display, age, treatment_word(treatment)),
        "genotype_clean": geno_key,
        "treatment_clean": treatment,
        "color": color,
        "marker": marker,
    }
    return spec


def _ffa_age_strip_cells(age: int) -> list:
    return [
        (
            group_label("WT", age, treatment_word("Vehicle")),
            "WT",
            "Vehicle",
            *ffa_cell_style("WT", age, "Vehicle"),
        ),
        (
            group_label("WT", age, treatment_word("FFA")),
            "WT",
            "FFA",
            *ffa_cell_style("WT", age, "FFA"),
        ),
        (
            group_label("Scn1a+/-", age, treatment_word("Vehicle")),
            "het",
            "Vehicle",
            *ffa_cell_style("het", age, "Vehicle"),
        ),
        (
            group_label("Scn1a+/-", age, treatment_word("FFA")),
            "het",
            "FFA",
            *ffa_cell_style("het", age, "FFA"),
        ),
    ]


def _ffa_drug_strip_cells(treatment: str) -> list:
    return [
        (
            group_label("WT", 19, treatment_word(treatment)),
            "WT",
            19,
            *ffa_cell_style("WT", 19, treatment),
        ),
        (
            group_label("WT", 22, treatment_word(treatment)),
            "WT",
            22,
            *ffa_cell_style("WT", 22, treatment),
        ),
        (
            group_label("Scn1a+/-", 19, treatment_word(treatment)),
            "het",
            19,
            *ffa_cell_style("het", 19, treatment),
        ),
        (
            group_label("Scn1a+/-", 22, treatment_word(treatment)),
            "het",
            22,
            *ffa_cell_style("het", 22, treatment),
        ),
    ]


def _ffa_geno_strip_cells(geno_value: str, geno_display: str) -> list:
    return [
        (
            group_label(geno_display, 19, treatment_word("Vehicle")),
            "Vehicle",
            19,
            *ffa_cell_style(geno_value, 19, "Vehicle"),
        ),
        (
            group_label(geno_display, 22, treatment_word("Vehicle")),
            "Vehicle",
            22,
            *ffa_cell_style(geno_value, 22, "Vehicle"),
        ),
        (
            group_label(geno_display, 19, treatment_word("FFA")),
            "FFA",
            19,
            *ffa_cell_style(geno_value, 19, "FFA"),
        ),
        (
            group_label(geno_display, 22, treatment_word("FFA")),
            "FFA",
            22,
            *ffa_cell_style(geno_value, 22, "FFA"),
        ),
    ]


def plot_ffa_subgroups(
    breathing_df: pd.DataFrame,
    output_dir: Path,
    *,
    parameters: Optional[Sequence[str]] = None,
) -> Dict[str, List[Path]]:
    """Generate the three FFA subgroup folders (Section 1 layout):

      - ``Time_periods_all_by_age/Timeseries_<slug>_{P19,P22}.png``
      - ``Time_periods_all_by_drug/Timeseries_<slug>_{FFA,Vehicle}.png``
      - ``Time_periods_all_by_genotype/Timeseries_<slug>_{Scn1a,WT}.png``

    Each file is a connected-line timeseries with 4 traces. ``output_dir``
    is the experiment's publication root.
    """
    output_dir = Path(output_dir)
    if parameters is None:
        parameters = _detect_parameters(breathing_df)
    saved: Dict[str, List[Path]] = {"by_age": [], "by_drug": [], "by_genotype": []}

    by_age_dir = output_dir / "Time_periods_all_by_age"
    by_drug_dir = output_dir / "Time_periods_all_by_drug"
    by_geno_dir = output_dir / "Time_periods_all_by_genotype"

    # By_age: fix age, vary (genotype, treatment). Marker carries age.
    for age_value, age_label in ((19, "P19"), (22, "P22")):
        for param in parameters:
            if param not in breathing_df.columns:
                continue
            specs = [
                {**_ffa_trace_spec("WT", "WT", age_value, "Vehicle"), },
                {**_ffa_trace_spec("WT", "WT", age_value, "FFA"), },
                {**_ffa_trace_spec("Scn1a+/-", "het", age_value, "Vehicle"), },
                {**_ffa_trace_spec("Scn1a+/-", "het", age_value, "FFA"), },
            ]
            sub = breathing_df[breathing_df["age_clean"] == age_value]
            path = _draw_ffa_timeseries(
                sub, param, specs,
                output_path=by_age_dir / f"Timeseries_{filename_slug(param)}_{age_label}.png",
            )
            if path is not None:
                saved["by_age"].append(path)

    # By_drug: fix treatment, vary (genotype, age). Marker carries age.
    for treatment in ("FFA", "Vehicle"):
        for param in parameters:
            if param not in breathing_df.columns:
                continue
            specs = [
                {**_ffa_trace_spec("WT", "WT", 19, treatment), "age_clean": 19},
                {**_ffa_trace_spec("WT", "WT", 22, treatment), "age_clean": 22},
                {**_ffa_trace_spec("Scn1a+/-", "het", 19, treatment), "age_clean": 19},
                {**_ffa_trace_spec("Scn1a+/-", "het", 22, treatment), "age_clean": 22},
            ]
            sub = breathing_df[breathing_df["treatment_clean"].astype(str) == treatment]
            path = _draw_ffa_timeseries(
                sub, param, specs,
                output_path=by_drug_dir / f"Timeseries_{filename_slug(param)}_{treatment}.png",
            )
            if path is not None:
                saved["by_drug"].append(path)

    # By_genotype: fix genotype, vary (treatment, age). Marker carries age.
    for geno_value, geno_label in (("het", "Scn1a"), ("WT", "WT")):
        for param in parameters:
            if param not in breathing_df.columns:
                continue
            geno_disp = "Scn1a+/-" if geno_value == "het" else "WT"
            specs = [
                {**_ffa_trace_spec(geno_disp, geno_value, 19, "Vehicle"), "age_clean": 19},
                {**_ffa_trace_spec(geno_disp, geno_value, 22, "Vehicle"), "age_clean": 22},
                {**_ffa_trace_spec(geno_disp, geno_value, 19, "FFA"), "age_clean": 19},
                {**_ffa_trace_spec(geno_disp, geno_value, 22, "FFA"), "age_clean": 22},
            ]
            sub = breathing_df[breathing_df["genotype_clean"].astype(str) == geno_value]
            path = _draw_ffa_timeseries(
                sub, param, specs,
                output_path=by_geno_dir / f"Timeseries_{filename_slug(param)}_{geno_label}.png",
            )
            if path is not None:
                saved["by_genotype"].append(path)

    return saved


# ---------------------------------------------------------------------------
# Per-period FFA strips (Section 1.3 exp2): same three facets as
# plot_ffa_subgroups, but each facet level becomes a strip plot at ONE
# period (4 cells per panel; no time axis). 4 periods x 3 facets x 2 levels
# = 24 PNGs per parameter for the chronic cohort.
# ---------------------------------------------------------------------------
def plot_ffa_per_period_strips(
    breathing_df: pd.DataFrame,
    output_dir: Path,
    *,
    parameters: Optional[Sequence[str]] = None,
) -> Dict[str, List[Path]]:
    """For each (period, facet, level) produce a 4-cell strip plot at that
    fixed period. Output layout:

      - ``Time_period_<period>_by_age/Strip_<slug>_{P19,P22}.png``
      - ``Time_period_<period>_by_drug/Strip_<slug>_{FFA,Vehicle}.png``
      - ``Time_period_<period>_by_genotype/Strip_<slug>_{Scn1a,WT}.png``

    Each strip has 4 cells (geno x treatment for by_age, geno x age for
    by_drug, treatment x age for by_genotype), drawn the same way as
    :func:`plot_within_period`'s ``_draw_across`` but with the fixed facet
    dimension swapped out.
    """
    output_dir = Path(output_dir)
    if parameters is None:
        parameters = _detect_parameters(breathing_df)
    saved: Dict[str, List[Path]] = {"by_age": [], "by_drug": [], "by_genotype": []}

    for period in _PERIODS_TO_PLOT:
        period_lower = _PERIOD_DISPLAY[period].lower()
        by_age_dir = output_dir / f"Time_period_{period_lower}_by_age"
        by_drug_dir = output_dir / f"Time_period_{period_lower}_by_drug"
        by_geno_dir = output_dir / f"Time_period_{period_lower}_by_genotype"
        display_period = _PERIOD_DISPLAY[period]
        period_data = breathing_df[breathing_df["period"] == period]
        if period_data.empty:
            continue

        # By_age: P19 and P22 facets share one y-axis within the period.
        by_age_mask = pd.Series(True, index=breathing_df.index)
        for age_value, age_label in ((19, "P19"), (22, "P22")):
            for param in within_style_params(parameters):
                if param not in breathing_df.columns:
                    continue
                ylim = period_ylim(
                    breathing_df, param, period,
                    condition_col="treatment_clean",
                    extra_mask=by_age_mask,
                )
                cells = _ffa_age_strip_cells(age_value)
                sub = period_data[period_data["age_clean"] == age_value]
                if sub.empty:
                    continue
                path = _draw_strip(
                    sub, param, cells,
                    match_cols=("genotype_clean", "treatment_clean"),
                    title_period=display_period,
                    output_path=by_age_dir / f"Strip_{filename_slug(param)}_{age_label}.png",
                    ylim=ylim,
                )
                if path is not None:
                    saved["by_age"].append(path)

        # By_drug: one y-axis per treatment facet within the period.
        for treatment in ("FFA", "Vehicle"):
            drug_mask = breathing_df["treatment_clean"].astype(str) == treatment
            for param in within_style_params(parameters):
                if param not in breathing_df.columns:
                    continue
                ylim = period_ylim(
                    breathing_df, param, period,
                    condition_col="treatment_clean",
                    extra_mask=drug_mask,
                )
                cells = _ffa_drug_strip_cells(treatment)
                sub = period_data[period_data["treatment_clean"].astype(str) == treatment]
                if sub.empty:
                    continue
                path = _draw_strip(
                    sub, param, cells,
                    match_cols=("genotype_clean", "age_clean"),
                    title_period=display_period,
                    output_path=by_drug_dir / f"Strip_{filename_slug(param)}_{treatment}.png",
                    ylim=ylim,
                )
                if path is not None:
                    saved["by_drug"].append(path)

        # By_genotype: one y-axis per genotype facet within the period.
        for geno_value, geno_label in (("het", "Scn1a"), ("WT", "WT")):
            geno_disp = "Scn1a+/-" if geno_value == "het" else "WT"
            geno_mask = breathing_df["genotype_clean"].astype(str) == geno_value
            for param in within_style_params(parameters):
                if param not in breathing_df.columns:
                    continue
                ylim = period_ylim(
                    breathing_df, param, period,
                    condition_col="treatment_clean",
                    extra_mask=geno_mask,
                )
                cells = _ffa_geno_strip_cells(geno_value, geno_disp)
                sub = period_data[period_data["genotype_clean"].astype(str) == geno_value]
                if sub.empty:
                    continue
                path = _draw_strip(
                    sub, param, cells,
                    match_cols=("treatment_clean", "age_clean"),
                    title_period=display_period,
                    output_path=by_geno_dir / f"Strip_{filename_slug(param)}_{geno_label}.png",
                    ylim=ylim,
                )
                if path is not None:
                    saved["by_genotype"].append(path)

    return saved


def _draw_strip(
    data: pd.DataFrame,
    parameter: str,
    cells: Sequence[Tuple[str, object, object, str, str]],
    *,
    match_cols: Tuple[str, str],
    title_period: str,
    output_path: Path,
    ylim: Optional[Tuple[float, float]] = None,
) -> Optional[Path]:
    """Generic 4-cell strip-plot draw helper for the FFA per-period facets.

    ``cells`` is a sequence of ``(label, value_a, value_b, color, marker)``;
    ``match_cols`` selects which DataFrame columns the cell values match
    against. Apnea-duration parameters carry the Item B 400 ms reference line
    (and the helper also re-anchors the y-axis at zero, Section 2.2d).
    """
    fig, ax = make_axes(figsize=(10, 10))
    rng = np.random.default_rng(7)
    means, sems, x_positions = [], [], []
    drew = False
    points_by_mouse: dict[str, dict[int, tuple[float, float, str]]] = {}
    has_mouse_id = "mouse_id" in data.columns
    ca, cb = match_cols
    ages_in_panel: set[int] = set()
    for _label, va, vb, _color, _marker in cells:
        if cb == "age_clean":
            ages_in_panel.add(int(vb))
        elif ca == "age_clean":
            ages_in_panel.add(int(va))
    pair_ages = has_mouse_id and 19 in ages_in_panel and 22 in ages_in_panel

    for i, (_label, va, vb, color, marker) in enumerate(cells):
        mask = (data[ca].astype(str) == str(va))
        if cb == "age_clean":
            mask &= data[cb] == vb
        else:
            mask &= data[cb].astype(str) == str(vb)
        if has_mouse_id:
            sub_df = data.loc[mask, [parameter, "mouse_id"]].dropna(subset=[parameter])
        else:
            sub_df = data.loc[mask, [parameter]].dropna(subset=[parameter])
        if sub_df.empty:
            continue
        for _, row in sub_df.iterrows():
            y = float(row[parameter])
            x = i + rng.uniform(-0.15, 0.15)
            ax.scatter(x, y, color=color, alpha=0.7, s=150, marker=marker,
                       edgecolors="black", linewidth=0.5, zorder=6)
            if pair_ages:
                age_val = int(vb if cb == "age_clean" else va)
                points_by_mouse.setdefault(str(row["mouse_id"]), {})[age_val] = (
                    x, y, color,
                )
        valid = sub_df[parameter]
        means.append(float(valid.mean()))
        sems.append(_sem(valid))
        x_positions.append(i)
        drew = True

    if pair_ages:
        draw_mouse_age_pair_lines(ax, points_by_mouse)

    if not drew:
        import matplotlib.pyplot as plt
        plt.close(fig)
        return None

    ax.errorbar(
        x_positions, means, yerr=sems, fmt="_", color="black",
        capsize=10, markersize=50, markeredgewidth=6, zorder=10,
    )
    ax.set_ylabel(f"{title_period}\n{display_label(parameter)}", fontsize=40)
    ax.tick_params(axis="both", labelsize=32)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(
        [italicize_scn1a(two_line_label(c[0])) for c in cells],
        fontsize=32, ha="center",
    )
    ax.xaxis.set_tick_params(rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    if parameter in APNEA_DURATION_PARAMS:
        add_apnea_duration_reference_line(ax)
    save_figure(fig, output_path)
    return output_path


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
    if parameter in APNEA_DURATION_PARAMS:
        add_apnea_duration_reference_line(ax)
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
    """Default parameter list for publication plots.

    Timing parameters use the ``*_no_apnea`` columns, and the apnea duration
    parameter uses ``apnea_mean_ms_imputed``, so plots aren't skewed by
    apneic breaths and 0-apnea traces still appear in the duration plots
    (see :mod:`plethysmography.analysis.breath_metrics` for rationale).
    The legacy columns stay in the breathing CSV for audit but are not
    plotted by default.
    """
    candidates = [
        "mean_ttot_ms_no_apnea", "mean_frequency_bpm_no_apnea",
        "mean_ti_ms_no_apnea", "mean_te_ms_no_apnea",
        "mean_pif_centered_ml_s", "mean_pef_centered_ml_s", "mean_pif_to_pef_ml_s",
        "mean_tv_ml", "sigh_rate_per_min", "mean_sigh_duration_ms",
        "cov", "pif_to_pef_cov",
        "apnea_rate_per_min", "apnea_mean_ms_imputed",
        "apnea_burden_s_per_min",
    ]
    return [c for c in candidates if c in breathing_df.columns]
