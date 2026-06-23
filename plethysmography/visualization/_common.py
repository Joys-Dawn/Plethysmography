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
#
# Both the legacy column keys (used by binned_plots, which compute per-bin
# values from raw signal) and the project-extension keys (``*_no_apnea``,
# ``apnea_mean_ms_imputed``, ``apnea_burden_s_per_min``) resolve to the
# same human-facing labels and slugs. They live in different output
# directories (``Postictal_Binned/`` vs ``Within each time period/``), so
# matching slugs do not collide on disk and the axis label reads cleanly
# in both cases.
_PARAM_DISPLAY: Dict[str, Tuple[str, str]] = {
    # Timing — legacy and no-apnea share labels.
    "mean_ttot_ms":                ("Ttot (ms)",       "Ttot_ms"),
    "mean_ttot_ms_no_apnea":       ("Ttot (ms)",       "Ttot_ms"),
    "mean_frequency_bpm":          ("Frequency (bpm)", "Frequency_bpm"),
    "mean_frequency_bpm_no_apnea": ("Frequency (bpm)", "Frequency_bpm"),
    "mean_ti_ms":                  ("Ti (ms)",         "Ti_ms"),
    "mean_ti_ms_no_apnea":         ("Ti (ms)",         "Ti_ms"),
    "mean_te_ms":                  ("Te (ms)",         "Te_ms"),
    "mean_te_ms_no_apnea":         ("Te (ms)",         "Te_ms"),
    # Apnea duration — V1 (real durations, >=1-apnea traces only) and V2
    # (imputed: real, or the longest min(10, n) Ttot when 0 apneas) are now
    # DISTINCT plots with distinct labels and slugs so the two PNGs no longer
    # collide on disk. Burden is its own parameter.
    "apnea_mean_ms":             ("Apnea duration (ms)",  "Apnea_duration_ms"),
    "apnea_mean_ms_imputed":     ("Apnea or longest-breaths\nduration (ms)",
                                  "Apnea_or_longest_breaths_duration_ms"),
    "apnea_burden_s_per_min":    ("Apnea burden (s/min)",  "Apnea_burden_s_per_min"),
    # Unchanged columns
    "mean_pif_centered_ml_s":    ("PIF (mL/s)",           "PIF_mL_s"),
    "mean_pef_centered_ml_s":    ("PEF (mL/s)",           "PEF_mL_s"),
    "mean_pif_to_pef_ml_s":      ("PIF-to-PEF (mL/s)",    "PIF_to_PEF_mL_s"),
    "mean_tv_ml":                ("Tv (mL)",              "Tv_mL"),
    "sigh_rate_per_min":         ("Sigh rate (/min)",     "Sigh_rate__min"),
    "mean_sigh_duration_ms":     ("Sigh duration (ms)",   "Sigh_duration_ms"),
    "cov":                       ("CoV_IBI",            "CoV_IBI"),
    "pif_to_pef_cov":            ("CoV (PIF-to-PEF)",     "CoV_PIF_to_PEF"),
    "apnea_rate_per_min":        ("Apnea rate (/min)",    "Apnea_rate__min"),
}


def display_label(parameter: str) -> str:
    """Old-code-compatible y-axis label for ``parameter``."""
    label, _ = _PARAM_DISPLAY.get(parameter, (parameter, parameter))
    return label


def filename_slug(parameter: str) -> str:
    """Old-code-compatible filename token for ``parameter``."""
    return _PARAM_DISPLAY.get(parameter, (parameter, parameter))[1]


# ---------------------------------------------------------------------------
# Apnea-duration reference line + parameter routing (Item B)
# ---------------------------------------------------------------------------
# The two apnea-DURATION parameters. The fixed 400 ms reference line is drawn
# only for these — never for apnea rate or apnea burden.
APNEA_DURATION_PARAMS: Tuple[str, str] = ("apnea_mean_ms", "apnea_mean_ms_imputed")

_APNEA_DURATION_FLOOR_MS = 400.0
_APNEA_DUR_SET = set(APNEA_DURATION_PARAMS)


def add_apnea_duration_reference_line(ax: "plt.Axes") -> None:
    """Draw the fixed 400 ms apnea-duration floor as a dashed grey horizontal
    line and anchor the y-axis lower bound at 0.

    The apnea threshold is ``max(2 x baseline median Ttot, 400 ms)``, so
    400 ms is the hard floor below which no breath can be flagged apneic.
    The reference line shows that anchor on every apnea-duration plot, and
    the 0-anchor lets the reader read durations as multiples of the floor
    rather than relative to whatever the cohort minimum happened to be.

    Call this only for parameters in :data:`APNEA_DURATION_PARAMS` (never on
    apnea rate or apnea burden).
    """
    ax.axhline(
        _APNEA_DURATION_FLOOR_MS, ls="--", color="grey",
        linewidth=1.5, zorder=1,
    )
    _, y1 = ax.get_ylim()
    ax.set_ylim(0.0, y1)


def _expand_apnea_slot(parameters, replacement):
    """Replace the single apnea-duration slot in ``parameters`` with
    ``replacement`` (a tuple of parameter names), preserving position and
    dropping any further duration tokens. If the input has no apnea-duration
    parameter the list is returned unchanged (explicit caller intent wins)."""
    out, inserted = [], False
    for p in parameters:
        if p in _APNEA_DUR_SET:
            if not inserted:
                out.extend(replacement)
                inserted = True
            # drop any additional duration tokens (de-dup)
        else:
            out.append(p)
    return out


def within_style_params(parameters):
    """Within-time-period strip plots emit BOTH apnea-duration variants:
    V1 ``apnea_mean_ms`` (real durations, >=1-apnea traces only) then V2
    ``apnea_mean_ms_imputed`` (imputed)."""
    return _expand_apnea_slot(
        parameters, ("apnea_mean_ms", "apnea_mean_ms_imputed")
    )


def across_style_params(parameters):
    """Across-time-period timeseries use the REAL apnea durations
    (``apnea_mean_ms``); the ``.dropna()`` in the trace builder already drops
    zero-apnea traces."""
    return _expand_apnea_slot(parameters, ("apnea_mean_ms",))


# ---------------------------------------------------------------------------
# Group labels (Item C) — one shared genotype -> age -> condition order
# ---------------------------------------------------------------------------
def group_label(genotype: str, age, condition: Optional[str] = None) -> str:
    """Canonical group label: ``"<Geno> P<age> <COND>"``.

    Word order is always genotype, then age, then condition. The literal
    ``"Scn1a+/-"`` is preserved so :func:`plethysmography.visualization.colors.italicize_scn1a`
    can still render it as ``$\\mathit{Scn1a}^{+/-}$`` downstream — only the
    ORDER of the words changes from the older free-form labels.

    ``age`` accepts an int (``22``), a numeric string (``"22"``) or an
    already-formatted token (``"P22"``). ``condition`` is omitted entirely
    when ``None``/empty (used for exp-2 within strips, which pool FFA +
    Vehicle into each genotype x age cell so no single treatment word
    applies).
    """
    parts = [str(genotype)]
    if age is not None and str(age) != "":
        a = str(age).strip()
        parts.append(a if a.upper().startswith("P") else f"P{int(float(a))}")
    if condition:
        parts.append(str(condition))
    return " ".join(parts)


def two_line_label(label: str) -> str:
    """Split a group label across two lines at the first space:
    ``"WT P22 LR"`` -> ``"WT\\nP22 LR"``, ``"Scn1a+/- P22"`` ->
    ``"Scn1a+/-\\nP22"``.

    Used for the crowded x-tick labels on within-period strip plots where
    four-cell genotype x age (or genotype x condition) labels otherwise
    overlap when rotated. The genotype goes on the top line, the
    age/condition tail on the bottom line. Returns the input unchanged when
    it has fewer than two parts.
    """
    parts = str(label).split(" ", 1)
    if len(parts) == 2:
        return f"{parts[0]}\n{parts[1]}"
    return label


def treatment_word(treatment) -> str:
    """Display form of the treatment condition for group labels: ``vehicle``
    (lower-case) / ``FFA`` (upper-case), matching the locked label spec
    (e.g. ``"WT P19 vehicle"``, ``"Scn1a+/- P22 FFA"``)."""
    return "vehicle" if str(treatment).strip().lower().startswith("veh") else "FFA"


def draw_mouse_age_pair_lines(
    ax,
    points_by_mouse: Dict[str, Dict[int, Tuple[float, float, str]]],
    *,
    line_alpha: float = 0.75,
    line_width: float = 2.0,
    zorder: int = 5,
) -> None:
    """Draw colored lines connecting P19 and P22 points for the same mouse.

    ``points_by_mouse`` maps ``mouse_id`` -> ``{19: (x, y, color), 22: ...}``.
    Lines use the P22 point color (full saturation) when both ages are present.
    """
    for by_age in points_by_mouse.values():
        if 19 not in by_age or 22 not in by_age:
            continue
        x19, y19, _ = by_age[19]
        x22, y22, c22 = by_age[22]
        ax.plot(
            [x19, x22], [y19, y22], color=c22, alpha=line_alpha,
            linewidth=line_width, zorder=zorder, solid_capstyle="round",
        )


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
    so across-period timeseries of the same parameter share an axis.

    For per-period strip plots prefer :func:`period_ylim`, which scales each
    ``Time_period_<x>/`` folder independently.
    """
    values = _collect_strip_ylim_values(
        data, parameter, periods, condition_col=condition_col,
    )
    return _ylim_from_values(values, parameter)


def period_ylim(
    data: pd.DataFrame,
    parameter: str,
    period: str,
    *,
    condition_col: str = "risk_clean",
    extra_mask: Optional[pd.Series] = None,
) -> Optional[Tuple[float, float]]:
    """Compute (y_min, y_max) for strip plots at one named period.

    When ``extra_mask`` is supplied it is AND-ed with the period filter so
    facet families (by_age / by_drug / by_genotype) can share an axis within
    one period while still scaling independently across periods.
    """
    values = _collect_strip_ylim_values(
        data, parameter, [period], condition_col=condition_col,
        extra_mask=extra_mask,
    )
    return _ylim_from_values(values, parameter)


def _cohort_high_values(condition_col: str) -> tuple[str, ...]:
    """Values of ``condition_col`` that define the within-period strip cohort."""
    if condition_col == "treatment_clean":
        return ("FFA", "Vehicle")
    return ("high_risk",)


def _collect_strip_ylim_values(
    data: pd.DataFrame,
    parameter: str,
    periods,
    *,
    condition_col: str,
    extra_mask: Optional[pd.Series] = None,
) -> list[float]:
    """Collect parameter values that will appear on strip plots.

    When ``extra_mask`` is set (FFA facet strips), every row matching the
    mask in each period is included — the caller defines the plotted cohort.

    Otherwise (standard ``plot_within_period`` pair), include exactly the rows
    shown on the across (P22) and within (high-cohort) panels:

      across rows: ``age_clean == 22``
      within rows: ``condition_col`` in :func:`_cohort_high_values`

    Using the union prevents the old failure mode where P19 treatment-cohort
    mice were omitted from the axis when ``condition_col`` defaulted to
    ``risk_clean`` on FFA experiments.
    """
    if parameter not in data.columns:
        return []
    high_values = _cohort_high_values(condition_col)

    all_values: list[float] = []
    for period in periods:
        sub = data[data["period"] == period]
        if extra_mask is not None:
            sub = sub.loc[extra_mask.reindex(sub.index, fill_value=False)]
        if sub.empty:
            continue
        if extra_mask is not None:
            plotted = sub
        else:
            plotted = sub[
                (sub["age_clean"] == 22)
                | sub[condition_col].astype(str).isin(high_values)
            ]
        all_values.extend(plotted[parameter].dropna().tolist())
    return all_values


def _ylim_from_values(
    all_values: list[float],
    parameter: str,
) -> Optional[Tuple[float, float]]:
    if not all_values:
        return None
    _zero_anchored = {"apnea_mean_ms_imputed", "apnea_burden_s_per_min"}
    if parameter in _zero_anchored:
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
