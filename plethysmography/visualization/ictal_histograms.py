"""
Per-trace and pooled ictal histograms.

Per-recording (one PNG per file basename):
  Side-by-side 2-panel figure — Ttot (ms) on the left, PIF-to-PEF amplitude
  (mL/s) on the right. Apneic breaths are overlaid in red on the Ttot panel
  so the eye separates "the whole distribution shifted" from "a few apneas
  pulled the tail".

Population (faceted combo plots per metric, per experiment layout):
  Layout mirrors the ``Time_period_*`` strip/timeseries facet structure so
  no more than four groups overlay on one figure:

    * ``across/``, ``within/``           — experiment 1
    * ``by_age/``, ``by_drug/``, ``by_genotype/`` — experiment 2
    * ``across/``                        — experiment 3 (P22-only)
    * (flat)                             — experiments 1b / 4 (≤2 groups)

  Each facet folder holds:

    * ``Population_ictal_Ttot.png``       — Ttot, capped at 1000 ms.
    * ``Population_ictal_PIF_to_PEF.png`` — PIF-to-PEF amplitude.

  Y-axes show ``% breaths`` (each group's histogram sums to 100%).

A per-breath CSV is also written per recording (one row per detected ictal
breath), making it easy to re-render with different bin counts without
re-segmenting.
"""

from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..analysis.breath_segmentation import Breath
from ._common import save_figure
from .colors import italicize_scn1a


# Larger figure than the one-panel publication plots; two stacked histograms
# render comfortably at 12 x 4. dpi=200 in save_figure keeps file sizes small.
_FIG_SIZE = (12.0, 4.5)
_NORMAL_COLOR = "#4C78A8"      # default seaborn blue
_APNEA_COLOR = "#FF0000"       # match HR Scn1a red used elsewhere
_BIN_COUNT = 40                # round number that reads cleanly for n in [50, 1500]


def write_ictal_breaths_csv(
    file_basename: str,
    breaths: Sequence[Breath],
    is_apnea: Sequence[bool],
    output_dir: Path,
) -> Optional[Path]:
    """Persist one row per breath to ``<output_dir>/<basename>_ictal_breaths.csv``.

    Returns the path written, or ``None`` if there are no breaths to write.
    The ``is_apnea`` flag is parallel to ``breaths`` and uses the same
    threshold as :mod:`plethysmography.analysis.apnea_detection`.
    """
    if not breaths:
        return None
    if len(is_apnea) != len(breaths):
        raise ValueError(
            f"is_apnea length {len(is_apnea)} != breaths length {len(breaths)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "file_basename": file_basename,
            "ti_ms": b.ti_ms,
            "te_ms": b.te_ms,
            "ttot_ms": b.ttot_ms,
            "pif_centered": b.pif_centered,
            "pef_centered": b.pef_centered,
            "peak_diff": b.peak_diff,
            "tv_ml": b.tv_ml,
            "ti_start_t": b.ti_start_t,
            "te_end_t": b.te_end_t,
            "is_apnea": bool(ap),
        }
        for b, ap in zip(breaths, is_apnea)
    ]
    out_path = output_dir / f"{file_basename}_ictal_breaths.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def _render_two_panel(
    title_label: str,
    breaths: Sequence[Breath],
    is_apnea: Sequence[bool],
    out_path: Path,
    *,
    bin_count: int = _BIN_COUNT,
) -> Optional[Path]:
    """Render the shared 2-panel ictal histogram (Ttot + PIF-to-PEF) to
    ``out_path``.

    This is the single rendering core used by both the per-recording
    :func:`plot_ictal_histograms` (``title_label`` = file basename) and the
    pooled :func:`plot_population_ictal_histograms` (``title_label`` = the
    canonical group label). Returns the written path, or ``None`` when there
    is nothing finite to plot (callers treat ``None`` as "skip this one").
    The parent directory is created only when a figure is actually saved, so
    a no-data call leaves the filesystem untouched.
    """
    if not breaths:
        return None
    if len(is_apnea) != len(breaths):
        raise ValueError(
            f"is_apnea length {len(is_apnea)} != breaths length {len(breaths)}"
        )

    ttot = np.array([b.ttot_ms for b in breaths], dtype=float)
    peak_diff = np.array([b.peak_diff for b in breaths], dtype=float)
    apnea_mask = np.array(is_apnea, dtype=bool)

    ttot_finite = np.isfinite(ttot)
    pd_finite = np.isfinite(peak_diff)
    if not ttot_finite.any() and not pd_finite.any():
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt  # local import keeps top-level import clean

    fig, (ax_ttot, ax_pd) = plt.subplots(1, 2, figsize=_FIG_SIZE)

    # --- Panel 1: Ttot --------------------------------------------------
    if ttot_finite.any():
        all_ttot = ttot[ttot_finite]
        bins = np.histogram_bin_edges(all_ttot, bins=bin_count)
        non_apnea = all_ttot[~apnea_mask[ttot_finite]]
        apnea = all_ttot[apnea_mask[ttot_finite]]
        ax_ttot.hist(
            non_apnea, bins=bins, color=_NORMAL_COLOR, alpha=0.85,
            edgecolor="black", linewidth=0.4, label=f"non-apneic (n={non_apnea.size})",
        )
        if apnea.size > 0:
            ax_ttot.hist(
                apnea, bins=bins, color=_APNEA_COLOR, alpha=0.85,
                edgecolor="black", linewidth=0.4,
                label=f"apneic (n={apnea.size})",
            )
        ax_ttot.legend(loc="upper right", fontsize=10, frameon=False)
    ax_ttot.set_title(f"{title_label} — Ictal Ttot")
    ax_ttot.set_xlabel("Ttot (ms)")
    ax_ttot.set_ylabel("Breath count")
    ax_ttot.spines["top"].set_visible(False)
    ax_ttot.spines["right"].set_visible(False)

    # --- Panel 2: PIF-to-PEF --------------------------------------------
    if pd_finite.any():
        all_pd = peak_diff[pd_finite]
        ax_pd.hist(
            all_pd, bins=bin_count, color=_NORMAL_COLOR, alpha=0.85,
            edgecolor="black", linewidth=0.4,
        )
    ax_pd.set_title(f"{title_label} — Ictal PIF-to-PEF")
    ax_pd.set_xlabel("PIF-to-PEF amplitude (mL/s)")
    ax_pd.set_ylabel("Breath count")
    ax_pd.spines["top"].set_visible(False)
    ax_pd.spines["right"].set_visible(False)

    save_figure(fig, out_path)
    return out_path


def plot_ictal_histograms(
    file_basename: str,
    breaths: Sequence[Breath],
    is_apnea: Sequence[bool],
    output_dir: Path,
    *,
    bin_count: int = _BIN_COUNT,
) -> Optional[Path]:
    """Render the per-trace 2-panel histogram (Ttot + PIF-to-PEF).

    Returns the saved PNG path, or ``None`` if there are no finite values to
    plot. The Ttot panel overlays apneic breaths in red so the reader can
    see how much of the distribution's right tail is driven by apneas vs.
    "merely long" non-apneic breaths.
    """
    output_dir = Path(output_dir)
    return _render_two_panel(
        file_basename, breaths, is_apnea,
        output_dir / f"{file_basename}_ictal_histograms.png",
        bin_count=bin_count,
    )


_TTOT_MAX_MS_DEFAULT = 1000.0
# Matplotlib's tab10 fallback for groups not in the caller's palette. The
# combo plot only needs distinct colors; downstream readability comes from
# the legend.
_DEFAULT_GROUP_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
)
_PERCENT_YLABEL = "% breaths"


def _label_parts(label: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse ``"WT P22 LR"`` -> (genotype, age token, condition token)."""
    parts = str(label).split()
    if len(parts) < 2:
        return (parts[0], None, None) if parts else ("", None, None)
    age = parts[1] if parts[1] in ("P19", "P22") else None
    cond = parts[2] if len(parts) >= 3 else None
    return parts[0], age, cond


def population_ictal_facets(
    layout: str,
) -> Sequence[Tuple[str, Callable[[str], bool]]]:
    """Return ``(subdir, label_predicate)`` pairs for one experiment layout."""

    def _age(age_token: str):
        return lambda label: _label_parts(label)[1] == age_token

    def _cond(*tokens: str):
        def _pred(label: str) -> bool:
            cond = _label_parts(label)[2]
            return cond is not None and cond in tokens
        return _pred

    def _geno(prefix: str):
        return lambda label: _label_parts(label)[0] == prefix

    if layout == "exp1":
        return (
            ("across", _age("P22")),
            ("within", _cond("HR")),
        )
    if layout == "exp2":
        return (
            ("by_age/P19", _age("P19")),
            ("by_age/P22", _age("P22")),
            ("by_drug/FFA", _cond("FFA")),
            ("by_drug/Vehicle", _cond("vehicle")),
            ("by_genotype/WT", _geno("WT")),
            ("by_genotype/Scn1a", _geno("Scn1a+/-")),
        )
    if layout == "exp3":
        return (("across", _age("P22")),)
    if layout == "single":
        return (("", lambda _label: True),)
    raise ValueError(f"unknown population_ictal layout: {layout!r}")


def _percent_weights(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    return np.full(n, 100.0 / n, dtype=float)


def plot_population_ictal_histograms(
    groups: Mapping[str, Tuple[Sequence[Breath], Sequence[bool]]],
    output_dir: Path,
    *,
    bin_count: int = _BIN_COUNT,
    group_colors: Optional[Mapping[str, str]] = None,
    ttot_max_ms: float = _TTOT_MAX_MS_DEFAULT,
) -> List[Path]:
    """Two combo population-ictal plots for one group subset, written into
    ``output_dir``:

      * ``Population_ictal_Ttot.png``       — Ttot (ms), capped at
        ``ttot_max_ms``; y-axis is ``% breaths``.
      * ``Population_ictal_PIF_to_PEF.png`` — PIF-to-PEF (mL/s); y-axis is
        ``% breaths``.

    Each group is drawn as a colored step outline. On the Ttot plot the
    apneic-breath subset is additionally overlaid as a hatched fill in the
    same color (percent of that group's total ictal breaths per bin).

    ``groups`` maps a canonical ``group_label`` string to pooled
    ``(breaths, is_apnea)`` for that group. Returns the saved PNG paths.
    """
    output_dir = Path(output_dir)
    if not groups:
        return []

    series: Dict[str, Dict[str, np.ndarray]] = {}
    for label, (breaths, is_apnea) in groups.items():
        if not breaths:
            continue
        if len(is_apnea) != len(breaths):
            raise ValueError(
                f"{label}: is_apnea length {len(is_apnea)} != breaths length "
                f"{len(breaths)}"
            )
        ttot = np.array([b.ttot_ms for b in breaths], dtype=float)
        peak_diff = np.array([b.peak_diff for b in breaths], dtype=float)
        apnea = np.array(is_apnea, dtype=bool)
        ttot_finite = np.isfinite(ttot)
        pd_finite = np.isfinite(peak_diff)
        series[label] = {
            "ttot_all": ttot[ttot_finite],
            "ttot_apnea": ttot[ttot_finite & apnea],
            "peak_diff": peak_diff[pd_finite],
        }
    if not series:
        return []

    fallback = cycle(_DEFAULT_GROUP_COLORS)
    color_for: Dict[str, str] = {}
    for label in series:
        c = (group_colors or {}).get(label)
        color_for[label] = c if c is not None else next(fallback)

    import matplotlib.pyplot as plt
    saved: List[Path] = []

    # ---- Ttot combo: capped to [0, ttot_max_ms] ------------------------
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ttot_bins = np.linspace(0.0, float(ttot_max_ms), int(bin_count) + 1)
    drew_anything = False
    for label, data in series.items():
        if data["ttot_all"].size == 0:
            continue
        color = color_for[label]
        clipped_all = np.clip(data["ttot_all"], 0.0, ttot_max_ms)
        n_group = clipped_all.size
        ax.hist(
            clipped_all, bins=ttot_bins, weights=_percent_weights(n_group),
            color=color, alpha=0.45,
            histtype="stepfilled", edgecolor=color, linewidth=1.5,
            label=italicize_scn1a(label),
        )
        if data["ttot_apnea"].size > 0:
            clipped_apnea = np.clip(data["ttot_apnea"], 0.0, ttot_max_ms)
            ax.hist(
                clipped_apnea, bins=ttot_bins,
                weights=np.full(clipped_apnea.size, 100.0 / n_group),
                facecolor="none",
                histtype="stepfilled",
                edgecolor=color, hatch="///", linewidth=1.5,
                label=f"{italicize_scn1a(label)} apneic (n={data['ttot_apnea'].size})",
            )
        drew_anything = True
    if drew_anything:
        ax.set_xlim(0.0, ttot_max_ms)
        ax.set_xlabel("Ttot (ms)")
        ax.set_ylabel(_PERCENT_YLABEL)
        ax.legend(loc="upper right", fontsize=10, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        out_path = output_dir / "Population_ictal_Ttot.png"
        save_figure(fig, out_path)
        saved.append(out_path)
    else:
        plt.close(fig)

    # ---- PIF-to-PEF combo ---------------------------------------------
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    all_pd = np.concatenate(
        [data["peak_diff"] for data in series.values() if data["peak_diff"].size > 0]
    ) if any(d["peak_diff"].size > 0 for d in series.values()) else np.array([])
    if all_pd.size > 0:
        pd_bins = np.histogram_bin_edges(all_pd, bins=int(bin_count))
        for label, data in series.items():
            if data["peak_diff"].size == 0:
                continue
            color = color_for[label]
            n_group = data["peak_diff"].size
            ax.hist(
                data["peak_diff"], bins=pd_bins, weights=_percent_weights(n_group),
                color=color, alpha=0.45,
                histtype="stepfilled", edgecolor=color, linewidth=1.5,
                label=italicize_scn1a(label),
            )
        ax.set_xlabel("PIF-to-PEF amplitude (mL/s)")
        ax.set_ylabel(_PERCENT_YLABEL)
        ax.legend(loc="upper right", fontsize=10, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        out_path = output_dir / "Population_ictal_PIF_to_PEF.png"
        save_figure(fig, out_path)
        saved.append(out_path)
    else:
        plt.close(fig)
    return saved


def plot_population_ictal_histograms_faceted(
    groups: Mapping[str, Tuple[Sequence[Breath], Sequence[bool]]],
    output_dir: Path,
    *,
    layout: str,
    bin_count: int = _BIN_COUNT,
    group_colors: Optional[Mapping[str, str]] = None,
    ttot_max_ms: float = _TTOT_MAX_MS_DEFAULT,
) -> List[Path]:
    """Render faceted population-ictal histograms for one experiment layout.

    See :func:`population_ictal_facets` for subfolder naming. Returns every
    PNG path written across all facets.
    """
    output_dir = Path(output_dir)
    if not groups:
        return []
    saved: List[Path] = []
    for subdir, predicate in population_ictal_facets(layout):
        subset = {k: v for k, v in groups.items() if predicate(k)}
        if not subset:
            continue
        facet_dir = output_dir if subdir == "" else output_dir / subdir
        saved.extend(
            plot_population_ictal_histograms(
                subset, facet_dir,
                bin_count=bin_count,
                group_colors=group_colors,
                ttot_max_ms=ttot_max_ms,
            )
        )
    return saved


def emit_ictal_histograms(
    file_basename: str,
    breaths: Sequence[Breath],
    is_apnea: Sequence[bool],
    output_dir: Path,
) -> List[Path]:
    """Convenience wrapper: write both the per-breath CSV and the histogram
    PNG for one recording's ictal period. Returns the list of files
    actually written (may be empty if there are no breaths)."""
    out: List[Path] = []
    csv_path = write_ictal_breaths_csv(file_basename, breaths, is_apnea, output_dir)
    if csv_path is not None:
        out.append(csv_path)
    png_path = plot_ictal_histograms(file_basename, breaths, is_apnea, output_dir)
    if png_path is not None:
        out.append(png_path)
    return out
