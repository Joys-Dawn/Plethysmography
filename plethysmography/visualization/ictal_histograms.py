"""
Per-trace ictal histograms for digesting the CoV signal.

For each recording's Ictal period, this module produces:

* A side-by-side PNG with two histograms — Ttot (ms) on the left, PIF-to-PEF
  amplitude (mL/s) on the right — saved as
  ``<output_dir>/<basename>_ictal_histograms.png``.
* A per-breath CSV with one row per detected breath in the Ictal period
  (file_basename, ti_ms, te_ms, ttot_ms, pif_centered, pef_centered,
  peak_diff, tv_ml, ti_start_t, te_end_t, is_apnea), saved as
  ``<output_dir>/<basename>_ictal_breaths.csv``. The CSV makes it easy to
  re-render histograms with different bin counts / overlays without
  re-running breath segmentation.

The histograms anchor the reader's intuition for the CoV plots in the
publication bundle: high CoV corresponds to a wide / multi-modal
distribution, low CoV to a concentrated one. Apneic breaths are highlighted
in red on the Ttot histogram (overlaid as a separate hatch) so the eye can
distinguish "the whole distribution shifted" from "a few apneas pulled the
tail".
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ..analysis.breath_segmentation import Breath
from ._common import save_figure


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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    ax_ttot.set_title(f"{file_basename} — Ictal Ttot")
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
    ax_pd.set_title(f"{file_basename} — Ictal PIF-to-PEF")
    ax_pd.set_xlabel("PIF-to-PEF amplitude (mL/s)")
    ax_pd.set_ylabel("Breath count")
    ax_pd.spines["top"].set_visible(False)
    ax_pd.spines["right"].set_visible(False)

    out_path = output_dir / f"{file_basename}_ictal_histograms.png"
    save_figure(fig, out_path)
    return out_path


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
