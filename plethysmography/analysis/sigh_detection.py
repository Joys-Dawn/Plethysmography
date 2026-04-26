"""
Sigh detection.

A sigh is a breath whose centered ``peak_diff`` (PEF − PIF) exceeds the threshold

    threshold = mean_peak_diff + sigma_multiplier * std_peak_diff

Per the docs (``pleth files meta data.xlsx`` sheet 'approach to analysis'),
``mean_peak_diff`` and ``std_peak_diff`` are computed over the BASELINE breaths
of THIS MOUSE — not over the current period.

That is the documented and intended behavior; old code computed them over the
current period instead. New code reads them from a :class:`BaselineCache`.
For the Baseline period itself (no cache yet), the threshold falls back to the
period's own peak_diff stats (which is numerically identical to old code's
behavior for Baseline).

Functions:
  - :func:`compute_sigh_threshold` returns the threshold given a list of
    breaths and (optional) baseline cache.
  - :func:`classify_sighs` returns a list of bool flags (one per breath) marking
    which breaths are sighs.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.config import SighConfig
from ..core.data_models import BaselineCache
from .breath_segmentation import Breath


def compute_sigh_threshold(
    breaths: List[Breath],
    config: SighConfig,
    baseline_cache: Optional[BaselineCache],
) -> float:
    """Return the sigh threshold for this period.

    If ``baseline_cache`` is provided and not degenerate, use the cached
    baseline mean / std of peak_diffs (this is the docs-correct behavior for
    every period EXCEPT Baseline, where the cache is built and so is None).
    Otherwise fall back to per-period stats and return ``+inf`` if there are
    no valid breaths to compute from.
    """
    if baseline_cache is not None and not baseline_cache.is_degenerate:
        return (
            baseline_cache.mean_pif_to_pef_ml_s
            + config.sigma_multiplier * baseline_cache.std_pif_to_pef_ml_s
        )

    peak_diffs = np.array(
        [b.peak_diff for b in breaths if not np.isnan(b.peak_diff)],
        dtype=float,
    )
    if peak_diffs.size == 0:
        return float("inf")
    mean = float(np.mean(peak_diffs))
    std = float(np.std(peak_diffs))
    return mean + config.sigma_multiplier * std


def classify_sighs(breaths: List[Breath], threshold: float) -> List[bool]:
    """Return ``[breath.peak_diff >= threshold for breath in breaths]``, with
    NaN ``peak_diff`` mapping to False (matches old code's filter at
    analyze_data.py:229-230)."""
    out: List[bool] = []
    for b in breaths:
        out.append(not np.isnan(b.peak_diff) and b.peak_diff >= threshold)
    return out
