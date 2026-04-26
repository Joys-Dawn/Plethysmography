"""
Per-mouse baseline statistics cache.

Computed once per recording, during pass 1 of the analysis driver, by running
the full breath segmentation on the Baseline period. The cache holds the three
quantities downstream periods need:

  - ``median_ttot_ms``        — used as the apnea threshold reference (existing
    behavior; old code already did this correctly).
  - ``mean_pif_to_pef_ml_s``  — used as the sigh threshold reference (NEW vs
    old code, which used per-period stats).
  - ``std_pif_to_pef_ml_s``   — same.

If the Baseline period had 0 breaths or zero std (degenerate), downstream
detectors fall back to per-period stats and log a warning.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ..core.config import BreathConfig
from ..core.data_models import BaselineCache, Period
from .breath_segmentation import Breath, segment_breaths


logger = logging.getLogger(__name__)


def cache_from_breaths(file_basename: str, breaths: List[Breath]) -> BaselineCache:
    """Build a :class:`BaselineCache` from an already-segmented list of Breath
    objects (the typical pass-1 path: we segment the Baseline period to compute
    its metrics, then reuse those breaths to derive the cache)."""
    if not breaths:
        return BaselineCache(
            file_basename=file_basename,
            median_ttot_ms=0.0,
            mean_pif_to_pef_ml_s=0.0,
            std_pif_to_pef_ml_s=0.0,
            n_breaths=0,
        )

    durations = np.array([b.ttot_ms for b in breaths], dtype=float)
    peak_diffs = np.array(
        [b.peak_diff for b in breaths if np.isfinite(b.peak_diff)],
        dtype=float,
    )

    median_ttot = float(np.median(durations)) if durations.size > 0 else 0.0
    mean_peak = float(np.mean(peak_diffs)) if peak_diffs.size > 0 else 0.0
    std_peak = float(np.std(peak_diffs)) if peak_diffs.size > 0 else 0.0

    cache = BaselineCache(
        file_basename=file_basename,
        median_ttot_ms=median_ttot,
        mean_pif_to_pef_ml_s=mean_peak,
        std_pif_to_pef_ml_s=std_peak,
        n_breaths=len(breaths),
    )
    if cache.is_degenerate:
        logger.warning(
            "Baseline cache for %s is degenerate (n_breaths=%d, std=%.4g). "
            "Downstream periods will fall back to per-period stats for sigh / apnea thresholds.",
            file_basename, cache.n_breaths, std_peak,
        )
    return cache


def build_baseline_cache(
    file_basename: str,
    baseline_period: Optional[Period],
    breath_config: BreathConfig,
) -> BaselineCache:
    """Convenience wrapper: segment the Baseline period and build the cache.
    Used when you don't already have the segmented Baseline breaths in hand."""
    if baseline_period is None:
        return BaselineCache(
            file_basename=file_basename,
            median_ttot_ms=0.0,
            mean_pif_to_pef_ml_s=0.0,
            std_pif_to_pef_ml_s=0.0,
            n_breaths=0,
        )
    breaths = segment_breaths(baseline_period, breath_config)
    return cache_from_breaths(file_basename, breaths)
