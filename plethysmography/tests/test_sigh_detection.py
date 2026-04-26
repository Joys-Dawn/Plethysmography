"""
Sigh detection tests. Confirm that:

  - When a baseline cache is provided and not degenerate, the threshold is
    derived from cache.mean + 2 * cache.std (the documented behavior — the
    main FIX vs old code).
  - When no cache is provided (or it is degenerate), fallback uses per-period
    stats.
  - ``classify_sighs`` flags only breaths whose peak_diff exceeds the
    threshold; NaN peak_diff stays False.
"""

from __future__ import annotations

import numpy as np

from plethysmography.analysis.breath_segmentation import Breath
from plethysmography.analysis.sigh_detection import classify_sighs, compute_sigh_threshold
from plethysmography.core.config import SighConfig
from plethysmography.core.data_models import BaselineCache


def _b(peak_diff: float) -> Breath:
    return Breath(
        ti_start_idx=0, ti_end_idx=50, te_start_idx=50, te_end_idx=100,
        ti_start_t=0.0, ti_end_t=0.05, te_start_t=0.05, te_end_t=0.1,
        ti_ms=50.0, te_ms=50.0, ttot_ms=100.0,
        pif_centered=1.0, pef_centered=peak_diff + 1.0,
        peak_diff=peak_diff, tv_ml=0.05,
    )


def test_threshold_uses_baseline_cache_when_present():
    cache = BaselineCache(
        file_basename="x", median_ttot_ms=200.0,
        mean_pif_to_pef_ml_s=2.0, std_pif_to_pef_ml_s=0.5, n_breaths=100,
    )
    breaths = [_b(1.0), _b(1.5), _b(2.0)]
    threshold = compute_sigh_threshold(breaths, SighConfig(), cache)
    # 2.0 + 2 * 0.5 = 3.0
    assert abs(threshold - 3.0) < 1e-9


def test_threshold_falls_back_to_period_stats_when_cache_missing():
    breaths = [_b(1.0), _b(2.0), _b(3.0)]
    threshold = compute_sigh_threshold(breaths, SighConfig(), None)
    # mean=2, std=sqrt(2/3) (population std), threshold = 2 + 2*sqrt(2/3)
    expected = 2.0 + 2.0 * np.sqrt(2.0 / 3.0)
    assert abs(threshold - expected) < 1e-9


def test_classify_sighs_basic():
    breaths = [_b(1.0), _b(5.0), _b(float("nan"))]
    flags = classify_sighs(breaths, threshold=3.0)
    assert flags == [False, True, False]
