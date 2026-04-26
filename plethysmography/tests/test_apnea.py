"""
Apnea detection unit tests. Confirm the threshold formula
``max(2 × baseline_median, 400)`` is applied as an AND condition, and that
post-sigh classification uses the configured 8 s window.
"""

from __future__ import annotations

import numpy as np

from plethysmography.analysis.apnea_detection import (
    compute_apnea_threshold,
    detect_apneas,
    split_apnea_counts,
)
from plethysmography.analysis.breath_segmentation import Breath
from plethysmography.core.config import ApneaConfig
from plethysmography.core.data_models import BaselineCache


def _make_breath(
    ttot_ms: float, *, ti_start_t: float = 0.0, ti_end_t: float = 0.05,
) -> Breath:
    return Breath(
        ti_start_idx=0, ti_end_idx=50, te_start_idx=50, te_end_idx=int(ttot_ms),
        ti_start_t=ti_start_t, ti_end_t=ti_end_t,
        te_start_t=ti_end_t, te_end_t=ti_end_t + (ttot_ms - 50) / 1000.0,
        ti_ms=50.0, te_ms=ttot_ms - 50.0, ttot_ms=ttot_ms,
        pif_centered=1.0, pef_centered=1.0, peak_diff=2.0, tv_ml=0.05,
    )


def test_threshold_uses_max_of_2x_median_and_400ms():
    cache_low = BaselineCache(
        file_basename="x", median_ttot_ms=150.0,
        mean_pif_to_pef_ml_s=2.0, std_pif_to_pef_ml_s=0.5, n_breaths=100,
    )
    cache_high = BaselineCache(
        file_basename="x", median_ttot_ms=300.0,
        mean_pif_to_pef_ml_s=2.0, std_pif_to_pef_ml_s=0.5, n_breaths=100,
    )
    cfg = ApneaConfig()
    # 2 * 150 = 300, but min is 400 -> threshold = 400
    assert compute_apnea_threshold([], cfg, cache_low) == 400.0
    # 2 * 300 = 600 > 400 -> threshold = 600
    assert compute_apnea_threshold([], cfg, cache_high) == 600.0


def test_detect_apneas_emits_only_long_breaths():
    cfg = ApneaConfig()
    breaths = [
        _make_breath(200.0, ti_start_t=0.0, ti_end_t=0.05),
        _make_breath(500.0, ti_start_t=1.0, ti_end_t=1.05),
        _make_breath(1000.0, ti_start_t=2.0, ti_end_t=2.05),
    ]
    apneas = detect_apneas(
        breaths=breaths,
        is_sigh=[False] * 3,
        threshold_ms=400.0,
        config=cfg,
        file_basename="x",
        period_name="Baseline",
        period_start_time=0.0,
        lid_closure_time=0.0,
    )
    assert len(apneas) == 2  # the 500 and 1000 ms breaths
    total, spont, post = split_apnea_counts(apneas)
    assert total == 2 and spont == 2 and post == 0


def test_post_sigh_classification_uses_8s_window():
    cfg = ApneaConfig()
    sigh_breath = _make_breath(150.0, ti_start_t=0.0, ti_end_t=0.05)
    long_breath = _make_breath(500.0, ti_start_t=5.0, ti_end_t=5.05)
    far_breath = _make_breath(500.0, ti_start_t=20.0, ti_end_t=20.05)
    breaths = [sigh_breath, long_breath, far_breath]
    is_sigh = [True, False, False]
    apneas = detect_apneas(
        breaths=breaths,
        is_sigh=is_sigh,
        threshold_ms=400.0,
        config=cfg,
        file_basename="x",
        period_name="Baseline",
        period_start_time=0.0,
        lid_closure_time=0.0,
    )
    by_t = {round(a.apnea_start_s_from_period_start, 2): a for a in apneas}
    # The 5s breath is within 8s of the sigh -> post-sigh
    assert by_t[5.05].is_post_sigh is True
    # The 20s breath is not -> spontaneous
    assert by_t[20.05].is_post_sigh is False
