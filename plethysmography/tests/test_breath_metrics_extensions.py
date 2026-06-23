"""
Unit tests for the project-extension columns on :class:`BreathMetrics`:

* ``mean_*_no_apnea`` — timing means with apneic breaths excluded.
* ``mean_frequency_bpm_no_apnea`` — derived from ``mean_ttot_ms_no_apnea``
  so it represents the cycle-rate of normal breaths (not raw count over
  duration).
* ``apnea_mean_ms_imputed`` — equal to ``apnea_mean_ms`` when ≥1 apneas
  are detected; otherwise the mean of the longest min(10, n) Ttots.
* ``apnea_burden_s_per_min`` — total real apnea time per minute of period,
  expressed in seconds per minute.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from plethysmography.analysis.breath_metrics import compute_breath_metrics
from plethysmography.analysis.breath_segmentation import Breath
from plethysmography.core.data_models import ApneaEvent


def _breath(ttot_ms: float, ti_start_t: float = 0.0) -> Breath:
    """Build a Breath with the requested Ttot. Other fields are placeholder
    constants — only Ttot/ti/te/peak_diff matter for compute_breath_metrics."""
    ti_ms = 50.0
    te_ms = ttot_ms - ti_ms
    return Breath(
        ti_start_idx=0, ti_end_idx=50, te_start_idx=50, te_end_idx=int(ttot_ms),
        ti_start_t=ti_start_t, ti_end_t=ti_start_t + ti_ms / 1000.0,
        te_start_t=ti_start_t + ti_ms / 1000.0,
        te_end_t=ti_start_t + ttot_ms / 1000.0,
        ti_ms=ti_ms, te_ms=te_ms, ttot_ms=ttot_ms,
        pif_centered=-1.0, pef_centered=1.0, peak_diff=2.0, tv_ml=0.05,
    )


def _apnea(ttot_ms: float, period_name: str = "Ictal") -> ApneaEvent:
    return ApneaEvent(
        file_basename="x", period=period_name,
        apnea_start_s_from_lid_closure=0.0,
        apnea_start_s_from_period_start=0.0,
        apnea_duration_ms=ttot_ms,
        is_post_sigh=False,
    )


# ---------------------------------------------------------------------------
# *_no_apnea timing means
# ---------------------------------------------------------------------------
def test_no_apnea_timing_excludes_apneic_breaths():
    breaths = [_breath(200.0), _breath(220.0), _breath(800.0)]
    is_apnea = [False, False, True]
    apneas = [_apnea(800.0)]

    metrics = compute_breath_metrics(
        file_basename="x", period_name="Ictal", period_duration_s=60.0,
        breaths=breaths, is_sigh=[False] * 3, apneas=apneas, is_apnea=is_apnea,
    )

    # Means over the two non-apneic breaths.
    assert metrics.mean_ttot_ms_no_apnea == pytest.approx(210.0)
    assert metrics.mean_ti_ms_no_apnea == pytest.approx(50.0)
    assert metrics.mean_te_ms_no_apnea == pytest.approx(160.0)
    # And the legacy means still include all breaths.
    assert metrics.mean_ttot_ms == pytest.approx((200 + 220 + 800) / 3)


def test_no_apnea_frequency_is_inverse_of_no_apnea_ttot():
    """``mean_frequency_bpm_no_apnea`` should be 60000 / mean_ttot_ms_no_apnea
    so it isn't depressed by apneic seconds in the denominator."""
    breaths = [_breath(300.0), _breath(300.0), _breath(900.0)]
    is_apnea = [False, False, True]
    apneas = [_apnea(900.0)]

    metrics = compute_breath_metrics(
        file_basename="x", period_name="Ictal", period_duration_s=60.0,
        breaths=breaths, is_sigh=[False] * 3, apneas=apneas, is_apnea=is_apnea,
    )

    assert metrics.mean_ttot_ms_no_apnea == pytest.approx(300.0)
    assert metrics.mean_frequency_bpm_no_apnea == pytest.approx(60000.0 / 300.0)


def test_no_apnea_means_are_nan_when_every_breath_is_apneic():
    breaths = [_breath(800.0), _breath(900.0)]
    is_apnea = [True, True]
    apneas = [_apnea(800.0), _apnea(900.0)]
    metrics = compute_breath_metrics(
        file_basename="x", period_name="Ictal", period_duration_s=60.0,
        breaths=breaths, is_sigh=[False] * 2, apneas=apneas, is_apnea=is_apnea,
    )
    assert math.isnan(metrics.mean_ttot_ms_no_apnea)
    assert math.isnan(metrics.mean_ti_ms_no_apnea)
    assert math.isnan(metrics.mean_te_ms_no_apnea)
    assert math.isnan(metrics.mean_frequency_bpm_no_apnea)


# ---------------------------------------------------------------------------
# apnea_mean_ms_imputed
# ---------------------------------------------------------------------------
def test_imputed_apnea_mean_equals_real_when_apneas_detected():
    """When ≥1 apnea is detected, the imputed value matches ``apnea_mean_ms``
    exactly — no fallback is applied."""
    breaths = [_breath(200.0), _breath(800.0)]
    is_apnea = [False, True]
    apneas = [_apnea(800.0)]
    metrics = compute_breath_metrics(
        file_basename="x", period_name="Ictal", period_duration_s=60.0,
        breaths=breaths, is_sigh=[False, False], apneas=apneas, is_apnea=is_apnea,
    )
    assert metrics.apnea_mean_ms == pytest.approx(800.0)
    assert metrics.apnea_mean_ms_imputed == pytest.approx(800.0)


def test_imputed_apnea_mean_uses_top_10_ttot_when_no_apneas():
    """When 0 apneas are detected, the imputed value is the mean of the 10
    longest Ttots in the period (or all breaths if n < 10)."""
    # 12 breaths with ascending Ttots; top 10 are 130..220 ms.
    ttots = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
             160.0, 170.0, 180.0, 190.0, 210.0, 220.0]
    breaths = [_breath(t) for t in ttots]
    metrics = compute_breath_metrics(
        file_basename="x", period_name="Baseline", period_duration_s=60.0,
        breaths=breaths, is_sigh=[False] * len(breaths), apneas=[],
        is_apnea=[False] * len(breaths),
    )
    expected = float(np.mean(sorted(ttots)[-10:]))  # mean of top-10
    assert math.isnan(metrics.apnea_mean_ms)              # legacy stays NaN
    assert metrics.apnea_mean_ms_imputed == pytest.approx(expected)
    # Per request: top-10 mean should typically land < 400 ms (the apnea floor).
    assert metrics.apnea_mean_ms_imputed < 400.0


def test_imputed_apnea_mean_uses_all_breaths_when_under_ten():
    """If n < 10, the imputation averages all available Ttots."""
    breaths = [_breath(200.0), _breath(220.0), _breath(180.0)]
    metrics = compute_breath_metrics(
        file_basename="x", period_name="Baseline", period_duration_s=60.0,
        breaths=breaths, is_sigh=[False] * 3, apneas=[], is_apnea=[False] * 3,
    )
    assert metrics.apnea_mean_ms_imputed == pytest.approx((200 + 220 + 180) / 3)


def test_imputed_apnea_mean_is_nan_when_no_breaths():
    metrics = compute_breath_metrics(
        file_basename="x", period_name="Baseline", period_duration_s=60.0,
        breaths=[], is_sigh=[], apneas=[], is_apnea=[],
    )
    assert math.isnan(metrics.apnea_mean_ms_imputed)


# ---------------------------------------------------------------------------
# apnea_burden_s_per_min
# ---------------------------------------------------------------------------
def test_apnea_burden_is_total_apnea_time_per_minute():
    """Two apneas of 800 + 1000 ms over a 60 s period -> burden = 1.8 s/min."""
    breaths = [_breath(200.0), _breath(800.0), _breath(1000.0)]
    is_apnea = [False, True, True]
    apneas = [_apnea(800.0), _apnea(1000.0)]
    metrics = compute_breath_metrics(
        file_basename="x", period_name="Ictal", period_duration_s=60.0,
        breaths=breaths, is_sigh=[False] * 3, apneas=apneas, is_apnea=is_apnea,
    )
    assert metrics.apnea_burden_s_per_min == pytest.approx(1.8)


def test_apnea_burden_scales_with_period_duration():
    """The same total apnea time should yield half the burden when the
    period is twice as long."""
    breaths = [_breath(200.0), _breath(800.0)]
    is_apnea = [False, True]
    apneas = [_apnea(800.0)]
    short = compute_breath_metrics(
        file_basename="x", period_name="Ictal", period_duration_s=60.0,
        breaths=breaths, is_sigh=[False, False], apneas=apneas, is_apnea=is_apnea,
    )
    long = compute_breath_metrics(
        file_basename="x", period_name="Ictal", period_duration_s=120.0,
        breaths=breaths, is_sigh=[False, False], apneas=apneas, is_apnea=is_apnea,
    )
    assert short.apnea_burden_s_per_min == pytest.approx(2.0 * long.apnea_burden_s_per_min)


def test_apnea_burden_zero_when_no_apneas():
    breaths = [_breath(200.0), _breath(220.0)]
    metrics = compute_breath_metrics(
        file_basename="x", period_name="Baseline", period_duration_s=60.0,
        breaths=breaths, is_sigh=[False, False], apneas=[], is_apnea=[False, False],
    )
    assert metrics.apnea_burden_s_per_min == 0.0


# ---------------------------------------------------------------------------
# is_apnea length validation
# ---------------------------------------------------------------------------
def test_compute_metrics_rejects_mismatched_is_apnea_length():
    breaths = [_breath(200.0), _breath(220.0)]
    with pytest.raises(ValueError, match="is_apnea"):
        compute_breath_metrics(
            file_basename="x", period_name="Baseline", period_duration_s=60.0,
            breaths=breaths, is_sigh=[False, False], apneas=[],
            is_apnea=[False],     # wrong length
        )
