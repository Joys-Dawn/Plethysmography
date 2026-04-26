"""
Synthetic-signal tests for breath segmentation. Use a small sinusoid as a
known-frequency input and confirm the segmenter recovers the right number of
breaths and the per-breath durations are within tolerance.
"""

from __future__ import annotations

import numpy as np

from plethysmography.analysis.breath_segmentation import segment_breaths
from plethysmography.core.config import BreathConfig
from plethysmography.core.data_models import Period


def _make_sinusoid_period(freq_hz: float, duration_s: float, fs: float = 1000.0) -> Period:
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    sig = np.sin(2.0 * np.pi * freq_hz * t)
    return Period(
        name="Synthetic",
        start_s=0.0,
        end_s=float(t[-1]),
        signal=sig,
        time_s=t,
        fs=fs,
        period_start_time=0.0,
        lid_closure_time=float("nan"),
    )


def test_segment_pure_sine_returns_expected_count():
    period = _make_sinusoid_period(freq_hz=4.0, duration_s=2.0)
    breaths = segment_breaths(period, BreathConfig())
    # 4 Hz × 2 s = 8 cycles, segmenter should find ~7-8 (boundary effects can
    # drop one).
    assert 6 <= len(breaths) <= 9


def test_breath_durations_match_input_freq():
    period = _make_sinusoid_period(freq_hz=4.0, duration_s=2.0)
    breaths = segment_breaths(period, BreathConfig())
    if not breaths:
        return
    expected_ms = 1000.0 / 4.0
    durs = [b.ttot_ms for b in breaths]
    median = float(np.median(durs))
    assert abs(median - expected_ms) < expected_ms * 0.2, (
        f"median duration {median:.1f} should be near {expected_ms:.1f}"
    )
