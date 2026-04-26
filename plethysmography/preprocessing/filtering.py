"""Per-period 0.5 Hz Butterworth high-pass, zero-phase filtered."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..core.config import FilterConfig
from ..core.data_models import Period


def filter_period(period: Period, config: FilterConfig) -> Period:
    """Return a new Period with a high-pass-filtered signal (zero-phase via
    ``sosfiltfilt``). If the segment is too short for the filter's padding
    requirement, return the period unchanged (matches old code behavior at
    pleth_preprocessing.py:519-530)."""
    if period.signal.size == 0:
        return period

    nyq = 0.5 * period.fs
    normalized = config.hpf_cutoff_hz / nyq
    if normalized <= 0 or normalized >= 1.0 or period.fs <= 0:
        return period  # invalid filter cutoff

    # sosfiltfilt's default padlen is 3*(filter_order+1) on the SOS coefficients
    # of a Butterworth, but to match old code exactly we use the same heuristic.
    padlen_required = 3 * (config.hpf_order + 1)
    if period.signal.size <= padlen_required:
        return period  # too short to filter

    sos = butter(
        config.hpf_order, normalized,
        btype="highpass", analog=False, output="sos",
    )
    filtered = sosfiltfilt(sos, period.signal).astype(float, copy=False)
    return replace(period, signal=filtered)
