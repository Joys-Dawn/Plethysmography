"""
Per-breath segmentation: signal -> List[Breath].

Pipeline (mirrors old_code/analyze_data.py:22-200):
  1. Running-mean centering (rolling window, ``local_window_ms``).
  2. Optional low-pass filter applied to the centered signal (per-file overrides only).
  3. Zero-crossings on the centered signal define raw segments.
  4. Spurious-inspiration merge: any inspiration whose magnitude is below
     ``max(spurious_inspiration_amp_floor, spurious_inspiration_sigma_frac * std)``
     is merged into adjacent expiration(s), then adjacent same-sign segments are collapsed.
  5. Short-segment merge: any segment shorter than ``short_segment_min_ms`` is merged
     into a neighbor (preferring same-sign neighbors).
  6. Build ``Breath`` objects from each (-1, +1) (inspiration, expiration) segment pair.

PIF / PEF are measured on the centered signal (per docs, decision 10);
TV is the ``abs(trapezoid(raw_signal))`` over the inspiration only (per old code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from ..core.config import BreathConfig
from ..core.data_models import Period


# Internal segment representation: (sign, start_idx, end_idx) where sign is
# -1 (inspiration) or +1 (expiration).
_Segment = Tuple[int, int, int]


@dataclass
class Breath:
    """One breath cycle: an inspiration segment followed by an expiration segment.

    Index fields are integer offsets into the period's signal/time arrays;
    time fields are the corresponding seconds-from-start-of-recording values.
    All amplitude / volume fields use the units of the underlying EDF channel.
    """
    ti_start_idx: int
    ti_end_idx: int
    te_start_idx: int
    te_end_idx: int
    ti_start_t: float
    ti_end_t: float
    te_start_t: float
    te_end_t: float
    ti_ms: float
    te_ms: float
    ttot_ms: float
    pif_centered: float
    pef_centered: float
    peak_diff: float
    tv_ml: float


def segment_breaths(
    period: Period,
    breath_config: BreathConfig,
    apply_post_center_lp_hz: Optional[float] = None,
) -> List[Breath]:
    """Segment a period into a list of Breath objects."""
    signal = period.signal.astype(float, copy=False)
    time_s = period.time_s
    fs = period.fs
    if signal.size < 2 or time_s.size < 2:
        return []

    centered = _running_mean_center(signal, fs, breath_config.local_window_ms)
    if apply_post_center_lp_hz is not None:
        centered = _butter_lowpass(centered, apply_post_center_lp_hz, fs, order=4)

    segments = _build_initial_segments(centered)
    segments = _merge_spurious_inspirations(segments, centered, breath_config)
    segments = _collapse_same_sign(segments)
    segments = _merge_short_segments(segments, fs, breath_config)
    return _segments_to_breaths(segments, signal, centered, time_s)


# ---------------------------------------------------------------------------
# Step 1: running-mean centering
# ---------------------------------------------------------------------------
def _running_mean_center(signal: np.ndarray, fs: float, local_window_ms: float) -> np.ndarray:
    """Subtract a centered rolling mean (matches old code's pandas rolling
    with ``center=True, min_periods=1``)."""
    win = max(1, int(local_window_ms * fs / 1000.0))
    local_mean = (
        pd.Series(signal)
        .rolling(window=win, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    return signal - local_mean


def _butter_lowpass(signal: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    sos = butter(order, cutoff_hz / nyq, btype="low", analog=False, output="sos")
    return sosfiltfilt(sos, signal)


# ---------------------------------------------------------------------------
# Step 2: zero-crossing segmentation
# ---------------------------------------------------------------------------
def _build_initial_segments(centered: np.ndarray) -> List[_Segment]:
    """Build (sign, start, end) tuples from zero-crossings of the centered signal."""
    crossings = np.where(np.diff(np.sign(centered)))[0]
    n = len(centered)
    phase_sign = 1 if centered[0] > 0 else -1   # +1 expiration, -1 inspiration

    segments: List[_Segment] = []
    if crossings.size == 0:
        segments.append((phase_sign, 0, n - 1))
        return segments

    prev_idx = 0
    for cx in crossings:
        segments.append((phase_sign, prev_idx, int(cx)))
        phase_sign *= -1
        prev_idx = int(cx)
    if prev_idx < n - 1:
        segments.append((phase_sign, prev_idx, n - 1))
    return segments


# ---------------------------------------------------------------------------
# Step 3: spurious-inspiration merge
# ---------------------------------------------------------------------------
def _merge_spurious_inspirations(
    segments: List[_Segment],
    centered: np.ndarray,
    config: BreathConfig,
) -> List[_Segment]:
    """Inspirations whose magnitude is below the spurious threshold are absorbed
    into the surrounding expiration(s). Mirrors old_code/analyze_data.py:69-104."""
    noise_sigma = float(np.std(centered))
    amp_thr = max(
        config.spurious_inspiration_amp_floor,
        config.spurious_inspiration_sigma_frac * noise_sigma,
    )

    merged: List[_Segment] = []
    i = 0
    while i < len(segments):
        sign, s_idx, e_idx = segments[i]
        if sign == -1:
            seg_slice = centered[s_idx:e_idx + 1]
            peak_abs = float(np.abs(np.min(seg_slice))) if seg_slice.size > 0 else 0.0
            if peak_abs < amp_thr:
                if merged and merged[-1][0] == 1:
                    prev_start = merged[-1][1]
                else:
                    prev_start = s_idx

                if (i + 1) < len(segments) and segments[i + 1][0] == 1:
                    e_idx = segments[i + 1][2]
                    i += 1   # skip the next expiration; we've absorbed it

                if merged and merged[-1][0] == 1:
                    merged[-1] = (1, prev_start, e_idx)
                else:
                    merged.append((1, prev_start, e_idx))
                i += 1
                continue

        merged.append((sign, s_idx, e_idx))
        i += 1
    return merged


def _collapse_same_sign(segments: List[_Segment]) -> List[_Segment]:
    """Defensively collapse adjacent same-sign segments into one."""
    cleaned: List[_Segment] = []
    for seg in segments:
        if cleaned and cleaned[-1][0] == seg[0]:
            cleaned[-1] = (seg[0], cleaned[-1][1], seg[2])
        else:
            cleaned.append(seg)
    return cleaned


# ---------------------------------------------------------------------------
# Step 4: short-segment merge
# ---------------------------------------------------------------------------
def _merge_short_segments(
    segments: List[_Segment],
    fs: float,
    config: BreathConfig,
) -> List[_Segment]:
    """Merge segments shorter than ``short_segment_min_ms`` into a neighbor.
    Mirrors old_code/analyze_data.py:120-156. The list is mutated in-place;
    we copy to avoid side effects on the caller."""
    segs = list(segments)
    dur_thr_samples = int(config.short_segment_min_ms * fs / 1000.0)

    idx = 0
    while idx < len(segs):
        sign, s_idx, e_idx = segs[idx]
        if (e_idx - s_idx) < dur_thr_samples:
            if 0 < idx < (len(segs) - 1):
                prev_sign, prev_s, _prev_e = segs[idx - 1]
                next_sign, _next_s, next_e = segs[idx + 1]
                if prev_sign == next_sign:
                    segs[idx - 1] = (prev_sign, prev_s, next_e)
                    segs.pop(idx + 1)
                    segs.pop(idx)
                    idx -= 1
                    continue
            if idx > 0:
                prev_sign, prev_s, _prev_e = segs[idx - 1]
                segs[idx - 1] = (prev_sign, prev_s, e_idx)
                segs.pop(idx)
                idx -= 1
                continue
            elif idx < len(segs) - 1:
                next_sign, _next_s, next_e = segs[idx + 1]
                segs[idx + 1] = (next_sign, s_idx, next_e)
                segs.pop(idx)
                continue
        idx += 1
    return segs


# ---------------------------------------------------------------------------
# Step 5: segments -> Breath objects
# ---------------------------------------------------------------------------
def _segments_to_breaths(
    segments: List[_Segment],
    raw_signal: np.ndarray,
    centered: np.ndarray,
    time_s: np.ndarray,
) -> List[Breath]:
    """Walk through segments matching (-1, +1) pairs into Breath dataclasses."""
    out: List[Breath] = []
    i = 0
    while i < len(segments) - 1:
        insp_sign, ti_start, ti_end = segments[i]
        exp_sign, te_start, te_end = segments[i + 1]
        if insp_sign != -1 or exp_sign != 1:
            i += 1
            continue

        ti_start_t = float(time_s[ti_start])
        ti_end_t = float(time_s[ti_end])
        te_start_t = float(time_s[te_start])
        te_end_t = float(time_s[te_end])
        ti_ms = (ti_end_t - ti_start_t) * 1000.0
        te_ms = (te_end_t - te_start_t) * 1000.0

        insp_centered_slice = centered[ti_start:ti_end + 1]
        exp_centered_slice = centered[te_start:te_end + 1]
        pif_c = float(np.min(insp_centered_slice)) if insp_centered_slice.size > 0 else float("nan")
        pef_c = float(np.max(exp_centered_slice)) if exp_centered_slice.size > 0 else float("nan")
        peak_diff = pef_c - pif_c

        # TV: integrate the RAW (uncentered) signal over the inspiration.
        tv_insp = float(np.trapezoid(
            raw_signal[ti_start:ti_end + 1],
            x=time_s[ti_start:ti_end + 1],
        )) if ti_end > ti_start else 0.0
        tv_ml = abs(tv_insp)

        out.append(Breath(
            ti_start_idx=ti_start, ti_end_idx=ti_end,
            te_start_idx=te_start, te_end_idx=te_end,
            ti_start_t=ti_start_t, ti_end_t=ti_end_t,
            te_start_t=te_start_t, te_end_t=te_end_t,
            ti_ms=ti_ms, te_ms=te_ms, ttot_ms=ti_ms + te_ms,
            pif_centered=pif_c, pef_centered=pef_c, peak_diff=peak_diff,
            tv_ml=tv_ml,
        ))
        i += 2

    return out
