"""
Lid open/close event detection.

3-pass spike algorithm + boundary-walk refinement, with per-file overrides for
the recordings whose default behavior was wrong. Mirrors
``old_code/pleth_preprocessing.py:159-261`` (the 3 passes + per-file branches)
and ``old_code/pleth_preprocessing.py:471-495`` (the boundary walk).

Public entry point: :func:`detect_lid_events`.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.config import LidDetectionConfig, PlethConfig
from ..core.data_models import LidEvents
from ..core.metadata import get_lid_override


def detect_lid_events(
    signal: np.ndarray,
    time_s: np.ndarray,
    fs: float,
    file_basename: str,
    config: PlethConfig,
) -> LidEvents:
    """Detect lid open/close events for a recording.

    Pipeline:
      1. If a 'hardcoded_spike_times' override exists, use those raw times directly.
         Otherwise run the 3-pass detector (threshold -> baseline-shift -> pair).
      2. Apply 'keep_only_index' override after pass 3 if present.
      3. For each surviving spike, walk it outward until the signal returns to
         within ±boundary_walk_threshold of the local mean, then offset further
         by boundary_walk_offset_samples. Even-indexed spikes (opens) walk
         backward; odd-indexed (closes) walk forward.

    The returned :class:`LidEvents` carries both the raw and the boundary-walked
    times. Period slicing uses the adjusted times.
    """
    override = get_lid_override(file_basename)
    lid_cfg = config.lid

    # ---- raw times ------------------------------------------------------
    if override is not None and override["type"] == "hardcoded_spike_times":
        raw_times = [float(t) for t in override["spike_times_s"]]
    else:
        first_pair_window = None
        if override is not None and override["type"] == "first_pair_close_window_s":
            first_pair_window = float(override["value"])

        pass1 = _pass1_threshold_spikes(signal, fs, lid_cfg)
        if pass1.size == 0:
            return LidEvents()
        pass2 = _pass2_baseline_shift(pass1, signal, fs, lid_cfg)
        if not pass2:
            return LidEvents()
        pass3 = _pass3_pair_open_close(pass2, time_s, lid_cfg, first_pair_window)
        raw_times = [float(time_s[idx]) for idx in pass3]

    if not raw_times:
        return LidEvents()

    # ---- post-pass3 override -------------------------------------------
    if override is not None and override["type"] == "keep_only_index":
        keep_idx = int(override["value"])
        if 0 <= keep_idx < len(raw_times):
            raw_times = [raw_times[keep_idx]]
        else:
            raw_times = []
    if not raw_times:
        return LidEvents(raw_spike_times_s=[], adjusted_spike_times_s=[])

    # ---- boundary walk --------------------------------------------------
    adjusted_times: List[float] = []
    for i, t in enumerate(raw_times):
        spike_idx = _find_time_index(time_s, t)
        is_close = (i % 2 == 1)
        new_idx = _walk_boundary(signal, spike_idx, is_close, lid_cfg)
        adjusted_times.append(float(time_s[new_idx]))

    return LidEvents(raw_spike_times_s=raw_times, adjusted_spike_times_s=adjusted_times)


# ----------------------------------------------------------------------------
# Pass 1: threshold detection with min spacing
# ----------------------------------------------------------------------------
def _pass1_threshold_spikes(
    signal: np.ndarray, fs: float, cfg: LidDetectionConfig,
) -> np.ndarray:
    """Find samples where ``|signal| > spike_sigma * std(signal)``, then drop
    any spike that follows the immediately-prior spike by less than
    ``min_spike_distance_ms`` (matches old code's consecutive-spike filter)."""
    threshold = cfg.spike_sigma * float(np.std(signal))
    spike_indices = np.where(np.abs(signal) > threshold)[0]
    if spike_indices.size == 0:
        return spike_indices

    min_gap = max(1, int(round(fs * cfg.min_spike_distance_ms / 1000.0)))
    kept = [int(spike_indices[0])]
    for i in range(1, len(spike_indices)):
        if (spike_indices[i] - spike_indices[i - 1]) > min_gap:
            kept.append(int(spike_indices[i]))
    return np.array(kept, dtype=int)


# ----------------------------------------------------------------------------
# Pass 2: baseline-shift filter
# ----------------------------------------------------------------------------
def _pass2_baseline_shift(
    candidates: np.ndarray,
    signal: np.ndarray,
    fs: float,
    cfg: LidDetectionConfig,
) -> List[int]:
    """Keep spike iff ``|mean(1-min pre) − mean(1-min post)| > tolerance``,
    where tolerance = ``baseline_shift_sigma_threshold * std(signal)``. If the
    spike is too close to either edge to compute a meaningful pre/post window,
    the spike is kept (matches old code line 191-192)."""
    window_samples = int(60 * fs)
    tolerance = cfg.baseline_shift_sigma_threshold * float(np.std(signal))
    n = len(signal)
    kept: List[int] = []
    for idx in candidates:
        idx = int(idx)
        pre_start = max(0, idx - window_samples)
        pre_end = idx
        post_start = idx + 1
        post_end = min(n, idx + 1 + window_samples)
        if pre_start < pre_end and post_start < post_end:
            avg_pre = float(np.mean(signal[pre_start:pre_end]))
            avg_post = float(np.mean(signal[post_start:post_end]))
            if abs(avg_pre - avg_post) > tolerance:
                kept.append(idx)
        else:
            kept.append(idx)
    return kept


# ----------------------------------------------------------------------------
# Pass 3: pair opens with closes
# ----------------------------------------------------------------------------
def _pass3_pair_open_close(
    candidates: List[int],
    time_s: np.ndarray,
    cfg: LidDetectionConfig,
    first_pair_close_window_override: Optional[float],
) -> List[int]:
    """For each candidate as a potential open, append it then look forward for the
    first valid close. A close is valid iff no further spike sits within
    ``cutoff_window`` after it. The cutoff is :attr:`pair_close_window_s` by
    default; if ``first_pair_close_window_override`` is provided, the FIRST pair
    uses that override and subsequent pairs revert to the default.

    Mirrors old_code/pleth_preprocessing.py:200-252.
    """
    result: List[int] = []
    first_pair_processed = False
    n = len(candidates)

    i = 0
    while i < n:
        open_idx = candidates[i]

        cutoff = cfg.pair_close_window_s
        if first_pair_close_window_override is not None and not first_pair_processed:
            cutoff = first_pair_close_window_override

        result.append(open_idx)

        j = i + 1
        found_close = False
        while j < n:
            close_idx = candidates[j]

            # Look one step ahead. If the very next candidate is within `cutoff`
            # of close_idx, this close is invalid; otherwise (or if there is no
            # further candidate) it's valid. Matches the inner-loop early-break
            # at old_code/pleth_preprocessing.py:227-235.
            is_valid = True
            if j + 1 < n:
                gap = float(time_s[candidates[j + 1]] - time_s[close_idx])
                if gap < cutoff:
                    is_valid = False

            if is_valid:
                result.append(close_idx)
                if first_pair_close_window_override is not None:
                    first_pair_processed = True
                i = j + 1
                found_close = True
                break
            j += 1

        if not found_close:
            i += 1

    return result


# ----------------------------------------------------------------------------
# Boundary walk
# ----------------------------------------------------------------------------
def _walk_boundary(
    signal: np.ndarray,
    spike_idx: int,
    is_close: bool,
    cfg: LidDetectionConfig,
) -> int:
    """Walk ``spike_idx`` outward (forward for closes, backward for opens) while
    the signal stays outside ±``boundary_walk_threshold`` of the local mean,
    then offset by ``boundary_walk_offset_samples`` further in the same
    direction.

    Mirrors old_code/pleth_preprocessing.py:471-495 exactly, including the
    quirk that the FIRST iteration compares ``signal[spike_idx]`` raw (with no
    local-mean subtraction) — the spike is so far above threshold that the
    loop always enters anyway.
    """
    n = len(signal)
    threshold = cfg.boundary_walk_threshold
    win = cfg.boundary_walk_local_mean_samples

    if is_close:
        window = signal[spike_idx:min(n, spike_idx + win)]
        step = 1
    else:
        window = signal[max(0, spike_idx - win):spike_idx]
        step = -1
    local_mean = float(np.mean(window)) if window.size > 0 else float(signal[spike_idx])

    val = float(signal[spike_idx])
    while val > threshold or val < -threshold:
        spike_idx += step
        if spike_idx < 0 or spike_idx >= n:
            # Walk ran off the end; clamp and break (old code would IndexError —
            # this defensive bounds check is the only deviation).
            spike_idx = max(0, min(n - 1, spike_idx))
            break
        val = float(signal[spike_idx]) - local_mean

    spike_idx += step * cfg.boundary_walk_offset_samples
    spike_idx = max(0, min(n - 1, spike_idx))
    return spike_idx


def _find_time_index(time_s: np.ndarray, t: float) -> int:
    """Index of the first sample whose timestamp equals ``t``. Falls back to
    nearest-sample search if exact match fails (e.g. if ``t`` came from a
    hardcoded override that doesn't sit on a sample boundary)."""
    matches = np.where(time_s == t)[0]
    if matches.size > 0:
        return int(matches[0])
    return int(np.argmin(np.abs(time_s - t)))
