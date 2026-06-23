"""
Lid open/close event detection.

3-pass spike algorithm + boundary-walk refinement, with per-file overrides for
the recordings whose default behavior was wrong. Mirrors
``old_code/pleth_preprocessing.py:159-261`` (the 3 passes + per-file branches)
and ``old_code/pleth_preprocessing.py:471-495`` (the boundary walk).

Public entry point: :func:`detect_lid_events`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import logging

import numpy as np

from ..core.config import LidDetectionConfig, PlethConfig
from ..core.data_models import LidEvents
from ..core.metadata import get_lid_override


logger = logging.getLogger(__name__)


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
        pass1b = _pass1b_baseline_shift_spikes(signal, fs, lid_cfg)
        merged = sorted(set(map(int, pass1)) | set(pass1b))
        if not merged:
            return LidEvents()
        pass2 = _pass2_baseline_shift(merged, signal, fs, lid_cfg)
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
        new_idx = _walk_boundary(signal, spike_idx, is_close, fs, lid_cfg)
        adjusted_times.append(float(time_s[new_idx]))

    if not _event_times_valid(adjusted_times):
        logger.warning(
            "%s: boundary walk produced invalid open/close ordering; "
            "using raw spike times instead.",
            file_basename,
        )
        adjusted_times = list(raw_times)

    return LidEvents(raw_spike_times_s=raw_times, adjusted_spike_times_s=adjusted_times)


# ----------------------------------------------------------------------------
# Pass 1: threshold detection with min spacing
# ----------------------------------------------------------------------------
def _pass1_threshold_spikes(
    signal: np.ndarray, fs: float, cfg: LidDetectionConfig,
) -> np.ndarray:
    """Find samples where ``|signal| > spike_sigma * std(signal)``, then drop
    any spike that follows the immediately-prior spike by less than
    ``min_spike_distance_ms`` (matches old code's consecutive-spike filter).

    Additionally — and this is the only behavioral deviation from the old
    algorithm — also keep the **last** above-threshold sample of any maximal
    consecutive above-threshold run lasting at least
    ``long_run_min_duration_s`` seconds. This catches the trailing edge of
    long, tall lid-open plateaus where the signal stays continuously above
    threshold (no mid-plateau dip), in which case the first-per-gap-subrun
    rule alone would emit just one candidate for the entire plateau and the
    close-edge transient would be invisible. Purely additive: short runs and
    plateaus that already produced multiple candidates via mid-run dips are
    unaffected (any duplicate index is removed by the final ``sorted(set(...))``).
    """
    threshold = cfg.spike_sigma * float(np.std(signal))
    above_mask = np.abs(signal) > threshold
    spike_indices = np.where(above_mask)[0]
    if spike_indices.size == 0:
        return spike_indices

    # Existing behavior: keep first per gap-separated sub-run.
    min_gap = max(1, int(round(fs * cfg.min_spike_distance_ms / 1000.0)))
    kept = [int(spike_indices[0])]
    for i in range(1, len(spike_indices)):
        if (spike_indices[i] - spike_indices[i - 1]) > min_gap:
            kept.append(int(spike_indices[i]))

    # Additive: trailing edge of long maximal-consecutive runs, gated by two
    # filters: (a) the gap to the next run, and (b) the distance from the end
    # of the recording. (a) guards against noisy plateaus that already produce
    # many Pass 1 candidates from mid-plateau dips (each sub-run's rising edge
    # already covers the previous sub-run's falling-edge vicinity).
    # (b) guards against recording-end artifacts where the chamber is being
    # moved as acquisition stops — those runs have no following run to filter
    # them via (a) and would otherwise smuggle in a spurious candidate near
    # EOF that confuses Pass 3's pairing.
    int_mask = above_mask.astype(np.int8)
    diff = np.diff(int_mask, prepend=0, append=0)
    rising = np.where(diff == 1)[0]            # first above-threshold sample of each run
    falling = np.where(diff == -1)[0] - 1      # last above-threshold sample of each run
    min_run_samples = int(round(fs * cfg.long_run_min_duration_s))
    n_samples = len(signal)
    for i, (r, f) in enumerate(zip(rising, falling)):
        if (f - r) < min_run_samples or f == r:
            continue
        # (a) Skip if another above-threshold run starts within the duration window.
        if i + 1 < len(rising) and (rising[i + 1] - f) < min_run_samples:
            continue
        # (b) Skip if the falling edge sits within the duration window of EOF.
        if (n_samples - 1 - f) < min_run_samples:
            continue
        kept.append(int(f))

    return np.array(sorted(set(kept)), dtype=int)


def _shift_magnitude(signal: np.ndarray, idx: int, window_samples: int) -> float:
    n = len(signal)
    pre_start = max(0, idx - window_samples)
    post_end = min(n, idx + 1 + window_samples)
    if pre_start >= idx or idx + 1 >= post_end:
        return 0.0
    avg_pre = float(np.mean(signal[pre_start:idx]))
    avg_post = float(np.mean(signal[idx + 1:post_end]))
    return abs(avg_pre - avg_post)


def _pass1b_baseline_shift_spikes(
    signal: np.ndarray, fs: float, cfg: LidDetectionConfig,
) -> List[int]:
    """Discover lid step edges that Pass 1 misses.

    Rectangular chamber shifts (e.g. plateau -> baseline without a tall spike)
    still show a large 1-min pre/post mean change. Scan coarsely, then keep the
    strongest shift in each ``min_spike_distance_ms`` cluster.
    """
    window_samples = int(60 * fs)
    tolerance = cfg.baseline_shift_sigma_threshold * float(np.std(signal))
    n = len(signal)
    if n <= 2 * window_samples:
        return []

    step = max(1, int(round(fs * 0.5)))
    found: List[int] = []
    for idx in range(window_samples, n - window_samples, step):
        if _shift_magnitude(signal, idx, window_samples) > tolerance:
            found.append(idx)
    if not found:
        return []

    min_gap = max(1, int(round(fs * cfg.min_spike_distance_ms / 1000.0)))
    clusters: List[List[int]] = [[found[0]]]
    for idx in found[1:]:
        if idx - clusters[-1][-1] <= min_gap:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return [
        max(cluster, key=lambda i: _shift_magnitude(signal, i, window_samples))
        for cluster in clusters
    ]


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
    first valid close within ``cutoff`` seconds of that open. A close is also
    invalid if another candidate sits within ``cutoff`` after it (the old
    trailing-spike guard).
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
        open_t = float(time_s[open_idx])

        j = i + 1
        found_close = False
        while j < n:
            close_idx = candidates[j]
            close_t = float(time_s[close_idx])

            is_valid = (close_t - open_t) <= cutoff
            if is_valid and j + 1 < n:
                gap = float(time_s[candidates[j + 1]] - close_t)
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
    fs: float,
    cfg: LidDetectionConfig,
) -> int:
    """Walk ``spike_idx`` outward (forward for closes, backward for opens) while
    the signal stays outside ±``boundary_walk_threshold`` of a reference mean,
    then offset by ``boundary_walk_offset_samples``.

    For closes the reference mean is taken *after* the lid artifact (skipping
    ~2 s past the spike). Using the artifact itself as the reference — the old
    behaviour — made ``signal - local_mean`` stay large for the entire baseline
    and the walk could run to EOF (030826 5445 p22).
    """
    n = len(signal)
    threshold = cfg.boundary_walk_threshold
    win = cfg.boundary_walk_local_mean_samples
    max_walk = int(round(fs * 60.0))
    start_idx = spike_idx

    if is_close:
        skip = min(n - spike_idx - 1, int(round(fs * 2.0)))
        ref_start = spike_idx + skip
        window = signal[ref_start:min(n, ref_start + win)]
        if window.size == 0:
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
            spike_idx = max(0, min(n - 1, spike_idx))
            break
        if abs(spike_idx - start_idx) >= max_walk:
            spike_idx = start_idx + step * max_walk
            spike_idx = max(0, min(n - 1, spike_idx))
            break
        val = float(signal[spike_idx]) - local_mean

    spike_idx += step * cfg.boundary_walk_offset_samples
    spike_idx = max(0, min(n - 1, spike_idx))
    return spike_idx


def _event_times_valid(times: Sequence[float]) -> bool:
    """Each open must precede its close; each close must precede the next open."""
    if len(times) < 2:
        return True
    for i in range(0, len(times) - 1, 2):
        if times[i] >= times[i + 1]:
            return False
        if i + 2 < len(times) and times[i + 1] >= times[i + 2]:
            return False
    return True


def _find_time_index(time_s: np.ndarray, t: float) -> int:
    """Index of the first sample whose timestamp equals ``t``. Falls back to
    nearest-sample search if exact match fails (e.g. if ``t`` came from a
    hardcoded override that doesn't sit on a sample boundary)."""
    matches = np.where(time_s == t)[0]
    if matches.size > 0:
        return int(matches[0])
    return int(np.argmin(np.abs(time_s - t)))
