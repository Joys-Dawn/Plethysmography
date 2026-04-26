"""
Apnea detection.

An apnea is a breath whose total cycle duration ``ttot_ms`` satisfies BOTH

    ttot_ms >= ttot_multiplier * baseline_median_ttot_ms
    ttot_ms >= ttot_minimum_ms

(combined as ``ttot_ms >= max(2 * baseline_median, 400)`` in the implementation).

The ``baseline_median_ttot_ms`` comes from the per-mouse :class:`BaselineCache`;
this is unchanged from old code, which already used the cache correctly. For
the Baseline period itself (no cache yet), we fall back to the period's own
median Ttot.

Each detected apnea is classified as either **post-sigh** (the immediately
preceding sigh's expiration ended within ``post_sigh_window_s`` seconds before
the apnea's inspiration start) or **spontaneous** (no qualifying sigh).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..core.config import ApneaConfig
from ..core.data_models import ApneaEvent, BaselineCache
from .breath_segmentation import Breath


def compute_apnea_threshold(
    breaths: List[Breath],
    config: ApneaConfig,
    baseline_cache: Optional[BaselineCache],
) -> Optional[float]:
    """Return the apnea threshold (ms) for this period, or None if no valid
    median Ttot is available (e.g. period has 0 breaths AND no cache).
    """
    if baseline_cache is not None and not baseline_cache.is_degenerate:
        median_ttot = baseline_cache.median_ttot_ms
    else:
        durations = [b.ttot_ms for b in breaths]
        if not durations:
            return None
        median_ttot = float(np.median(durations))

    if median_ttot <= 0:
        return None
    return max(config.ttot_multiplier * median_ttot, config.ttot_minimum_ms)


def detect_apneas(
    breaths: List[Breath],
    is_sigh: List[bool],
    threshold_ms: Optional[float],
    config: ApneaConfig,
    file_basename: str,
    period_name: str,
    period_start_time: float,
    lid_closure_time: float,
) -> List[ApneaEvent]:
    """Return the list of detected apneas for this period.

    Each apnea's ``apnea_start_s_from_lid_closure`` is measured from the ABSOLUTE
    lid closure timestamp of the period's anchor (matches old code), and
    ``apnea_start_s_from_period_start`` is measured from the period's first
    timestamp. Apnea start time is the end of the breath's inspiration
    (``ti_end_t``) — same as old code.
    """
    if threshold_ms is None:
        return []

    # Sigh expiration end times for post-sigh classification.
    sigh_end_times: List[float] = [
        b.te_end_t for b, s in zip(breaths, is_sigh) if s
    ]

    out: List[ApneaEvent] = []
    for b in breaths:
        if not (np.isfinite(b.ttot_ms) and b.ttot_ms >= threshold_ms):
            continue

        # Find most recent sigh ending at or before this breath's inspiration start.
        is_post_sigh = False
        for st in reversed(sigh_end_times):
            if st <= b.ti_start_t:
                is_post_sigh = (b.ti_start_t - st) <= config.post_sigh_window_s
                break

        # Apnea spans ti_end -> te_end (the "silent" portion); start = ti_end.
        apnea_start_t = b.ti_end_t
        out.append(ApneaEvent(
            file_basename=file_basename,
            period=period_name,
            apnea_start_s_from_lid_closure=apnea_start_t - lid_closure_time,
            apnea_start_s_from_period_start=apnea_start_t - period_start_time,
            apnea_duration_ms=b.ttot_ms,
            is_post_sigh=is_post_sigh,
        ))
    return out


def split_apnea_counts(apneas: List[ApneaEvent]) -> Tuple[int, int, int]:
    """Return (total_count, spontaneous_count, post_sigh_count)."""
    spont = sum(1 for a in apneas if not a.is_post_sigh)
    post = sum(1 for a in apneas if a.is_post_sigh)
    return (len(apneas), spont, post)
