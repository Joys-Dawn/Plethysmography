"""
Slice a recording into named periods.

Period boundaries (in temporal order):
  - **Acclimation** (0 to 5 min) — DROPPED upstream; never returned.
  - **Habituation** (5 min to min(15 min, first lid open)) — analyzed.
  - **Baseline** (close 1 to open 2).
  - **Ictal** (close 2 to close 2 + seizure_offset_s; or close 2 to end of file
    if offset is missing).
  - **Immediate Postictal** (seizure_offset_end to + 10 min). Only if offset known.
  - **Recovery** (postictal_end to end of file). Only if offset known.

Match old_code/pleth_preprocessing.py:63-157 (define_periods) for the
Baseline / Ictal / Postictal / Recovery boundaries; Habituation differs
(new code applies the 5-15 min envelope upstream rather than dropping the
first 5 min downstream during breath analysis — per docs and user decision 6).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.config import PeriodConfig
from ..core.data_models import (
    BASELINE,
    HABITUATION,
    ICTAL,
    IMMEDIATE_POSTICTAL,
    LidEvents,
    Period,
    RECOVERY,
)


def slice_periods(
    signal: np.ndarray,
    time_s: np.ndarray,
    fs: float,
    lid_events: LidEvents,
    seizure_offset_s: Optional[float],
    config: PeriodConfig,
) -> List[Period]:
    """Return the list of non-empty Period segments for a recording."""
    if signal.size == 0 or time_s.size == 0:
        return []

    opens = lid_events.open_times_s
    closes = lid_events.close_times_s
    out: List[Period] = []

    # --- Habituation (5-15 min, capped by first lid open) -------------------
    hab_start = config.habituation_start_s
    hab_end = config.habituation_end_s
    if opens:
        hab_end = min(hab_end, opens[0])
    p = _make_period(
        HABITUATION, hab_start, hab_end, signal, time_s, fs,
        start_inclusive=True, end_inclusive=False,
        lid_closure_time=float("nan"),
    )
    if p is not None:
        out.append(p)

    # --- Baseline (close 1 to open 2) ---------------------------------------
    if len(closes) >= 1 and len(opens) >= 2:
        s = closes[0]
        e = opens[1]
        p = _make_period(
            BASELINE, s, e, signal, time_s, fs,
            start_inclusive=False, end_inclusive=False,
            lid_closure_time=float(s),
        )
        if p is not None:
            out.append(p)

    # --- Ictal / Postictal / Recovery (anchored on close 2) -----------------
    if len(closes) >= 2:
        close2 = float(closes[1])
        offset_known = seizure_offset_s is not None and not (
            isinstance(seizure_offset_s, float) and np.isnan(seizure_offset_s)
        )
        if offset_known:
            seizure_end_t = close2 + float(seizure_offset_s)
        else:
            seizure_end_t = None

        # Ictal: close2 -> seizure_end_t (or end of file if offset missing)
        if seizure_end_t is not None:
            p = _make_period(
                ICTAL, close2, seizure_end_t, signal, time_s, fs,
                start_inclusive=False, end_inclusive=False,
                lid_closure_time=close2,
            )
        else:
            ictal_e = float(time_s[-1])
            p = _make_period(
                ICTAL, close2, ictal_e, signal, time_s, fs,
                start_inclusive=False, end_inclusive=True,
                lid_closure_time=close2,
            )
        if p is not None:
            out.append(p)

        # Postictal + Recovery only when offset is known
        if seizure_end_t is not None:
            postictal_e = seizure_end_t + config.postictal_duration_s
            p = _make_period(
                IMMEDIATE_POSTICTAL, seizure_end_t, postictal_e, signal, time_s, fs,
                start_inclusive=True, end_inclusive=False,
                lid_closure_time=close2,
            )
            if p is not None:
                out.append(p)

            recovery_e = float(time_s[-1])
            p = _make_period(
                RECOVERY, postictal_e, recovery_e, signal, time_s, fs,
                start_inclusive=True, end_inclusive=True,
                lid_closure_time=close2,
            )
            if p is not None:
                out.append(p)

    return out


def _make_period(
    name: str,
    start_s: float,
    end_s: float,
    signal: np.ndarray,
    time_s: np.ndarray,
    fs: float,
    start_inclusive: bool,
    end_inclusive: bool,
    lid_closure_time: float,
) -> Optional[Period]:
    """Build a Period for [start_s, end_s] (inclusivity per flags). Returns None
    if the resulting segment is empty or boundaries are degenerate."""
    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        return None

    lo_op = (time_s >= start_s) if start_inclusive else (time_s > start_s)
    hi_op = (time_s <= end_s) if end_inclusive else (time_s < end_s)
    mask = lo_op & hi_op
    if not mask.any():
        return None

    seg_t = time_s[mask]
    return Period(
        name=name,
        start_s=float(start_s),
        end_s=float(end_s),
        signal=signal[mask].astype(float, copy=True),
        time_s=seg_t.astype(float, copy=True),
        fs=fs,
        period_start_time=float(seg_t[0]),
        lid_closure_time=lid_closure_time,
    )
