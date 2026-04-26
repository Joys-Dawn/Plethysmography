"""
Single-entry preprocessing orchestrator.

``preprocess_recording`` is the only function the rest of the codebase needs
to call to take a Recording from raw EDF to a list of filtered, artifact-cleaned
Periods. It also writes per-period CSVs in the same column shape as old code
(time / signal / period_start_time / lid_closure_time) so downstream analysis
can load them with pandas.read_csv.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.config import PlethConfig
from ..core.data_models import LidEvents, Period, Recording
from ..core.metadata import get_preprocess_override, should_skip_preprocess
from ..data_loading.edf_reader import read_edf_signal
from ..data_loading.lid_detection import detect_lid_events
from .artifacts import remove_artifacts_from_period
from .filtering import filter_period
from .periods import slice_periods


logger = logging.getLogger(__name__)


def preprocess_recording(
    recording: Recording,
    config: PlethConfig,
    save_dir: Optional[str | Path] = None,
    *,
    traces_dir: Optional[str | Path] = None,
) -> Tuple[List[Period], LidEvents]:
    """Run the full preprocessing pipeline on a single Recording.

    Steps:
      1. If the file is in EXCLUSIONS["preprocess"], return ``([], LidEvents())``.
      2. Read EDF channel 0; set recording.fs.
      3. Detect lid events (3-pass + boundary walk + per-file overrides).
      4. If a 'remove_segment_between_first_open_and_close' preprocess override
         applies, drop the segment and the first two spikes.
      5. Slice into periods.
      6. Filter each period (0.5 Hz HPF zero-phase).
      7. Remove +/-8sigma outliers per period via linear interpolation.
      8. If ``save_dir`` is given, write one CSV per period.
      9. If ``traces_dir`` is given, save ``<basename>_spikes.png`` (raw signal
         + lid markers) and ``<basename>_periods.png`` (filtered periods
         overlay) for visual QC.

    Returns ``(periods, lid_events)``. The caller can use the returned LidEvents
    for trace plotting. The Recording dataclass is mutated in place to set ``fs``.
    """
    if should_skip_preprocess(recording.file_basename):
        return [], LidEvents()

    signal, time_s, fs = read_edf_signal(recording.edf_path)
    recording.fs = fs

    lid_events = detect_lid_events(
        signal=signal, time_s=time_s, fs=fs,
        file_basename=recording.file_basename, config=config,
    )

    if traces_dir is not None:
        from ..visualization.trace_plots import plot_lid_spikes
        plot_lid_spikes(
            signal=signal, time_s=time_s, lid_events=lid_events,
            file_basename=recording.file_basename, output_dir=Path(traces_dir),
        )

    override = get_preprocess_override(recording.file_basename)
    if override is not None and override["type"] == "remove_segment_between_first_open_and_close":
        signal, time_s, lid_events = _remove_segment_between_first_pair(signal, time_s, lid_events)

    periods = slice_periods(
        signal=signal, time_s=time_s, fs=fs,
        lid_events=lid_events,
        seizure_offset_s=recording.seizure_offset_s,
        config=config.period,
    )

    if not periods:
        n_events = len(lid_events.adjusted_spike_times_s)
        logger.warning(
            "preprocess: %s produced 0 periods (lid detection found %d event(s); "
            "need 4 for full slicing). Signal mean=%.2f std=%.4f. "
            "If these look anomalous compared to siblings, the EDF may be corrupted.",
            recording.file_basename, n_events, float(np.mean(signal)), float(np.std(signal)),
        )

    periods = [filter_period(p, config.filter) for p in periods]
    periods = [remove_artifacts_from_period(p, config.filter) for p in periods]

    if save_dir is not None:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        for period in periods:
            save_period_csv(period, recording.file_basename, save_dir_path)

    if traces_dir is not None and periods:
        from ..visualization.trace_plots import plot_periods_overlay
        plot_periods_overlay(periods, recording.file_basename, Path(traces_dir))

    return periods, lid_events


def save_period_csv(period: Period, file_basename: str, save_dir: Path) -> Path:
    """Write one period to CSV with columns matching old code:
    ``time, signal, period_start_time, lid_closure_time``. Returns the path written.
    """
    n = len(period.time_s)
    df = pd.DataFrame({
        "time": period.time_s,
        "signal": period.signal,
        "period_start_time": np.full(n, period.period_start_time),
        "lid_closure_time": np.full(n, period.lid_closure_time),
    })
    name = f"{file_basename}_{period.name.replace(' ', '_')}.csv"
    out = save_dir / name
    df.to_csv(out, index=False)
    return out


def _remove_segment_between_first_pair(
    signal: np.ndarray,
    time_s: np.ndarray,
    lid_events: LidEvents,
) -> Tuple[np.ndarray, np.ndarray, LidEvents]:
    """Implements PER_FILE_PREPROCESS_OVERRIDES['250304 4056 p22']:
    drop all samples in [adjusted[0], adjusted[1]] and remove those two events
    from the LidEvents list.

    Mirrors old_code/pleth_preprocessing.py:498-508.
    """
    if len(lid_events.adjusted_spike_times_s) < 2:
        return signal, time_s, lid_events

    t0 = lid_events.adjusted_spike_times_s[0]
    t1 = lid_events.adjusted_spike_times_s[1]
    keep_mask = ~((time_s >= t0) & (time_s <= t1))
    new_signal = signal[keep_mask]
    new_time = time_s[keep_mask]
    new_lid = LidEvents(
        raw_spike_times_s=lid_events.raw_spike_times_s[2:],
        adjusted_spike_times_s=lid_events.adjusted_spike_times_s[2:],
    )
    return new_signal, new_time, new_lid
