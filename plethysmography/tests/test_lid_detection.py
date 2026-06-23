"""Regression tests for lid open/close detection on known-problem recordings."""

from __future__ import annotations

from pathlib import Path

import pytest

from plethysmography.core.config import PlethConfig
from plethysmography.core.data_models import BASELINE
from plethysmography.data_loading.data_log import resolve_raw_edf
from plethysmography.data_loading.edf_reader import read_edf_signal
from plethysmography.data_loading.lid_detection import detect_lid_events
from plethysmography.preprocessing.periods import slice_periods

_EXP2_RAW = (
    Path("Data")
    / "experiment 2 - chronic FFA vs vehicle"
    / "experiment 2 - raw data"
)

def _edf_path(basename: str) -> Path:
    resolved = resolve_raw_edf(basename, Path("Data"))
    if resolved is not None and resolved.exists():
        return resolved
    exp3 = (
        Path("Data")
        / "experiment 3 - acute FFA vs vehicle"
        / "experiment 3 - raw data"
        / f"{basename}.EDF"
    )
    if exp3.exists():
        return exp3
    return _EXP2_RAW / f"{basename}.EDF"

_REGRESSION_CASES = (
    "030826 5445 p22",
    "033026 5514 p22",
    "041326 5549 p22",
    "042826 5609 p22",
)


@pytest.mark.parametrize("basename", _REGRESSION_CASES)
def test_problem_recordings_yield_baseline_period(basename: str) -> None:
    edf_path = _edf_path(basename)
    if not edf_path.exists():
        pytest.skip(f"EDF not on disk: {edf_path}")

    config = PlethConfig()
    signal, time_s, fs = read_edf_signal(edf_path)
    events = detect_lid_events(signal, time_s, fs, basename, config)
    periods = slice_periods(
        signal=signal,
        time_s=time_s,
        fs=fs,
        lid_events=events,
        seizure_offset_s=300.0,
        config=config.period,
    )

    assert len(events.adjusted_spike_times_s) == 4
    assert events.open_times_s[0] < events.close_times_s[0]
    assert events.open_times_s[1] < events.close_times_s[1]
    assert events.close_times_s[0] < events.open_times_s[1]
    assert any(p.name == BASELINE for p in periods), [
        p.name for p in periods
    ]
