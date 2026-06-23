"""Missing Baseline must abort preprocessing and analysis — never fall back."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plethysmography.analysis.pipeline import analyze_recording
from plethysmography.core.config import PlethConfig
from plethysmography.core.data_models import HABITUATION, LidEvents, Period, Recording
from plethysmography.core.errors import MissingBaselineError
from plethysmography.preprocessing.pipeline import _require_baseline_period


def test_require_baseline_period_raises_when_baseline_missing() -> None:
    lid = LidEvents(raw_spike_times_s=[100.0], adjusted_spike_times_s=[100.0])
    periods = [
        Period(
            name=HABITUATION,
            start_s=300.0,
            end_s=600.0,
            signal=np.array([0.0]),
            time_s=np.array([300.0]),
            fs=250.0,
            period_start_time=300.0,
            lid_closure_time=float("nan"),
        ),
    ]
    with pytest.raises(MissingBaselineError, match="no Baseline period"):
        _require_baseline_period("bad_mouse p22", periods, lid)


def test_analyze_recording_raises_without_baseline_csv() -> None:
    recording = Recording(
        file_basename="bad_mouse p22",
        edf_path=Path("unused.EDF"),
        mouse_id="1",
        age="P22",
        genotype="het",
        cohort="test",
        risk="HR",
        fs=1000.0,
    )
    periods = {
        "Habituation": Period(
            name="Habituation",
            start_s=300.0,
            end_s=600.0,
            signal=np.sin(np.linspace(0, 10, 1000)),
            time_s=np.linspace(300, 600, 1000),
            fs=1000.0,
            period_start_time=300.0,
            lid_closure_time=float("nan"),
        ),
    }
    with pytest.raises(MissingBaselineError, match="no Baseline preprocessed CSV"):
        analyze_recording(recording, periods, PlethConfig())
