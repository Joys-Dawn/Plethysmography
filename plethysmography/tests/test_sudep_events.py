"""
Item H — SUDEP fatal-seizure pipeline.

Covers the three pieces the plan calls out:
  * the Column J ``(MM.SS-MM.SS)`` window parser, including the edge
    values (empty -> None; unparseable / out-of-range / non-increasing
    -> ValueError, never a guess);
  * the recording-relative window slice produces the right sample range
    (inclusive both ends) and rejects degenerate windows;
  * the cohort is built straight from Column J and is exactly the 5
    expected mice — the near-SUDEP ``250304 4056 p22`` is present
    despite being G=1/G-filtered, and the WT blank-J row is absent.

Plus the net-new ``plot_fatal_seizure_strips``: all mice pool into a
single "fatal seizures" column (one PNG per parameter, not per mouse)
and the apnea-duration strip uses the Item B distinct slug.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plethysmography.data_loading import data_log
from plethysmography.data_loading.data_log import (
    COL_AGE,
    COL_FATAL_SEIZURE,
    COL_FILENAME,
    COL_GENOTYPE,
    COL_INCLUDE,
    SudepEvent,
    load_sudep_event_cohort,
    parse_fatal_seizure_window,
    resolve_raw_edf,
)
from plethysmography.pipelines.sudep_events import (
    _build_window_period,
    plot_fatal_seizure_strips,
)


# ---------------------------------------------------------------------------
# parse_fatal_seizure_window
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell, expected",
    [
        ("evoked SUDEP (58.21-58.53)", (3501.0, 3533.0)),
        ("near spont SUDEP (6.27-6.53)", (387.0, 413.0)),
        ("spont SUDEP (10.04-10.38)", (604.0, 638.0)),
        ("(0.00-0.30)", (0.0, 30.0)),               # zero-minute edge
        ("( 6.27 - 6.53 )", (387.0, 413.0)),        # whitespace tolerance
        ("x (0.01-0.59)", (1.0, 59.0)),             # max valid seconds
        ("evoked SUDEP (120.00-120.45)", (7200.0, 7245.0)),  # 3-digit min
    ],
)
def test_parse_window_valid(cell, expected):
    assert parse_fatal_seizure_window(cell) == expected


@pytest.mark.parametrize(
    "cell",
    [float("nan"), None, pd.NA, " ", "", "   ", np.nan],
)
def test_parse_window_empty_returns_none(cell):
    assert parse_fatal_seizure_window(cell) is None


@pytest.mark.parametrize(
    "cell",
    [
        "no window here",          # no parenthesised time at all
        "evoked SUDEP (58.21)",    # single time, not a range
        "(58.21 58.53)",          # missing dash
        "(1.60-1.70)",             # seconds component >= 60 (start)
        "(1.10-1.75)",             # seconds component >= 60 (end)
        "(5.30-5.30)",             # end == start
        "(5.30-4.10)",             # end < start
    ],
)
def test_parse_window_unparseable_raises(cell):
    with pytest.raises(ValueError):
        parse_fatal_seizure_window(cell)


# ---------------------------------------------------------------------------
# _build_window_period — recording-relative slice
# ---------------------------------------------------------------------------
def _synthetic_recording(fs: float = 100.0, seconds: int = 120):
    n = int(fs * seconds)
    signal = np.arange(n, dtype=float)        # value == sample index
    time_s = np.arange(n) / fs                # == read_edf_signal basis
    return signal, time_s, fs


def test_window_slice_inclusive_both_ends():
    signal, time_s, fs = _synthetic_recording()
    ev = SudepEvent("m", "x", "het", "P22", 10.0, 20.0, None)
    period = _build_window_period(ev, signal, time_s, fs)
    assert period is not None
    # [10, 20] s at 100 Hz, inclusive both ends -> 1001 samples
    assert period.signal.size == 1001
    assert period.time_s[0] == 10.0
    assert math.isclose(period.time_s[-1], 20.0, abs_tol=1e-9)
    # value == sample index -> 10 s*100 Hz = 1000 .. 20 s*100 Hz = 2000
    assert period.signal[0] == 1000.0
    assert period.signal[-1] == 2000.0
    assert period.name == "Fatal Seizure"
    assert period.period_start_time == 10.0


@pytest.mark.parametrize(
    "start_s, end_s",
    [
        (50.0, 50.0),     # zero-length
        (60.0, 50.0),     # inverted (defensive; parser would also reject)
        (999.0, 1000.0),  # entirely past the recording
    ],
)
def test_window_slice_degenerate_returns_none(start_s, end_s):
    signal, time_s, fs = _synthetic_recording()
    ev = SudepEvent("m", "x", "het", "P22", start_s, end_s, None)
    assert _build_window_period(ev, signal, time_s, fs) is None


# ---------------------------------------------------------------------------
# resolve_raw_edf — exp1 then exp2 fallback
# ---------------------------------------------------------------------------
def test_resolve_raw_edf_prefers_exp1_then_exp2_then_none(tmp_path: Path):
    reg1 = data_log.get_experiment_registry(1)
    reg2 = data_log.get_experiment_registry(2)
    exp1_raw = tmp_path / reg1["cohort_folder"] / reg1["raw_subfolder"]
    exp2_raw = tmp_path / reg2["cohort_folder"] / reg2["raw_subfolder"]
    exp1_raw.mkdir(parents=True)
    exp2_raw.mkdir(parents=True)

    # absent in both -> None
    assert resolve_raw_edf("250307 4051 p22", tmp_path) is None

    # only in exp2 -> resolves to exp2
    (exp2_raw / "033026 5514 p22.EDF").write_bytes(b"x")
    got = resolve_raw_edf("033026 5514 p22", tmp_path)
    assert got is not None and got.parent == exp2_raw

    # present in exp1 too -> exp1 wins (tried first)
    (exp1_raw / "033026 5514 p22.EDF").write_bytes(b"x")
    got = resolve_raw_edf("033026 5514 p22", tmp_path)
    assert got is not None and got.parent == exp1_raw


# ---------------------------------------------------------------------------
# load_sudep_event_cohort — built from Column J, bypasses Column G
# ---------------------------------------------------------------------------
def _fake_data_log() -> pd.DataFrame:
    """4056 is G=1 (near-SUDEP) and MUST still appear; the four G=0 SUDEP
    rows appear; the WT blank-J row and the plain G=0 row do not."""
    return pd.DataFrame(
        {
            COL_FILENAME: [
                "250304 4056 p22",   # G=1 near-SUDEP  -> include
                "250307 4051 p22",   # G=0 evoked      -> include
                "033026 5514 p22",   # G=0 evoked      -> include
                "250325 4173 p19",   # WT, blank J     -> exclude
                "999999 9999 p22",   # G=0, no J cell  -> exclude
            ],
            COL_GENOTYPE: ["het", "het", "het", "WT", "het"],
            COL_AGE: [22, 22, 22, 19, 22],
            COL_INCLUDE: [1, 0, 0, 1, 0],
            COL_FATAL_SEIZURE: [
                "near spont SUDEP (6.27-6.53)",
                "evoked SUDEP (58.21-58.53)",
                "evoked SUDEP (56.42-57.22)",
                " ",
                float("nan"),
            ],
        }
    )


def test_cohort_from_fake_log_bypasses_g_filter(monkeypatch):
    monkeypatch.setattr(data_log, "load_data_log", lambda *a, **k: _fake_data_log())
    cohort = load_sudep_event_cohort()
    names = {e.file_basename for e in cohort}
    assert names == {
        "250304 4056 p22",
        "250307 4051 p22",
        "033026 5514 p22",
    }
    by_name = {e.file_basename: e for e in cohort}
    # 4056 is G=1 yet present — the explicit H-only override of Item A.
    assert "250304 4056 p22" in by_name
    assert by_name["250304 4056 p22"].start_s == 387.0
    assert by_name["250304 4056 p22"].end_s == 413.0
    for ev in cohort:
        assert ev.end_s > ev.start_s


def test_cohort_missing_column_j_fails_fast(monkeypatch):
    bad = _fake_data_log().drop(columns=[COL_FATAL_SEIZURE])
    monkeypatch.setattr(data_log, "load_data_log", lambda *a, **k: bad)
    with pytest.raises(KeyError, match="SUDEP captured on pleth"):
        load_sudep_event_cohort()


def test_cohort_unparseable_cell_raises(monkeypatch):
    df = _fake_data_log()
    df.loc[0, COL_FATAL_SEIZURE] = "evoked SUDEP no-window"
    monkeypatch.setattr(data_log, "load_data_log", lambda *a, **k: df)
    with pytest.raises(ValueError):
        load_sudep_event_cohort()


def test_cohort_against_real_data_log():
    """Against the real docs/pleth data log.xlsx: exactly the 5 mice,
    4056 (G=1 near-SUDEP) present, WT blank-J row absent."""
    cohort = load_sudep_event_cohort(data_root=".")
    names = {e.file_basename for e in cohort}
    assert len(cohort) == 5
    assert names == {
        "250304 4056 p22",
        "250307 4051 p22",
        "250423 4269 p22",
        "033026 5514 p22",
        "051626 5676 p22",
    }
    assert "250325 4173 p19" not in names  # WT, blank Column J
    for ev in cohort:
        assert ev.end_s > ev.start_s
        assert ev.genotype == "het"


# ---------------------------------------------------------------------------
# plot_fatal_seizure_strips — one pooled "fatal seizures" column
# ---------------------------------------------------------------------------
def _strip_df() -> pd.DataFrame:
    """5 mice; 3 have apneas (V1 keeps them), 2 do not (V1 drops, V2 imputes
    via 10-longest-breaths). Exercises the within-period V1+V2 split."""
    return pd.DataFrame(
        {
            "file_basename": [f"m{i}" for i in range(5)],
            "mean_ttot_ms_no_apnea": [200.0, 210, 205, 198, 220],
            # V1 real durations — NaN where no apneas detected.
            "apnea_mean_ms":         [410.0, 500, 450, np.nan, np.nan],
            # V2 imputed — real values for apneic mice, 10-longest-breaths
            # fallback for the zero-apnea ones (mice 3, 4).
            "apnea_mean_ms_imputed": [410.0, 500, 450, 230, 240],
            "apnea_rate_per_min": [1.0, 2, 3, 0.0, 0.0],
            "mean_tv_ml": [0.20, 0.21, 0.19, 0.22, 0.20],
        }
    )


def test_strips_one_png_per_parameter_not_per_mouse(tmp_path: Path):
    out = plot_fatal_seizure_strips(_strip_df(), tmp_path / "plots")
    # 5 mice, 4 plottable columns; apnea_mean_ms_imputed expands via
    # within_style_params to (apnea_mean_ms, apnea_mean_ms_imputed) so the
    # apnea-duration slot emits TWO PNGs (V1 + V2) -> 5 files, pooled
    # (never 5 mice * 5 params).
    assert len(out) == 5
    for p in out:
        assert p.exists() and p.stat().st_size > 0
        assert p.name.startswith("Fatal_Seizure_")


def test_strips_emit_both_apnea_duration_versions(tmp_path: Path):
    """SUDEP fatal-seizure windows are isolated (no time-period comparison),
    so use the within-period V1+V2 split: emit BOTH ``apnea_mean_ms`` (real
    durations, dropping zero-apnea mice) and ``apnea_mean_ms_imputed`` (every
    mouse, 10-longest-breaths fallback) under the Item B distinct slugs so
    they never collide on disk."""
    out = plot_fatal_seizure_strips(_strip_df(), tmp_path / "plots")
    names = {p.name for p in out}
    assert "Fatal_Seizure_Apnea_duration_ms.png" in names
    assert "Fatal_Seizure_Apnea_or_longest_breaths_duration_ms.png" in names


def test_strips_v1_dropped_when_no_mouse_has_apneas(tmp_path: Path):
    """Edge case: if no mouse in the cohort has any detected apnea, the V1
    real-durations strip has nothing to plot and is silently skipped, while
    V2 still emits via the 10-longest-breaths fallback. Belt-and-braces for
    future cohorts where the current "every mouse has apneas" assumption
    may not hold."""
    df = _strip_df()
    df["apnea_mean_ms"] = np.nan          # zero-apnea cohort
    out = plot_fatal_seizure_strips(df, tmp_path / "plots")
    names = {p.name for p in out}
    assert "Fatal_Seizure_Apnea_duration_ms.png" not in names
    assert "Fatal_Seizure_Apnea_or_longest_breaths_duration_ms.png" in names


def test_strips_skip_empty_and_all_nan_params(tmp_path: Path):
    df = _strip_df()
    df["apnea_rate_per_min"] = np.nan          # all-NaN -> skipped
    out = plot_fatal_seizure_strips(
        df, tmp_path / "plots",
        parameters=["mean_ttot_ms_no_apnea", "apnea_rate_per_min", "absent_col"],
    )
    names = {p.name for p in out}
    assert names == {"Fatal_Seizure_Ttot_ms.png"}
