"""
Smoke tests for the ictal histogram emitter: confirm the expected output
files appear in ``output_dir``, the per-breath CSV has one row per breath,
and the apnea-flag column is correctly populated.

Includes an integration test that exercises ``analyze_experiment`` end to
end with ``ictal_histograms_dir`` set — this guards against the
"plumbed but never passed through" regression where the kwarg is accepted
at one layer but dropped at the next call site.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plethysmography.analysis.breath_segmentation import Breath
from plethysmography.visualization.ictal_histograms import (
    emit_ictal_histograms,
    plot_ictal_histograms,
    write_ictal_breaths_csv,
)


def _breath(ttot_ms: float, peak_diff: float = 2.0, ti_start_t: float = 0.0) -> Breath:
    return Breath(
        ti_start_idx=0, ti_end_idx=50, te_start_idx=50, te_end_idx=int(ttot_ms),
        ti_start_t=ti_start_t, ti_end_t=ti_start_t + 0.05,
        te_start_t=ti_start_t + 0.05, te_end_t=ti_start_t + ttot_ms / 1000.0,
        ti_ms=50.0, te_ms=ttot_ms - 50.0, ttot_ms=ttot_ms,
        pif_centered=-1.0, pef_centered=peak_diff - 1.0, peak_diff=peak_diff,
        tv_ml=0.05,
    )


def test_write_csv_produces_one_row_per_breath(tmp_path: Path):
    breaths = [_breath(200.0), _breath(800.0)]
    is_apnea = [False, True]

    csv_path = write_ictal_breaths_csv("250304 4055 p22", breaths, is_apnea, tmp_path)
    assert csv_path is not None and csv_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert df.loc[0, "is_apnea"] is False or df.loc[0, "is_apnea"] == False  # noqa: E712
    assert df.loc[1, "is_apnea"] is True or df.loc[1, "is_apnea"] == True    # noqa: E712
    assert "ttot_ms" in df.columns
    assert "peak_diff" in df.columns


def test_write_csv_returns_none_for_empty_breaths(tmp_path: Path):
    assert write_ictal_breaths_csv("x", [], [], tmp_path) is None


def test_csv_rejects_mismatched_is_apnea_length(tmp_path: Path):
    with pytest.raises(ValueError, match="is_apnea"):
        write_ictal_breaths_csv("x", [_breath(200.0)], [], tmp_path)


def test_plot_writes_png(tmp_path: Path):
    breaths = [_breath(200.0), _breath(220.0), _breath(800.0, peak_diff=4.0)]
    is_apnea = [False, False, True]

    png_path = plot_ictal_histograms("250304 4055 p22", breaths, is_apnea, tmp_path)
    assert png_path is not None and png_path.exists()
    assert png_path.suffix == ".png"
    assert png_path.stat().st_size > 0


def test_emit_writes_both_csv_and_png(tmp_path: Path):
    breaths = [_breath(200.0), _breath(800.0)]
    is_apnea = [False, True]

    written = emit_ictal_histograms("250304 4055 p22", breaths, is_apnea, tmp_path)
    suffixes = sorted(p.suffix for p in written)
    assert suffixes == [".csv", ".png"]


def test_emit_with_no_breaths_returns_empty_list(tmp_path: Path):
    assert emit_ictal_histograms("x", [], [], tmp_path) == []


# ---------------------------------------------------------------------------
# Integration: analyze_experiment must actually emit histograms when given
# ``ictal_histograms_dir``. Prevents the regression where the kwarg was
# accepted at the analyze_experiment layer but dropped before reaching
# analyze_recording.
# ---------------------------------------------------------------------------
def _write_synthetic_period_csv(
    path: Path,
    *,
    duration_s: float,
    fs: float,
    period_start_time: float,
    lid_closure_time: float,
) -> None:
    """Write a CSV in the same shape as the real preprocessed period CSVs:
    a low-frequency sine wave that the breath segmenter will resolve into
    a clean train of breaths."""
    n = int(duration_s * fs)
    t = np.arange(n, dtype=float) / fs + period_start_time
    signal = np.sin(2 * np.pi * 2.0 * (t - period_start_time))
    df = pd.DataFrame({
        "time": t,
        "signal": signal,
        "period_start_time": np.full(n, period_start_time),
        "lid_closure_time": np.full(n, lid_closure_time),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_analyze_experiment_emits_ictal_histograms(tmp_path: Path):
    """End-to-end: feed analyze_experiment a tiny synthetic preprocessed
    dataset and confirm the Ictal_Histograms output dir is populated."""
    from plethysmography.analysis.pipeline import analyze_experiment
    from plethysmography.core.config import PlethConfig
    from plethysmography.core.data_models import Recording

    preprocessed = tmp_path / "preprocessed"
    histograms = tmp_path / "Ictal_Histograms"

    fs = 1000.0
    basename = "test_subject"
    for period_token, duration_s, period_start, lid_close in (
        ("Habituation",         60.0, 300.0, float("nan")),
        ("Baseline",            60.0, 900.0, 900.0),
        ("Ictal",               30.0, 1500.0, 1500.0),
        ("Immediate_Postictal", 60.0, 1530.0, 1500.0),
        ("Recovery",            60.0, 1590.0, 1500.0),
    ):
        _write_synthetic_period_csv(
            preprocessed / f"{basename}_{period_token}.csv",
            duration_s=duration_s, fs=fs,
            period_start_time=period_start, lid_closure_time=lid_close,
        )

    recording = Recording(
        file_basename=basename,
        edf_path=tmp_path / f"{basename}.EDF",   # not read in analyze stage
        mouse_id="test", age="P22", genotype="het", cohort="test",
        risk="HR", fs=fs,
    )

    breathing_df, _ = analyze_experiment(
        [recording], preprocessed, PlethConfig(),
        ictal_histograms_dir=histograms,
    )

    assert not breathing_df.empty, "stats sanity: synthetic data produced no metrics"
    # The integration assertion: the histogram dir must exist and contain
    # both the PNG and the per-breath CSV for the synthetic recording.
    assert histograms.exists(), "Ictal_Histograms directory was not created"
    expected_png = histograms / f"{basename}_ictal_histograms.png"
    expected_csv = histograms / f"{basename}_ictal_breaths.csv"
    assert expected_png.exists(), f"missing histogram PNG at {expected_png}"
    assert expected_csv.exists(), f"missing per-breath CSV at {expected_csv}"
    # And only the Ictal period gets a CSV — no Baseline_breaths.csv etc.
    assert not (histograms / f"{basename}_baseline_breaths.csv").exists()


def test_analyze_experiment_skips_histograms_when_dir_is_none(tmp_path: Path):
    """The default (no ictal_histograms_dir kwarg) path should not create
    any directories. Lock in the opt-in semantics."""
    from plethysmography.analysis.pipeline import analyze_experiment
    from plethysmography.core.config import PlethConfig
    from plethysmography.core.data_models import Recording

    preprocessed = tmp_path / "preprocessed"
    fs = 1000.0
    basename = "test_subject"
    for period_token, duration_s, period_start, lid_close in (
        ("Baseline", 60.0, 900.0, 900.0),
        ("Ictal", 30.0, 1500.0, 1500.0),
    ):
        _write_synthetic_period_csv(
            preprocessed / f"{basename}_{period_token}.csv",
            duration_s=duration_s, fs=fs,
            period_start_time=period_start, lid_closure_time=lid_close,
        )

    recording = Recording(
        file_basename=basename, edf_path=tmp_path / f"{basename}.EDF",
        mouse_id="test", age="P22", genotype="het", cohort="test",
        risk="HR", fs=fs,
    )
    analyze_experiment([recording], preprocessed, PlethConfig())
    # No Ictal_Histograms folder anywhere under tmp_path.
    assert not (tmp_path / "Ictal_Histograms").exists()
