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
    _population_slug,
    emit_ictal_histograms,
    plot_ictal_histograms,
    plot_population_ictal_histograms,
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


# ---------------------------------------------------------------------------
# Item F: population (pooled-per-group) ictal histograms.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("WT P22 LR", "WT_P22_LR"),
        ("Scn1a+/- P22 vehicle", "Scn1a_P22_vehicle"),
        ("Scn1a+/- P19 SUDEP", "Scn1a_P19_SUDEP"),
        ("Scn1a+/- P19 survivor", "Scn1a_P19_survivor"),
    ],
)
def test_population_slug(label, expected):
    assert _population_slug(label) == expected


def test_plot_population_writes_one_png_per_group(tmp_path: Path):
    groups = {
        "WT P22 LR": ([_breath(200.0), _breath(210.0)], [False, False]),
        "Scn1a+/- P22 HR": (
            [_breath(220.0), _breath(900.0, peak_diff=4.0)],
            [False, True],
        ),
    }
    saved = plot_population_ictal_histograms(groups, tmp_path)
    assert sorted(p.name for p in saved) == [
        "Scn1a_P22_HR_ictal_histograms.png",
        "WT_P22_LR_ictal_histograms.png",
    ]
    for p in saved:
        assert p.exists() and p.stat().st_size > 0


def test_plot_population_skips_group_with_no_breaths(tmp_path: Path):
    """A group that pooled zero breaths (or no finite values) is skipped —
    no PNG, not in the returned list (reuses the _render_two_panel None
    contract)."""
    groups = {
        "WT P22 LR": ([_breath(200.0)], [False]),
        "Scn1a+/- P22 HR": ([], []),
    }
    saved = plot_population_ictal_histograms(groups, tmp_path)
    assert [p.name for p in saved] == ["WT_P22_LR_ictal_histograms.png"]
    assert not (tmp_path / "Scn1a_P22_HR_ictal_histograms.png").exists()


def test_population_accumulation_skips_g0_basename(tmp_path: Path, monkeypatch):
    """The Item A↔F coupling guard: analyze_experiment's pool must drop a
    recording whose basename is NOT in population_included_basenames()
    (Column G == 0), even though that recording still gets a breathing row.
    The two recordings map to distinct group labels so the skip is provable
    by the presence/absence of each group's pooled PNG."""
    from plethysmography.analysis.pipeline import analyze_experiment
    from plethysmography.core.config import PlethConfig
    from plethysmography.core.data_models import Recording
    import plethysmography.data_loading.data_log as data_log

    preprocessed = tmp_path / "preprocessed"
    pop_dir = tmp_path / "Ictal_Histograms_population"
    fs = 1000.0
    for basename in ("inc_subject", "exc_subject"):
        for period_token, duration_s, period_start, lid_close in (
            ("Baseline", 60.0, 900.0, 900.0),
            ("Ictal", 30.0, 1500.0, 1500.0),
        ):
            _write_synthetic_period_csv(
                preprocessed / f"{basename}_{period_token}.csv",
                duration_s=duration_s, fs=fs,
                period_start_time=period_start, lid_closure_time=lid_close,
            )

    inc = Recording(
        file_basename="inc_subject", edf_path=tmp_path / "inc_subject.EDF",
        mouse_id="i", age="P22", genotype="het", cohort="test", risk="HR", fs=fs,
    )
    exc = Recording(
        file_basename="exc_subject", edf_path=tmp_path / "exc_subject.EDF",
        mouse_id="e", age="P22", genotype="WT", cohort="test", risk="LR", fs=fs,
    )

    # Column G says only inc_subject is included.
    monkeypatch.setattr(
        data_log, "population_included_basenames", lambda *a, **k: {"inc_subject"}
    )

    breathing_df, _ = analyze_experiment(
        [inc, exc], preprocessed, PlethConfig(), population_ictal_dir=pop_dir,
    )

    # exc_subject still produced a breathing row (G filter is a population
    # gate, not an analyze gate) ...
    assert set(breathing_df["file_basename"]) == {"inc_subject", "exc_subject"}
    # ... but only the included recording's group histogram exists.
    assert (pop_dir / "Scn1a_P22_HR_ictal_histograms.png").exists()
    assert not (pop_dir / "WT_P22_LR_ictal_histograms.png").exists()


def test_population_pool_off_by_default_does_not_read_data_log(
    tmp_path: Path, monkeypatch
):
    """When population_ictal_dir is not given, population_included_basenames
    must NOT be called — otherwise every existing analyze_experiment caller
    /test would start depending on the real data-log xlsx."""
    from plethysmography.analysis.pipeline import analyze_experiment
    from plethysmography.core.config import PlethConfig
    from plethysmography.core.data_models import Recording
    import plethysmography.data_loading.data_log as data_log

    preprocessed = tmp_path / "preprocessed"
    fs = 1000.0
    for period_token, duration_s, period_start, lid_close in (
        ("Baseline", 60.0, 900.0, 900.0),
        ("Ictal", 30.0, 1500.0, 1500.0),
    ):
        _write_synthetic_period_csv(
            preprocessed / f"test_subject_{period_token}.csv",
            duration_s=duration_s, fs=fs,
            period_start_time=period_start, lid_closure_time=lid_close,
        )

    def _boom(*a, **k):
        raise AssertionError("population_included_basenames must not be called")

    monkeypatch.setattr(data_log, "population_included_basenames", _boom)

    recording = Recording(
        file_basename="test_subject", edf_path=tmp_path / "test_subject.EDF",
        mouse_id="t", age="P22", genotype="het", cohort="test", risk="HR", fs=fs,
    )
    # No population_ictal_dir -> _boom must never fire.
    analyze_experiment([recording], preprocessed, PlethConfig())
    assert not (tmp_path / "Ictal_Histograms_population").exists()
