"""
Helpers shared by the experiment-1 and experiment-2 pipeline drivers.
Experiment 4 reuses these but skips preprocessing / analysis (it loads
existing breathing CSVs from experiments 1 + 2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

from ..analysis.pipeline import analyze_experiment
from ..core.config import PlethConfig
from ..core.data_models import Recording
from ..core.metadata import should_skip_preprocess
from ..data_loading.edf_reader import read_edf_signal
from ..preprocessing.pipeline import preprocess_recording


logger = logging.getLogger(__name__)

DATA_ROOT = Path("Data")
RESULTS_ROOT = Path("results")


def preprocess_all(
    recordings: Sequence[Recording],
    config: PlethConfig,
    preprocessed_dir: Path,
    *,
    skip_existing: bool = True,
    traces_dir: Path | None = None,
) -> List[Recording]:
    """Run preprocessing for every recording. Returns the list of recordings
    that completed successfully (skipped + failed are dropped).

    ``skip_existing`` short-circuits files whose 5 period CSVs already exist on
    disk -- useful for re-running the analysis without re-doing the (slow) EDF
    read + filter + slice steps.

    ``traces_dir``, when given, makes preprocessing also write the per-recording
    ``_spikes.png`` and ``_periods.png`` verification plots there. Cached
    recordings still produce traces (the saved CSVs are reread to rebuild the
    period overlay).
    """
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    if traces_dir is not None:
        traces_dir.mkdir(parents=True, exist_ok=True)
    completed: List[Recording] = []

    for rec in recordings:
        if should_skip_preprocess(rec.file_basename):
            logger.info("Skipping preprocess (excluded): %s", rec.file_basename)
            continue
        if not rec.edf_path.exists():
            logger.warning("EDF missing for %s: %s", rec.file_basename, rec.edf_path)
            continue
        if skip_existing and _has_preprocessed_outputs(rec.file_basename, preprocessed_dir):
            logger.info("Skipping preprocess (cached): %s", rec.file_basename)
            if traces_dir is not None:
                _emit_cached_trace_plots(rec, preprocessed_dir, traces_dir)
            completed.append(rec)
            continue
        try:
            _, _, fs = read_edf_signal(rec.edf_path)
            rec.fs = fs
            preprocess_recording(
                rec, config,
                save_dir=preprocessed_dir,
                traces_dir=traces_dir,
            )
            completed.append(rec)
        except Exception as exc:  # pragma: no cover -- log and keep going
            logger.exception("Preprocess failed for %s: %s", rec.file_basename, exc)
    return completed


def _emit_cached_trace_plots(
    recording: Recording,
    preprocessed_dir: Path,
    traces_dir: Path,
) -> None:
    """For recordings whose preprocessed CSVs are already on disk, rebuild the
    period overlay PNG only (the spikes plot needs the raw EDF signal, which we
    deliberately do not re-read here in cache-hit mode)."""
    import numpy as np
    from ..core.data_models import Period
    from ..visualization.trace_plots import plot_periods_overlay

    periods: List[Period] = []
    for path in sorted(preprocessed_dir.glob(f"{recording.file_basename}_*.csv")):
        df = pd.read_csv(path, usecols=["time", "signal", "period_start_time", "lid_closure_time"])
        if df.empty:
            continue
        period_token = path.stem.split(recording.file_basename + "_", 1)[-1]
        period_name = period_token.replace("_", " ")
        time_s = df["time"].to_numpy(dtype=float)
        signal = df["signal"].to_numpy(dtype=float)
        if recording.fs is not None:
            fs = float(recording.fs)
        else:
            dt = float(np.median(np.diff(time_s[:64]))) if len(time_s) > 1 else 1.0e-3
            fs = 1.0 / dt if dt > 0 else 1000.0
        periods.append(Period(
            name=period_name, start_s=float(time_s[0]), end_s=float(time_s[-1]),
            signal=signal, time_s=time_s, fs=fs,
            period_start_time=float(df["period_start_time"].iloc[0]) if not pd.isna(df["period_start_time"].iloc[0]) else float(time_s[0]),
            lid_closure_time=float(df["lid_closure_time"].iloc[0]) if not pd.isna(df["lid_closure_time"].iloc[0]) else float("nan"),
        ))
    if periods:
        plot_periods_overlay(periods, recording.file_basename, traces_dir)


def _has_preprocessed_outputs(file_basename: str, preprocessed_dir: Path) -> bool:
    pattern = f"{file_basename}_*.csv"
    return any(preprocessed_dir.glob(pattern))


def analyze_all(
    recordings: Sequence[Recording],
    config: PlethConfig,
    preprocessed_dir: Path,
    *,
    interactive_dir: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Wrap :func:`plethysmography.analysis.pipeline.analyze_experiment`."""
    return analyze_experiment(
        list(recordings), preprocessed_dir, config,
        interactive_dir=interactive_dir,
    )


def write_breathing_outputs(
    breathing_df: pd.DataFrame,
    apnea_df: pd.DataFrame,
    results_dir: Path,
) -> Tuple[Path, Path]:
    """Persist the breathing and apnea DataFrames in the per-experiment
    results directory. Returns the two written paths."""
    results_dir.mkdir(parents=True, exist_ok=True)
    breathing_path = results_dir / "breathing_analysis_results.csv"
    apnea_path = results_dir / "apnea_list.xlsx"
    breathing_df.to_csv(breathing_path, index=False)
    if not apnea_df.empty:
        with pd.ExcelWriter(apnea_path, engine="openpyxl") as writer:
            for period_name, group in apnea_df.groupby("period"):
                # Excel sheet names cap at 31 chars; "Immediate Postictal" fits.
                # Match old layout: keep all columns including 'period'.
                group.to_excel(writer, sheet_name=period_name, index=False)
    return breathing_path, apnea_path


def metadata_for_bins(recordings: Sequence[Recording]) -> dict[str, dict[str, str]]:
    """Build the ``{file_basename: {genotype, risk_clean / treatment_clean,
    age}}`` mapping used by the binned-plot generators."""
    out: dict[str, dict[str, str]] = {}
    for rec in recordings:
        if rec.risk is not None:
            condition_value = "high_risk" if rec.risk == "HR" else "low_risk"
            cond_key = "risk_clean"
        elif rec.treatment is not None:
            condition_value = rec.treatment
            cond_key = "treatment_clean"
        else:
            continue
        out[rec.file_basename] = {
            "genotype": rec.genotype,
            cond_key: condition_value,
            "age": rec.age,
        }
    return out


def load_period_data_for_bins(
    recordings: Sequence[Recording],
    preprocessed_dir: Path,
    period_name: str,
) -> List[Tuple[str, "np.ndarray", "np.ndarray", float]]:
    """Load (file_basename, time_s, signal, fs) tuples for the named period.

    Used by the binned plot generators which need the raw time/signal arrays
    rather than the per-breath summary stored in the breathing CSV.
    """
    import numpy as np
    csv_period_token = period_name.replace(" ", "_")
    out: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
    for rec in recordings:
        path = preprocessed_dir / f"{rec.file_basename}_{csv_period_token}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["time", "signal"])
        if df.empty:
            continue
        time_s = df["time"].to_numpy(dtype=float)
        signal = df["signal"].to_numpy(dtype=float)
        if rec.fs is not None:
            fs = float(rec.fs)
        else:
            dt = float(np.median(np.diff(time_s[:64])))
            fs = 1.0 / dt if dt > 0 else 1000.0
        out.append((rec.file_basename, time_s, signal, fs))
    return out


def baseline_median_ttot_by_basename(
    recordings: Sequence[Recording],
    preprocessed_dir: Path,
    config: PlethConfig,
) -> dict[str, float]:
    """Per-recording baseline median Ttot (ms), used as the apnea-threshold
    reference inside the binned plot generators (which match
    old_code/postictal_binned_plots.py:collect_baseline_medians)."""
    from ..analysis.baseline_cache import build_baseline_cache
    from ..core.data_models import Period
    import numpy as np

    out: dict[str, float] = {}
    for rec in recordings:
        path = preprocessed_dir / f"{rec.file_basename}_Baseline.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["time", "signal"])
        if df.empty:
            continue
        time_s = df["time"].to_numpy(dtype=float)
        signal = df["signal"].to_numpy(dtype=float)
        if rec.fs is not None:
            fs = float(rec.fs)
        else:
            dt = float(np.median(np.diff(time_s[:64])))
            fs = 1.0 / dt if dt > 0 else 1000.0
        period = Period(
            name="Baseline",
            start_s=float(time_s[0]), end_s=float(time_s[-1]),
            signal=signal, time_s=time_s, fs=fs,
            period_start_time=float(time_s[0]),
            lid_closure_time=float("nan"),
        )
        cache = build_baseline_cache(rec.file_basename, period, config.breath)
        if cache.median_ttot_ms > 0:
            out[rec.file_basename] = cache.median_ttot_ms
    return out


