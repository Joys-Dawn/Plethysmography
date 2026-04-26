"""
Two-pass analysis driver.

Pass 1 — for each recording with a Baseline preprocessed CSV, segment its
breaths and build a :class:`BaselineCache`. The Baseline period's metrics row
is also produced here (its sigh / apnea thresholds use per-period stats since
the cache is built FROM Baseline; for the Baseline period this is identical to
old code).

Pass 2 — for every other (file, period) CSV, segment breaths, then derive the
sigh / apnea thresholds from the recording's :class:`BaselineCache`. Each
period contributes one :class:`BreathMetrics` row plus zero-or-more
:class:`ApneaEvent` rows.

Public entry points:
  - :func:`analyze_period`     — one (file, period) -> (metrics, apneas, breaths)
  - :func:`analyze_recording`  — all periods of one recording
  - :func:`analyze_experiment` — all recordings of an experiment cohort

Output handling (CSV / xlsx writes) lives at the pipeline-driver level, not here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.config import PlethConfig
from ..core.data_models import (
    ApneaEvent,
    BaselineCache,
    BASELINE,
    BreathMetrics,
    PERIOD_NAMES,
    Period,
    Recording,
)
from ..core.metadata import get_analysis_override
from .apnea_detection import compute_apnea_threshold, detect_apneas
from .baseline_cache import cache_from_breaths
from .breath_metrics import compute_breath_metrics
from .breath_segmentation import Breath, segment_breaths
from .sigh_detection import classify_sighs, compute_sigh_threshold


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class PeriodAnalysisResult:
    """Output of analyzing a single (file, period)."""
    metrics: BreathMetrics
    apneas: List[ApneaEvent]
    breaths: List[Breath]


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------
def _load_period_csv(path: Path, fs: float) -> Optional[Period]:
    """Reconstruct a :class:`Period` from a preprocessed CSV. Returns None if
    the file is missing or empty.

    The CSV columns are: ``time, signal, period_start_time, lid_closure_time``
    (matches preprocessing/pipeline.py:save_period_csv).
    """
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "time" not in df.columns or "signal" not in df.columns:
        return None
    period_name = _period_from_filename(path.stem)
    if period_name is None:
        return None
    time_s = df["time"].to_numpy(dtype=float)
    signal = df["signal"].to_numpy(dtype=float)
    pst = df.get("period_start_time")
    lct = df.get("lid_closure_time")
    return Period(
        name=period_name,
        start_s=float(time_s[0]),
        end_s=float(time_s[-1]),
        signal=signal,
        time_s=time_s,
        fs=fs,
        period_start_time=float(pst.iloc[0]) if pst is not None and not pd.isna(pst.iloc[0]) else float(time_s[0]),
        lid_closure_time=float(lct.iloc[0]) if lct is not None and not pd.isna(lct.iloc[0]) else float("nan"),
    )


def _period_from_filename(stem: str) -> Optional[str]:
    """Extract period name from ``<basename>_<period>.csv`` filename. Period
    name has its space replaced with underscore (Immediate_Postictal)."""
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-2].lower() == "immediate" and parts[-1].lower() == "postictal":
        return "Immediate Postictal"
    last = parts[-1] if parts else ""
    for name in PERIOD_NAMES:
        if last == name:
            return name
    return None


def _basename_from_filename(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-2].lower() == "immediate" and parts[-1].lower() == "postictal":
        return "_".join(parts[:-2])
    return "_".join(parts[:-1]) if len(parts) > 1 else stem


# ---------------------------------------------------------------------------
# Per-period analysis
# ---------------------------------------------------------------------------
def analyze_period(
    period: Period,
    file_basename: str,
    config: PlethConfig,
    baseline_cache: Optional[BaselineCache],
) -> PeriodAnalysisResult:
    """Analyze one (file, period). Per-file analysis overrides (e.g. the 6 Hz
    LPF for ``250304 4056 p22`` Recovery) are applied here."""
    override = get_analysis_override(file_basename, period.name)
    apply_lp_hz: Optional[float] = None
    if override is not None and "apply_lowpass_hz" in override:
        apply_lp_hz = float(override["apply_lowpass_hz"])

    breaths = segment_breaths(period, config.breath, apply_post_center_lp_hz=apply_lp_hz)
    sigh_threshold = compute_sigh_threshold(breaths, config.sigh, baseline_cache)
    is_sigh = classify_sighs(breaths, sigh_threshold)
    apnea_threshold = compute_apnea_threshold(breaths, config.apnea, baseline_cache)
    apneas = detect_apneas(
        breaths=breaths,
        is_sigh=is_sigh,
        threshold_ms=apnea_threshold,
        config=config.apnea,
        file_basename=file_basename,
        period_name=period.name,
        period_start_time=period.period_start_time,
        lid_closure_time=period.lid_closure_time if np.isfinite(period.lid_closure_time) else period.period_start_time,
    )
    metrics = compute_breath_metrics(
        file_basename=file_basename,
        period_name=period.name,
        period_duration_s=period.duration_s,
        breaths=breaths,
        is_sigh=is_sigh,
        apneas=apneas,
    )
    return PeriodAnalysisResult(metrics=metrics, apneas=apneas, breaths=breaths)


# ---------------------------------------------------------------------------
# Per-recording analysis
# ---------------------------------------------------------------------------
def analyze_recording(
    recording: Recording,
    periods_by_name: Dict[str, Period],
    config: PlethConfig,
    *,
    interactive_dir: Optional[Path] = None,
    apneas_for_interactive: bool = True,
) -> Tuple[List[BreathMetrics], List[ApneaEvent], Dict[str, List[Breath]]]:
    """Analyze every Period of one recording, two-pass.

    The breathing CSV produced here matches the old code's output: it includes
    rows for every (file, period) that has data, regardless of whether the
    file/period is later excluded from cohort statistics. The
    ``metadata.EXCLUSIONS`` dict is consulted only by the stats layer
    (``stats/helpers.py:prepare_breathing_data``), which drops excluded rows
    before running tests.

    When ``interactive_dir`` is given, also writes one
    ``<basename>_<period>_interactive_breaths.html`` per period (the breath-
    segmentation plotly view), with apnea bars when ``apneas_for_interactive``
    is True.

    Returns:
        (list of BreathMetrics rows, list of ApneaEvent rows, dict period_name -> breaths)
    """
    baseline_period = periods_by_name.get(BASELINE)
    baseline_result: Optional[PeriodAnalysisResult] = None
    if baseline_period is not None:
        baseline_result = analyze_period(
            period=baseline_period,
            file_basename=recording.file_basename,
            config=config,
            baseline_cache=None,
        )
        cache = cache_from_breaths(recording.file_basename, baseline_result.breaths)
    else:
        cache = BaselineCache(
            file_basename=recording.file_basename,
            median_ttot_ms=0.0,
            mean_pif_to_pef_ml_s=0.0,
            std_pif_to_pef_ml_s=0.0,
            n_breaths=0,
        )
        if baseline_period is None:
            logger.info(
                "%s has no Baseline period; sigh and apnea thresholds for "
                "other periods will fall back to per-period stats.",
                recording.file_basename,
            )

    metrics_rows: List[BreathMetrics] = []
    apnea_rows: List[ApneaEvent] = []
    breaths_by_period: Dict[str, List[Breath]] = {}

    def _maybe_emit_interactive(period: Period, result: PeriodAnalysisResult) -> None:
        if interactive_dir is None:
            return
        from ..visualization.interactive_plots import plot_breath_segmentation
        out_dir = Path(interactive_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        period_token = period.name.replace(" ", "_")
        out_path = out_dir / f"{recording.file_basename}_{period_token}_interactive_breaths.html"
        ap_periods = None
        if apneas_for_interactive and result.apneas:
            ap_periods = [
                {
                    "start": float(period.period_start_time + ap.apnea_start_s_from_period_start),
                    "end":   float(period.period_start_time + ap.apnea_start_s_from_period_start
                                   + ap.apnea_duration_ms / 1000.0),
                    "duration": float(ap.apnea_duration_ms),
                }
                for ap in result.apneas
            ]
        plot_breath_segmentation(
            time_s=period.time_s,
            signal=period.signal,
            breaths=result.breaths,
            output_path=out_path,
            title=f"Interactive Breath Analysis: {recording.file_basename} - {period.name}",
            apnea_periods=ap_periods,
            allow_split=True,
            period_name=period.name,
        )

    for name in PERIOD_NAMES:
        if name == BASELINE:
            if baseline_result is not None:
                metrics_rows.append(baseline_result.metrics)
                apnea_rows.extend(baseline_result.apneas)
                breaths_by_period[BASELINE] = baseline_result.breaths
                if baseline_period is not None:
                    _maybe_emit_interactive(baseline_period, baseline_result)
            continue
        period = periods_by_name.get(name)
        if period is None:
            continue
        result = analyze_period(
            period=period,
            file_basename=recording.file_basename,
            config=config,
            baseline_cache=cache,
        )
        metrics_rows.append(result.metrics)
        apnea_rows.extend(result.apneas)
        breaths_by_period[name] = result.breaths
        _maybe_emit_interactive(period, result)

    return metrics_rows, apnea_rows, breaths_by_period


# ---------------------------------------------------------------------------
# Experiment-level driver (loading from preprocessed CSVs)
# ---------------------------------------------------------------------------
def analyze_experiment(
    recordings: List[Recording],
    preprocessed_dir: str | Path,
    config: PlethConfig,
    *,
    interactive_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed CSVs from ``preprocessed_dir`` and run the two-pass
    breath analysis for every recording in ``recordings``.

    When ``interactive_dir`` is given, also writes the per-(file, period)
    plotly HTML trace plots used for visual QC of breath segmentation.

    Returns ``(breathing_df, apnea_df)`` ready to write to disk. The breathing
    DataFrame's column order matches old_results/breathing_analysis_results.csv.
    """
    preprocessed_dir = Path(preprocessed_dir)
    interactive_path = Path(interactive_dir) if interactive_dir is not None else None
    by_basename = {r.file_basename: r for r in recordings}

    # Group all *.csv in the preprocessed dir by basename + period.
    period_files: Dict[str, Dict[str, Path]] = {}
    for path in preprocessed_dir.glob("*.csv"):
        basename = _basename_from_filename(path.stem)
        period_name = _period_from_filename(path.stem)
        if period_name is None or basename not in by_basename:
            continue
        period_files.setdefault(basename, {})[period_name] = path

    metrics_rows: List[BreathMetrics] = []
    apnea_rows: List[ApneaEvent] = []

    for basename, files in period_files.items():
        recording = by_basename[basename]
        if recording.fs is None:
            # Recording.fs is set during preprocessing; if missing, infer from
            # the first CSV's mean dt.
            recording.fs = _infer_fs_from_csv(next(iter(files.values())))

        periods: Dict[str, Period] = {}
        for period_name, path in files.items():
            period = _load_period_csv(path, recording.fs)
            if period is not None:
                periods[period_name] = period

        m_rows, a_rows, _ = analyze_recording(
            recording, periods, config,
            interactive_dir=interactive_path,
        )
        metrics_rows.extend(m_rows)
        apnea_rows.extend(a_rows)

    breathing_df = _metrics_to_dataframe(metrics_rows)
    apnea_df = _apneas_to_dataframe(apnea_rows)
    return breathing_df, apnea_df


def _infer_fs_from_csv(path: Path) -> float:
    """Estimate sampling frequency from the time column's median dt."""
    df = pd.read_csv(path, usecols=["time"], nrows=64)
    if df.empty or len(df) < 2:
        return 1000.0  # plausible default
    dt = float(np.median(np.diff(df["time"].to_numpy(dtype=float))))
    if dt <= 0:
        return 1000.0
    return 1.0 / dt


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------
_METRICS_COLUMNS: Tuple[str, ...] = (
    "file_basename", "period", "period_duration_s", "num_breaths_detected",
    "mean_ttot_ms", "mean_frequency_bpm", "mean_ti_ms", "mean_te_ms",
    "mean_pif_centered_ml_s", "mean_pef_centered_ml_s", "mean_pif_to_pef_ml_s",
    "mean_tv_ml", "sigh_rate_per_min", "mean_sigh_duration_ms",
    "cov_instant_freq", "alternate_cov", "pif_to_pef_cov",
    "apnea_rate_per_min", "apnea_mean_ms",
    "apnea_spont_rate_per_min", "apnea_spont_mean_ms",
    "apnea_postsigh_rate_per_min", "apnea_postsigh_mean_ms",
)

_APNEA_COLUMNS: Tuple[str, ...] = (
    "file_basename", "period",
    "apnea_start_s_from_lid_closure", "apnea_start_s_from_period_start",
    "apnea_duration_ms",
)


def _metrics_to_dataframe(rows: Iterable[BreathMetrics]) -> pd.DataFrame:
    data = [
        {col: getattr(r, col) for col in _METRICS_COLUMNS} for r in rows
    ]
    return pd.DataFrame(data, columns=list(_METRICS_COLUMNS))


def _apneas_to_dataframe(rows: Iterable[ApneaEvent]) -> pd.DataFrame:
    data = [
        {col: getattr(r, col) for col in _APNEA_COLUMNS} for r in rows
    ]
    return pd.DataFrame(data, columns=list(_APNEA_COLUMNS))
