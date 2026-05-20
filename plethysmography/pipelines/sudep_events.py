"""
SUDEP fatal-seizure pipeline (Item H).

For each mouse with a parseable Column J ``(MM.SS-MM.SS)`` fatal-seizure
window, read its raw EDF (resolved across the exp1/exp2 raw folders),
slice the recording-relative ``[start_s, end_s]`` window, run the same
filter -> artifact-removal -> breath-segmentation -> per-period-metrics
chain the main pipeline uses (per-period thresholds, i.e.
``baseline_cache=None`` — an isolated fatal window has no Baseline), and
emit three artifacts:

  (a) one interactive plotly HTML per mouse — the windowed, post-filter
      trace with segmented breaths + apnea spans;
  (b) one breathing-parameter spreadsheet, one row per mouse, in the
      identical 29-column schema as the main breathing CSV (built via the
      shared ``_metrics_to_dataframe``);
  (c) one-column publication strip plots, one per breathing parameter,
      pooling **all** mice into a single ``"fatal seizures"`` group
      (jittered scatter + black mean +/- SEM); apnea-duration parameters
      carry the Item B 400 ms reference line.

Cohort = the ONE pooled group returned by
:func:`plethysmography.data_loading.data_log.load_sudep_event_cohort`
(built from Column J only). This deliberately **bypasses** the Item A
Column-G population filter: the near-SUDEP ``250304 4056 p22`` is G=1 yet
must be included. A mouse whose raw EDF is absent in both raw folders is
logged and skipped (the run continues).

Reuses exp1/exp2 raw EDFs only via bounded windowed reads — it never
re-preprocesses and never touches the main breathing CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Reuse the canonical 29-col metrics->frame builder so the SUDEP
# spreadsheet schema can never drift from the main breathing CSV.
from ..analysis.pipeline import (
    PeriodAnalysisResult,
    _metrics_to_dataframe,
    analyze_period,
)
from ..core.config import PlethConfig
from ..core.data_models import BreathMetrics, Period
from ..data_loading.data_log import (
    DEFAULT_DATA_LOG_PATH,
    SudepEvent,
    load_sudep_event_cohort,
)
from ..data_loading.edf_reader import read_edf_signal
from ..preprocessing.artifacts import remove_artifacts_from_period
from ..preprocessing.filtering import filter_period
# Canonical window->Period builder (same masking / period_start_time
# semantics slice_periods uses); reused so analyze_period's apnea timing
# stays identical to the main pipeline.
from ..preprocessing.periods import _make_period
from ..visualization._common import (
    APNEA_DURATION_PARAMS,
    add_apnea_duration_reference_line,
    display_label,
    filename_slug,
    make_axes,
    mean_sem,
    save_figure,
    within_style_params,
)
from ..visualization.interactive_plots import plot_breath_segmentation
from ..visualization.publication_plots import _detect_parameters
from ._common import DATA_ROOT, RESULTS_ROOT, experiment_output_dirs


logger = logging.getLogger(__name__)

_FATAL_PERIOD_NAME = "Fatal Seizure"
# Logical (non-experiment-numbered) registry stub so the Item E folder
# helper yields the two sibling SUDEP folders verbatim (its leading-token
# title-casing only fires on names that start with "experiment ").
_SUDEP_REGISTRY = {"results_folder": "SUDEP fatal seizures"}
_FIG_SIZE = (5.0, 7.0)
_JITTER = 0.15
_POINT_COLOR = "#800080"


def _build_window_period(
    event: SudepEvent,
    signal: np.ndarray,
    time_s: np.ndarray,
    fs: float,
) -> Optional[Period]:
    """Slice the recording-relative ``[start_s, end_s]`` window into a
    Period via the canonical builder (``time_s`` from
    :func:`read_edf_signal` is ``np.arange(N)/fs`` — recording-relative
    seconds, exactly the basis Column J's window is expressed in)."""
    return _make_period(
        _FATAL_PERIOD_NAME,
        event.start_s,
        event.end_s,
        signal,
        time_s,
        fs,
        start_inclusive=True,
        end_inclusive=True,
        lid_closure_time=float("nan"),
    )


def _analyze_event(
    event: SudepEvent, config: PlethConfig
) -> Optional[Tuple[PeriodAnalysisResult, Period]]:
    """read -> slice -> filter -> de-artifact -> analyze one fatal-seizure
    window. Returns ``(result, post-filter period)`` or ``None`` when the
    EDF is missing, the window is degenerate, or no breaths segment."""
    if event.edf_path is None:
        logger.warning(
            "SUDEP: raw EDF for %s not found in exp1/exp2 raw folders — "
            "skipping this mouse.",
            event.file_basename,
        )
        return None

    signal, time_s, fs = read_edf_signal(event.edf_path)
    period = _build_window_period(event, signal, time_s, fs)
    if period is None:
        rec_len = int(time_s[-1]) if time_s.size else 0
        logger.warning(
            "SUDEP: %s window [%.0f, %.0f]s is empty/degenerate "
            "(recording is %ds @ %.1f Hz) — skipping.",
            event.file_basename, event.start_s, event.end_s, rec_len, fs,
        )
        return None

    period = filter_period(period, config.filter)
    period = remove_artifacts_from_period(period, config.filter)
    result = analyze_period(
        period=period,
        file_basename=event.file_basename,
        config=config,
        baseline_cache=None,
    )
    if not result.breaths:
        logger.warning(
            "SUDEP: %s — no breaths segmented in the fatal window; "
            "skipping.",
            event.file_basename,
        )
        return None
    return result, period


def _emit_interactive(
    event: SudepEvent,
    result: PeriodAnalysisResult,
    period: Period,
    interactive_root: Path,
) -> Path:
    """One plotly HTML for the windowed fatal seizure (apnea spans drawn
    the same way :func:`analyze_recording` builds them)."""
    interactive_root.mkdir(parents=True, exist_ok=True)
    out_path = (
        interactive_root
        / f"{event.file_basename}_Fatal_Seizure_interactive_breaths.html"
    )
    ap_periods = [
        {
            "start": float(
                period.period_start_time
                + ap.apnea_start_s_from_period_start
            ),
            "end": float(
                period.period_start_time
                + ap.apnea_start_s_from_period_start
                + ap.apnea_duration_ms / 1000.0
            ),
            "duration": float(ap.apnea_duration_ms),
        }
        for ap in result.apneas
    ]
    plot_breath_segmentation(
        time_s=period.time_s,
        signal=period.signal,
        breaths=result.breaths,
        output_path=out_path,
        title=f"Fatal Seizure: {event.file_basename}",
        apnea_periods=ap_periods or None,
        allow_split=False,
        period_name=_FATAL_PERIOD_NAME,
    )
    return out_path


def plot_fatal_seizure_strips(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    parameters: Optional[Sequence[str]] = None,
) -> List[Path]:
    """One-column publication strip plot per breathing parameter: every
    mouse pooled into a single ``"fatal seizures"`` column (jittered
    scatter + black mean +/- SEM). Apnea-duration parameters carry the
    Item B 400 ms reference line; rate / burden never do.

    Defaults to the publication parameter set
    (:func:`publication_plots._detect_parameters` — ``*_no_apnea`` timing
    + imputed apnea duration) so the SUDEP strips stay parameter-aligned
    with the exp1/2/4 publication plots.
    """
    output_dir = Path(output_dir)
    if parameters is None:
        # Each mouse is a single isolated fatal-seizure window (no time-period
        # comparison), so use within-period semantics: emit BOTH V1
        # ``apnea_mean_ms`` (real durations, >=1-apnea mice only) and V2
        # ``apnea_mean_ms_imputed`` (imputed via 10-longest-breaths when 0
        # apneas). The current 4-mouse cohort all have apneas so V1 == V2,
        # but future cohorts may include zero-apnea windows where V1 drops
        # those mice while V2 keeps them via the imputed fallback.
        parameters = within_style_params(_detect_parameters(df))

    saved: List[Path] = []
    rng = np.random.default_rng(2)
    for param in parameters:
        if param not in df.columns:
            continue
        values = pd.to_numeric(df[param], errors="coerce").dropna()
        if values.empty:
            continue

        fig, ax = make_axes(figsize=_FIG_SIZE)
        xs = rng.uniform(-_JITTER, _JITTER, size=len(values))
        ax.scatter(
            xs, values, color=_POINT_COLOR, alpha=0.7, s=150,
            marker="o", edgecolors="black", linewidth=0.5,
        )
        mean, sem, _n = mean_sem(values)
        ax.errorbar(
            [0], [mean],
            yerr=[sem if np.isfinite(sem) else 0.0],
            fmt="_", color="black", capsize=10, markersize=50,
            markeredgewidth=6, zorder=10,
        )
        ax.set_ylabel(display_label(param), fontsize=40)
        ax.tick_params(axis="both", labelsize=32)
        ax.set_xticks([0])
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticklabels(["fatal seizures"], fontsize=32)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if param in APNEA_DURATION_PARAMS:
            add_apnea_duration_reference_line(ax)

        out_path = output_dir / f"Fatal_Seizure_{filename_slug(param)}.png"
        save_figure(fig, out_path)
        saved.append(out_path)
    return saved


def run(
    config: Optional[PlethConfig] = None,
    *,
    data_log_path: str | Path = DEFAULT_DATA_LOG_PATH,
    data_root: Path = DATA_ROOT,
    results_root: Path = RESULTS_ROOT,
) -> None:
    """Run the SUDEP fatal-seizure pipeline (Item H).

    Builds the cohort straight from Column J (no Column-G filter),
    windows + analyzes each mouse's raw EDF, and writes the interactive
    HTMLs, the one-row-per-mouse breathing spreadsheet, and the pooled
    ``"fatal seizures"`` strip plots into the Item E sibling folders."""
    config = config or PlethConfig()
    interactive_root, pub_root = experiment_output_dirs(
        _SUDEP_REGISTRY, Path(results_root)
    )

    events = load_sudep_event_cohort(
        data_log_path=data_log_path, data_root=data_root
    )
    logger.info("SUDEP: %d fatal-seizure events in cohort", len(events))
    if not events:
        logger.warning(
            "SUDEP: empty cohort (no parseable Column J windows) — "
            "nothing to do."
        )
        return

    metrics_rows: List[BreathMetrics] = []
    for event in events:
        analyzed = _analyze_event(event, config)
        if analyzed is None:
            continue
        result, period = analyzed
        _emit_interactive(event, result, period, interactive_root)
        metrics_rows.append(result.metrics)

    if not metrics_rows:
        logger.warning(
            "SUDEP: no events produced metrics (all EDFs missing or "
            "windows empty) — no spreadsheet / strips written."
        )
        return

    df = _metrics_to_dataframe(metrics_rows)
    pub_root.mkdir(parents=True, exist_ok=True)
    csv_path = pub_root / "fatal_seizure_breathing_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("SUDEP: wrote %s (%d mice)", csv_path, len(df))

    strips = plot_fatal_seizure_strips(df, pub_root / "plots")
    logger.info(
        "SUDEP: wrote %d strip plots under %s",
        len(strips), pub_root / "plots",
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
