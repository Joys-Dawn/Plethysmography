"""
Experiment 1b driver (Item G): a 2-group developmental slice of experiment 1
— HR Scn1a+/- P19 vs LR Scn1a+/- P22.

Experiment 1's within / across / binned templates are hardwired to 4-cell
genotype x age / genotype x condition layouts with a P22-only "across", so
simply filtering a DataFrame through ``generate_publication_plots`` would
yield degenerate panels. Instead this driver is a *full mirror*: every exp1
plot family has an explicit 2-group analog.

It **reuses** experiment 1's artifacts (raw EDFs, preprocessed period CSVs,
and the breathing CSV) — it never re-preprocesses and never recomputes the
breathing parameters. The only recomputation is the bounded, cohort-limited
re-segmentation that the interactive / ictal-histogram / binned drivers do
from the cached preprocessed CSVs.

Plot-family mapping (exp1 -> exp1b):
  - within-period strip          -> ``plot_developmental_comparison``
  - across-periods timeseries    -> ``draw_developmental_timeseries`` (net-new)
  - postictal + ictal binned     -> ``plot_*_binned(condition_col=
                                     "developmental")`` (net-new variant)
  - interactive HTML / per-rec + population ictal histograms / trace plots
                                 -> the same drivers, restricted to the
                                    2-group basenames
  - stats                        -> developmental t-test sheet
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ..core.config import PlethConfig
from ..data_loading.data_log import (
    get_experiment_registry,
    load_data_log,
    load_recordings_for_experiment1b,
)
from ..stats import (
    prepare_breathing_data,
    run_statistics,
    write_stats_xlsx,
)
from ..visualization import (
    draw_developmental_timeseries,
    plot_developmental_comparison,
)
from ..visualization.binned_plots import plot_ictal_binned, plot_postictal_binned
from ._common import (
    DATA_ROOT,
    RESULTS_ROOT,
    analyze_all,
    baseline_median_ttot_by_basename,
    experiment_output_dirs,
    load_period_data_for_bins,
    metadata_for_bins,
    preprocess_all,
)


logger = logging.getLogger(__name__)
EXPERIMENT_ID = "1b"


def _filter_developmental(merged: pd.DataFrame) -> pd.DataFrame:
    """Restrict a prepared breathing frame to the two developmental groups:
    het HR P19 and het LR P22 (mirrors the cohort loader's rule, applied to
    the cleaned grouping columns)."""
    is_het = merged["genotype_clean"].astype(str).str.lower() == "het"
    hr_p19 = (merged["risk_clean"] == "high_risk") & (merged["age_clean"] == 19)
    lr_p22 = (merged["risk_clean"] == "low_risk") & (merged["age_clean"] == 22)
    return merged[is_het & (hr_p19 | lr_p22)].reset_index(drop=True)


def run(
    config: Optional[PlethConfig] = None,
    *,
    data_root: Path = DATA_ROOT,
    results_root: Path = RESULTS_ROOT,
    do_analyze: bool = True,
    do_stats: bool = True,
    do_plots: bool = True,
) -> None:
    """Run the experiment-1b developmental pipeline.

    ``do_analyze`` here means the bounded reuse pass (trace plots from cached
    CSVs + interactive HTML + per-recording / population ictal histograms for
    the 2-group cohort); it never re-reads EDFs or recomputes breathing
    parameters. Experiment 1 must have been run first (its breathing CSV and
    preprocessed CSVs are the inputs)."""
    config = config or PlethConfig()
    registry = get_experiment_registry(EXPERIMENT_ID)
    exp1_registry = get_experiment_registry(1)

    cohort_dir = data_root / exp1_registry["cohort_folder"]
    preprocessed_dir = cohort_dir / exp1_registry["preprocessed_subfolder"]
    interactive_root, pub_root = experiment_output_dirs(registry, results_root)
    _, exp1_pub = experiment_output_dirs(exp1_registry, results_root)

    recordings = load_recordings_for_experiment1b(data_root=data_root)
    logger.info("experiment 1b: %d developmental recordings", len(recordings))

    traces_dir = pub_root / "trace_plots"
    interactive_dir = interactive_root
    ictal_histograms_dir = pub_root / "Ictal_Histograms"
    population_ictal_dir = pub_root / "plots" / "Ictal_Histograms_population"

    exp1_breathing_csv = exp1_pub / "breathing_analysis_results.csv"
    if not exp1_breathing_csv.exists():
        logger.warning(
            "experiment 1b: %s not found — run `python run.py exp1` first.",
            exp1_breathing_csv,
        )
        return
    breathing_df = pd.read_csv(exp1_breathing_csv)

    if do_analyze:
        # Trace plots: skip_existing=True hits the preprocessed-CSV cache, so
        # this rebuilds only the period-overlay PNGs (no EDF re-read, no
        # re-filter) for the 2-group cohort.
        preprocess_all(
            recordings, config, preprocessed_dir,
            skip_existing=True, traces_dir=traces_dir,
        )
        # Interactive HTML + per-recording & pooled ictal histograms,
        # restricted to the 2-group basenames (only these recordings are
        # passed, so analyze_experiment ignores every other cached CSV).
        analyze_all(
            recordings, config, preprocessed_dir,
            interactive_dir=interactive_dir,
            ictal_histograms_dir=ictal_histograms_dir,
            population_ictal_dir=population_ictal_dir,
        )

    merged = prepare_breathing_data(breathing_df, load_data_log())
    merged = _filter_developmental(merged)
    if merged.empty:
        logger.warning("experiment 1b: no developmental rows after filtering")
        return

    if do_stats:
        rows = run_statistics(
            merged,
            condition_col="risk_clean",
            run_anova=False,
            run_gee=False,
            run_developmental=True,
            run_survival=False,
            run_across_periods=False,
        )
        stats_dir = pub_root / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        write_stats_xlsx(rows, stats_dir / "statistical_results.xlsx")

    if do_plots:
        plot_dir = pub_root / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        # within-style strip (HR P19 vs LR P22) — reuse the exp1 driver.
        plot_developmental_comparison(merged, plot_dir / "HR P19 vs LR P22")
        # across-periods timeseries — net-new 2-trace driver.
        draw_developmental_timeseries(merged, plot_dir / "Across time periods")
        # binned line plots — net-new 2-group "developmental" variant.
        postictal_data = load_period_data_for_bins(
            recordings, preprocessed_dir, "Immediate Postictal",
        )
        ictal_data = load_period_data_for_bins(
            recordings, preprocessed_dir, "Ictal",
        )
        bin_meta = metadata_for_bins(recordings)
        baseline_ttot = baseline_median_ttot_by_basename(
            recordings, preprocessed_dir, config,
        )
        plot_postictal_binned(
            postictal_data, bin_meta, plot_dir / "Postictal_Binned",
            condition_col="developmental",
            baseline_median_ttot_ms=baseline_ttot,
        )
        plot_ictal_binned(
            ictal_data, bin_meta, plot_dir / "Ictal_Binned",
            condition_col="developmental",
            baseline_median_ttot_ms=baseline_ttot,
        )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
