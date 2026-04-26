"""
Experiment 2 driver: chronic FFA vs vehicle.

Same shape as experiment 1 but the third factor is ``treatment_clean``
(FFA vs Vehicle) instead of ``risk_clean``.
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
    load_recordings_for_experiment,
)
from ..stats import (
    prepare_breathing_data,
    run_statistics,
    write_stats_xlsx,
)
from ..visualization import generate_publication_plots, plot_ffa_subgroups
from ._common import (
    DATA_ROOT,
    RESULTS_ROOT,
    analyze_all,
    baseline_median_ttot_by_basename,
    load_period_data_for_bins,
    metadata_for_bins,
    preprocess_all,
    write_breathing_outputs,
)


logger = logging.getLogger(__name__)
EXPERIMENT_ID = 2


def run(
    config: Optional[PlethConfig] = None,
    *,
    data_root: Path = DATA_ROOT,
    results_root: Path = RESULTS_ROOT,
    do_preprocess: bool = True,
    do_analyze: bool = True,
    do_stats: bool = True,
    do_plots: bool = True,
) -> None:
    config = config or PlethConfig()
    registry = get_experiment_registry(EXPERIMENT_ID)
    cohort_dir = data_root / registry["cohort_folder"]
    preprocessed_dir = cohort_dir / registry["preprocessed_subfolder"]
    results_dir = results_root / registry["results_folder"]

    recordings = load_recordings_for_experiment(EXPERIMENT_ID, data_root=data_root)
    logger.info("experiment 2: %d recordings loaded", len(recordings))

    traces_dir = results_dir / "trace_plots"
    interactive_dir = results_dir / "interactive"

    if do_preprocess:
        recordings = preprocess_all(
            recordings, config, preprocessed_dir,
            traces_dir=traces_dir,
        )

    breathing_df = pd.DataFrame()
    apnea_df = pd.DataFrame()
    if do_analyze:
        breathing_df, apnea_df = analyze_all(
            recordings, config, preprocessed_dir,
            interactive_dir=interactive_dir,
        )
        write_breathing_outputs(breathing_df, apnea_df, results_dir)
    elif (results_dir / "breathing_analysis_results.csv").exists():
        breathing_df = pd.read_csv(results_dir / "breathing_analysis_results.csv")

    if breathing_df.empty:
        logger.warning("breathing_df is empty; skipping stats and plots")
        return

    if do_stats:
        merged = prepare_breathing_data(breathing_df, load_data_log())
        rows = run_statistics(
            merged,
            condition_col="treatment_clean",
            condition_levels=("FFA", "Vehicle"),
        )
        stats_dir = results_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        write_stats_xlsx(rows, stats_dir / "statistical_results.xlsx")

    if do_plots:
        merged = prepare_breathing_data(breathing_df, load_data_log())
        plot_dir = results_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
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
        generate_publication_plots(
            merged, plot_dir,
            condition_col="treatment_clean",
            postictal_period_data=postictal_data,
            ictal_period_data=ictal_data,
            metadata_for_bins=bin_meta,
            baseline_median_ttot_ms=baseline_ttot,
        )
        plot_ffa_subgroups(merged, plot_dir / "FFA")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
