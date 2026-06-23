"""
Experiment 1 driver: LR vs HR comparison (HR vs LR Scn1a, P19 vs P22).

Preprocesses every "HR vs LR" cohort recording, runs the two-pass breath
analysis, the full stats suite (10-sheet xlsx), and the publication plot
bundle.
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
from ..visualization import generate_publication_plots
from ..visualization._common import group_label
from ..visualization.colors import DEFAULT_PALETTE, DS_PALE_ORANGE, WT_PALE_BLUE
from ._common import (
    DATA_ROOT,
    RESULTS_ROOT,
    analyze_all,
    baseline_median_ttot_by_basename,
    experiment_output_dirs,
    load_period_data_for_bins,
    metadata_for_bins,
    preprocess_all,
    write_breathing_outputs,
)


_EXP1_POPULATION_PALETTE = {
    # P22: full LR vs HR palette
    group_label("WT", 22, "LR"): DEFAULT_PALETTE[("WT", "low_risk")],
    group_label("WT", 22, "HR"): DEFAULT_PALETTE[("WT", "high_risk")],
    group_label("Scn1a+/-", 22, "LR"): DEFAULT_PALETTE[("het", "low_risk")],
    group_label("Scn1a+/-", 22, "HR"): DEFAULT_PALETTE[("het", "high_risk")],
    # P19 cohort (HR only) reuses the P22 HR full-saturation colors so the
    # overlay reads the same hue per genotype across ages.
    group_label("WT", 19, "HR"): WT_PALE_BLUE,
    group_label("Scn1a+/-", 19, "HR"): DS_PALE_ORANGE,
}


logger = logging.getLogger(__name__)
EXPERIMENT_ID = 1


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
    """Run the full experiment-1 pipeline. Each ``do_*`` flag lets you replay
    a later stage without re-doing the slow earlier ones (preprocessing in
    particular, which reads every EDF)."""
    config = config or PlethConfig()
    registry = get_experiment_registry(EXPERIMENT_ID)
    cohort_dir = data_root / registry["cohort_folder"]
    preprocessed_dir = cohort_dir / registry["preprocessed_subfolder"]
    interactive_root, pub_root = experiment_output_dirs(registry, results_root)

    recordings = load_recordings_for_experiment(EXPERIMENT_ID, data_root=data_root)
    logger.info("experiment 1: %d recordings loaded", len(recordings))

    # Section 1 layout: plotly HTML -> interactive_root; everything else
    # lives directly under pub_root (no plots/ wrapper, no stats/ subfolder).
    traces_dir = pub_root / "Trace_plots"
    interactive_dir = interactive_root
    ictal_histograms_dir = pub_root / "Histograms_ictal_individual"
    population_ictal_dir = pub_root / "Histograms_ictal_population"

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
            ictal_histograms_dir=ictal_histograms_dir,
            population_ictal_dir=population_ictal_dir,
            population_palette=_EXP1_POPULATION_PALETTE,
            population_ictal_layout="exp1",
        )
        write_breathing_outputs(breathing_df, apnea_df, pub_root)
    elif (pub_root / "breathing_analysis_results.csv").exists():
        breathing_df = pd.read_csv(pub_root / "breathing_analysis_results.csv")

    if breathing_df.empty:
        logger.warning("breathing_df is empty; skipping stats and plots")
        return

    if do_stats:
        merged = prepare_breathing_data(breathing_df, load_data_log())
        rows = run_statistics(merged, condition_col="risk_clean")
        pub_root.mkdir(parents=True, exist_ok=True)
        write_stats_xlsx(rows, pub_root / "statistical_results.xlsx")

    if do_plots:
        merged = prepare_breathing_data(breathing_df, load_data_log())
        pub_root.mkdir(parents=True, exist_ok=True)
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
            merged, pub_root,
            condition_col="risk_clean",
            postictal_period_data=postictal_data,
            ictal_period_data=ictal_data,
            metadata_for_bins=bin_meta,
            baseline_median_ttot_ms=baseline_ttot,
        )
        # Section 1.3 exp1: HR P19 vs LR P22 developmental strips are
        # produced by experiment1b's dedicated pipeline; no duplication here.


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
