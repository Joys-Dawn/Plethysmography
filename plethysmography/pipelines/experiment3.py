"""
Experiment 3 driver: acute FFA vs vehicle.

Mirrors the experiment-2 (chronic FFA) flow exactly:
``preprocess_all -> analyze_all -> write_breathing_outputs ->
prepare_breathing_data -> run_statistics -> write_stats_xlsx ->
generate_publication_plots``. The factorial design is genotype x treatment
at P22 only; ``condition_col="treatment_clean"`` with
``condition_levels=("FFA", "Vehicle")`` is identical to exp2.

Two intentional design differences vs experiment 2:

  1. **Palette swap.** ``generate_publication_plots`` is called with
     ``palette=ACUTE_FFA_PALETTE`` so the FFA traces render in the pink
     family (``#FFB6C1`` / ``#C71585``) rather than the chronic
     gray/purple. The Vehicle pair (gray) is shared across acute and
     chronic so Vehicle reads the same.
  2. **No ``plot_ffa_subgroups`` call.** That driver's three subfolders
     (``By_age`` / ``By_drug`` / ``By_genotype``) were designed for the
     chronic P19+P22 cohort. The acute cohort is P22-only — those
     subfolders would either replay the main ``Across_periods`` plot
     (``By_age/_P22``) or render sparse 2-trace panels (the P19 half is
     empty). Omitting the call is the cleanest defense against the
     Item G "ship degenerate panels" risk.

Stats writer auto-skips empty sheets. For the acute single-age design the
sheets that fill are: "P22 across groups ANOVA" (+ posthocs) and "P22
groups and periods GEE" (+ posthocs). Convergence notes for the 3-way
GEE land in the ``notes`` column of the All Results sheet — review per
parameter; the smaller WT FFA / het Vehicle cells (n=4 each) sit at the
lower end of stable identifiability for some apnea parameters.
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
from ..visualization.colors import ACUTE_FFA_PALETTE
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


logger = logging.getLogger(__name__)
EXPERIMENT_ID = 3


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
    interactive_root, pub_root = experiment_output_dirs(registry, results_root)

    recordings = load_recordings_for_experiment(EXPERIMENT_ID, data_root=data_root)
    logger.info("experiment 3: %d recordings loaded", len(recordings))
    if not recordings:
        logger.warning(
            "experiment 3: no recordings — nothing to do. "
            "Check the data log for 'FFA vs vehicle - acute' rows."
        )
        return

    # Item E: plotly HTML -> interactive_root; everything else -> pub_root.
    traces_dir = pub_root / "trace_plots"
    interactive_dir = interactive_root
    ictal_histograms_dir = pub_root / "Ictal_Histograms"
    # Item F: pooled per-group ictal histograms live under plots/.
    population_ictal_dir = pub_root / "plots" / "Ictal_Histograms_population"

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
        )
        write_breathing_outputs(breathing_df, apnea_df, pub_root)
    elif (pub_root / "breathing_analysis_results.csv").exists():
        breathing_df = pd.read_csv(pub_root / "breathing_analysis_results.csv")

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
        stats_dir = pub_root / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        write_stats_xlsx(rows, stats_dir / "statistical_results.xlsx")

    if do_plots:
        merged = prepare_breathing_data(breathing_df, load_data_log())
        plot_dir = pub_root / "plots"
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
            palette=ACUTE_FFA_PALETTE,
        )
        # plot_ffa_subgroups is intentionally NOT called for exp3 (acute
        # single-age cohort would render degenerate / redundant panels).


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
