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
from ..visualization import (
    generate_publication_plots,
    plot_ffa_per_period_strips,
    plot_ffa_subgroups,
)
from ..visualization._common import group_label, treatment_word
from ..visualization.colors import TREATMENT_PALETTE, ffa_cell_color
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


def _chronic_population_palette():
    """Map every chronic ``group_label(geno, age, treatment_word(t))`` to its
    color from ``TREATMENT_PALETTE``. Covers both ages (the chronic cohort
    spans P19 + P22)."""
    out: dict[str, str] = {}
    for geno_display, geno_key in (("WT", "WT"), ("Scn1a+/-", "het")):
        for age in (19, 22):
            for treatment in ("Vehicle", "FFA"):
                label = group_label(geno_display, age, treatment_word(treatment))
                out[label] = ffa_cell_color(geno_key, age, treatment)
    return out


_EXP2_POPULATION_PALETTE = _chronic_population_palette()


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
    interactive_root, pub_root = experiment_output_dirs(registry, results_root)

    recordings = load_recordings_for_experiment(EXPERIMENT_ID, data_root=data_root)
    logger.info("experiment 2: %d recordings loaded", len(recordings))

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
            population_palette=_EXP2_POPULATION_PALETTE,
            population_ictal_layout="exp2",
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
        # P22 across strips / timeseries duplicate the by_age facet outputs;
        # keep only the within-age panels in the plain Time_period_* folders
        # and Timeseries_*_within in Time_periods_all/.
        generate_publication_plots(
            merged, pub_root,
            condition_col="treatment_clean",
            postictal_period_data=postictal_data,
            ictal_period_data=ictal_data,
            metadata_for_bins=bin_meta,
            baseline_median_ttot_ms=baseline_ttot,
            do_across_strips=False,
            do_timeseries_across=False,
        )
        # Section 1.3 exp2: P22 across-period FFA timeseries land in
        # Time_periods_all_by_{age,drug,genotype}/ and per-period facet
        # strips land in Time_period_<period>_by_{age,drug,genotype}/.
        plot_ffa_subgroups(merged, pub_root)
        plot_ffa_per_period_strips(merged, pub_root)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
