"""
Experiment 4 driver: P19 het — survivors vs eventual SUDEP.

Sources:
  - Cohort comes from the data log's ``Survivors vs eventual SUDEP (P19 trace)``
    column. Per the plan: 13 survivors + 10 SUDEP, all P19 het, drawn from
    experiment 1 (HR mice: 7 + 6) and experiment 2 (Vehicle-treated: 6 + 4).
  - Breathing CSVs come from experiments 1 and 2 results — this driver does
    not recompute breathing parameters. If those CSVs are missing, the user
    should run experiments 1 and 2 first.
  - Trace plots, per-recording / population ictal histograms, and binned
    postictal/ictal plots reuse each mouse's cached preprocessed CSVs from
    its source experiment (same bounded reuse pattern as experiment 1b).

Outputs:
  - A merged breathing CSV restricted to the cohort.
  - Survival t-tests (one row per parameter × period) in xlsx.
  - Publication plot bundle (within-period strip, across-period timeseries)
    showing survivor vs SUDEP for each parameter.
  - ``Trace_plots/``, ``Histograms_ictal_individual/``,
    ``Histograms_ictal_population/``, ``Binned_postictal/``, ``Binned_ictal/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ..core.config import PlethConfig
from ..data_loading.data_log import (
    get_experiment_registry,
    load_data_log,
    load_exp4_cohort,
)
from ..stats import (
    prepare_breathing_data,
    run_statistics,
    write_stats_xlsx,
)
from ..visualization import plot_survivor_publication
from ..visualization._common import group_label
from ..visualization.binned_plots import plot_ictal_binned, plot_postictal_binned
from ..visualization.colors import DS_PALE_ORANGE
from ._common import (
    DATA_ROOT,
    RESULTS_ROOT,
    analyze_all,
    baseline_median_ttot_by_basename_mixed,
    experiment_output_dirs,
    load_period_data_for_bins_mixed,
    metadata_for_bins,
    preprocessed_dir_for_recording,
    preprocess_all,
)


_EXP4_POPULATION_PALETTE = {
    group_label("Scn1a+/-", 19, "Survivor"): DS_PALE_ORANGE,
    group_label("Scn1a+/-", 19, "SUDEP"): "#FF0000",
}


logger = logging.getLogger(__name__)
EXPERIMENT_ID = 4


def run(
    config: Optional[PlethConfig] = None,
    *,
    data_root: Path = DATA_ROOT,
    results_root: Path = RESULTS_ROOT,
    exp1_breathing_csv: Optional[Path] = None,
    exp2_breathing_csv: Optional[Path] = None,
    do_analyze: bool = True,
    do_stats: bool = True,
    do_plots: bool = True,
) -> None:
    """Run the experiment-4 stats + plots."""
    config = config or PlethConfig()
    registry = get_experiment_registry(EXPERIMENT_ID)
    # Item E: exp4 produces no plotly HTML, so only pub_root is used.
    _interactive_root, pub_root = experiment_output_dirs(registry, results_root)
    pub_root.mkdir(parents=True, exist_ok=True)

    cohort = load_exp4_cohort(data_root=data_root)
    logger.info(
        "experiment 4: %d cohort recordings (%d survivors, %d SUDEP)",
        len(cohort),
        sum(1 for r in cohort if r.is_survivor),
        sum(1 for r in cohort if r.is_sudep),
    )
    if not cohort:
        logger.warning("experiment 4: empty cohort — nothing to do")
        return

    traces_dir = pub_root / "Trace_plots"
    ictal_histograms_dir = pub_root / "Histograms_ictal_individual"
    population_ictal_dir = pub_root / "Histograms_ictal_population"

    if do_analyze:
        exp2_folder = get_experiment_registry(2)["cohort_folder"]
        by_source: dict[str, list] = {get_experiment_registry(1)["cohort_folder"]: [], exp2_folder: []}
        for rec in cohort:
            by_source.setdefault(rec.cohort, []).append(rec)
        for source_recs in by_source.values():
            if not source_recs:
                continue
            preprocessed_dir = preprocessed_dir_for_recording(source_recs[0], data_root)
            preprocess_all(
                source_recs, config, preprocessed_dir,
                skip_existing=True, traces_dir=traces_dir,
            )
            analyze_all(
                source_recs, config, preprocessed_dir,
                interactive_dir=None,
                ictal_histograms_dir=ictal_histograms_dir,
                population_ictal_dir=population_ictal_dir,
                population_palette=_EXP4_POPULATION_PALETTE,
                population_ictal_layout="single",
            )

    breathing_df = _load_cohort_breathing(
        cohort, results_root,
        exp1_csv=exp1_breathing_csv, exp2_csv=exp2_breathing_csv,
    )
    if breathing_df.empty:
        logger.warning(
            "experiment 4: no breathing rows found — run experiments 1 and 2 first.",
        )
        return

    breathing_df.to_csv(pub_root / "breathing_analysis_results.csv", index=False)

    merged = prepare_breathing_data(breathing_df, load_data_log())
    merged = merged[merged["sudep_status"].isin(["survivor", "sudep"])]
    merged = merged[merged["age_clean"] == 19]
    if merged.empty:
        logger.warning("experiment 4: no P19 survivor/SUDEP rows after filtering")
        return

    if do_stats:
        rows = run_statistics(
            merged,
            run_anova=False,
            run_gee=False,
            run_developmental=False,
            run_across_periods=False,
            run_survival=True,
        )
        pub_root.mkdir(parents=True, exist_ok=True)
        write_stats_xlsx(rows, pub_root / "statistical_results.xlsx")

    if do_plots:
        pub_root.mkdir(parents=True, exist_ok=True)
        plot_survivor_publication(merged, pub_root)
        postictal_data = load_period_data_for_bins_mixed(
            cohort, data_root, "Immediate Postictal",
        )
        ictal_data = load_period_data_for_bins_mixed(
            cohort, data_root, "Ictal",
        )
        bin_meta = metadata_for_bins(cohort)
        baseline_ttot = baseline_median_ttot_by_basename_mixed(
            cohort, data_root, config,
        )
        plot_postictal_binned(
            postictal_data, bin_meta, pub_root / "Binned_postictal",
            condition_col="sudep_status",
            baseline_median_ttot_ms=baseline_ttot,
        )
        plot_ictal_binned(
            ictal_data, bin_meta, pub_root / "Binned_ictal",
            condition_col="sudep_status",
            baseline_median_ttot_ms=baseline_ttot,
        )


def _load_cohort_breathing(
    cohort: Sequence,
    results_root: Path,
    *,
    exp1_csv: Optional[Path],
    exp2_csv: Optional[Path],
) -> pd.DataFrame:
    """Load the breathing CSVs from experiments 1 and 2 and concatenate the
    rows whose ``file_basename`` is in the experiment-4 cohort.

    Item E moved the breathing CSV under each experiment's ``pub_root``
    (``Experiment N - … - publication plots and stats/``), so exp4 must
    resolve those new locations rather than the old flat folders."""
    if exp1_csv is None:
        _, exp1_pub = experiment_output_dirs(get_experiment_registry(1), results_root)
        exp1_csv = exp1_pub / "breathing_analysis_results.csv"
    if exp2_csv is None:
        _, exp2_pub = experiment_output_dirs(get_experiment_registry(2), results_root)
        exp2_csv = exp2_pub / "breathing_analysis_results.csv"
    cohort_basenames = {r.file_basename for r in cohort}

    dfs = []
    for csv_path in (exp1_csv, exp2_csv):
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df[df["file_basename"].isin(cohort_basenames)])
        else:
            logger.warning("missing breathing CSV: %s", csv_path)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
