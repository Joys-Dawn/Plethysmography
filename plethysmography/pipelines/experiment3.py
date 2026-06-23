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
sheets that fill are: "P22 across groups ANOVA" (+ posthocs) and the
independent "P22 groups and periods GEE" (+ posthocs). Per-period strip
plots use the P22 ``_*_across.png`` layout only (no within-age strips or
``Timeseries_*_within.png``); the per-period GEE and developmental t-tests
and the dependent across-periods GEE are disabled because the cohort is
P22-only.
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
from ..visualization._common import group_label, treatment_word
from ..visualization.colors import ACUTE_FFA_PALETTE, ffa_cell_color
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


def _acute_population_palette():
    """Map each acute P22 ``group_label(geno, 22, treatment_word(t))`` to its
    color from ``ACUTE_FFA_PALETTE``. Acute cohort is P22-only."""
    out: dict[str, str] = {}
    for geno_display, geno_key in (("WT", "WT"), ("Scn1a+/-", "het")):
        for treatment in ("Vehicle", "FFA"):
            label = group_label(geno_display, 22, treatment_word(treatment))
            out[label] = ffa_cell_color(geno_key, 22, treatment)
    return out


_EXP3_POPULATION_PALETTE = _acute_population_palette()


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
            population_palette=_EXP3_POPULATION_PALETTE,
            population_ictal_layout="exp3",
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
            run_gee=False,
            run_developmental=False,
            run_across_periods_dependent=False,
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
        # Section 1.3 exp3: skip within-age strips and within timeseries
        # (acute cohort is P22-only); still emit per-period _across strips.
        generate_publication_plots(
            merged, pub_root,
            condition_col="treatment_clean",
            postictal_period_data=postictal_data,
            ictal_period_data=ictal_data,
            metadata_for_bins=bin_meta,
            baseline_median_ttot_ms=baseline_ttot,
            palette=ACUTE_FFA_PALETTE,
            do_within=False,
            do_timeseries_within=False,
        )
        # plot_ffa_subgroups / plot_ffa_per_period_strips are intentionally
        # NOT called for exp3 (the acute single-age cohort would render
        # degenerate / redundant panels — Section 1.3 exp3).


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
