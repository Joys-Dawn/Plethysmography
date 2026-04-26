"""
Experiment 4 driver: P19 het — survivors vs eventual SUDEP.

Sources:
  - Cohort comes from the data log's ``Survivors vs eventual SUDEP (P19 trace)``
    column. Per the plan: 13 survivors + 10 SUDEP, all P19 het, drawn from
    experiment 1 (HR mice: 7 + 6) and experiment 2 (Vehicle-treated: 6 + 4).
  - Breathing CSVs come from experiments 1 and 2 results — this driver does
    not preprocess or re-analyze EDFs. If those CSVs are missing, the user
    should run experiments 1 and 2 first.

Outputs:
  - A merged breathing CSV restricted to the cohort.
  - Survival t-tests (one row per parameter × period) in xlsx.
  - Publication plot bundle (within-period strip, across-period timeseries)
    showing survivor vs SUDEP for each parameter.
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
from ._common import DATA_ROOT, RESULTS_ROOT


logger = logging.getLogger(__name__)
EXPERIMENT_ID = 4


def run(
    config: Optional[PlethConfig] = None,
    *,
    data_root: Path = DATA_ROOT,
    results_root: Path = RESULTS_ROOT,
    exp1_breathing_csv: Optional[Path] = None,
    exp2_breathing_csv: Optional[Path] = None,
) -> None:
    """Run the experiment-4 stats + plots."""
    del config  # not used for exp 4 — it has no preprocess/analyze stage
    registry = get_experiment_registry(EXPERIMENT_ID)
    results_dir = results_root / registry["results_folder"]
    results_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_exp4_cohort(data_root=data_root)
    logger.info(
        "experiment 4: %d cohort recordings (%d survivors, %d SUDEP)",
        len(cohort),
        sum(1 for r in cohort if r.is_survivor),
        sum(1 for r in cohort if r.is_sudep),
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

    breathing_df.to_csv(results_dir / "breathing_analysis_results.csv", index=False)

    merged = prepare_breathing_data(breathing_df, load_data_log())
    merged = merged[merged["sudep_status"].isin(["survivor", "sudep"])]
    merged = merged[merged["age_clean"] == 19]

    rows = run_statistics(
        merged,
        run_anova=False,
        run_gee=False,
        run_developmental=False,
        run_across_periods=False,
        run_survival=True,
    )
    stats_dir = results_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    write_stats_xlsx(rows, stats_dir / "statistical_results.xlsx")

    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_survivor_publication(merged, plot_dir)


def _load_cohort_breathing(
    cohort: Sequence,
    results_root: Path,
    *,
    exp1_csv: Optional[Path],
    exp2_csv: Optional[Path],
) -> pd.DataFrame:
    """Load the breathing CSVs from experiments 1 and 2 and concatenate the
    rows whose ``file_basename`` is in the experiment-4 cohort."""
    if exp1_csv is None:
        exp1_csv = results_root / "experiment 1 - LR vs HR comparison" / "breathing_analysis_results.csv"
    if exp2_csv is None:
        exp2_csv = results_root / "experiment 2 - chronic FFA vs vehicle" / "breathing_analysis_results.csv"
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
