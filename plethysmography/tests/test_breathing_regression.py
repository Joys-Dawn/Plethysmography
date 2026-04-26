"""
Regression test: compare the new ``breathing_analysis_results.csv`` produced
by the experiment-1 pipeline against ``old_results/breathing_analysis_results.csv``.

Two periods are exempted from comparison because they reflect documented
intentional changes:

* **Habituation** — the new pipeline caps Habituation at 10 min (5-15 min);
  the old pipeline ran from start to first lid open then dropped the first
  5 min downstream. Period_duration and num_breaths therefore differ for
  most rows.
* **Ictal** — the new pipeline reads the seizure offset from the
  ``Seizure offset or equivalent time (sec from lid closure) -- all trials``
  column (decision 1 of the rewrite plan); the old pipeline used a different
  column ("Krystal's offset times" or earlier equivalent), which produces a
  different period boundary.

For Baseline / Immediate Postictal / Recovery rows we expect numerical
agreement within 5 %. Sigh- and apnea-related columns are intentionally
freed everywhere (sigh threshold is now baseline-derived per the docs;
apnea downstream metrics shift slightly when sigh classification changes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_PAIRS: tuple[tuple[str, Path, Path], ...] = (
    (
        "experiment 1 (LR vs HR)",
        Path("results/experiment 1 - LR vs HR comparison/breathing_analysis_results.csv"),
        Path("old_results/breathing_analysis_results.csv"),
    ),
    (
        "experiment 2 (chronic FFA)",
        Path("results/experiment 2 - chronic FFA vs vehicle/breathing_analysis_results.csv"),
        Path("old_results/breathing_analysis_results_FFA.csv"),
    ),
)

_EXEMPT_PERIODS = {"Habituation", "Ictal"}

_FREE_COLUMNS = {
    "sigh_rate_per_min", "mean_sigh_duration_ms",
    "apnea_rate_per_min", "apnea_mean_ms",
    "apnea_spont_rate_per_min", "apnea_spont_mean_ms",
    "apnea_postsigh_rate_per_min", "apnea_postsigh_mean_ms",
}


@pytest.mark.parametrize("label,new_csv,old_csv", _PAIRS)
def test_breathing_csv_matches_old_within_tolerance(
    label: str, new_csv: Path, old_csv: Path,
) -> None:
    if not (new_csv.exists() and old_csv.exists()):
        pytest.skip(f"missing breathing CSV(s) for {label}")
    new = pd.read_csv(new_csv)
    old = pd.read_csv(old_csv)

    new = new[~new["period"].isin(_EXEMPT_PERIODS)]
    old = old[~old["period"].isin(_EXEMPT_PERIODS)]

    key_cols = ["file_basename", "period"]
    new = new.set_index(key_cols).sort_index()
    old = old.set_index(key_cols).sort_index()

    common = new.index.intersection(old.index)
    assert len(common) > 0, f"{label}: no overlapping (file, period) keys"

    new_c = new.loc[common]
    old_c = old.loc[common]

    columns_to_check = [
        c for c in old_c.columns
        if c in new_c.columns and c not in _FREE_COLUMNS
        and pd.api.types.is_numeric_dtype(old_c[c])
    ]
    bad: list[str] = []
    for col in columns_to_check:
        old_vals = old_c[col].to_numpy(dtype=float)
        new_vals = new_c[col].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.where(np.abs(old_vals) > 1e-9, np.abs(old_vals), 1.0)
            rel_diff = np.abs(new_vals - old_vals) / denom
        mask = ~(np.isnan(rel_diff) | np.isnan(old_vals))
        if mask.any():
            max_diff = float(np.max(rel_diff[mask]))
            if max_diff > 0.05:
                bad.append(f"{col}: max_rel_diff={max_diff:.3g}")
    assert not bad, f"{label}: columns drifted >5%: " + "; ".join(bad)
