"""
Survival t-test: do P19 het breathing parameters predict whether the mouse
survives to P22?

Two classification strategies (controlled by ``classification``):

  - ``"sudep_column"`` — use the explicit ``sudep_status`` column (the
    "Survivors vs eventual SUDEP (P19 trace)" column from the data log).
    Mice marked ``"sudep"`` are non-survivors; ``"survivor"`` are survivors.
    Cleanest source of truth, but only populated for the experiment 4 cohort.
  - ``"p22_presence"`` — fall back to the old code's heuristic: a P19 mouse is
    a survivor iff it also has a P22 row in ``data``. Used when the SUDEP
    column is not populated (e.g. early experiment-1 runs).

Default is ``"sudep_column"`` with automatic fallback to ``"p22_presence"`` if
the column has no values for the het cohort being tested.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd
from scipy import stats


def perform_survival_comparison(
    data: pd.DataFrame,
    parameter: str,
    period: str,
    *,
    classification: str = "auto",
) -> Optional[Dict[str, Any]]:
    """Welch t-test on P19 het mice, survivors vs non-survivors, for one
    (parameter, period). Returns ``None`` if either group has <2 observations.
    """
    het = data[data["genotype_clean"] == "het"]
    if het.empty:
        return None

    survivors, non_survivors = _classify_survival(het, classification)
    if len(survivors) < 2 or len(non_survivors) < 2:
        return None

    p19_period = het[(het["age_clean"] == 19) & (het["period"] == period)]
    if p19_period.empty or p19_period[parameter].isna().all():
        return None

    surv = p19_period[p19_period["mouse_id"].isin(survivors)][parameter].dropna()
    non_surv = p19_period[p19_period["mouse_id"].isin(non_survivors)][parameter].dropna()
    if len(surv) < 2 or len(non_surv) < 2:
        return None

    try:
        t_stat, p_value = stats.ttest_ind(surv, non_surv, nan_policy="omit", equal_var=False)
    except Exception:
        return None
    if pd.isna(p_value) or pd.isna(t_stat):
        return None
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "survivors_n": int(len(surv)),
        "non_survivors_n": int(len(non_surv)),
        "survivors_mean": float(surv.mean()),
        "non_survivors_mean": float(non_surv.mean()),
        "survivors_std": float(surv.std(ddof=1)) if len(surv) > 1 else float("nan"),
        "non_survivors_std": float(non_surv.std(ddof=1)) if len(non_surv) > 1 else float("nan"),
        "survivor_mouse_ids": sorted(survivors),
        "non_survivor_mouse_ids": sorted(non_survivors),
    }


def _classify_survival(
    het_data: pd.DataFrame, classification: str,
) -> tuple[Set[str], Set[str]]:
    """Resolve the (survivors, non_survivors) mouse-id sets for the het cohort.
    For ``"auto"`` mode, prefer the SUDEP column and fall back to the P22
    presence heuristic if it yields no labels for either group."""
    if classification not in {"auto", "sudep_column", "p22_presence"}:
        raise ValueError(f"unknown classification mode: {classification!r}")

    if classification in {"auto", "sudep_column"} and "sudep_status" in het_data.columns:
        survivors = set(
            het_data.loc[het_data["sudep_status"] == "survivor", "mouse_id"].dropna().astype(str)
        )
        non_survivors = set(
            het_data.loc[het_data["sudep_status"] == "sudep", "mouse_id"].dropna().astype(str)
        )
        if survivors and non_survivors:
            return survivors, non_survivors
        if classification == "sudep_column":
            return survivors, non_survivors  # caller must accept empties

    p22_mice = set(
        het_data.loc[het_data["age_clean"] == 22, "mouse_id"].dropna().astype(str)
    )
    p19_mice = set(
        het_data.loc[het_data["age_clean"] == 19, "mouse_id"].dropna().astype(str)
    )
    survivors = p19_mice & p22_mice
    non_survivors = p19_mice - p22_mice
    return survivors, non_survivors
