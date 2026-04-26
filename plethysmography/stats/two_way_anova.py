"""
Two-way ANOVA for P22 mice: ``parameter ~ C(genotype) * C(condition)``.

The condition axis is configurable via ``condition_col`` so the same machinery
can run experiment 1 (``risk_clean`` ∈ {high_risk, low_risk}) or experiment 2
(``treatment_clean`` ∈ {FFA, Vehicle}).

Model fit uses statsmodels OLS with HC3 heteroskedasticity-robust covariance.
The post-hoc step performs three Welch's t-tests on the four group cells:

  - het vs WT at low_risk (or Vehicle)
  - het vs WT at high_risk (or FFA)
  - low_risk vs high_risk within het (or Vehicle vs FFA within het)

These are gated upstream — only run when the corrected interaction p is
significant.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def perform_two_way_anova(
    data: pd.DataFrame,
    parameter: str,
    period: str,
    *,
    condition_col: str = "risk_clean",
    condition_levels: tuple[str, str] = ("high_risk", "low_risk"),
) -> Optional[Dict[str, Any]]:
    """Fit a 2-way ANOVA on P22 mice for one parameter at one period.

    Returns a dict with the ANOVA table, the fitted (HC3-robust) model, the
    filtered DataFrame, and the observation count — or ``None`` when there is
    not enough data (fewer than 4 non-NaN rows, or one of the factor levels
    missing). The ``data`` payload lets the caller run the post-hoc step
    without re-filtering.
    """
    p22_data = data[(data["age_clean"] == 22) & (data["period"] == period)].copy()
    if p22_data.empty or p22_data[parameter].isna().all():
        return None
    p22_data = p22_data.dropna(subset=[parameter])
    if len(p22_data) < 4:
        return None
    if p22_data["genotype_clean"].nunique() < 2:
        return None
    if p22_data[condition_col].nunique() < 2:
        return None

    formula = f"{parameter} ~ C(genotype_clean) * C({condition_col})"
    try:
        model = ols(formula, data=p22_data).fit()
        model = model.get_robustcov_results(cov_type="HC3")
        anova_table = anova_lm(model, typ=2)
    except Exception:
        return None

    return {
        "anova_table": anova_table,
        "model": model,
        "data": p22_data,
        "n_observations": len(p22_data),
        "condition_col": condition_col,
        "condition_levels": tuple(condition_levels),
    }


def perform_anova_posthoc(
    anova_result: Dict[str, Any],
    parameter: str,
) -> Dict[str, Dict[str, Any]]:
    """Run the three pairwise post-hoc Welch t-tests on the 2x2 cells.

    Comparison labels are ``f"{group1}_vs_{group2}"`` with groups encoded as
    ``f"{genotype}_{condition_value}"`` (e.g. ``het_low_risk_vs_WT_low_risk``).
    Only comparisons with at least 2 observations per cell are emitted.
    """
    if not anova_result or "data" not in anova_result:
        return {}
    p22_data = anova_result["data"].copy()
    condition_col = anova_result.get("condition_col", "risk_clean")
    cond_low, cond_high = anova_result.get("condition_levels", ("high_risk", "low_risk"))

    p22_data["group"] = (
        p22_data["genotype_clean"].astype(str) + "_" + p22_data[condition_col].astype(str)
    )

    comparisons = [
        (f"het_{cond_high}", f"WT_{cond_high}"),  # het vs WT (high or low — first listed)
        (f"het_{cond_low}", f"WT_{cond_low}"),
        (f"het_{cond_high}", f"het_{cond_low}"),
    ]

    results: Dict[str, Dict[str, Any]] = {}
    for g1, g2 in comparisons:
        d1 = p22_data.loc[p22_data["group"] == g1, parameter].dropna()
        d2 = p22_data.loc[p22_data["group"] == g2, parameter].dropna()
        if len(d1) < 2 or len(d2) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(d1, d2, nan_policy="omit", equal_var=False)
        if pd.isna(p_val) or pd.isna(t_stat):
            continue
        results[f"{g1}_vs_{g2}"] = {
            "mean_diff": float(d1.mean() - d2.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "n1": int(len(d1)),
            "n2": int(len(d2)),
            "mean1": float(d1.mean()),
            "mean2": float(d2.mean()),
        }
    return results
