"""
Per-period GEE: ``parameter ~ C(genotype) * C(age)`` clustered on ``mouse_id``
with an exchangeable working correlation. Used for the longitudinal
"P19 vs P22 within group" comparison; the cohort filter (high-risk only for
exp 1, FFA-treated for exp 2, etc.) is taken from the caller via
``cohort_filter``.

Post-hocs are split into two kinds (matching old code):

  - **Genotype contrasts** at each age: independent Welch t-tests because the
    comparison is between two unrelated groups of mice.
  - **Age contrast within het**: GEE sub-model fitted on the het subset,
    handles unpaired data (mice that died at P19) properly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

from .helpers import capture_warnings, get_convergence_notes


CohortFilter = Callable[[pd.DataFrame], pd.DataFrame]


def _default_cohort_filter(data: pd.DataFrame) -> pd.DataFrame:
    """Default to high-risk mice only (matches old experiment-1 behavior)."""
    if "risk_clean" not in data.columns:
        return data
    return data[data["risk_clean"] == "high_risk"]


def perform_gee(
    data: pd.DataFrame,
    parameter: str,
    period: str,
    *,
    cohort_filter: CohortFilter = _default_cohort_filter,
) -> Optional[Dict[str, Any]]:
    """Fit a GEE on the cohort-filtered subset for one parameter at one period.

    Returns dict with ``model_result``, ``pvalues`` (genotype, age,
    Interaction), ``data`` for post-hocs, and convergence ``notes`` — or
    ``None`` if there is not enough data for a stable fit.
    """
    subset = cohort_filter(data)
    subset = subset[subset["period"] == period].copy()
    if subset.empty or subset[parameter].isna().all():
        return None
    subset = subset.dropna(subset=[parameter])
    if len(subset) < 4:
        return None
    if subset["genotype_clean"].nunique() < 2:
        return None

    formula = f"{parameter} ~ C(genotype_clean) * C(age_clean)"
    try:
        model = GEE.from_formula(
            formula,
            groups="mouse_id",
            data=subset,
            family=Gaussian(),
            cov_struct=Exchangeable(),
        )
        with capture_warnings() as caught:
            result = model.fit(cov_type="robust")
    except Exception:
        return None

    notes = get_convergence_notes(caught)
    pvalues: Dict[str, float] = {}
    pnames = result.params.index
    if "C(genotype_clean)[T.het]" in pnames:
        pvalues["genotype_clean"] = float(result.pvalues["C(genotype_clean)[T.het]"])
    if "C(age_clean)[T.22]" in pnames:
        pvalues["age_clean"] = float(result.pvalues["C(age_clean)[T.22]"])
    interaction = "C(genotype_clean)[T.het]:C(age_clean)[T.22]"
    if interaction in pnames:
        pvalues["Interaction"] = float(result.pvalues[interaction])

    return {
        "model_result": result,
        "pvalues": pvalues,
        "data": subset,
        "n_observations": int(len(subset)),
        "n_mice": int(subset["mouse_id"].nunique()),
        "formula": formula,
        "notes": notes,
    }


def perform_gee_posthoc(
    gee_result: Dict[str, Any],
    parameter: str,
) -> Dict[str, Dict[str, Any]]:
    """Run the genotype-at-each-age t-tests plus the age-within-het GEE
    sub-model. Comparisons emitted:

      - ``het_vs_WT_P19``, ``het_vs_WT_P22`` (independent t-tests)
      - ``P19_vs_P22_het`` (GEE sub-model on het only)
    """
    if not gee_result or "data" not in gee_result:
        return {}
    subset = gee_result["data"].copy()
    out: Dict[str, Dict[str, Any]] = {}

    for age in (19, 22):
        age_data = subset[subset["age_clean"] == age]
        het = age_data.loc[age_data["genotype_clean"] == "het", parameter].dropna()
        wt = age_data.loc[age_data["genotype_clean"] == "WT", parameter].dropna()
        if len(het) < 2 or len(wt) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(het, wt, nan_policy="omit", equal_var=False)
        if pd.isna(p_val) or pd.isna(t_stat):
            continue
        out[f"het_vs_WT_P{age}"] = {
            "mean_diff": float(het.mean() - wt.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "test_type": "independent",
            "n_het": int(len(het)),
            "n_wt": int(len(wt)),
            "mean_het": float(het.mean()),
            "mean_wt": float(wt.mean()),
            "notes": None,
        }

    het_data = subset[subset["genotype_clean"] == "het"].copy()
    if len(het_data) >= 2 and het_data["age_clean"].nunique() >= 2:
        try:
            sub_model = GEE.from_formula(
                f"{parameter} ~ C(age_clean)",
                groups="mouse_id",
                data=het_data,
                family=Gaussian(),
                cov_struct=Exchangeable(),
            )
            with capture_warnings() as caught:
                sub_result = sub_model.fit(cov_type="robust")
            sub_notes = get_convergence_notes(caught)
            age_term = "C(age_clean)[T.22]"
            if age_term in sub_result.params.index:
                z = sub_result.tvalues[age_term]
                p = sub_result.pvalues[age_term]
                if not pd.isna(z) and not pd.isna(p):
                    p19 = het_data.loc[het_data["age_clean"] == 19, parameter].mean()
                    p22 = het_data.loc[het_data["age_clean"] == 22, parameter].mean()
                    out["P19_vs_P22_het"] = {
                        "mean_diff": float(p19 - p22),
                        "z_statistic": float(z),
                        "p_value": float(p),
                        "test_type": "gee_submodel",
                        "n_obs": int(len(het_data)),
                        "n_mice": int(het_data["mouse_id"].nunique()),
                        "mean_p19": float(p19),
                        "mean_p22": float(p22),
                        "notes": sub_notes,
                    }
        except Exception:
            pass
    return out
