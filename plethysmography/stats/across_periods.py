"""
Across-periods 3-way GEE: ``parameter ~ C(genotype) * C(other) * C(period)``.

Two designs:

  - **Independent** — P22 only, third factor = ``risk_clean`` (or
    ``treatment_clean`` for exp 2). All effects are between-subjects.
  - **Dependent** — high-risk only (or FFA-treated), third factor =
    ``age_clean``. Mice that died at P19 are unpaired; GEE handles this.

Period (4 levels) requires joint Wald tests over the 3 dummy coefficients to
get a single omnibus p-value; the helper :func:`_joint_wald` wraps that.

Post-hocs compare each non-Baseline period back to Baseline. When no
interaction with period is significant (after FDR correction), the post-hoc
runs once on pooled data; otherwise it runs separately per group cell.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

from .helpers import capture_warnings, get_convergence_notes


CohortFilter = Callable[[pd.DataFrame], pd.DataFrame]
_PERIODS_TO_COMPARE: Tuple[str, ...] = ("Ictal", "Immediate Postictal", "Recovery")


def _joint_wald(result, terms: Sequence[str]) -> float:
    """Joint Wald test over all listed parameter names; returns the scalar
    p-value or NaN if the test cannot be constructed."""
    if not terms:
        return float("nan")
    try:
        return float(result.wald_test(list(terms), scalar=True).pvalue)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Independent design: P22, geno × condition × period
# ---------------------------------------------------------------------------
def perform_across_periods_independent_gee(
    data: pd.DataFrame,
    parameter: str,
    *,
    condition_col: str = "risk_clean",
) -> Optional[Dict[str, Any]]:
    """Fit the 3-way GEE on P22 mice across all 4 periods. Returns a dict with
    p-values for the 3 main effects, 3 two-way interactions, and the 3-way
    interaction (joint Wald where the factor has >2 levels), or ``None`` when
    there isn't enough data."""
    subset = data[data["age_clean"] == 22].copy()
    if subset.empty or subset[parameter].isna().all():
        return None
    subset = subset.dropna(subset=[parameter])
    if len(subset) < 8:
        return None
    if subset["genotype_clean"].nunique() < 2:
        return None
    if subset[condition_col].nunique() < 2:
        return None

    formula = f"{parameter} ~ C(genotype_clean) * C({condition_col}) * C(period)"
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
    pnames = list(result.params.index)
    pvalues: Dict[str, float] = {}

    geno_term = "C(genotype_clean)[T.het]"
    if geno_term in pnames:
        pvalues["genotype_clean"] = float(result.pvalues[geno_term])
    cond_term_candidates = [n for n in pnames if n.startswith(f"C({condition_col})[T.")
                            and ":" not in n]
    if len(cond_term_candidates) == 1:
        pvalues[condition_col] = float(result.pvalues[cond_term_candidates[0]])
    elif cond_term_candidates:
        pvalues[condition_col] = _joint_wald(result, cond_term_candidates)

    period_main = [n for n in pnames if n.startswith("C(period)") and ":" not in n]
    if period_main:
        pvalues["period"] = _joint_wald(result, period_main)

    geno_cond_terms = [
        n for n in pnames
        if "genotype_clean" in n and condition_col in n and "period" not in n
    ]
    if len(geno_cond_terms) == 1:
        pvalues[f"genotype_x_{_short(condition_col)}"] = float(result.pvalues[geno_cond_terms[0]])
    elif geno_cond_terms:
        pvalues[f"genotype_x_{_short(condition_col)}"] = _joint_wald(result, geno_cond_terms)

    geno_period_terms = [
        n for n in pnames
        if "genotype_clean" in n and "period" in n and condition_col not in n
    ]
    if geno_period_terms:
        pvalues["genotype_x_period"] = _joint_wald(result, geno_period_terms)

    cond_period_terms = [
        n for n in pnames
        if condition_col in n and "period" in n and "genotype_clean" not in n
    ]
    if cond_period_terms:
        pvalues[f"{_short(condition_col)}_x_period"] = _joint_wald(result, cond_period_terms)

    three_way_terms = [
        n for n in pnames
        if "genotype_clean" in n and condition_col in n and "period" in n
    ]
    if three_way_terms:
        pvalues[f"genotype_x_{_short(condition_col)}_x_period"] = _joint_wald(result, three_way_terms)

    return {
        "model_result": result,
        "pvalues": pvalues,
        "data": subset,
        "n_observations": int(len(subset)),
        "n_mice": int(subset["mouse_id"].nunique()),
        "formula": formula,
        "notes": notes,
        "condition_col": condition_col,
    }


def _short(condition_col: str) -> str:
    """Map the data column name to the short label used in p-value keys (so
    callers see ``genotype_x_risk`` for ``risk_clean``)."""
    if condition_col == "risk_clean":
        return "risk"
    if condition_col == "treatment_clean":
        return "treatment"
    return condition_col


# ---------------------------------------------------------------------------
# Dependent design: HR / treatment-fixed, geno × age × period
# ---------------------------------------------------------------------------
def _default_dep_filter(data: pd.DataFrame) -> pd.DataFrame:
    if "risk_clean" not in data.columns:
        return data
    return data[data["risk_clean"] == "high_risk"]


def perform_across_periods_dependent_gee(
    data: pd.DataFrame,
    parameter: str,
    *,
    cohort_filter: CohortFilter = _default_dep_filter,
) -> Optional[Dict[str, Any]]:
    """Fit the 3-way GEE on the cohort-filtered subset across all 4 periods,
    with age (P19 vs P22) as the third factor. Same shape of return as the
    independent version."""
    subset = cohort_filter(data).copy()
    if subset.empty or subset[parameter].isna().all():
        return None
    subset = subset.dropna(subset=[parameter])
    if len(subset) < 8:
        return None
    if subset["genotype_clean"].nunique() < 2:
        return None

    formula = f"{parameter} ~ C(genotype_clean) * C(age_clean) * C(period)"
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
    pnames = list(result.params.index)
    pvalues: Dict[str, float] = {}

    if "C(genotype_clean)[T.het]" in pnames:
        pvalues["genotype_clean"] = float(result.pvalues["C(genotype_clean)[T.het]"])
    if "C(age_clean)[T.22]" in pnames:
        pvalues["age_clean"] = float(result.pvalues["C(age_clean)[T.22]"])

    period_main = [n for n in pnames if n.startswith("C(period)") and ":" not in n]
    if period_main:
        pvalues["period"] = _joint_wald(result, period_main)

    geno_age_term = "C(genotype_clean)[T.het]:C(age_clean)[T.22]"
    if geno_age_term in pnames:
        pvalues["genotype_x_age"] = float(result.pvalues[geno_age_term])

    geno_period_terms = [
        n for n in pnames
        if "genotype_clean" in n and "period" in n and "age_clean" not in n
    ]
    if geno_period_terms:
        pvalues["genotype_x_period"] = _joint_wald(result, geno_period_terms)

    age_period_terms = [
        n for n in pnames
        if "age_clean" in n and "period" in n and "genotype_clean" not in n
    ]
    if age_period_terms:
        pvalues["age_x_period"] = _joint_wald(result, age_period_terms)

    three_way_terms = [
        n for n in pnames
        if "genotype_clean" in n and "age_clean" in n and "period" in n
    ]
    if three_way_terms:
        pvalues["genotype_x_age_x_period"] = _joint_wald(result, three_way_terms)

    return {
        "model_result": result,
        "pvalues": pvalues,
        "data": subset,
        "n_observations": int(len(subset)),
        "n_mice": int(subset["mouse_id"].nunique()),
        "formula": formula,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Post-hocs
# ---------------------------------------------------------------------------
def _fit_period_only_gee(
    subset: pd.DataFrame, parameter: str,
) -> Tuple[Any, Optional[str]]:
    """Helper: fit ``parameter ~ C(period)`` clustered on mouse_id and return
    (result, notes)."""
    model = GEE.from_formula(
        f"{parameter} ~ C(period)",
        groups="mouse_id",
        data=subset,
        family=Gaussian(),
        cov_struct=Exchangeable(),
    )
    with capture_warnings() as caught:
        result = model.fit(cov_type="robust")
    return result, get_convergence_notes(caught)


def _period_vs_baseline_rows(
    sub_result, parameter: str, group_label: str,
    subset: pd.DataFrame, notes: Optional[str],
    test_type: str = "gee_submodel",
) -> Dict[str, Dict[str, Any]]:
    """Build the per-period vs Baseline rows from a fitted period-only GEE.
    Comparison labels are ``f"{group_label}_{period}_vs_Baseline"``."""
    out: Dict[str, Dict[str, Any]] = {}
    periods_present = set(subset["period"].unique())
    if "Baseline" not in periods_present:
        return out
    base_mean = float(subset.loc[subset["period"] == "Baseline", parameter].mean())
    for period in _PERIODS_TO_COMPARE:
        if period not in periods_present:
            continue
        term = f"C(period)[T.{period}]"
        if term not in sub_result.params.index:
            continue
        z = sub_result.tvalues[term]
        p = sub_result.pvalues[term]
        if pd.isna(z) or pd.isna(p):
            continue
        period_mean = float(subset.loc[subset["period"] == period, parameter].mean())
        comparison = f"{group_label}_{period}_vs_Baseline"
        out[comparison] = {
            "mean_diff": float(period_mean - base_mean),
            "z_statistic": float(z),
            "p_value": float(p),
            "test_type": test_type,
            "n_obs": int(len(subset)),
            "mean_baseline": base_mean,
            "mean_period": period_mean,
            "group": group_label,
            "period": period,
            "notes": notes,
        }
    return out


def perform_across_periods_independent_posthoc(
    gee_result: Dict[str, Any],
    parameter: str,
    *,
    any_interaction_significant: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """When ``any_interaction_significant=True``, emit one post-hoc per (group
    cell × period vs Baseline). Otherwise emit a single pooled set of
    period-vs-Baseline contrasts. Group cells use the labels
    ``f"{genotype}_{condition}"``."""
    if not gee_result or "data" not in gee_result:
        return {}
    subset = gee_result["data"].copy()
    condition_col = gee_result.get("condition_col", "risk_clean")
    out: Dict[str, Dict[str, Any]] = {}
    if subset.empty:
        return out

    if not any_interaction_significant:
        if "Baseline" not in subset["period"].unique() or len(subset) < 4:
            return {}
        try:
            sub_result, notes = _fit_period_only_gee(subset, parameter)
        except Exception:
            return {}
        out.update(_period_vs_baseline_rows(
            sub_result, parameter, "pooled", subset, notes, test_type="gee_pooled",
        ))
        return out

    for genotype in ("WT", "het"):
        for cond in subset[condition_col].dropna().unique():
            cell = subset[
                (subset["genotype_clean"] == genotype) & (subset[condition_col] == cond)
            ].copy()
            if len(cell) < 4 or "Baseline" not in cell["period"].unique():
                continue
            try:
                sub_result, notes = _fit_period_only_gee(cell, parameter)
            except Exception:
                continue
            out.update(_period_vs_baseline_rows(
                sub_result, parameter, f"{genotype}_{cond}", cell, notes,
            ))
    return out


def perform_across_periods_dependent_posthoc(
    gee_result: Dict[str, Any],
    parameter: str,
    *,
    any_interaction_significant: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Same as the independent version but cells = (genotype × age) instead of
    (genotype × condition); group labels are ``f"{genotype}_P{age}"``."""
    if not gee_result or "data" not in gee_result:
        return {}
    subset = gee_result["data"].copy()
    out: Dict[str, Dict[str, Any]] = {}
    if subset.empty:
        return out

    if not any_interaction_significant:
        if "Baseline" not in subset["period"].unique() or len(subset) < 4:
            return {}
        try:
            sub_result, notes = _fit_period_only_gee(subset, parameter)
        except Exception:
            return {}
        out.update(_period_vs_baseline_rows(
            sub_result, parameter, "pooled", subset, notes, test_type="gee_pooled",
        ))
        return out

    for genotype in ("WT", "het"):
        for age in (19, 22):
            cell = subset[
                (subset["genotype_clean"] == genotype) & (subset["age_clean"] == age)
            ].copy()
            if len(cell) < 4 or "Baseline" not in cell["period"].unique():
                continue
            try:
                sub_result, notes = _fit_period_only_gee(cell, parameter)
            except Exception:
                continue
            out.update(_period_vs_baseline_rows(
                sub_result, parameter, f"{genotype}_P{age}", cell, notes,
            ))
    return out
