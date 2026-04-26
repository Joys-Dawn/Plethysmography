"""
Two-pass statistical analysis driver.

Pass 1 — fit all main models for every (parameter, period) combination
(ANOVA, GEE, developmental t-test, survival t-test, across-periods independent
GEE, across-periods dependent GEE) and emit one row per individual effect (so
a 2-way ANOVA emits 3 rows: genotype, condition, interaction).

Apply BH-FDR correction grouped by ``(category, test_type, period, effect)``
so each parameter category is its own family for each effect type.

Pass 2 — for every main row whose corrected p crosses the alpha threshold and
which represents a gating effect (ANOVA / GEE interaction, or any
period-related across-periods effect), run the appropriate post-hoc. Apply BH
FDR correction to the post-hocs within parameter (mirroring old code).

The output is a flat list of result-row dicts ready for
:func:`plethysmography.stats.writer.write_stats_xlsx`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .across_periods import (
    perform_across_periods_dependent_gee,
    perform_across_periods_dependent_posthoc,
    perform_across_periods_independent_gee,
    perform_across_periods_independent_posthoc,
)
from .developmental import perform_developmental_comparison
from .families import define_parameter_categories, get_parameter_to_category
from .gee import perform_gee, perform_gee_posthoc
from .helpers import (
    compute_group_summaries,
    format_mean_sem_n_scalar,
)
from .posthoc import (
    clean_comparison_name,
    posthoc_group_summaries_across_periods,
    posthoc_group_summaries_anova,
    posthoc_group_summaries_gee,
)
from .survival import perform_survival_comparison
from .two_way_anova import perform_anova_posthoc, perform_two_way_anova


_PERIODS: Tuple[str, ...] = ("Baseline", "Ictal", "Immediate Postictal", "Recovery")
_INDEP_EFFECT_LABELS: Dict[str, str] = {
    "genotype_clean": "genotype_main",
    "risk_clean": "risk_main",
    "treatment_clean": "treatment_main",
    "period": "period_main",
    "genotype_x_risk": "genotype_x_risk",
    "genotype_x_treatment": "genotype_x_treatment",
    "genotype_x_period": "genotype_x_period",
    "risk_x_period": "risk_x_period",
    "treatment_x_period": "treatment_x_period",
    "genotype_x_risk_x_period": "genotype_x_risk_x_period",
    "genotype_x_treatment_x_period": "genotype_x_treatment_x_period",
}
_DEP_EFFECT_LABELS: Dict[str, str] = {
    "genotype_clean": "genotype_main",
    "age_clean": "age_main",
    "period": "period_main",
    "genotype_x_age": "genotype_x_age",
    "genotype_x_period": "genotype_x_period",
    "age_x_period": "age_x_period",
    "genotype_x_age_x_period": "genotype_x_age_x_period",
}


def run_statistics(
    data: pd.DataFrame,
    *,
    parameters: Optional[Sequence[str]] = None,
    condition_col: str = "risk_clean",
    condition_levels: Tuple[str, str] = ("high_risk", "low_risk"),
    run_anova: bool = True,
    run_gee: bool = True,
    run_developmental: bool = True,
    run_survival: bool = True,
    run_across_periods: bool = True,
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Run the full stats suite on a prepared breathing DataFrame.

    Args:
        data: Output of
            :func:`plethysmography.stats.helpers.prepare_breathing_data`.
        parameters: Restrict the analysis to a subset of parameters; if
            ``None``, every parameter from :func:`define_parameter_categories`
            that exists in ``data.columns`` is analyzed.
        condition_col: Either ``"risk_clean"`` (exp 1) or ``"treatment_clean"``
            (exp 2). Drives the ANOVA / across-periods-independent third
            factor.
        condition_levels: Tuple of the two values ``condition_col`` takes,
            used by the post-hoc to label group cells.
        run_*: Toggle entire test families off (e.g. exp 4 only runs
            ``run_survival=True``).

    Returns the flat list of output rows, with corrected p-values, significance
    flags, and pretty comparison names already applied.
    """
    categories = define_parameter_categories()
    param_to_cat = get_parameter_to_category()
    all_params = [p for params in categories.values() for p in params]
    if parameters is None:
        parameters = [p for p in all_params if p in data.columns]
    else:
        parameters = [p for p in parameters if p in data.columns]

    results_dict: Dict[str, Any] = {p: {"across_periods": {}} for p in parameters}
    all_results: List[Dict[str, Any]] = []

    # =======================================================================
    # PASS 1 — main effects
    # =======================================================================
    for param in parameters:
        for period in _PERIODS:
            results_dict[param][period] = {}

            if run_anova:
                anova_result = perform_two_way_anova(
                    data, param, period,
                    condition_col=condition_col,
                    condition_levels=condition_levels,
                )
                results_dict[param][period]["anova"] = anova_result
                if anova_result is not None:
                    _emit_anova_main_rows(all_results, param, period, anova_result, condition_col)

            if run_gee:
                gee_result = perform_gee(data, param, period)
                results_dict[param][period]["gee"] = gee_result
                if gee_result is not None:
                    _emit_gee_main_rows(all_results, param, period, gee_result)

            if run_developmental:
                dev_result = perform_developmental_comparison(data, param, period)
                results_dict[param][period]["developmental"] = dev_result
                if dev_result is not None:
                    _emit_developmental_row(all_results, param, period, dev_result)

            if run_survival:
                surv_result = perform_survival_comparison(data, param, period)
                results_dict[param][period]["survival"] = surv_result
                if surv_result is not None:
                    _emit_survival_row(all_results, param, period, surv_result)

        if run_across_periods:
            ap_indep = perform_across_periods_independent_gee(
                data, param, condition_col=condition_col,
            )
            results_dict[param]["across_periods"]["independent"] = ap_indep
            if ap_indep is not None:
                _emit_across_periods_rows(
                    all_results, param, ap_indep,
                    analysis_type="across_periods_independent",
                    condition_col=condition_col,
                )
            ap_dep = perform_across_periods_dependent_gee(data, param)
            results_dict[param]["across_periods"]["dependent"] = ap_dep
            if ap_dep is not None:
                _emit_across_periods_rows(
                    all_results, param, ap_dep,
                    analysis_type="across_periods_dependent",
                    condition_col="age_clean",
                )

    # =======================================================================
    # FDR correction on main effects
    # =======================================================================
    _correct_main_rows(all_results, results_dict, categories, param_to_cat, alpha)

    # =======================================================================
    # PASS 2 — post-hocs gated on significant gating effects
    # =======================================================================
    posthoc_rows: List[Dict[str, Any]] = []
    for param in parameters:
        for period in _PERIODS:
            anova_result = results_dict[param][period].get("anova")
            if anova_result is not None and _gating_effect_significant(
                all_results, param, period, "anova",
                comparison_predicate=lambda c: " x " in c or "_x_" in c,
            ):
                ph = perform_anova_posthoc(anova_result, param)
                results_dict[param][period].setdefault("anova_posthoc", {}).update(ph)
                _emit_anova_posthoc_rows(
                    posthoc_rows, anova_result["data"], param, period, ph, condition_col,
                )
            gee_result = results_dict[param][period].get("gee")
            if gee_result is not None and _gating_effect_significant(
                all_results, param, period, "gee",
                comparison_predicate=lambda c: c == "Interaction",
            ):
                ph = perform_gee_posthoc(gee_result, param)
                results_dict[param][period].setdefault("gee_posthoc", {}).update(ph)
                _emit_gee_posthoc_rows(
                    posthoc_rows, gee_result["data"], param, period, ph,
                )

        if run_across_periods:
            for design_key, fit_key, posthoc_fn in (
                ("independent", "across_periods_independent", perform_across_periods_independent_posthoc),
                ("dependent", "across_periods_dependent", perform_across_periods_dependent_posthoc),
            ):
                ap_result = results_dict[param]["across_periods"].get(design_key)
                if ap_result is None:
                    continue
                period_main_sig, any_inter_sig = _across_periods_gating_status(
                    all_results, param, fit_key,
                )
                if not (period_main_sig or any_inter_sig):
                    continue
                ph = posthoc_fn(ap_result, param, any_interaction_significant=any_inter_sig)
                results_dict[param]["across_periods"].setdefault(
                    f"{design_key}_posthoc", {}
                ).update(ph)
                _emit_across_periods_posthoc_rows(
                    posthoc_rows, ap_result.get("data"), param, ph,
                    analysis_type=f"{fit_key}_posthoc",
                    design=design_key,
                    condition_col=condition_col if design_key == "independent" else "age_clean",
                )

    all_results.extend(posthoc_rows)
    _correct_posthoc_rows(posthoc_rows, alpha)

    # Final cleanup: pretty comparison names + fill in category for posthoc rows.
    for row in all_results:
        row["comparison"] = clean_comparison_name(row["comparison"])
        if not row.get("category"):
            row["category"] = param_to_cat.get(row["parameter"])
    return all_results


# ---------------------------------------------------------------------------
# Row emitters (Pass 1)
# ---------------------------------------------------------------------------
def _emit_anova_main_rows(
    sink: List[Dict[str, Any]],
    parameter: str,
    period: str,
    anova_result: Dict[str, Any],
    condition_col: str,
) -> None:
    table = anova_result["anova_table"]
    summaries = compute_group_summaries(
        anova_result["data"], parameter,
        ["genotype_clean", condition_col], label_sep="_",
    )
    effects = (
        ("C(genotype_clean)", "genotype"),
        (f"C({condition_col})", condition_col.replace("_clean", "")),
        (f"C(genotype_clean):C({condition_col})", f"genotype_x_{condition_col.replace('_clean', '')}"),
    )
    for effect_term, label in effects:
        if effect_term not in table.index:
            continue
        p_val = table.loc[effect_term, "PR(>F)"]
        if pd.isna(p_val):
            continue
        sink.append({
            "parameter": parameter,
            "period": period,
            "analysis_type": "anova",
            "comparison": label,
            "group_summaries": summaries,
            "p_value": float(p_val),
        })


def _emit_gee_main_rows(
    sink: List[Dict[str, Any]],
    parameter: str,
    period: str,
    gee_result: Dict[str, Any],
) -> None:
    summaries = compute_group_summaries(
        gee_result["data"], parameter,
        ["genotype_clean", "age_clean"], label_sep="_",
    )
    notes = gee_result.get("notes")
    for source, p_val in gee_result["pvalues"].items():
        if pd.isna(p_val):
            continue
        sink.append({
            "parameter": parameter,
            "period": period,
            "analysis_type": "gee",
            "comparison": source.replace(" ", "_"),
            "group_summaries": summaries,
            "p_value": float(p_val),
            "notes": notes,
        })


def _emit_developmental_row(
    sink: List[Dict[str, Any]],
    parameter: str,
    period: str,
    dev: Dict[str, Any],
) -> None:
    g1 = format_mean_sem_n_scalar(dev["hr_p19_mean"], dev["hr_p19_std"], dev["hr_p19_n"])
    g2 = format_mean_sem_n_scalar(dev["lr_p22_mean"], dev["lr_p22_std"], dev["lr_p22_n"])
    sink.append({
        "parameter": parameter,
        "period": period,
        "analysis_type": "developmental_ttest",
        "comparison": "hr_p19_vs_lr_p22",
        "group_summaries": f"{g1} HR P19; {g2} LR P22",
        "p_value": float(dev["p_value"]),
    })


def _emit_survival_row(
    sink: List[Dict[str, Any]],
    parameter: str,
    period: str,
    surv: Dict[str, Any],
) -> None:
    g1 = format_mean_sem_n_scalar(
        surv["survivors_mean"], surv["survivors_std"], surv["survivors_n"],
    )
    g2 = format_mean_sem_n_scalar(
        surv["non_survivors_mean"], surv["non_survivors_std"], surv["non_survivors_n"],
    )
    sink.append({
        "parameter": parameter,
        "period": period,
        "analysis_type": "survival_prediction",
        "comparison": "survived_vs_died",
        "group_summaries": f"{g1} Survivors; {g2} Non-survivors",
        "p_value": float(surv["p_value"]),
    })


def _emit_across_periods_rows(
    sink: List[Dict[str, Any]],
    parameter: str,
    ap_result: Dict[str, Any],
    *,
    analysis_type: str,
    condition_col: str,
) -> None:
    summaries = compute_group_summaries(
        ap_result.get("data"), parameter,
        ["genotype_clean", condition_col], label_sep="_",
    )
    notes = ap_result.get("notes")
    for source, p_val in ap_result["pvalues"].items():
        if pd.isna(p_val):
            continue
        sink.append({
            "parameter": parameter,
            "period": "all_periods",
            "analysis_type": analysis_type,
            "comparison": source.replace(" ", "_"),
            "group_summaries": summaries,
            "p_value": float(p_val),
            "notes": notes,
        })


# ---------------------------------------------------------------------------
# Row emitters (Pass 2 — post-hocs)
# ---------------------------------------------------------------------------
def _emit_anova_posthoc_rows(
    sink: List[Dict[str, Any]],
    data: pd.DataFrame,
    parameter: str,
    period: str,
    posthoc: Dict[str, Dict[str, Any]],
    condition_col: str,
) -> None:
    for comp, info in posthoc.items():
        sink.append({
            "parameter": parameter,
            "period": period,
            "analysis_type": "posthoc",
            "comparison": comp,
            "group_summaries": posthoc_group_summaries_anova(
                data, parameter, comp, condition_col=condition_col,
            ),
            "p_value": float(info["p_value"]),
        })


def _emit_gee_posthoc_rows(
    sink: List[Dict[str, Any]],
    data: pd.DataFrame,
    parameter: str,
    period: str,
    posthoc: Dict[str, Dict[str, Any]],
) -> None:
    for comp, info in posthoc.items():
        if pd.isna(info.get("p_value")):
            continue
        sink.append({
            "parameter": parameter,
            "period": period,
            "analysis_type": "gee_posthoc",
            "comparison": comp,
            "group_summaries": posthoc_group_summaries_gee(data, parameter, comp),
            "p_value": float(info["p_value"]),
            "notes": info.get("notes"),
        })


def _emit_across_periods_posthoc_rows(
    sink: List[Dict[str, Any]],
    data: Optional[pd.DataFrame],
    parameter: str,
    posthoc: Dict[str, Dict[str, Any]],
    *,
    analysis_type: str,
    design: str,
    condition_col: str,
) -> None:
    for comp, info in posthoc.items():
        if pd.isna(info.get("p_value")):
            continue
        sink.append({
            "parameter": parameter,
            "period": "all_periods",
            "analysis_type": analysis_type,
            "comparison": comp,
            "group_summaries": posthoc_group_summaries_across_periods(
                data, parameter, comp, design=design, condition_col=condition_col,
            ),
            "p_value": float(info["p_value"]),
            "notes": info.get("notes"),
        })


# ---------------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------------
def _correct_main_rows(
    all_results: List[Dict[str, Any]],
    results_dict: Dict[str, Any],
    categories: Dict[str, List[str]],
    param_to_cat: Dict[str, str],
    alpha: float,
) -> None:
    """Group main-effect rows by ``(category, analysis_type, period_key,
    effect_label)`` and apply BH-FDR. Mutates each row to add ``p_corrected``,
    ``significant_corrected``, ``category``, ``correction_group``."""
    families: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(list)

    for idx, row in enumerate(all_results):
        cat = param_to_cat.get(row["parameter"])
        if cat is None:
            continue
        key = _family_key_for_main_row(row, cat)
        if key is None:
            continue
        families[key].append(idx)

    for (cat, analysis_type, period_key, effect_label), indices in families.items():
        pvals = [all_results[i]["p_value"] for i in indices]
        rejected, p_corr, _, _ = multipletests(pvals, method="fdr_bh", alpha=alpha)
        for j, i in enumerate(indices):
            row = all_results[i]
            row["p_corrected"] = float(p_corr[j])
            row["significant_corrected"] = bool(rejected[j])
            row["category"] = cat
            clean_test = analysis_type.replace("_main", "")
            if analysis_type.startswith("across_periods"):
                row["correction_group"] = f"{cat}_{clean_test}_{effect_label}"
            else:
                row["correction_group"] = f"{cat}_{clean_test}_{period_key}_{effect_label}"


def _family_key_for_main_row(
    row: Dict[str, Any], category: str,
) -> Optional[Tuple[str, str, str, str]]:
    """Pick the ``(category, analysis_type, period_key, effect_label)`` family
    for one main-effect row; ``None`` for posthoc / unknown rows."""
    a_type = row["analysis_type"]
    period = row["period"]
    comp = row["comparison"]
    if a_type == "anova":
        label = _anova_effect_label(comp)
        return (category, "anova", period, label)
    if a_type == "gee":
        label = _gee_effect_label(comp)
        return (category, "gee", period, label)
    if a_type == "developmental_ttest":
        return (category, "developmental_ttest", period, "hr_p19_vs_lr_p22")
    if a_type == "survival_prediction":
        return (category, "survival_prediction", period, "survived_vs_died")
    if a_type == "across_periods_independent":
        return (category, "across_periods_independent", "all_periods", comp)
    if a_type == "across_periods_dependent":
        return (category, "across_periods_dependent", "all_periods", comp)
    return None


def _anova_effect_label(comparison: str) -> str:
    if "_x_" in comparison or " x " in comparison:
        return "interaction"
    if comparison.startswith("genotype"):
        return "genotype_main"
    if "risk" in comparison or "treatment" in comparison:
        return f"{comparison.split('_')[0]}_main"
    return f"{comparison}_main"


def _gee_effect_label(comparison: str) -> str:
    if comparison == "Interaction":
        return "interaction"
    if comparison.startswith("genotype"):
        return "genotype_main"
    if comparison.startswith("age"):
        return "age_main"
    return f"{comparison}_main"


def _correct_posthoc_rows(
    posthoc_rows: List[Dict[str, Any]],
    alpha: float,
) -> None:
    """FDR correction on post-hocs is grouped within parameter (× period for
    ANOVA / GEE post-hocs, × analysis_type for across_periods)."""
    families: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for idx, row in enumerate(posthoc_rows):
        a_type = row["analysis_type"]
        if a_type == "posthoc":
            families[(row["parameter"], row["period"], "anova_posthoc")].append(idx)
        elif a_type == "gee_posthoc":
            families[(row["parameter"], row["period"], "gee_posthoc")].append(idx)
        elif a_type == "across_periods_independent_posthoc":
            families[(row["parameter"], "all_periods", "across_periods_independent_posthoc")].append(idx)
        elif a_type == "across_periods_dependent_posthoc":
            families[(row["parameter"], "all_periods", "across_periods_dependent_posthoc")].append(idx)

    for (param, period, kind), indices in families.items():
        pvals = [posthoc_rows[i]["p_value"] for i in indices]
        rejected, p_corr, _, _ = multipletests(pvals, method="fdr_bh", alpha=alpha)
        for j, i in enumerate(indices):
            row = posthoc_rows[i]
            row["p_corrected"] = float(p_corr[j])
            row["significant_corrected"] = bool(rejected[j])
            row["correction_group"] = f"{param}_{period}_{kind}"


# ---------------------------------------------------------------------------
# Pass-1 → Pass-2 gating
# ---------------------------------------------------------------------------
def _gating_effect_significant(
    all_results: List[Dict[str, Any]],
    parameter: str,
    period: str,
    analysis_type: str,
    *,
    comparison_predicate,
) -> bool:
    """True if any row matching (parameter, period, analysis_type) has
    significant_corrected=True AND its comparison matches the predicate."""
    for row in all_results:
        if (row["parameter"] == parameter and row["period"] == period
                and row["analysis_type"] == analysis_type
                and row.get("significant_corrected")
                and comparison_predicate(row["comparison"])):
            return True
    return False


def _across_periods_gating_status(
    all_results: List[Dict[str, Any]],
    parameter: str,
    analysis_type: str,
) -> Tuple[bool, bool]:
    """Return ``(period_main_significant, any_period_interaction_significant)``
    for the across_periods rows of ``parameter`` × ``analysis_type``."""
    period_main = False
    any_inter = False
    for row in all_results:
        if (row["parameter"] != parameter or row["analysis_type"] != analysis_type):
            continue
        if not row.get("significant_corrected"):
            continue
        comp = row["comparison"]
        if comp == "period":
            period_main = True
        elif "period" in comp and comp != "period":
            any_inter = True
    return period_main, any_inter
