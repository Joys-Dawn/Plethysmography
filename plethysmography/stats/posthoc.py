"""
Post-hoc helpers: building the ``group_summaries`` string for each kind of
post-hoc row, and cleaning the ``comparison`` field for human-readable display.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .helpers import format_mean_sem_n


# ---------------------------------------------------------------------------
# Group-summary builders, one per post-hoc test type
# ---------------------------------------------------------------------------
def posthoc_group_summaries_anova(
    data: Optional[pd.DataFrame],
    parameter: str,
    comparison_name: str,
    *,
    condition_col: str = "risk_clean",
) -> str:
    """For an ANOVA post-hoc comparison like ``"het_low_risk_vs_WT_low_risk"``,
    extract the two groups and format ``mean ± SEM (n)`` for each."""
    if data is None or data.empty or parameter not in data.columns:
        return ""
    if "_vs_" not in comparison_name:
        return ""
    g1, g2 = comparison_name.split("_vs_", 1)
    if "genotype_clean" not in data.columns or condition_col not in data.columns:
        return ""
    df = data.dropna(subset=[parameter]).copy()
    df["_grp"] = df["genotype_clean"].astype(str) + "_" + df[condition_col].astype(str)
    s1 = df.loc[df["_grp"] == g1, parameter]
    s2 = df.loc[df["_grp"] == g2, parameter]
    return f"{format_mean_sem_n(s1)} {g1}; {format_mean_sem_n(s2)} {g2}"


def posthoc_group_summaries_gee(
    data: Optional[pd.DataFrame],
    parameter: str,
    comparison_name: str,
) -> str:
    """For GEE post-hoc comparisons (``het_vs_WT_P19``, ``het_vs_WT_P22``,
    ``P19_vs_P22_het``, ``P19_vs_P22_WT``)."""
    if data is None or data.empty or parameter not in data.columns:
        return ""
    if "genotype_clean" not in data.columns or "age_clean" not in data.columns:
        return ""
    df = data.dropna(subset=[parameter])

    if comparison_name == "het_vs_WT_P19":
        d = df[df["age_clean"] == 19]
        return (
            f"{format_mean_sem_n(d.loc[d['genotype_clean'] == 'het', parameter])} het P19; "
            f"{format_mean_sem_n(d.loc[d['genotype_clean'] == 'WT', parameter])} WT P19"
        )
    if comparison_name == "het_vs_WT_P22":
        d = df[df["age_clean"] == 22]
        return (
            f"{format_mean_sem_n(d.loc[d['genotype_clean'] == 'het', parameter])} het P22; "
            f"{format_mean_sem_n(d.loc[d['genotype_clean'] == 'WT', parameter])} WT P22"
        )
    if comparison_name == "P19_vs_P22_het":
        d = df[df["genotype_clean"] == "het"]
        return (
            f"{format_mean_sem_n(d.loc[d['age_clean'] == 19, parameter])} het P19; "
            f"{format_mean_sem_n(d.loc[d['age_clean'] == 22, parameter])} het P22"
        )
    if comparison_name == "P19_vs_P22_WT":
        d = df[df["genotype_clean"] == "WT"]
        return (
            f"{format_mean_sem_n(d.loc[d['age_clean'] == 19, parameter])} WT P19; "
            f"{format_mean_sem_n(d.loc[d['age_clean'] == 22, parameter])} WT P22"
        )
    return ""


def posthoc_group_summaries_across_periods(
    data: Optional[pd.DataFrame],
    parameter: str,
    comparison_name: str,
    *,
    design: str = "independent",
    condition_col: str = "risk_clean",
) -> str:
    """For across-periods post-hoc comparisons.

    ``comparison_name`` looks like:

      - ``"pooled_Ictal_vs_Baseline"`` (no significant interaction)
      - ``"het_high_risk_Ictal_vs_Baseline"`` (independent design)
      - ``"het_P22_Ictal_vs_Baseline"`` (dependent design)
    """
    if data is None or data.empty or parameter not in data.columns or "period" not in data.columns:
        return ""
    if "_vs_Baseline" not in comparison_name:
        return ""
    df = data.dropna(subset=[parameter])
    rest = comparison_name.replace("_vs_Baseline", "")
    if "_" not in rest:
        return ""
    group_part, period_name = rest.rsplit("_", 1)

    if group_part == "pooled":
        sub = df
    elif design == "independent":
        if "_" not in group_part:
            sub = df
        else:
            geno, cond = group_part.split("_", 1)
            sub = df[
                (df["genotype_clean"].astype(str) == geno)
                & (df[condition_col].astype(str) == cond)
            ]
    else:  # dependent
        if "_" not in group_part:
            sub = df
        else:
            geno, age_str = group_part.split("_", 1)
            try:
                age = int(age_str.replace("P", ""))
            except (ValueError, AttributeError):
                age = 22
            sub = df[
                (df["genotype_clean"].astype(str) == geno)
                & (df["age_clean"] == age)
            ]
    if sub.empty:
        return ""
    s_base = sub.loc[sub["period"] == "Baseline", parameter]
    s_per = sub.loc[sub["period"] == period_name, parameter]
    return f"{format_mean_sem_n(s_base)} Baseline; {format_mean_sem_n(s_per)} {period_name}"


# ---------------------------------------------------------------------------
# Comparison-name pretty-printer (matches old breathing_statistics.py)
# ---------------------------------------------------------------------------
def clean_comparison_name(comparison_name: str) -> str:
    """Replace internal underscore patterns with the more readable forms used
    in the published xlsx (e.g. ``"het_vs_WT_P19"`` → ``"het vs WT (P19)"``)."""
    cleaned = comparison_name.replace("_clean", "")
    replacements = {
        "genotype_het": "genotype",
        "age_22": "age",
        "_x_": " x ",
        "het_vs_WT_P19": "het vs WT (P19)",
        "het_vs_WT_P22": "het vs WT (P22)",
        "P19_vs_P22_het": "P19 vs P22 (het)",
        "P19_vs_P22_WT": "P19 vs P22 (WT)",
        "het_low_risk_vs_WT_low_risk": "het vs WT (low risk)",
        "het_high_risk_vs_WT_high_risk": "het vs WT (high risk)",
        "het_low_risk_vs_het_high_risk": "low vs high risk (het)",
        "het_high_risk_vs_het_low_risk": "low vs high risk (het)",
        "het_FFA_vs_WT_FFA": "het vs WT (FFA)",
        "het_Vehicle_vs_WT_Vehicle": "het vs WT (Vehicle)",
        "het_FFA_vs_het_Vehicle": "FFA vs Vehicle (het)",
        "hr_p19_vs_lr_p22": "het HR P19 vs het LR P22 (developmental)",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    if " x " not in cleaned:
        cleaned = cleaned.replace("risk_", "")
    return cleaned
