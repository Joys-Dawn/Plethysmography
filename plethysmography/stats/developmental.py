"""
Developmental t-test: Welch's t-test on a single period comparing het HR P19
vs het LR P22 (genotype-matched mice both experiencing seizure #1, but at
different developmental ages). Produces one row per (parameter, period).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def perform_developmental_comparison(
    data: pd.DataFrame,
    parameter: str,
    period: str,
) -> Optional[Dict[str, Any]]:
    """Compare het HR P19 vs het LR P22 at one period. Returns ``None`` when
    either group has fewer than 2 valid observations."""
    period_data = data[data["period"] == period]
    scn1a = period_data[period_data["genotype_clean"] == "het"]

    hr_p19 = scn1a[
        (scn1a["risk_clean"] == "high_risk") & (scn1a["age_clean"] == 19)
    ][parameter].dropna()
    lr_p22 = scn1a[
        (scn1a["risk_clean"] == "low_risk") & (scn1a["age_clean"] == 22)
    ][parameter].dropna()

    if len(hr_p19) < 2 or len(lr_p22) < 2:
        return None
    try:
        t_stat, p_value = stats.ttest_ind(hr_p19, lr_p22, nan_policy="omit", equal_var=False)
    except Exception:
        return None
    if pd.isna(p_value) or pd.isna(t_stat):
        return None
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "hr_p19_n": int(len(hr_p19)),
        "lr_p22_n": int(len(lr_p22)),
        "hr_p19_mean": float(hr_p19.mean()),
        "lr_p22_mean": float(lr_p22.mean()),
        "hr_p19_std": float(hr_p19.std(ddof=1)) if len(hr_p19) > 1 else float("nan"),
        "lr_p22_std": float(lr_p22.std(ddof=1)) if len(lr_p22) > 1 else float("nan"),
    }
