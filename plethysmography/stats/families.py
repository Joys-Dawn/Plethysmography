"""
Parameter category definitions used as the families for Benjamini-Hochberg FDR
correction. Categories are taken from old_code/breathing_statistics.py
:func:`define_parameter_categories` with project-driven changes (see the
breath_metrics module docstring for the underlying rationale):

* The ``Timing`` parameters now point at the ``*_no_apnea`` columns, so the
  means are no longer skewed by long apneic Ttots.
* ``Pauses_duration`` carries the imputed apnea mean
  (``apnea_mean_ms_imputed``) plus the new aggregate
  ``apnea_burden_s_per_min``. Both are apnea-time measures, so grouping
  them in the same FDR family is the natural fit.

The original columns (``mean_ttot_ms``, ``apnea_mean_ms``, etc.) remain in the
breathing CSV for transparency / audit but are not analyzed by default.

Each category groups parameters that measure the same physiological dimension;
FDR correction is applied within a (category, test_type, period, effect)
family.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "Timing": (
        "mean_ti_ms_no_apnea",
        "mean_te_ms_no_apnea",
        "mean_ttot_ms_no_apnea",
        "mean_frequency_bpm_no_apnea",
    ),
    "Pauses": (
        "sigh_rate_per_min",
        "apnea_rate_per_min",
    ),
    "Pauses_duration": (
        "apnea_mean_ms_imputed",
        "apnea_burden_s_per_min",
    ),
    "Irregularity_frequency": (
        "cov",
    ),
    "Amplitudes": (
        "mean_pif_centered_ml_s",
        "mean_pef_centered_ml_s",
        "mean_pif_to_pef_ml_s",
        "mean_tv_ml",
    ),
    "Irregularity_amplitude": (
        "pif_to_pef_cov",
    ),
}


def define_parameter_categories() -> Dict[str, List[str]]:
    """Return a fresh ``category_name -> [parameter_names]`` mapping. Returned
    lists are mutable copies so callers may filter out parameters that are
    missing from the breathing CSV without polluting the module-level constant.
    """
    return {cat: list(params) for cat, params in _CATEGORIES.items()}


def get_parameter_to_category() -> Dict[str, str]:
    """Inverted mapping: ``parameter_name -> category_name``. Returns a fresh
    dict each call."""
    return {param: cat for cat, params in _CATEGORIES.items() for param in params}
