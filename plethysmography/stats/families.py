"""
Parameter category definitions used as the families for Benjamini-Hochberg FDR
correction. Categories are taken verbatim from old_code/breathing_statistics.py
:func:`define_parameter_categories`. Each category groups parameters that
measure the same physiological dimension; FDR correction is applied within a
(category, test_type, period, effect) family.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "Timing": (
        "mean_ti_ms",
        "mean_te_ms",
        "mean_ttot_ms",
        "mean_frequency_bpm",
    ),
    "Pauses": (
        "sigh_rate_per_min",
        "apnea_rate_per_min",
    ),
    "Pauses_duration": (
        "apnea_mean_ms",
    ),
    "Irregularity_frequency": (
        "cov_instant_freq",
        "alternate_cov",
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
