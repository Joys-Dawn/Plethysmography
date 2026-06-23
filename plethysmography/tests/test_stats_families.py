"""
Test that the parameter→category mapping has the right shape: every parameter
listed in :func:`define_parameter_categories` exists as a field on the current
:class:`BreathMetrics` dataclass (the source of truth for the breathing CSV
schema), and the inverse lookup is consistent.

Note: the old regression CSV is not the source of truth for parameter names
anymore — the project introduced the ``*_no_apnea`` / ``_imputed`` extension
columns which are absent from the legacy CSV. The dataclass is the canonical
schema, so we validate against it.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from plethysmography.core.data_models import BreathMetrics
from plethysmography.stats.families import (
    define_parameter_categories,
    get_parameter_to_category,
)


def test_categories_nonempty_and_covered():
    cats = define_parameter_categories()
    assert cats, "categories dict must not be empty"
    expected = {
        "Timing", "Pauses", "Pauses_duration",
        "Irregularity_frequency", "Amplitudes", "Irregularity_amplitude",
    }
    assert set(cats.keys()) == expected


def test_inverse_mapping_is_consistent():
    cats = define_parameter_categories()
    inverse = get_parameter_to_category()
    for cat, params in cats.items():
        for p in params:
            assert inverse[p] == cat, f"{p} should map to {cat}, got {inverse[p]}"


def test_parameters_present_in_breath_metrics_schema():
    """Every category parameter must be a field of :class:`BreathMetrics`."""
    valid = {f.name for f in fields(BreathMetrics)}
    cats = define_parameter_categories()
    missing = [
        p for params in cats.values() for p in params if p not in valid
    ]
    assert not missing, (
        f"Stats parameters missing from BreathMetrics dataclass: {missing}"
    )


def test_extension_columns_are_in_their_expected_categories():
    """Lock in the project's extension column → family wiring so a future
    accidental edit to families.py won't silently regress the experimental
    setup."""
    inverse = get_parameter_to_category()
    assert inverse["mean_ti_ms_no_apnea"] == "Timing"
    assert inverse["mean_te_ms_no_apnea"] == "Timing"
    assert inverse["mean_ttot_ms_no_apnea"] == "Timing"
    assert inverse["mean_frequency_bpm_no_apnea"] == "Timing"
    assert inverse["apnea_mean_ms_imputed"] == "Pauses_duration"
    assert inverse["apnea_burden_s_per_min"] == "Pauses_duration"
    # And the legacy columns must NOT be in any FDR family (otherwise they
    # would compete for the same correction with their replacements).
    for legacy in (
        "mean_ti_ms", "mean_te_ms", "mean_ttot_ms", "mean_frequency_bpm",
        "apnea_mean_ms",
    ):
        assert legacy not in inverse, (
            f"{legacy} should not be in any FDR family — its replacement is."
        )
