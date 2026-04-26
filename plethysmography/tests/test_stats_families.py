"""
Test that the parameter→category mapping has the right shape: every parameter
listed in :func:`define_parameter_categories` is a valid old-CSV column, and
the inverse lookup is consistent.
"""

from __future__ import annotations

import pandas as pd
import pytest

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


def test_parameters_present_in_old_csv(tmp_path):
    """Skip if the regression CSV isn't in the working directory."""
    import os
    if not os.path.exists("old_results/breathing_analysis_results.csv"):
        pytest.skip("old_results CSV not present")
    df = pd.read_csv("old_results/breathing_analysis_results.csv")
    cats = define_parameter_categories()
    missing = [
        p for params in cats.values() for p in params if p not in df.columns
    ]
    assert not missing, f"Stats parameters missing from breathing CSV: {missing}"
