"""
Tests for cohort color palettes and the binned-plot tuple builder.

Markers are age-based everywhere (P19 triangle, P22 circle). Vehicle colors
match the no-drug risk palette; acute and chronic FFA share one FFA family.
"""

from __future__ import annotations

from plethysmography.visualization.binned_plots import (
    _RISK_GROUPS,
    _TREATMENT_GROUPS,
    _build_treatment_groups,
)
from plethysmography.visualization.colors import (
    ACUTE_FFA_PALETTE,
    DEFAULT_PALETTE,
    DS_FFA_BRIGHT_PURPLE,
    DS_BRIGHT_RED,
    DS_FFA_PALE_PURPLE,
    DS_PALE_ORANGE,
    MARKERS_BY_AGE,
    TREATMENT_MARKERS,
    TREATMENT_PALETTE,
    WT_DARK_BLUE,
    WT_FFA_BLACK,
    WT_FFA_PALE_GRAY,
    WT_PALE_BLUE,
    ffa_cell_color,
    ffa_cell_style,
)


def test_default_palette_exact():
    assert DEFAULT_PALETTE == {
        ("WT", "low_risk"): WT_PALE_BLUE,
        ("WT", "high_risk"): WT_DARK_BLUE,
        ("het", "low_risk"): DS_PALE_ORANGE,
        ("het", "high_risk"): DS_BRIGHT_RED,
    }


def test_treatment_palette_exact():
    assert TREATMENT_PALETTE == {
        ("WT", "Vehicle"): WT_DARK_BLUE,
        ("WT", "FFA"): WT_FFA_BLACK,
        ("het", "Vehicle"): DS_BRIGHT_RED,
        ("het", "FFA"): DS_FFA_BRIGHT_PURPLE,
    }


def test_acute_ffa_palette_matches_chronic():
    assert ACUTE_FFA_PALETTE == TREATMENT_PALETTE


def test_treatment_markers_are_p22_circles():
    assert TREATMENT_MARKERS == {
        "Vehicle": MARKERS_BY_AGE[22],
        "FFA": MARKERS_BY_AGE[22],
    }


def test_ffa_cell_style_vehicle_matches_risk_colors():
    assert ffa_cell_style("WT", 19, "Vehicle") == (WT_PALE_BLUE, "^")
    assert ffa_cell_style("WT", 22, "Vehicle") == (WT_DARK_BLUE, "o")
    assert ffa_cell_style("het", 19, "Vehicle") == (DS_PALE_ORANGE, "^")
    assert ffa_cell_style("het", 22, "Vehicle") == (DS_BRIGHT_RED, "o")


def test_ffa_cell_style_ffa_colors():
    assert ffa_cell_color("WT", 19, "FFA") == WT_FFA_PALE_GRAY
    assert ffa_cell_color("WT", 22, "FFA") == WT_FFA_BLACK
    assert ffa_cell_color("het", 19, "FFA") == DS_FFA_PALE_PURPLE
    assert ffa_cell_color("het", 22, "FFA") == DS_FFA_BRIGHT_PURPLE


def test_binned_treatment_groups_aligned_with_palette():
    for label, genotype, treatment, color, marker in _TREATMENT_GROUPS:
        assert color == TREATMENT_PALETTE[(genotype, treatment)]
        assert marker == MARKERS_BY_AGE[22]


def test_binned_risk_groups_unchanged_against_default_palette():
    for label, genotype, condition, color, marker in _RISK_GROUPS:
        assert color == DEFAULT_PALETTE[(genotype, condition)]
        assert marker == MARKERS_BY_AGE[22]


def test_build_treatment_groups_default_matches_module_constant():
    assert _build_treatment_groups(TREATMENT_PALETTE) == _TREATMENT_GROUPS


def test_build_treatment_groups_acute_matches_chronic():
    assert _build_treatment_groups(ACUTE_FFA_PALETTE) == _TREATMENT_GROUPS
