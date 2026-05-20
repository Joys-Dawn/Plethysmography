"""
Tests for Item D: the chronic-FFA ``TREATMENT_PALETTE`` rewrite, the new
deferred ``ACUTE_FFA_PALETTE`` constant, and the requirement that the binned
``_TREATMENT_GROUPS`` hexes stay aligned (by ``(genotype, treatment)`` key)
with ``TREATMENT_PALETTE`` after the Item C relabel.

A regression guard also pins ``_RISK_GROUPS`` to ``DEFAULT_PALETTE`` so the
C/D changes provably did not perturb the experiment-1 risk colors.
"""

from __future__ import annotations

from plethysmography.visualization.binned_plots import (
    _RISK_GROUPS,
    _TREATMENT_GROUPS,
)
from plethysmography.visualization.colors import (
    ACUTE_FFA_PALETTE,
    DEFAULT_PALETTE,
    TREATMENT_PALETTE,
)


# ---------------------------------------------------------------------------
# Exact hex mappings (the locked Item D spec)
# ---------------------------------------------------------------------------
def test_treatment_palette_exact():
    assert TREATMENT_PALETTE == {
        ("WT", "Vehicle"): "#D3D3D3",
        ("WT", "FFA"): "#D8BFD8",
        ("het", "Vehicle"): "#696969",
        ("het", "FFA"): "#800080",
    }


def test_acute_ffa_palette_exact():
    assert ACUTE_FFA_PALETTE == {
        ("WT", "Vehicle"): "#D3D3D3",
        ("WT", "FFA"): "#FFB6C1",
        ("het", "Vehicle"): "#696969",
        ("het", "FFA"): "#C71585",
    }


def test_palettes_use_genotype_tokens_not_label_strings():
    """Consumers key these palettes by the ``"WT"``/``"het"`` genotype token
    (not the C label string). Item D must not have changed the key shape."""
    for palette in (TREATMENT_PALETTE, ACUTE_FFA_PALETTE):
        assert set(palette) == {
            ("WT", "Vehicle"),
            ("WT", "FFA"),
            ("het", "Vehicle"),
            ("het", "FFA"),
        }


def test_acute_shares_vehicle_grays_but_differs_on_ffa():
    """Acute reuses the chronic gray Vehicle pair (so Vehicle reads the same
    across experiments) but swaps the FFA family to pink so acute vs chronic
    FFA plots are visually distinguishable."""
    assert ACUTE_FFA_PALETTE[("WT", "Vehicle")] == TREATMENT_PALETTE[("WT", "Vehicle")]
    assert ACUTE_FFA_PALETTE[("het", "Vehicle")] == TREATMENT_PALETTE[("het", "Vehicle")]
    assert ACUTE_FFA_PALETTE[("WT", "FFA")] != TREATMENT_PALETTE[("WT", "FFA")]
    assert ACUTE_FFA_PALETTE[("het", "FFA")] != TREATMENT_PALETTE[("het", "FFA")]


# ---------------------------------------------------------------------------
# binned tuple order aligned with the palette (the Item D risk note)
# ---------------------------------------------------------------------------
def test_binned_treatment_groups_aligned_with_palette():
    """Each ``_TREATMENT_GROUPS`` row's color (g[3]) must equal
    ``TREATMENT_PALETTE[(genotype g[1], treatment g[2])]`` — i.e. the C
    relabel did not desynchronize the tuple order from the palette keys."""
    for label, genotype, treatment, color, marker in _TREATMENT_GROUPS:
        assert color == TREATMENT_PALETTE[(genotype, treatment)], (
            f"{label}: {color} != palette[{(genotype, treatment)}]"
        )
        assert marker == "o"


def test_binned_risk_groups_unchanged_against_default_palette():
    """Regression: C/D must not have perturbed the exp1 risk colors. Each
    ``_RISK_GROUPS`` row's color must still equal
    ``DEFAULT_PALETTE[(genotype, condition)]``."""
    for label, genotype, condition, color, marker in _RISK_GROUPS:
        assert color == DEFAULT_PALETTE[(genotype, condition)], (
            f"{label}: {color} != palette[{(genotype, condition)}]"
        )
        assert marker == "o"
