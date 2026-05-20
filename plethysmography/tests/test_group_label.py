"""
Tests for the shared label helpers added in Item C:
``_common.group_label`` and ``_common.treatment_word``.

The canonical group label is always ``"<Geno> P<age> <COND>"`` (genotype ->
age -> condition). The literal ``Scn1a+/-`` must survive verbatim so the
downstream ``colors.italicize_scn1a`` still renders it as
``$\\mathit{Scn1a}^{+/-}$``.
"""

from __future__ import annotations

import pytest

from plethysmography.visualization._common import group_label, treatment_word
from plethysmography.visualization.colors import italicize_scn1a


# ---------------------------------------------------------------------------
# group_label — exact canonical strings (the locked Item C spec)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("genotype", "age", "condition", "expected"),
    [
        # Exp1: risk
        ("WT", 22, "LR", "WT P22 LR"),
        ("WT", 22, "HR", "WT P22 HR"),
        ("Scn1a+/-", 22, "LR", "Scn1a+/- P22 LR"),
        ("Scn1a+/-", 22, "HR", "Scn1a+/- P22 HR"),
        ("WT", 19, "HR", "WT P19 HR"),
        ("Scn1a+/-", 19, "HR", "Scn1a+/- P19 HR"),
        # Exp2: treatment word
        ("WT", 19, "vehicle", "WT P19 vehicle"),
        ("Scn1a+/-", 22, "FFA", "Scn1a+/- P22 FFA"),
        # Exp4: survivor / SUDEP (note case: lowercase survivor, upper SUDEP)
        ("Scn1a+/-", 19, "survivor", "Scn1a+/- P19 survivor"),
        ("Scn1a+/-", 19, "SUDEP", "Scn1a+/- P19 SUDEP"),
    ],
)
def test_group_label_exact(genotype, age, condition, expected):
    assert group_label(genotype, age, condition) == expected


def test_group_label_condition_none_omits_condition():
    """exp2-within pools FFA+Vehicle per genotype x age cell, so no treatment
    word applies — condition=None must drop the trailing token entirely."""
    assert group_label("WT", 22, None) == "WT P22"
    assert group_label("Scn1a+/-", 22, None) == "Scn1a+/- P22"
    # default arg is None
    assert group_label("Scn1a+/-", 19) == "Scn1a+/- P19"


def test_group_label_empty_condition_omitted():
    """An empty-string condition is falsy and is dropped (not appended as a
    trailing space)."""
    assert group_label("WT", 22, "") == "WT P22"


def test_group_label_age_accepts_int_str_float_and_p_token():
    assert group_label("WT", 22, "LR") == "WT P22 LR"
    assert group_label("WT", "22", "LR") == "WT P22 LR"
    assert group_label("WT", 22.0, "LR") == "WT P22 LR"
    assert group_label("WT", "22.0", "LR") == "WT P22 LR"
    # already-formatted token is passed through, not double-prefixed
    assert group_label("WT", "P22", "LR") == "WT P22 LR"
    assert group_label("WT", "p19", "HR") == "WT p19 HR"


def test_group_label_age_none_or_empty_omits_age():
    assert group_label("WT", None) == "WT"
    assert group_label("WT", "") == "WT"
    assert group_label("WT", None, "LR") == "WT LR"


def test_group_label_scn1a_literal_preserved_for_italicize():
    """The literal ``Scn1a+/-`` must remain intact so the downstream
    matplotlib mathtext substitution still fires."""
    label = group_label("Scn1a+/-", 22, "LR")
    assert "Scn1a+/-" in label
    assert italicize_scn1a(label) == r"$\mathit{Scn1a}^{+/-}$ P22 LR"
    # WT labels are untouched by italicize_scn1a
    assert italicize_scn1a(group_label("WT", 22, "LR")) == "WT P22 LR"


# ---------------------------------------------------------------------------
# treatment_word — vehicle/FFA normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Vehicle", "vehicle"),
        ("vehicle", "vehicle"),
        ("veh", "vehicle"),
        ("VEHICLE", "vehicle"),
        ("  Vehicle  ", "vehicle"),
        ("FFA", "FFA"),
        ("ffa", "FFA"),
        ("fenfluramine", "FFA"),
        ("anything-else", "FFA"),
    ],
)
def test_treatment_word(raw, expected):
    assert treatment_word(raw) == expected


def test_treatment_word_composes_with_group_label():
    """The exact exp2 labels the binned/bar/timeseries drivers emit."""
    assert group_label("WT", 22, treatment_word("Vehicle")) == "WT P22 vehicle"
    assert group_label("Scn1a+/-", 22, treatment_word("FFA")) == "Scn1a+/- P22 FFA"
