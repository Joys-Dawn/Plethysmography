"""
Tests for the shared label helpers added in Item C:
``_common.group_label``, ``_common.two_line_label``, and
``_common.treatment_word``.

The canonical group label is always ``"<Geno> P<age> <COND>"`` (genotype ->
age -> condition). The literal ``Scn1a+/-`` must survive verbatim through
``group_label`` so the downstream ``colors.italicize_scn1a`` can swap it for
the shorter ``DS`` display abbreviation at the last moment before the
label hits matplotlib.
"""

from __future__ import annotations

import pytest

from plethysmography.visualization._common import (
    group_label,
    treatment_word,
    two_line_label,
)
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


def test_group_label_scn1a_literal_preserved_for_display_substitution():
    """The literal ``Scn1a+/-`` must remain intact in the internal label so
    the downstream ``italicize_scn1a`` substitution can swap it for the
    shorter ``DS`` (Dravet Syndrome) abbreviation."""
    label = group_label("Scn1a+/-", 22, "LR")
    assert "Scn1a+/-" in label
    assert italicize_scn1a(label) == "DS P22 LR"
    # WT labels are untouched by italicize_scn1a
    assert italicize_scn1a(group_label("WT", 22, "LR")) == "WT P22 LR"


# ---------------------------------------------------------------------------
# two_line_label — first-space split for crowded within-period strip x-ticks
# ---------------------------------------------------------------------------
def test_two_line_label_splits_at_first_space():
    """Genotype goes on line 1, age + condition tail on line 2."""
    assert two_line_label("WT P22 LR") == "WT\nP22 LR"
    assert two_line_label("Scn1a+/- P22 HR") == "Scn1a+/-\nP22 HR"
    # Two-part labels still split, putting genotype alone on top
    assert two_line_label("WT P22") == "WT\nP22"


def test_two_line_label_passes_through_single_token():
    """A label without spaces has no split point and is returned unchanged."""
    assert two_line_label("WT") == "WT"
    assert two_line_label("") == ""


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
