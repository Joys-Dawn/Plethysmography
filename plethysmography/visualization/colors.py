"""
Color and marker palette for cohort plots.

Organizing principles (current spec):

  1. P19 -> triangle (``"^"``), P22 -> circle (``"o"``) on every plot type.
  2. Vehicle uses the same color family as the no-drug risk cohort
     (pale vs saturated by age for WT blue and DS orange/red).
  3. Acute and chronic FFA share one purple / gray FFA family.

Risk / vehicle (no-drug) colors:

  - WT P22 LR / WT P19 HR / WT P19 vehicle : pale blue triangle or circle
  - WT P22 HR / WT P22 vehicle             : dark blue circle
  - DS P22 LR / DS P19 HR / DS P19 vehicle : pale orange triangle or circle
  - DS P22 HR / DS P22 vehicle             : bright red circle

FFA colors (acute + chronic unified):

  - WT P19 FFA : medium pale gray triangle
  - WT P22 FFA : black circle
  - DS P19 FFA : medium pale purple triangle
  - DS P22 FFA : bright purple circle
"""

from __future__ import annotations

from typing import Dict, Tuple


# Risk / vehicle base colors
WT_PALE_BLUE = "#6EC4E8"
WT_DARK_BLUE = "#0000FF"
DS_PALE_ORANGE = "#FFB08A"
DS_BRIGHT_RED = "#FF0000"

# FFA colors (shared acute + chronic)
WT_FFA_PALE_GRAY = "#A9A9A9"
WT_FFA_BLACK = "#000000"
DS_FFA_PALE_PURPLE = "#C8A2C8"
DS_FFA_BRIGHT_PURPLE = "#9400D3"

# Experiment 1 palette: (genotype, risk) -> color (P22 across plots)
DEFAULT_PALETTE: Dict[Tuple[str, str], str] = {
    ("WT", "low_risk"): WT_PALE_BLUE,
    ("WT", "high_risk"): WT_DARK_BLUE,
    ("het", "low_risk"): DS_PALE_ORANGE,
    ("het", "high_risk"): DS_BRIGHT_RED,
}

# P22-only treatment palette for across / binned plots (Vehicle = HR colors,
# FFA = unified FFA colors). Acute reuses the same mapping.
TREATMENT_PALETTE: Dict[Tuple[str, str], str] = {
    ("WT", "Vehicle"): WT_DARK_BLUE,
    ("WT", "FFA"): WT_FFA_BLACK,
    ("het", "Vehicle"): DS_BRIGHT_RED,
    ("het", "FFA"): DS_FFA_BRIGHT_PURPLE,
}

ACUTE_FFA_PALETTE: Dict[Tuple[str, str], str] = dict(TREATMENT_PALETTE)

MARKERS_BY_AGE: Dict[int, str] = {19: "^", 22: "o"}

# Legacy name kept for call sites that still import it; markers are age-based
# everywhere now, not treatment-based.
TREATMENT_MARKERS: Dict[str, str] = {
    "Vehicle": MARKERS_BY_AGE[22],
    "FFA": MARKERS_BY_AGE[22],
}

# Within bar / timeseries plots: pale P19, saturated P22.
HR_BAR_PALETTE: Dict[Tuple[str, int], str] = {
    ("WT", 19): WT_PALE_BLUE,
    ("WT", 22): WT_DARK_BLUE,
    ("het", 19): DS_PALE_ORANGE,
    ("het", 22): DS_BRIGHT_RED,
}

HR_TIMESERIES_PALETTE: Dict[Tuple[str, int], str] = dict(HR_BAR_PALETTE)


def ffa_cell_color(genotype: str, age: int, treatment: str) -> str:
    """Return the plot color for one (genotype, age, treatment) cell."""
    age = int(age)
    if str(treatment) == "Vehicle":
        if genotype == "WT":
            return WT_PALE_BLUE if age == 19 else WT_DARK_BLUE
        return DS_PALE_ORANGE if age == 19 else DS_BRIGHT_RED
    if genotype == "WT":
        return WT_FFA_PALE_GRAY if age == 19 else WT_FFA_BLACK
    return DS_FFA_PALE_PURPLE if age == 19 else DS_FFA_BRIGHT_PURPLE


def ffa_cell_style(genotype: str, age: int, treatment: str) -> Tuple[str, str]:
    """Return ``(color, marker)`` for one FFA-cohort cell."""
    return ffa_cell_color(genotype, age, treatment), MARKERS_BY_AGE[int(age)]


def get_group_color(genotype: str, condition: str) -> str:
    """Look up a hex color for one (genotype, condition) cell, falling back to
    a neutral gray when the pair is not in the default palette."""
    key = (genotype, condition)
    if key in DEFAULT_PALETTE:
        return DEFAULT_PALETTE[key]
    if key in TREATMENT_PALETTE:
        return TREATMENT_PALETTE[key]
    return "#6b7280"


def treatment_palette() -> Dict[Tuple[str, str], str]:
    """Return a copy of the unified FFA palette."""
    return dict(TREATMENT_PALETTE)


def italicize_scn1a(label: str) -> str:
    """Display-time substitution of the literal ``Scn1a+/-`` genotype token
    with the shorter ``DS`` (Dravet Syndrome) abbreviation."""
    return label.replace("Scn1a+/-", "DS")
