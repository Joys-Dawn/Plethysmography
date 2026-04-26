"""
Color palette for cohort plots, matching ``old_code/analyze_data.py`` and
``old_code/fenfluramine_plots.py`` byte-for-byte.

  - HR WT          (P19 or P22)  : ``#0000FF``  standard blue
  - HR Scn1a+/-    (P19 or P22)  : ``#FF0000``  standard red
  - LR WT          (P22)         : ``#87CEEB``  pale blue (sky blue)
  - LR Scn1a+/-    (P22)         : ``#FFA07A``  pale red  (light salmon)

For experiment 2 (FFA cohort), Vehicle reuses the HR palette and FFA gets a
distinct orange/purple pair:

  - Vehicle WT     : ``#0000FF``
  - Vehicle Scn1a  : ``#FF0000``
  - FFA WT         : ``#9370DB``  purple
  - FFA Scn1a      : ``#E67E22``  orange

Marker convention (used by within / timeseries plots):

  - P19 -> ``"^"`` (triangle)
  - P22 -> ``"o"`` (circle)
"""

from __future__ import annotations

from typing import Dict, Tuple


# Experiment 1 palette: (genotype, condition) -> color
DEFAULT_PALETTE: Dict[Tuple[str, str], str] = {
    ("WT", "low_risk"): "#87CEEB",
    ("WT", "high_risk"): "#0000FF",
    ("het", "low_risk"): "#FFA07A",
    ("het", "high_risk"): "#FF0000",
}

# Experiment 2 palette: (genotype, treatment) -> color
TREATMENT_PALETTE: Dict[Tuple[str, str], str] = {
    ("WT", "Vehicle"): "#0000FF",
    ("WT", "FFA"): "#9370DB",
    ("het", "Vehicle"): "#FF0000",
    ("het", "FFA"): "#E67E22",
}

# (genotype, age) -> color for **bar within** plots (old_code/analyze_data.py
# create_within_plot): solid colors regardless of age, markers carry the age
# distinction.
HR_BAR_PALETTE: Dict[Tuple[str, int], str] = {
    ("WT", 19): "#0000FF",
    ("WT", 22): "#0000FF",
    ("het", 19): "#FF0000",
    ("het", 22): "#FF0000",
}

# (genotype, age) -> color for **timeseries within** plots
# (old_code/analyze_data.py create_timeseries_within_plot): P19 cells use the
# pale LR colors so the connected line plot can distinguish ages by hue too.
HR_TIMESERIES_PALETTE: Dict[Tuple[str, int], str] = {
    ("WT", 19): "#87CEEB",
    ("WT", 22): "#0000FF",
    ("het", 19): "#FFA07A",
    ("het", 22): "#FF0000",
}

MARKERS_BY_AGE: Dict[int, str] = {19: "^", 22: "o"}


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
    """Return a copy of the experiment-2 palette."""
    return dict(TREATMENT_PALETTE)


def italicize_scn1a(label: str) -> str:
    """Replace the literal ``Scn1a+/-`` substring with the matplotlib mathtext
    italicized form ``$\\mathit{Scn1a}^{+/-}$``. Mirrors old code's label
    formatting in every plot."""
    return label.replace("Scn1a+/-", r"$\mathit{Scn1a}^{+/-}$")
