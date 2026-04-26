"""Plot generation: bar plots per period (within), timeseries across periods,
postictal-binned line plots, developmental-comparison plots, and interactive
plotly traces of segmented breaths.

Static plots use matplotlib (with the Agg backend); interactive plots use
plotly. Color palettes are centralized in :mod:`colors`.
"""

from .colors import (
    DEFAULT_PALETTE,
    HR_BAR_PALETTE,
    HR_TIMESERIES_PALETTE,
    MARKERS_BY_AGE,
    TREATMENT_PALETTE,
    get_group_color,
    italicize_scn1a,
    treatment_palette,
)
from .bar_plots import plot_within_period
from .timeseries_plots import plot_across_periods
from .binned_plots import plot_postictal_binned, plot_ictal_binned
from .interactive_plots import plot_breath_segmentation
from .trace_plots import plot_lid_spikes, plot_periods_overlay
from .publication_plots import (
    generate_publication_plots,
    plot_developmental_comparison,
    plot_ffa_subgroups,
)
from .survivor_plots import plot_survivor_publication

__all__ = [
    "DEFAULT_PALETTE",
    "HR_BAR_PALETTE",
    "HR_TIMESERIES_PALETTE",
    "MARKERS_BY_AGE",
    "TREATMENT_PALETTE",
    "get_group_color",
    "italicize_scn1a",
    "treatment_palette",
    "plot_within_period",
    "plot_across_periods",
    "plot_postictal_binned",
    "plot_ictal_binned",
    "plot_breath_segmentation",
    "plot_lid_spikes",
    "plot_periods_overlay",
    "generate_publication_plots",
    "plot_developmental_comparison",
    "plot_ffa_subgroups",
    "plot_survivor_publication",
]
