"""
Statistical analysis: parameter families, two-way ANOVA, GEE, across-periods
GEE, developmental and survival t-tests, post-hoc gating, and multi-sheet xlsx
writer.

The structure mirrors old_code/breathing_statistics.py but split into focused
modules. The two-pass driver lives in :mod:`runner`; output formatting in
:mod:`writer`.
"""

from .families import define_parameter_categories, get_parameter_to_category
from .helpers import (
    capture_warnings,
    compute_group_summaries,
    format_mean_sem_n,
    format_mean_sem_n_scalar,
    get_convergence_notes,
    prepare_breathing_data,
)
from .two_way_anova import perform_two_way_anova, perform_anova_posthoc
from .gee import perform_gee, perform_gee_posthoc
from .across_periods import (
    perform_across_periods_independent_gee,
    perform_across_periods_dependent_gee,
    perform_across_periods_independent_posthoc,
    perform_across_periods_dependent_posthoc,
)
from .developmental import perform_developmental_comparison
from .survival import perform_survival_comparison
from .runner import run_statistics
from .writer import write_stats_xlsx

__all__ = [
    "define_parameter_categories",
    "get_parameter_to_category",
    "capture_warnings",
    "compute_group_summaries",
    "format_mean_sem_n",
    "format_mean_sem_n_scalar",
    "get_convergence_notes",
    "prepare_breathing_data",
    "perform_two_way_anova",
    "perform_anova_posthoc",
    "perform_gee",
    "perform_gee_posthoc",
    "perform_across_periods_independent_gee",
    "perform_across_periods_dependent_gee",
    "perform_across_periods_independent_posthoc",
    "perform_across_periods_dependent_posthoc",
    "perform_developmental_comparison",
    "perform_survival_comparison",
    "run_statistics",
    "write_stats_xlsx",
]
