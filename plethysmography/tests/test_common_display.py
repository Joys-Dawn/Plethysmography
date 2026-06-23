"""
Item B — the apnea-duration slug collision is resolved and the fixed 400 ms
reference-line helper behaves correctly.

  * ``apnea_mean_ms`` (V1) and ``apnea_mean_ms_imputed`` (V2) now have
    DISTINCT display labels and filename slugs, so the two PNGs no longer
    overwrite each other on disk.
  * ``APNEA_DURATION_PARAMS`` contains exactly the two duration parameters
    (the gate that keeps the 400 ms line off apnea rate / burden plots).
  * ``add_apnea_duration_reference_line`` draws one dashed line at 400 ms and
    keeps it visible even when the data autoscales entirely above the floor.
  * ``within_style_params`` / ``across_style_params`` expand the apnea slot.
"""

from __future__ import annotations

import numpy as np

from plethysmography.visualization._common import (
    APNEA_DURATION_PARAMS,
    across_style_params,
    add_apnea_duration_reference_line,
    display_label,
    filename_slug,
    make_axes,
    within_style_params,
)


def test_apnea_duration_label_and_slug_no_longer_collide():
    assert display_label("apnea_mean_ms") == "Apnea duration (ms)"
    assert filename_slug("apnea_mean_ms") == "Apnea_duration_ms"
    assert display_label("apnea_mean_ms_imputed") == (
        "Apnea or longest-breaths\nduration (ms)"
    )
    assert filename_slug("apnea_mean_ms_imputed") == (
        "Apnea_or_longest_breaths_duration_ms"
    )
    # The collision (both -> Apnea_duration_ms) is gone.
    assert filename_slug("cov") == "CoV_IBI"
    assert display_label("cov") == "CoV_IBI"
    assert filename_slug("pif_to_pef_cov") == "CoV_PIF_to_PEF"


def test_apnea_duration_params_gate_excludes_rate_and_burden():
    assert set(APNEA_DURATION_PARAMS) == {"apnea_mean_ms", "apnea_mean_ms_imputed"}
    assert "apnea_rate_per_min" not in APNEA_DURATION_PARAMS
    assert "apnea_burden_s_per_min" not in APNEA_DURATION_PARAMS


def _axhlines_at(ax, y):
    return [
        ln for ln in ax.lines
        if len(ln.get_ydata()) == 2 and np.allclose(ln.get_ydata(), y)
    ]


def test_reference_line_anchors_y_axis_at_zero_when_autoscaled_above_floor():
    """Section 2.2d: apnea-duration y-axis always starts at 0. The reference
    line helper drops the lower bound to 0 regardless of the autoscaled
    minimum so durations read as multiples of the 400 ms floor."""
    fig, ax = make_axes()
    ax.set_ylim(450.0, 2000.0)            # V1 autoscaled entirely above 400
    add_apnea_duration_reference_line(ax)
    assert len(_axhlines_at(ax, 400.0)) == 1
    y0, y1 = ax.get_ylim()
    assert y0 == 0.0                       # always anchored at 0
    assert y1 == 2000.0                    # upper bound unchanged
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_reference_line_does_not_raise_lower_bound_when_already_below():
    fig, ax = make_axes()
    ax.set_ylim(0.0, 1000.0)              # V2 zero-anchored: 400 already in view
    add_apnea_duration_reference_line(ax)
    assert len(_axhlines_at(ax, 400.0)) == 1
    y0, y1 = ax.get_ylim()
    assert y0 == 0.0 and y1 == 1000.0      # unchanged
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_within_style_params_emits_v1_then_v2():
    base = ["mean_tv_ml", "apnea_rate_per_min", "apnea_mean_ms_imputed",
            "apnea_burden_s_per_min"]
    out = within_style_params(base)
    assert out == [
        "mean_tv_ml", "apnea_rate_per_min",
        "apnea_mean_ms", "apnea_mean_ms_imputed",
        "apnea_burden_s_per_min",
    ]


def test_across_style_params_uses_real_duration_only():
    base = ["mean_tv_ml", "apnea_mean_ms_imputed", "apnea_burden_s_per_min"]
    assert across_style_params(base) == [
        "mean_tv_ml", "apnea_mean_ms", "apnea_burden_s_per_min",
    ]


def test_style_params_dedupe_when_both_present():
    base = ["apnea_mean_ms", "apnea_mean_ms_imputed"]
    assert within_style_params(base) == ["apnea_mean_ms", "apnea_mean_ms_imputed"]
    assert across_style_params(base) == ["apnea_mean_ms"]


def test_style_params_noop_without_duration_param():
    base = ["mean_tv_ml", "apnea_rate_per_min"]
    assert within_style_params(base) == base
    assert across_style_params(base) == base


def test_period_ylim_includes_p19_treatment_cohort():
    """Regression: FFA strip plots must not build y limits from P22-only data.

    The old ``global_ylim(..., condition_col='risk_clean')`` path on exp-2
    omitted P19 Vehicle/FFA rows, so P19 points (e.g. high postictal apnea
    burden) could render above ``ylim[1]``.
    """
    import pandas as pd

    from plethysmography.visualization._common import period_ylim

    df = pd.DataFrame({
        "period": ["Immediate Postictal"] * 4,
        "age_clean": [19, 19, 22, 22],
        "genotype_clean": ["het", "het", "het", "het"],
        "treatment_clean": ["Vehicle", "FFA", "Vehicle", "FFA"],
        "risk_clean": [None, None, None, None],
        "apnea_burden_s_per_min": [120.0, 100.0, 10.0, 8.0],
    })

    wrong = period_ylim(
        df, "apnea_burden_s_per_min", "Immediate Postictal",
        condition_col="risk_clean",
    )
    right = period_ylim(
        df, "apnea_burden_s_per_min", "Immediate Postictal",
        condition_col="treatment_clean",
    )

    assert wrong is not None and right is not None
    assert wrong[1] < 120.0, "risk_clean ylim must not cover P19 high values"
    assert right[1] >= 120.0, "treatment_clean ylim must cover P19 high values"


def test_period_ylim_matches_plot_within_period_cohort():
    """Standard strip y limits must cover both across (P22) and within panels."""
    import pandas as pd

    from plethysmography.visualization._common import period_ylim

    df = pd.DataFrame({
        "period": ["Ictal"] * 4,
        "age_clean": [19, 22, 22, 22],
        "genotype_clean": ["het", "het", "het", "het"],
        "risk_clean": ["high_risk", "high_risk", "low_risk", "high_risk"],
        "mean_tv_ml": [500.0, 400.0, 50.0, 450.0],
    })
    ylim = period_ylim(df, "mean_tv_ml", "Ictal", condition_col="risk_clean")
    assert ylim is not None
    assert ylim[0] <= 50.0
    assert ylim[1] >= 500.0

