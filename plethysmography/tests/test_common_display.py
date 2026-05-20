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
        "Apnea or longest-breaths duration (ms)"
    )
    assert filename_slug("apnea_mean_ms_imputed") == (
        "Apnea_or_longest_breaths_duration_ms"
    )
    # The collision (both -> Apnea_duration_ms) is gone.
    assert filename_slug("apnea_mean_ms") != filename_slug("apnea_mean_ms_imputed")
    assert display_label("apnea_mean_ms") != display_label("apnea_mean_ms_imputed")


def test_apnea_duration_params_gate_excludes_rate_and_burden():
    assert set(APNEA_DURATION_PARAMS) == {"apnea_mean_ms", "apnea_mean_ms_imputed"}
    assert "apnea_rate_per_min" not in APNEA_DURATION_PARAMS
    assert "apnea_burden_ms_per_min" not in APNEA_DURATION_PARAMS


def _axhlines_at(ax, y):
    return [
        ln for ln in ax.lines
        if len(ln.get_ydata()) == 2 and np.allclose(ln.get_ydata(), y)
    ]


def test_reference_line_drawn_and_kept_visible_when_autoscaled_above_floor():
    fig, ax = make_axes()
    ax.set_ylim(450.0, 2000.0)            # V1 autoscaled entirely above 400
    add_apnea_duration_reference_line(ax)
    assert len(_axhlines_at(ax, 400.0)) == 1
    y0, _ = ax.get_ylim()
    assert y0 < 400.0                      # lower bound dropped so line shows
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
            "apnea_burden_ms_per_min"]
    out = within_style_params(base)
    assert out == [
        "mean_tv_ml", "apnea_rate_per_min",
        "apnea_mean_ms", "apnea_mean_ms_imputed",
        "apnea_burden_ms_per_min",
    ]


def test_across_style_params_uses_real_duration_only():
    base = ["mean_tv_ml", "apnea_mean_ms_imputed", "apnea_burden_ms_per_min"]
    assert across_style_params(base) == [
        "mean_tv_ml", "apnea_mean_ms", "apnea_burden_ms_per_min",
    ]


def test_style_params_dedupe_when_both_present():
    base = ["apnea_mean_ms", "apnea_mean_ms_imputed"]
    assert within_style_params(base) == ["apnea_mean_ms", "apnea_mean_ms_imputed"]
    assert across_style_params(base) == ["apnea_mean_ms"]


def test_style_params_noop_without_duration_param():
    base = ["mean_tv_ml", "apnea_rate_per_min"]
    assert within_style_params(base) == base
    assert across_style_params(base) == base
