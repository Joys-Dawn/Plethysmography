"""
Plotly interactive trace plots for one (file, period) — used to QC breath
segmentation, sigh classification, and apnea detection.

Mirrors ``old_code/analyze_data.py:plot_interactive_breath_analysis`` very
carefully because the original implementation took several iterations to
reach a fast-rendering form. The performance-critical decisions are:

  1. Build ``breath_phase_shapes`` as a single Python list, then bulk-assign
     once via ``fig.layout.shapes = tuple(shapes)``. Calling
     :py:meth:`plotly.graph_objects.Figure.add_shape` once per breath was
     orders of magnitude slower on long recordings (Recovery has ~5000
     breaths).
  2. Apnea bars are emitted as **one** ``Scatter`` trace, with ``None``
     values used as separators between bars in the x/y arrays. Adding one
     trace per apnea inflated the JSON layout by O(n_apnea) and was the
     biggest bottleneck for high-apnea recordings.
  3. ``add_hline`` is used (not ``add_shape``) so the y=0 reference line
     stays on the layout's reserved fast path.
  4. Long-form Baseline / Recovery periods (~1700 s) are split in half and
     re-emitted as ``..._first_half.html`` and ``..._second_half.html``;
     a single >1500 s plot was already in the slow zone for the in-browser
     plotly renderer.

Don't reorder these steps without measuring -- it's tempting to "clean up"
this function and accidentally fall back to an O(n_breath) per-shape
codepath.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from ..analysis.breath_segmentation import Breath


def plot_breath_segmentation(
    time_s: np.ndarray,
    signal: np.ndarray,
    breaths: Sequence[Breath],
    output_path: Path,
    *,
    title: str = "Interactive Breath Analysis",
    sigh_indices: Optional[Iterable[int]] = None,
    apnea_periods: Optional[Sequence[dict]] = None,
    allow_split: bool = True,
    period_name: Optional[str] = None,
) -> Path:
    """Render the interactive HTML breath-segmentation plot. ``apnea_periods``
    is a sequence of ``{"start": float_s, "end": float_s, "duration": float_ms}``
    dicts, in absolute time. Setting ``allow_split=False`` disables the
    long-period split (used internally for the recursive halves). Returns the
    saved path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if time_s.size == 0:
        return output_path

    apnea_periods = list(apnea_periods or [])
    sigh_set = set(sigh_indices or [])

    # ---- step 1: hline (background) ---------------------------------------
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="grey", layer="below")

    # ---- step 2: bulk-assign all breath phase rectangles -----------------
    breath_phase_shapes: List[dict] = []
    for breath in breaths:
        breath_phase_shapes.append(
            dict(
                type="rect",
                x0=float(breath.ti_start_t), y0=0,
                x1=float(breath.ti_end_t),   y1=1,
                xref="x", yref="paper",
                fillcolor="rgba(0, 0, 255, 0.2)",
                layer="below", line_width=0,
            )
        )
        breath_phase_shapes.append(
            dict(
                type="rect",
                x0=float(breath.te_start_t), y0=0,
                x1=float(breath.te_end_t),   y1=1,
                xref="x", yref="paper",
                fillcolor="rgba(255, 0, 0, 0.2)",
                layer="below", line_width=0,
            )
        )
    fig.layout.shapes = tuple(breath_phase_shapes)

    # ---- step 3: signal trace --------------------------------------------
    fig.add_trace(go.Scatter(
        x=time_s, y=signal, mode="lines",
        name="Respiratory Signal", line=dict(color="black"),
    ))

    # ---- step 4: apnea bars as one trace, None-separated -----------------
    apnea_periods_in_segment = [
        ap for ap in apnea_periods
        if (time_s[0] <= ap["start"] <= time_s[-1]
            and time_s[0] <= ap["end"]   <= time_s[-1])
    ]
    if apnea_periods_in_segment:
        y_bar = float(np.max(signal)) * 1.15 if signal.size else 1.0
        apnea_x: List[Optional[float]] = []
        apnea_y: List[Optional[float]] = []
        apnea_text: List[str] = []
        for ap in apnea_periods_in_segment:
            apnea_x.extend([ap["start"], ap["end"], None])
            apnea_y.extend([y_bar, y_bar, None])
            apnea_text.extend([f"Apnea {ap['duration']:.0f}ms", "", ""])
        fig.add_trace(go.Scatter(
            x=apnea_x, y=apnea_y, mode="lines",
            line=dict(color="red", width=8),
            hovertext=apnea_text, hoverinfo="text",
            name="Apneas",
        ))

    # ---- step 5: optional sigh markers (single trace) --------------------
    if sigh_set:
        sigh_x = [float(breaths[i].ti_start_t) for i in sigh_set if 0 <= i < len(breaths)]
        sigh_y = [float(signal[breaths[i].ti_start_idx])
                  for i in sigh_set if 0 <= i < len(breaths)]
        if sigh_x:
            fig.add_trace(go.Scatter(
                x=sigh_x, y=sigh_y, mode="markers", name="Sigh",
                marker=dict(color="gold", size=10, symbol="triangle-up",
                            line=dict(color="black", width=0.5)),
            ))

    # ---- step 6: dummy legend swatches -----------------------------------
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="rgba(0, 0, 255, 0.5)", width=10),
        name="Inspiration",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="rgba(255, 0, 0, 0.5)", width=10),
        name="Expiration",
    ))

    # ---- step 7: layout + write -----------------------------------------
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Signal Amplitude (mL/s)",
        legend_title_text="Phases & Signal",
        hovermode="x unified",
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)

    # ---- step 8: split long Baseline / Recovery into halves --------------
    if allow_split and period_name in {"Baseline", "Recovery"}:
        total_span = float(time_s[-1] - time_s[0])
        if total_span > 0:
            half = time_s[0] + total_span / 2.0
            first_mask = time_s <= half
            second_mask = time_s >= half
            base = output_path.with_suffix("")
            stem = base.name
            first_breaths  = [b for b in breaths if b.te_end_t <= half]
            second_breaths = [b for b in breaths if b.ti_start_t >= half]
            first_apneas  = [ap for ap in apnea_periods if ap["end"]   <= half]
            second_apneas = [ap for ap in apnea_periods if ap["start"] >= half]
            plot_breath_segmentation(
                time_s[first_mask], signal[first_mask], first_breaths,
                output_path.with_name(stem + "_first_half.html"),
                title=title + " (first half)",
                sigh_indices=None,
                apnea_periods=first_apneas,
                allow_split=False, period_name=period_name,
            )
            plot_breath_segmentation(
                time_s[second_mask], signal[second_mask], second_breaths,
                output_path.with_name(stem + "_second_half.html"),
                title=title + " (second half)",
                sigh_indices=None,
                apnea_periods=second_apneas,
                allow_split=False, period_name=period_name,
            )
    return output_path
