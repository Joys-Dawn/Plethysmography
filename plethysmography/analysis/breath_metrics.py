"""
Aggregate breath metrics for a single (file, period).

Given:
  - the list of :class:`Breath` for a period,
  - a list of bool sigh flags (same length as breaths),
  - the list of :class:`ApneaEvent` for that period,

produce a :class:`BreathMetrics` row matching the column schema of
``old_results/breathing_analysis_results.csv`` (23 columns).

Frequency:
  ``mean_frequency_bpm = (num_breaths / period_duration_s) * 60``
  (matches old code at analyze_data.py:1811-1812 — count-over-duration
  rather than mean of instantaneous 1/Ttot.)

Variability:
  - ``cov_instant_freq``: SD/mean of instantaneous breathing frequency
    (1000/Ttot in Hz, computed across all breaths).
  - ``alternate_cov``: mean of |D_n − D_{n+1}| / D_{n+1} for successive
    Ttot values (per old code's "alternate" CoV).
  - ``pif_to_pef_cov``: SD/mean of peak_diff = PEF − PIF.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..core.data_models import ApneaEvent, BreathMetrics
from .apnea_detection import split_apnea_counts
from .breath_segmentation import Breath


def compute_breath_metrics(
    file_basename: str,
    period_name: str,
    period_duration_s: float,
    breaths: List[Breath],
    is_sigh: List[bool],
    apneas: List[ApneaEvent],
) -> BreathMetrics:
    """Aggregate per-breath data into a single :class:`BreathMetrics` row."""
    n = len(breaths)

    if n == 0:
        return _empty_metrics(file_basename, period_name, period_duration_s)

    ti = np.array([b.ti_ms for b in breaths], dtype=float)
    te = np.array([b.te_ms for b in breaths], dtype=float)
    ttot = np.array([b.ttot_ms for b in breaths], dtype=float)
    pif_c = np.array([b.pif_centered for b in breaths], dtype=float)
    pef_c = np.array([b.pef_centered for b in breaths], dtype=float)
    peak_diff = np.array([b.peak_diff for b in breaths], dtype=float)
    tv = np.array([b.tv_ml for b in breaths], dtype=float)

    # Counts -> per-minute rates
    sigh_count = int(sum(1 for s in is_sigh if s))
    sigh_durations = ttot[np.array(is_sigh, dtype=bool)] if is_sigh else np.array([])
    mean_sigh_dur_ms = float(np.mean(sigh_durations)) if sigh_durations.size > 0 else float("nan")

    apnea_total, apnea_spont, apnea_post = split_apnea_counts(apneas)
    apnea_durations = np.array([a.apnea_duration_ms for a in apneas], dtype=float)
    apnea_spont_durations = np.array(
        [a.apnea_duration_ms for a in apneas if not a.is_post_sigh], dtype=float,
    )
    apnea_post_durations = np.array(
        [a.apnea_duration_ms for a in apneas if a.is_post_sigh], dtype=float,
    )

    if period_duration_s > 0:
        sigh_rate = (sigh_count / period_duration_s) * 60.0
        apnea_rate = (apnea_total / period_duration_s) * 60.0
        apnea_spont_rate = (apnea_spont / period_duration_s) * 60.0
        apnea_post_rate = (apnea_post / period_duration_s) * 60.0
    else:
        sigh_rate = apnea_rate = apnea_spont_rate = apnea_post_rate = 0.0

    # Mean frequency: count-over-duration (matches old code line 1811).
    mean_frequency_bpm = (n / period_duration_s) * 60.0 if period_duration_s > 0 else 0.0

    # Variability.
    inst_freq_hz = 1000.0 / ttot[ttot > 0] if (ttot > 0).any() else np.array([])
    if inst_freq_hz.size > 0:
        mean_freq_hz = float(np.mean(inst_freq_hz))
        std_freq_hz = float(np.std(inst_freq_hz))
        cov_instant_freq = std_freq_hz / mean_freq_hz if mean_freq_hz else float("nan")
    else:
        cov_instant_freq = float("nan")

    if n > 1:
        rel_diffs = []
        for i in range(n - 1):
            if ttot[i + 1] > 0:
                rel_diffs.append(abs((ttot[i] - ttot[i + 1]) / ttot[i + 1]))
        alternate_cov = float(np.mean(rel_diffs)) if rel_diffs else float("nan")
    else:
        alternate_cov = float("nan")

    valid_pd = peak_diff[np.isfinite(peak_diff)]
    if valid_pd.size > 0:
        mean_pd = float(np.mean(valid_pd))
        std_pd = float(np.std(valid_pd))
        pif_to_pef_cov = std_pd / mean_pd if mean_pd != 0 else float("nan")
    else:
        pif_to_pef_cov = float("nan")

    return BreathMetrics(
        file_basename=file_basename,
        period=period_name,
        period_duration_s=float(period_duration_s),
        num_breaths_detected=n,
        mean_ttot_ms=float(np.mean(ttot)) if ttot.size > 0 else 0.0,
        mean_frequency_bpm=mean_frequency_bpm,
        mean_ti_ms=float(np.mean(ti)) if ti.size > 0 else float("nan"),
        mean_te_ms=float(np.mean(te)) if te.size > 0 else float("nan"),
        mean_pif_centered_ml_s=float(np.mean(pif_c)) if pif_c.size > 0 else float("nan"),
        mean_pef_centered_ml_s=float(np.mean(pef_c)) if pef_c.size > 0 else float("nan"),
        mean_pif_to_pef_ml_s=float(np.mean(peak_diff)) if peak_diff.size > 0 else 0.0,
        mean_tv_ml=float(np.mean(tv)) if tv.size > 0 else 0.0,
        sigh_rate_per_min=sigh_rate,
        mean_sigh_duration_ms=mean_sigh_dur_ms,
        cov_instant_freq=cov_instant_freq,
        alternate_cov=alternate_cov,
        pif_to_pef_cov=pif_to_pef_cov,
        apnea_rate_per_min=apnea_rate,
        apnea_mean_ms=float(np.mean(apnea_durations)) if apnea_durations.size > 0 else float("nan"),
        apnea_spont_rate_per_min=apnea_spont_rate,
        apnea_spont_mean_ms=float(np.mean(apnea_spont_durations)) if apnea_spont_durations.size > 0 else float("nan"),
        apnea_postsigh_rate_per_min=apnea_post_rate,
        apnea_postsigh_mean_ms=float(np.mean(apnea_post_durations)) if apnea_post_durations.size > 0 else float("nan"),
    )


def _empty_metrics(file_basename: str, period_name: str, period_duration_s: float) -> BreathMetrics:
    nan = float("nan")
    return BreathMetrics(
        file_basename=file_basename,
        period=period_name,
        period_duration_s=float(period_duration_s),
        num_breaths_detected=0,
        mean_ttot_ms=0.0,
        mean_frequency_bpm=0.0,
        mean_ti_ms=nan,
        mean_te_ms=nan,
        mean_pif_centered_ml_s=nan,
        mean_pef_centered_ml_s=nan,
        mean_pif_to_pef_ml_s=0.0,
        mean_tv_ml=0.0,
        sigh_rate_per_min=0.0,
        mean_sigh_duration_ms=nan,
        cov_instant_freq=nan,
        alternate_cov=nan,
        pif_to_pef_cov=nan,
        apnea_rate_per_min=0.0,
        apnea_mean_ms=nan,
        apnea_spont_rate_per_min=0.0,
        apnea_spont_mean_ms=nan,
        apnea_postsigh_rate_per_min=0.0,
        apnea_postsigh_mean_ms=nan,
    )
