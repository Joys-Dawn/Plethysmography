"""
Time-binned line plots for the postictal (30 s bins, first 5 min, P22-only)
and ictal (1 s bins) periods.

Mirrors ``old_code/postictal_binned_plots.py``: P22-only, 4 fixed groups (LR
WT / HR WT / LR Scn1a / HR Scn1a), per-bin mean and SEM across recordings,
SEM shaded as a translucent band. Figure size, fonts, and legend formatting
match old.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..analysis.breath_segmentation import segment_breaths, Breath
from ..core.config import BreathConfig
from ..core.data_models import Period
from ._common import display_label, filename_slug, make_axes, save_figure
from .colors import italicize_scn1a


# Postictal binned plot config: 30 s bins for the first 5 min (10 bins).
# Mirrors old_code/postictal_binned_plots.py:22-24.
_POSTICTAL_BIN_S = 30.0
_POSTICTAL_N_BINS = 10
_POSTICTAL_DURATION_S = _POSTICTAL_BIN_S * _POSTICTAL_N_BINS  # 300 s

# Group definitions for postictal. Verbatim from old_code GROUPS at line 28.
# (display_name, genotype, condition_value, color, marker)
_RISK_GROUPS: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("LR WT P22",        "WT",  "low_risk",  "#87CEEB", "o"),
    ("HR WT P22",        "WT",  "high_risk", "#0000FF", "o"),
    ("LR Scn1a+/- P22",  "het", "low_risk",  "#FFA07A", "o"),
    ("HR Scn1a+/- P22",  "het", "high_risk", "#FF0000", "o"),
)
_TREATMENT_GROUPS: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("WT Vehicle P22",         "WT",  "Vehicle", "#87CEEB", "o"),
    ("WT FFA P22",             "WT",  "FFA",     "#0000FF", "o"),
    ("Scn1a+/- Vehicle P22",   "het", "Vehicle", "#FFA07A", "o"),
    ("Scn1a+/- FFA P22",       "het", "FFA",     "#FF0000", "o"),
)


_DEFAULT_PARAMETERS: Tuple[str, ...] = (
    "mean_ttot_ms",
    "mean_frequency_bpm",
    "mean_ti_ms",
    "mean_te_ms",
    "mean_pif_centered_ml_s",
    "mean_pef_centered_ml_s",
    "mean_pif_to_pef_ml_s",
    "mean_tv_ml",
    "sigh_rate_per_min",
    "mean_sigh_duration_ms",
    "apnea_rate_per_min",
    "apnea_mean_ms",
    "cov_instant_freq",
    "alternate_cov",
    "pif_to_pef_cov",
)


def plot_postictal_binned(
    period_data: Sequence[Tuple[str, np.ndarray, np.ndarray, float]],
    metadata: Dict[str, Dict[str, str]],
    output_dir: Path,
    *,
    baseline_median_ttot_ms: Optional[Dict[str, float]] = None,
    breath_config: Optional[BreathConfig] = None,
    condition_col: str = "risk_clean",
) -> List[Path]:
    """Render the postictal-binned line plot family. P22 only, 30 s bins
    covering the first 5 min, 4 fixed groups, mean + SEM band per group.

    Args:
        period_data: One entry per recording's Immediate Postictal period as
            ``(file_basename, time_s, signal, fs)``.
        metadata: ``{file_basename: {"genotype": ..., "risk_clean" /
            "treatment_clean": ..., "age": ...}}``. P19 recordings are
            silently dropped.
        baseline_median_ttot_ms: Per-recording baseline median Ttot (ms). Used
            as the apnea-threshold reference. If a recording is absent from
            this map, falls back to the bin-local median.
    Returns the list of saved file paths."""
    return _plot_binned(
        period_data, metadata, output_dir,
        bin_s=_POSTICTAL_BIN_S, n_bins=_POSTICTAL_N_BINS,
        title_prefix="Postictal_5min_30s_bins",
        x_label="Time from postictal start (30 s bins)",
        breath_config=breath_config or BreathConfig(),
        condition_col=condition_col,
        baseline_median_ttot_ms=baseline_median_ttot_ms or {},
        x_tick_labels=[f"{int((b * _POSTICTAL_BIN_S) + _POSTICTAL_BIN_S / 2)}s"
                        for b in range(_POSTICTAL_N_BINS)],
    )


def plot_ictal_binned(
    period_data: Sequence[Tuple[str, np.ndarray, np.ndarray, float]],
    metadata: Dict[str, Dict[str, str]],
    output_dir: Path,
    *,
    bin_s: float = 1.0,
    baseline_median_ttot_ms: Optional[Dict[str, float]] = None,
    breath_config: Optional[BreathConfig] = None,
    condition_col: str = "risk_clean",
) -> List[Path]:
    """Render the ictal-binned line plot family with 1 s bins (matches the
    docs' specification, which the old code mistakenly used 5 s for)."""
    if not period_data:
        return []
    duration_s = max(t[-1] - t[0] for _, t, _, _ in period_data if t.size > 0)
    if duration_s <= 0:
        return []
    n_bins = max(1, int(np.ceil(duration_s / bin_s)))
    return _plot_binned(
        period_data, metadata, output_dir,
        bin_s=bin_s, n_bins=n_bins,
        title_prefix=f"Ictal_{int(bin_s)}s_bins",
        x_label=f"Time from ictal start ({int(bin_s)} s bins)",
        breath_config=breath_config or BreathConfig(),
        condition_col=condition_col,
        baseline_median_ttot_ms=baseline_median_ttot_ms or {},
        x_tick_labels=None,
    )


def _plot_binned(
    period_data: Sequence[Tuple[str, np.ndarray, np.ndarray, float]],
    metadata: Dict[str, Dict[str, str]],
    output_dir: Path,
    *,
    bin_s: float,
    n_bins: int,
    title_prefix: str,
    x_label: str,
    breath_config: BreathConfig,
    condition_col: str,
    baseline_median_ttot_ms: Dict[str, float],
    x_tick_labels: Optional[List[str]],
) -> List[Path]:
    if n_bins <= 0:
        return []
    groups = _TREATMENT_GROUPS if condition_col == "treatment_clean" else _RISK_GROUPS

    # Collect per-recording per-bin values:
    # by_group[group_key][parameter] = list of (n_bins,) arrays, one per recording.
    by_group: Dict[str, Dict[str, List[np.ndarray]]] = {
        g[0]: {p: [] for p in _DEFAULT_PARAMETERS} for g in groups
    }
    for basename, time_s, signal, fs in period_data:
        meta = metadata.get(basename)
        if meta is None:
            continue
        # P22 only -- old code excludes P19 entirely from postictal binned.
        if str(meta.get("age", "")).upper().lstrip("P") != "22":
            continue
        cond_value = str(meta.get(condition_col, ""))
        genotype = str(meta.get("genotype", ""))
        group_match = next(
            (g for g in groups if g[1] == genotype and g[2] == cond_value), None,
        )
        if group_match is None:
            continue
        period = _make_period(time_s, signal, fs)
        breaths = segment_breaths(period, breath_config)
        baseline_ttot = baseline_median_ttot_ms.get(basename)
        binned = _bin_metrics_per_recording(
            breaths, time_s[0], bin_s, n_bins,
            baseline_median_ttot_ms=baseline_ttot,
        )
        for p in _DEFAULT_PARAMETERS:
            by_group[group_match[0]][p].append(binned[p])

    saved: List[Path] = []
    bin_centers = np.arange(n_bins, dtype=float)
    for parameter in _DEFAULT_PARAMETERS:
        fig, ax = make_axes(figsize=(12.0, 10.0))
        any_drawn = False
        for group_name, _, _, color, marker in groups:
            arrays = by_group[group_name][parameter]
            if not arrays:
                continue
            stacked = np.vstack(arrays)
            with np.errstate(invalid="ignore"):
                means = np.nanmean(stacked, axis=0)
                ns = np.sum(~np.isnan(stacked), axis=0)
                sds = np.nanstd(stacked, axis=0, ddof=1)
            sems = np.where(ns > 1, sds / np.sqrt(np.maximum(ns, 1)), 0.0)
            valid = ~np.isnan(means)
            if not valid.any():
                continue
            ax.plot(bin_centers[valid], means[valid], color=color, linewidth=3,
                    marker=marker, markersize=8)
            ax.fill_between(
                bin_centers[valid],
                means[valid] - sems[valid],
                means[valid] + sems[valid],
                color=color, alpha=0.25,
            )
            any_drawn = True

        if not any_drawn:
            import matplotlib.pyplot as plt
            plt.close(fig)
            continue

        ax.set_ylabel(display_label(parameter), fontsize=24)
        ax.set_xlabel(x_label, fontsize=20)
        ax.set_xticks(bin_centers)
        if x_tick_labels is not None:
            ax.set_xticklabels(x_tick_labels, fontsize=16)
        else:
            ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        handles = [
            Line2D([0], [0], color=g[3], marker=g[4], linewidth=3, markersize=10)
            for g in groups
        ]
        labels = [italicize_scn1a(g[0]) for g in groups]
        ax.legend(handles, labels, fontsize=14, frameon=False, loc="best")
        out = output_dir / f"{title_prefix}_{filename_slug(parameter)}.png"
        save_figure(fig, out)
        saved.append(out)
    return saved


def _make_period(time_s: np.ndarray, signal: np.ndarray, fs: float) -> Period:
    return Period(
        name="Bin",
        start_s=float(time_s[0]),
        end_s=float(time_s[-1]),
        signal=signal,
        time_s=time_s,
        fs=fs,
        period_start_time=float(time_s[0]),
        lid_closure_time=float("nan"),
    )


def _bin_metrics_per_recording(
    breaths: List[Breath], t0: float, bin_s: float, n_bins: int,
    *,
    baseline_median_ttot_ms: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Aggregate per-breath metrics into per-bin scalars (one value per bin
    per parameter). Bins are anchored at ``t0``.

    ``baseline_median_ttot_ms`` is used as the apnea threshold reference; an
    apnea is a breath with ``ttot >= 2 * baseline`` AND ``ttot >= 400 ms``.
    Falls back to the bin-local median when not supplied (matches old code's
    fallback in ``analyze_data.py:240-243``).
    """
    out = {p: np.full((n_bins,), np.nan) for p in _DEFAULT_PARAMETERS}
    if not breaths:
        return out
    starts = np.array([b.ti_start_t - t0 for b in breaths])
    bin_idx = (starts // bin_s).astype(int)

    for b_idx in range(n_bins):
        in_bin = [b for b, i in zip(breaths, bin_idx) if i == b_idx]
        if not in_bin:
            continue
        ttot = np.array([b.ttot_ms for b in in_bin], dtype=float)
        ti = np.array([b.ti_ms for b in in_bin], dtype=float)
        te = np.array([b.te_ms for b in in_bin], dtype=float)
        pif = np.array([b.pif_centered for b in in_bin], dtype=float)
        pef = np.array([b.pef_centered for b in in_bin], dtype=float)
        peak_diff = np.array([b.peak_diff for b in in_bin], dtype=float)
        tv = np.array([b.tv_ml for b in in_bin], dtype=float)

        out["mean_ttot_ms"][b_idx] = float(np.nanmean(ttot))
        out["mean_frequency_bpm"][b_idx] = (len(in_bin) / bin_s) * 60.0
        out["mean_ti_ms"][b_idx] = float(np.nanmean(ti))
        out["mean_te_ms"][b_idx] = float(np.nanmean(te))
        out["mean_pif_centered_ml_s"][b_idx] = float(np.nanmean(pif))
        out["mean_pef_centered_ml_s"][b_idx] = float(np.nanmean(pef))
        out["mean_pif_to_pef_ml_s"][b_idx] = float(np.nanmean(peak_diff))
        out["mean_tv_ml"][b_idx] = float(np.nanmean(tv))
        # Per-bin sigh detection. Mirrors old_code/analyze_data.py:208-233 ---
        # mean+2SD threshold computed *within* this bin (not from the baseline
        # cache) -- old code's binned plots use bin-local stats.
        clean_pd = peak_diff[np.isfinite(peak_diff)]
        if clean_pd.size > 0:
            mean_amp = float(np.mean(clean_pd))
            std_amp = float(np.std(clean_pd))
            thresh = mean_amp + 2.0 * std_amp
            sigh_mask = np.isfinite(peak_diff) & (peak_diff >= thresh)
            sigh_durations = ttot[sigh_mask]
            n_sighs = int(sigh_mask.sum())
            out["sigh_rate_per_min"][b_idx] = (n_sighs / bin_s) * 60.0
            out["mean_sigh_duration_ms"][b_idx] = (
                float(np.mean(sigh_durations)) if sigh_durations.size > 0 else float("nan")
            )
        else:
            out["sigh_rate_per_min"][b_idx] = 0.0
        # Apnea: ttot >= 2*baseline_median AND ttot >= 400 ms.
        # Mirrors old_code/analyze_data.py:240-260 (with bin-local fallback
        # when baseline_median_ttot_ms is None).
        ref_ttot = baseline_median_ttot_ms
        if ref_ttot is None or ref_ttot <= 0:
            valid_for_med = ttot[ttot > 0]
            ref_ttot = float(np.median(valid_for_med)) if valid_for_med.size > 0 else None
        if ref_ttot is not None and ref_ttot > 0:
            apnea_thresh = 2.0 * ref_ttot
            apnea_mask = (ttot >= apnea_thresh) & (ttot >= 400.0)
            n_apneas = int(apnea_mask.sum())
            out["apnea_rate_per_min"][b_idx] = (n_apneas / bin_s) * 60.0
            out["apnea_mean_ms"][b_idx] = (
                float(np.mean(ttot[apnea_mask])) if n_apneas > 0 else float("nan")
            )
        else:
            out["apnea_rate_per_min"][b_idx] = 0.0
        valid_ttot = ttot[ttot > 0]
        if valid_ttot.size > 0:
            inst = 1000.0 / valid_ttot
            out["cov_instant_freq"][b_idx] = (
                float(np.std(inst) / np.mean(inst)) if np.mean(inst) else float("nan")
            )
            if valid_ttot.size > 1:
                rel = np.abs(np.diff(valid_ttot) / valid_ttot[1:])
                out["alternate_cov"][b_idx] = float(np.mean(rel)) if rel.size else float("nan")
        valid_pd = peak_diff[np.isfinite(peak_diff)]
        if valid_pd.size > 0 and np.mean(valid_pd) != 0:
            out["pif_to_pef_cov"][b_idx] = float(np.std(valid_pd) / np.mean(valid_pd))
    return out
