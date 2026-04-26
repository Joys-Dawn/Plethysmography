"""
Configuration dataclasses for the plethysmography pipeline.

Every scientific parameter is exposed here so a full analysis is reproducible
from a single config object. Sub-configs are frozen so values cannot be
mutated mid-run by accident; the top-level PlethConfig is also frozen.

Mirrors the architecture used in the sibling Fiber_Photometry_ECoG project
(see fiber_photometry_ecog/core/config.py).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PeriodConfig:
    """Period-boundary parameters.

    Acclimation (0 to acclimation_duration_s) is dropped upstream and never
    analyzed. Habituation runs from habituation_start_s to habituation_end_s,
    truncated by the first lid-open event if it occurs before habituation_end_s.
    """
    acclimation_duration_s: float = 300.0          # 0-5 min, dropped
    habituation_start_s: float = 300.0             # 5 min
    habituation_end_s: float = 900.0               # 15 min
    postictal_duration_s: float = 600.0            # 10 min
    postictal_bin_s: float = 30.0                  # bin width for postictal binned plots
    ictal_bin_s: float = 1.0                       # bin width for ictal binned plots (per docs; old code used 5)


@dataclass(frozen=True)
class FilterConfig:
    """Per-period high-pass filter and artifact rejection."""
    hpf_cutoff_hz: float = 0.5
    hpf_order: int = 4
    artifact_sigma: float = 8.0                    # |x - mean| > artifact_sigma * std -> interpolate


@dataclass(frozen=True)
class LidDetectionConfig:
    """3-pass spike detection for lid open/close events.

    Pass 1: |signal| > spike_sigma * std, drop spikes closer than min_spike_distance_ms.
    Pass 2: keep spike only if |1-min pre-mean - 1-min post-mean| > baseline_shift_sigma_threshold * std.
    Pass 3: pair opens and closes; close is valid only if no further spike within pair_close_window_s.
    Boundary walk: from each spike, walk samples (forward for closes, backward for opens) while
    |signal - local_mean| > boundary_walk_threshold, then offset by boundary_walk_offset_samples to
    anchor the period boundary cleanly outside the lid-event artifact.
    """
    spike_sigma: float = 2.5
    min_spike_distance_ms: float = 1.0
    baseline_shift_sigma_threshold: float = 0.5
    pair_close_window_s: float = 900.0             # 15 min default; 5 min for one specific file (see metadata.py)
    boundary_walk_threshold: float = 5.0           # |signal - local_mean| threshold while walking
    boundary_walk_offset_samples: int = 1000       # additional offset away from spike after walking
    boundary_walk_local_mean_samples: int = 250    # window for the local mean used during walk


@dataclass(frozen=True)
class BreathConfig:
    """Breath segmentation parameters (zero-crossing on running-mean-centered signal)."""
    local_window_ms: float = 300.0                 # rolling-mean window for centering
    spurious_inspiration_amp_floor: float = 0.02   # absolute amp threshold below which an inspiration is spurious
    spurious_inspiration_sigma_frac: float = 0.5   # OR sigma fraction; whichever is larger
    short_segment_min_ms: float = 20.0             # segments shorter than this are merged into neighbors


@dataclass(frozen=True)
class SighConfig:
    """Sigh detection threshold (mean + sigma_multiplier * std of BASELINE PIF-to-PEF, per mouse).

    Differs from old code which used per-period stats. Per docs (pleth files meta data.xlsx,
    sheet 'approach to analysis'), sigh threshold is mouse-specific and baseline-derived.
    """
    sigma_multiplier: float = 2.0


@dataclass(frozen=True)
class ApneaConfig:
    """Apnea threshold (Ttot >= ttot_multiplier * baseline_median_ttot AND Ttot >= ttot_minimum_ms).

    `post_sigh_window_s` is the time after a sigh's expiration end within which an
    apnea is classified as post-sigh rather than spontaneous (matches old code's
    8-second window at analyze_data.py:271).
    """
    ttot_multiplier: float = 2.0
    ttot_minimum_ms: float = 400.0
    post_sigh_window_s: float = 8.0


@dataclass(frozen=True)
class StatsConfig:
    """Statistical analysis parameters."""
    fdr_method: str = "fdr_bh"                     # statsmodels multipletests method tag
    alpha: float = 0.05
    gee_correlation: str = "exchangeable"


@dataclass(frozen=True)
class PlethConfig:
    """Top-level config aggregating all subsystem configs."""
    period: PeriodConfig = field(default_factory=PeriodConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    lid: LidDetectionConfig = field(default_factory=LidDetectionConfig)
    breath: BreathConfig = field(default_factory=BreathConfig)
    sigh: SighConfig = field(default_factory=SighConfig)
    apnea: ApneaConfig = field(default_factory=ApneaConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlethConfig":
        sub_types = {f.name: f.default_factory() for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for name, default_inst in sub_types.items():
            sub_data = data.get(name)
            if sub_data is None:
                kwargs[name] = default_inst
            else:
                kwargs[name] = _replace_dataclass(default_inst, sub_data)
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> "PlethConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _replace_dataclass(instance: Any, overrides: Mapping[str, Any]) -> Any:
    if not is_dataclass(instance):
        raise TypeError(f"Expected dataclass instance, got {type(instance).__name__}")
    valid = {f.name for f in fields(instance)}
    bad = set(overrides) - valid
    if bad:
        raise ValueError(
            f"Unknown fields {sorted(bad)} for {type(instance).__name__}; "
            f"valid fields: {sorted(valid)}"
        )
    return replace(instance, **dict(overrides))
