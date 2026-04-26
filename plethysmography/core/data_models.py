"""
Core data containers for the plethysmography pipeline.

Each Recording carries metadata derived from the data log; LidEvents holds the
detected lid open/close times; Periods are the temporal segments after slicing,
filtering, and artifact removal; BreathMetrics, ApneaEvent, and BaselineCache
hold the analysis outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# Period name constants. Use these instead of bare string literals everywhere.
HABITUATION = "Habituation"
BASELINE = "Baseline"
ICTAL = "Ictal"
IMMEDIATE_POSTICTAL = "Immediate Postictal"
RECOVERY = "Recovery"
PERIOD_NAMES: Tuple[str, ...] = (
    HABITUATION,
    BASELINE,
    ICTAL,
    IMMEDIATE_POSTICTAL,
    RECOVERY,
)


@dataclass
class Recording:
    """One recording session = one EDF file + its data-log metadata.

    Fields populated at load time:
      - file_basename: e.g. "250304 4055 p22" (no extension, no extra whitespace)
      - edf_path: absolute path to the EDF
      - mouse_id: e.g. "4055" (token 1 of the basename)
      - age: "P19" or "P22" (token 2)
      - genotype: "het" or "WT"
      - risk: "HR" / "LR" / None (experiment 1 only)
      - treatment: "FFA" / "Vehicle" / None (experiment 2 only)
      - cohort: experiment folder name (e.g. "experiment 1 - LR vs HR comparison")
      - seizure_offset_s: from the all-trials offset column; NaN if SUDEP/missing
      - n_seizures: 0 / 1 / >1
      - is_sudep / is_survivor: parsed from the P19-trace SUDEP column (mutually exclusive,
        both False if outside experiment 4 cohort)
      - racine_max: maximum Racine score, if recorded
      - fs: sampling rate (set after EDF read)
    """
    file_basename: str
    edf_path: Path
    mouse_id: str
    age: str
    genotype: str
    cohort: str
    risk: Optional[str] = None
    treatment: Optional[str] = None
    seizure_offset_s: Optional[float] = None
    n_seizures: int = 0
    is_sudep: bool = False
    is_survivor: bool = False
    racine_max: Optional[int] = None
    fs: Optional[float] = None


@dataclass
class LidEvents:
    """Detected lid open/close times for a recording.

    Both raw (from the 3-pass spike detector) and adjusted (after the boundary-walk
    refinement that pushes the boundary clear of the lid artifact) are kept. Periods
    are sliced using adjusted times; raw times are kept for plotting/debugging.
    Even-indexed entries are opens, odd-indexed are closes (consistent with old code).
    """
    raw_spike_times_s: List[float] = field(default_factory=list)
    adjusted_spike_times_s: List[float] = field(default_factory=list)

    @property
    def open_times_s(self) -> List[float]:
        return self.adjusted_spike_times_s[0::2]

    @property
    def close_times_s(self) -> List[float]:
        return self.adjusted_spike_times_s[1::2]


@dataclass
class Period:
    """A single time segment of a recording (Habituation / Baseline / Ictal / etc).

    `period_start_time` is the first timestamp actually present in time_s after
    slicing (may differ slightly from start_s due to strict-inequality masks);
    `lid_closure_time` is the lid closure that anchors apnea timing for this
    period (NaN for Habituation; close 1 for Baseline; close 2 for
    Ictal / Immediate Postictal / Recovery). Both are stored as scalars and
    written to the per-sample CSV as constant columns to match the old format.
    """
    name: str
    start_s: float
    end_s: float
    signal: np.ndarray
    time_s: np.ndarray
    fs: float
    period_start_time: float = np.nan
    lid_closure_time: float = np.nan

    @property
    def duration_s(self) -> float:
        if self.time_s.size < 2:
            return 0.0
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def is_empty(self) -> bool:
        return self.signal.size == 0


@dataclass
class BreathMetrics:
    """One row's worth of breath-summary statistics for (file, period).

    Field names match old_results/breathing_analysis_results.csv columns exactly
    so the regression test compares like-for-like. The output CSV is written by
    serializing a list[BreathMetrics] in this column order.
    """
    file_basename: str
    period: str
    period_duration_s: float
    num_breaths_detected: int
    mean_ttot_ms: float
    mean_frequency_bpm: float
    mean_ti_ms: float
    mean_te_ms: float
    mean_pif_centered_ml_s: float
    mean_pef_centered_ml_s: float
    mean_pif_to_pef_ml_s: float
    mean_tv_ml: float
    sigh_rate_per_min: float
    mean_sigh_duration_ms: float
    cov_instant_freq: float
    alternate_cov: float
    pif_to_pef_cov: float
    apnea_rate_per_min: float
    apnea_mean_ms: float
    apnea_spont_rate_per_min: float
    apnea_spont_mean_ms: float
    apnea_postsigh_rate_per_min: float
    apnea_postsigh_mean_ms: float


@dataclass
class ApneaEvent:
    """One detected apnea — written to apnea_list.xlsx.

    `is_post_sigh` distinguishes post-sigh apneas (apnea immediately following a
    detected sigh) from spontaneous apneas. Used for the apnea_postsigh / apnea_spont
    breakdown in BreathMetrics.
    """
    file_basename: str
    period: str
    apnea_start_s_from_lid_closure: float
    apnea_start_s_from_period_start: float
    apnea_duration_ms: float
    is_post_sigh: bool = False


@dataclass
class BaselineCache:
    """Per-recording statistics derived from the Baseline period only.

    Computed in pass 1 of the analysis driver, then reused for every other period of
    the same recording. Two roles:
      1. Apnea threshold: Ttot >= max(ttot_multiplier * median_ttot_ms, ttot_minimum_ms).
         (Same as old code, line 240-260 of analyze_data.py.)
      2. Sigh threshold: peak_diff > mean_pif_to_pef_ml_s + sigma_multiplier * std_pif_to_pef_ml_s.
         (Replaces the old per-period definition.)
    n_breaths == 0 or std == 0 indicates a degenerate baseline; callers fall back to
    per-period stats with a warning.
    """
    file_basename: str
    median_ttot_ms: float
    mean_pif_to_pef_ml_s: float
    std_pif_to_pef_ml_s: float
    n_breaths: int

    @property
    def is_degenerate(self) -> bool:
        return self.n_breaths == 0 or self.std_pif_to_pef_ml_s == 0.0
