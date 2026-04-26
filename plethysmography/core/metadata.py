"""
Project-wide overrides and exclusions.

Encoded as data so that all per-file special-cases live in one place. Each entry
references the original location in old_code/ so the historical reasoning is
recoverable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .data_models import (
    BASELINE,
    HABITUATION,
    ICTAL,
    IMMEDIATE_POSTICTAL,
    RECOVERY,
)


# ----------------------------------------------------------------------------
# Exclusions
# ----------------------------------------------------------------------------
# Periods listed are the ones to EXCLUDE for that file.
#   ["preprocess"] = skip preprocessing entirely (file never enters the pipeline)
#   ["all"]        = preprocess but exclude from every analysis / cohort
#   [period, ...]  = exclude only from those specific periods
EXCLUSIONS: Dict[str, List[str]] = {
    # Two recordings that did not have a seizure and should not be analyzed at all.
    # See old_code/pleth_preprocessing.py:438 (skip_basenames).
    "260117 5308 p22": ["preprocess"],
    "260117 5310 p22": ["preprocess"],
    # SUDEP — exclude from every analysis. Boundary-walk also keeps only spike[1]
    # for this file (see PER_FILE_LID_OVERRIDES).
    # See old_code/analyze_data.py per-file override at line ~1785 area.
    "250423 4269 p22": ["all"],
    # Mouse died in the chamber during postictal; first half of recording is OK.
    # Exclude only the late-recording periods.
    "250307 4051 p22": [IMMEDIATE_POSTICTAL, RECOVERY],
}


# ----------------------------------------------------------------------------
# Per-file lid detection overrides
# ----------------------------------------------------------------------------
# Verbatim from old_code/pleth_preprocessing.py:159-261 (calculate_lid_open_and_close).
# Each entry tells lid_detection.py how to short-circuit or modify the standard
# 3-pass + boundary-walk algorithm for that file.
PER_FILE_LID_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # The 3-pass detector returns the wrong spikes for this file; replaced wholesale
    # with hardcoded times that were validated by hand.
    # old_code/pleth_preprocessing.py:162-163.
    "250307 4051 p22": {
        "type": "hardcoded_spike_times",
        "spike_times_s": [928.654, 1307.571, 3182.366, 3500.115, 3559.647],
    },
    # First close happened sooner than the 15-min default cutoff would allow.
    # Apply 5 min cutoff for the FIRST open/close pair only; subsequent pairs
    # use the default 15 min.
    # old_code/pleth_preprocessing.py:211-214.
    "250304 4056 p22": {
        "type": "first_pair_close_window_s",
        "value": 300.0,
    },
    # After the 3-pass detector, keep only spike index 1 (drop the others).
    # old_code/pleth_preprocessing.py:258-259.
    "250423 4269 p22": {
        "type": "keep_only_index",
        "value": 1,
    },
}


# ----------------------------------------------------------------------------
# Per-file preprocessing overrides
# ----------------------------------------------------------------------------
# Special-case signal/time edits applied to the raw EDF data BEFORE period slicing.
# Verbatim from old_code/pleth_preprocessing.py:498-508.
PER_FILE_PREPROCESS_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Variable-length segment between adjusted_spike_times[0] and [1] is removed,
    # and those two spikes are dropped from the list (so the original [2:] becomes the
    # new spike sequence). This file had a spurious open/close pair during habituation.
    # old_code/pleth_preprocessing.py:498-508.
    "250304 4056 p22": {
        "type": "remove_segment_between_first_open_and_close",
    },
}


# ----------------------------------------------------------------------------
# Per-file analysis overrides
# ----------------------------------------------------------------------------
# Applied during breath segmentation for specific (file, period) combinations.
# Verbatim from old_code/analyze_data.py:1785.
PER_FILE_ANALYSIS_OVERRIDES: Dict[Tuple[str, str], Dict[str, Any]] = {
    # Apply a 6 Hz low-pass after running-mean centering for this file's Recovery
    # period only (signal had high-frequency noise that broke segmentation).
    ("250304 4056 p22", RECOVERY): {"apply_lowpass_hz": 6.0},
}


_VALID_PERIOD_TOKENS = frozenset({
    "preprocess",
    "all",
    HABITUATION,
    BASELINE,
    ICTAL,
    IMMEDIATE_POSTICTAL,
    RECOVERY,
})


def _validate_exclusions() -> None:
    """Defensive check: any period name in EXCLUSIONS must be valid."""
    for basename, periods in EXCLUSIONS.items():
        unknown = set(periods) - _VALID_PERIOD_TOKENS
        if unknown:
            raise ValueError(
                f"EXCLUSIONS[{basename!r}] contains unknown period tokens: {sorted(unknown)}. "
                f"Valid: {sorted(_VALID_PERIOD_TOKENS)}"
            )


_validate_exclusions()


# ----------------------------------------------------------------------------
# Lookup helpers
# ----------------------------------------------------------------------------

def should_skip_preprocess(file_basename: str) -> bool:
    """True if the file should not even be preprocessed."""
    return "preprocess" in EXCLUSIONS.get(file_basename, [])


def is_excluded(file_basename: str, period_name: Optional[str] = None) -> bool:
    """True if this file (and optionally this period) should be excluded from analysis.

    Excluded means do not include in cohort statistics / plots. Files marked
    "preprocess" are also excluded from analysis. Files marked "all" are excluded
    from every period.
    """
    periods = EXCLUSIONS.get(file_basename, [])
    if not periods:
        return False
    if "preprocess" in periods or "all" in periods:
        return True
    if period_name is None:
        return False
    return period_name in periods


def get_lid_override(file_basename: str) -> Optional[Dict[str, Any]]:
    return PER_FILE_LID_OVERRIDES.get(file_basename)


def get_preprocess_override(file_basename: str) -> Optional[Dict[str, Any]]:
    return PER_FILE_PREPROCESS_OVERRIDES.get(file_basename)


def get_analysis_override(file_basename: str, period_name: str) -> Optional[Dict[str, Any]]:
    return PER_FILE_ANALYSIS_OVERRIDES.get((file_basename, period_name))
