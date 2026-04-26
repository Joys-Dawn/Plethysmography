"""
Tests for the encoded exclusions and per-file overrides. These are pure data
checks — they confirm the metadata module is internally consistent (period
tokens are valid, helper functions return the expected types) without
loading any signal data.
"""

from __future__ import annotations

from plethysmography.core.metadata import (
    EXCLUSIONS,
    PER_FILE_LID_OVERRIDES,
    PER_FILE_PREPROCESS_OVERRIDES,
    PER_FILE_ANALYSIS_OVERRIDES,
    get_analysis_override,
    get_lid_override,
    get_preprocess_override,
    is_excluded,
    should_skip_preprocess,
)


def test_known_exclusions_are_present():
    assert "260117 5308 p22" in EXCLUSIONS
    assert "260117 5310 p22" in EXCLUSIONS
    assert "250423 4269 p22" in EXCLUSIONS
    assert "250307 4051 p22" in EXCLUSIONS


def test_should_skip_preprocess_only_for_seizureless_files():
    assert should_skip_preprocess("260117 5308 p22") is True
    assert should_skip_preprocess("260117 5310 p22") is True
    assert should_skip_preprocess("250423 4269 p22") is False  # excluded all but does preprocess
    assert should_skip_preprocess("250307 4051 p22") is False
    assert should_skip_preprocess("nonexistent file") is False


def test_is_excluded_period_specific():
    # 250307 4051 p22 is excluded only from Immediate Postictal + Recovery
    assert is_excluded("250307 4051 p22", "Baseline") is False
    assert is_excluded("250307 4051 p22", "Immediate Postictal") is True
    assert is_excluded("250307 4051 p22", "Recovery") is True

    # 250423 4269 p22 is excluded everywhere
    assert is_excluded("250423 4269 p22", "Baseline") is True
    assert is_excluded("250423 4269 p22") is True


def test_lid_overrides_have_expected_shapes():
    assert get_lid_override("250307 4051 p22")["type"] == "hardcoded_spike_times"
    assert get_lid_override("250304 4056 p22")["type"] == "first_pair_close_window_s"
    assert get_lid_override("250423 4269 p22")["type"] == "keep_only_index"
    assert get_lid_override("nonexistent") is None


def test_preprocess_override_shape():
    o = get_preprocess_override("250304 4056 p22")
    assert o is not None
    assert o["type"] == "remove_segment_between_first_open_and_close"


def test_analysis_override_shape():
    o = get_analysis_override("250304 4056 p22", "Recovery")
    assert o is not None
    assert o["apply_lowpass_hz"] == 6.0
    assert get_analysis_override("250304 4056 p22", "Baseline") is None
