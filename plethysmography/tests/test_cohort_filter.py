"""
Item A — Column G ('include for population analysis') is the authoritative
file-level population filter applied in
``stats.helpers.prepare_breathing_data`` after the data-log merge.

These tests confirm:
  * G != 1 recordings are dropped (independently of the period-level SUDEP
    exclusions — proven by dropping ``250307 4051 p22`` Baseline, which the
    period-level EXCLUSIONS would keep);
  * G == 1 recordings survive and the COL_INCLUDE column survives the merge;
  * a data log missing Column G fails fast with a clear error.
"""

from __future__ import annotations

import pandas as pd
import pytest

from plethysmography.data_loading.data_log import (
    COL_AGE,
    COL_CONDITION,
    COL_FILENAME,
    COL_GENOTYPE,
    COL_INCLUDE,
    COL_SUDEP,
)
from plethysmography.stats.helpers import prepare_breathing_data


def _breathing_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_basename": [
                "250304 4055 p22",   # G=1  -> keep
                "250307 4051 p22",   # G=0  -> drop (Baseline NOT in EXCLUSIONS)
                "250423 4269 p22",   # G=0  -> drop
                "999999 9999 p19",   # G=1  -> keep
            ],
            "period": ["Baseline", "Baseline", "Baseline", "Ictal"],
            "mean_frequency_bpm": [120.0, 130.0, 140.0, 150.0],
        }
    )


def _data_log_df(include_col: bool = True) -> pd.DataFrame:
    cols = {
        COL_FILENAME: [
            "250304 4055 p22",
            "250307 4051 p22",
            "250423 4269 p22",
            "999999 9999 p19",
        ],
        COL_GENOTYPE: ["het", "het", "het", "WT"],
        COL_AGE: [22, 22, 22, 19],
        COL_CONDITION: ["high risk", "high risk", "high risk", "low risk"],
        COL_SUDEP: [None, None, None, None],
    }
    if include_col:
        cols[COL_INCLUDE] = [1, 0, 0, 1]
    return pd.DataFrame(cols)


def test_g_filter_drops_include_zero_rows():
    merged = prepare_breathing_data(_breathing_df(), _data_log_df())
    kept = sorted(merged["file_basename"].unique())
    assert kept == ["250304 4055 p22", "999999 9999 p19"]
    assert len(merged) == 2


def test_g_filter_is_independent_of_period_exclusions():
    # 250307 4051 p22 Baseline is NOT excluded by core.metadata.EXCLUSIONS
    # (only its Immediate Postictal + Recovery are). Without the Column G
    # filter the Baseline row would survive — its absence proves G did it.
    merged = prepare_breathing_data(
        _breathing_df(), _data_log_df(), apply_sudep_exclusions=False
    )
    assert "250307 4051 p22" not in set(merged["file_basename"])
    assert sorted(merged["file_basename"].unique()) == [
        "250304 4055 p22",
        "999999 9999 p19",
    ]


def test_col_include_survives_the_merge():
    merged = prepare_breathing_data(_breathing_df(), _data_log_df())
    assert COL_INCLUDE in merged.columns
    # every surviving row is include == 1
    assert (pd.to_numeric(merged[COL_INCLUDE]) == 1).all()


def test_missing_column_g_fails_fast():
    with pytest.raises(KeyError, match="include for population analysis"):
        prepare_breathing_data(_breathing_df(), _data_log_df(include_col=False))
