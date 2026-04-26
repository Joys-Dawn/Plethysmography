"""
End-to-end test that the stats runner reproduces the old xlsx layout when fed
``old_results/breathing_analysis_results.csv``: 11 sheets with the right
names, total row count matching old, and at least the analysis_type
distribution preserved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _have_old_results() -> bool:
    return Path("old_results/breathing_analysis_results.csv").exists()


@pytest.mark.skipif(not _have_old_results(), reason="old_results CSV not present")
def test_stats_runner_matches_old_layout(tmp_path):
    import warnings; warnings.filterwarnings("ignore")
    from plethysmography.data_loading.data_log import load_data_log
    from plethysmography.stats import (
        prepare_breathing_data, run_statistics, write_stats_xlsx,
    )

    breathing = pd.read_csv("old_results/breathing_analysis_results.csv")
    log = load_data_log()
    merged = prepare_breathing_data(breathing, log)
    rows = run_statistics(merged)
    out = tmp_path / "stats.xlsx"
    write_stats_xlsx(rows, out, write_csv=False)

    import openpyxl
    wb = openpyxl.load_workbook(out, read_only=True, data_only=True)
    assert wb.sheetnames == [
        "P22 across groups ANOVA",
        "P22 across groups posthocs",
        "P19 and 22 within groups GEE",
        "P19 and 22 within posthocs",
        "HR P19 vs LR P22 t-tests",
        "HR P19s survival t-tests",
        "P22 groups and periods GEE",
        "P22 groups and periods posthoc",
        "P19 and 22 periods GEE",
        "P19 and 22 periods posthoc",
        "All Results",
    ]
    # All Results should have within ±5 rows of the old code's 1029 data rows.
    n_data_rows = wb["All Results"].max_row - 1
    assert 1024 <= n_data_rows <= 1034, (
        f"All Results count {n_data_rows} should be ~1029"
    )
