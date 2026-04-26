"""
Multi-sheet xlsx writer for the stats output. Sheet layout matches
``old_results/statistical_results.xlsx``:

  1. P22 across groups ANOVA          (analysis_type=anova)
  2. P22 across groups posthocs       (analysis_type=posthoc)
  3. P19 and 22 within groups GEE     (analysis_type=gee)
  4. P19 and 22 within posthocs       (analysis_type=gee_posthoc)
  5. HR P19 vs LR P22 t-tests         (analysis_type=developmental_ttest)
  6. HR P19s survival t-tests         (analysis_type=survival_prediction)
  7. P22 groups and periods GEE       (analysis_type=across_periods_independent)
  8. P22 groups and periods posthoc   (analysis_type=across_periods_independent_posthoc)
  9. P19 and 22 periods GEE           (analysis_type=across_periods_dependent)
  10. P19 and 22 periods posthoc      (analysis_type=across_periods_dependent_posthoc)
  11. All Results                     (everything concatenated)

Empty per-analysis sheets are skipped so the output stays clean for
experiments that only run a subset of the suite (e.g. experiment 4 produces
only a "HR P19s survival t-tests" sheet plus "All Results").
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "parameter", "period", "analysis_type", "comparison",
    "group_summaries", "p_value",
    "p_corrected", "significant_corrected",
    "category", "correction_group", "notes",
)

_SHEET_LAYOUT: Tuple[Tuple[str, str], ...] = (
    ("P22 across groups ANOVA", "anova"),
    ("P22 across groups posthocs", "posthoc"),
    ("P19 and 22 within groups GEE", "gee"),
    ("P19 and 22 within posthocs", "gee_posthoc"),
    ("HR P19 vs LR P22 t-tests", "developmental_ttest"),
    ("HR P19s survival t-tests", "survival_prediction"),
    ("P22 groups and periods GEE", "across_periods_independent"),
    ("P22 groups and periods posthoc", "across_periods_independent_posthoc"),
    ("P19 and 22 periods GEE", "across_periods_dependent"),
    ("P19 and 22 periods posthoc", "across_periods_dependent_posthoc"),
)


def write_stats_xlsx(
    results_rows: Sequence[dict],
    output_path: str | Path,
    *,
    write_csv: bool = True,
) -> Path:
    """Serialize ``results_rows`` to a multi-sheet xlsx (and optionally a flat
    csv side-by-side). Returns the resolved xlsx path. The output schema is
    fixed by ``_OUTPUT_COLUMNS``; any extra fields on input rows are dropped."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(list(results_rows))
    if df.empty:
        # Still emit an empty workbook so downstream tooling has a known path.
        df = pd.DataFrame(columns=list(_OUTPUT_COLUMNS))
    else:
        for col in _OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df.reindex(columns=list(_OUTPUT_COLUMNS))

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, analysis_type in _SHEET_LAYOUT:
            sheet_df = df[df["analysis_type"] == analysis_type]
            if not sheet_df.empty:
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        df.to_excel(writer, sheet_name="All Results", index=False)

    if write_csv:
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

    return output_path
