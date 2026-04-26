"""
Shared utilities for the stats package: data preparation, group-summary
formatting (``mean ± SEM (n)``), and statsmodels warning capture.

Behavior matches old_code/breathing_statistics.py — the data preparation step
applies the same SUDEP exclusions and column normalization, but reads the
breathing CSV and merges with the cleaned data log produced by
:mod:`plethysmography.data_loading.data_log`.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.metadata import is_excluded


# ---------------------------------------------------------------------------
# Warning capture (statsmodels GEE / OLS convergence chatter)
# ---------------------------------------------------------------------------
_WARNING_KEYWORDS = (
    "singular", "boundary", "Hessian", "converge",
    "Gradient", "positive definite", "Retrying",
    "iteration", "covariance", "Iteration",
)


@contextmanager
def capture_warnings() -> Iterator[List[str]]:
    """Context manager that captures statsmodels-style warnings emitted while
    fitting models. Yields a list that gets populated with the captured
    messages (one entry per qualifying warning). Intermediate "Retrying"
    messages are skipped."""
    captured: List[str] = []

    def handler(message, category, filename, lineno, file=None, line=None):
        msg_str = str(message)
        if any(key in msg_str for key in _WARNING_KEYWORDS):
            if "Retrying" not in msg_str:
                captured.append(msg_str)

    old_showwarning = warnings.showwarning
    warnings.showwarning = handler
    try:
        yield captured
    finally:
        warnings.showwarning = old_showwarning


def get_convergence_notes(warnings_list: Sequence[str]) -> Optional[str]:
    """Reduce a list of captured warnings to a deduplicated, semicolon-separated
    short-form notes string. Returns ``None`` if there were no warnings."""
    if not warnings_list:
        return None
    seen: set[str] = set()
    out: List[str] = []
    for w in warnings_list:
        lw = w.lower()
        if "singular" in lw:
            key = "singular covariance"
        elif "boundary" in lw:
            key = "MLE on boundary"
        elif "hessian" in lw and "positive definite" in lw:
            key = "Hessian not positive definite"
        elif "converge" in lw:
            key = "convergence issue"
        elif "gradient" in lw:
            key = "gradient optimization failed"
        elif "iteration" in lw:
            key = "iteration limit"
        else:
            key = w[:50]
        if key not in seen:
            seen.add(key)
            out.append(key)
    return "; ".join(out) if out else None


# ---------------------------------------------------------------------------
# Group-summary formatting
# ---------------------------------------------------------------------------
def format_mean_sem_n(series: pd.Series) -> str:
    """Format one group as ``mean ± SEM (n=N)``. SEM = std / sqrt(n).

    For n=0 returns ``"— (0)"``; for n=1 the SEM is omitted because std is
    undefined."""
    series = series.dropna()
    n = len(series)
    if n == 0:
        return "— (0)"
    mean_val = float(series.mean())
    if n < 2:
        return f"{mean_val:.4g} (n={n})"
    sem = float(series.std(ddof=1)) / (n ** 0.5)
    return f"{mean_val:.4g} ± {sem:.4g} (n={n})"


def format_mean_sem_n_scalar(
    mean_val: Optional[float], std_val: Optional[float], n: Optional[int],
) -> str:
    """Same format as :func:`format_mean_sem_n` but from precomputed scalars
    (used by the developmental / survival t-test result-row builders)."""
    if n is None or n == 0:
        return "— (0)"
    if mean_val is None:
        mean_val = float("nan")
    if n < 2:
        return f"{float(mean_val):.4g} (n={n})"
    if std_val is None or (isinstance(std_val, float) and np.isnan(std_val)):
        return f"{float(mean_val):.4g} (n={n})"
    sem = float(std_val) / (n ** 0.5)
    return f"{float(mean_val):.4g} ± {sem:.4g} (n={n})"


def compute_group_summaries(
    data: Optional[pd.DataFrame],
    param: str,
    group_cols: Sequence[str],
    label_sep: str = " ",
) -> str:
    """Build a single ``"mean ± SEM (n) group; ..."`` string by grouping
    ``data`` on ``group_cols``. Used to populate the ``group_summaries`` column
    in the stats output. Returns an empty string when the input is empty or
    columns are missing."""
    if data is None or data.empty or param not in data.columns:
        return ""
    data = data.dropna(subset=[param])
    if data.empty:
        return ""
    if any(c not in data.columns for c in group_cols):
        return ""
    grouped = data.groupby(list(group_cols), sort=True)[param]
    parts: List[str] = []
    for names, grp in grouped:
        if not isinstance(names, tuple):
            names = (names,)
        label = label_sep.join(str(x) for x in names)
        parts.append(f"{format_mean_sem_n(grp)} {label}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Data preparation: merge breathing CSV with metadata
# ---------------------------------------------------------------------------
def prepare_breathing_data(
    breathing_df: pd.DataFrame,
    data_log_df: pd.DataFrame,
    *,
    apply_sudep_exclusions: bool = True,
) -> pd.DataFrame:
    """Merge ``breathing_df`` (one row per (file, period)) with ``data_log_df``
    (one row per recording) and add the cleaned columns the downstream stats
    functions expect:

      - ``mouse_id``: parsed from ``file_basename`` (second whitespace token).
      - ``genotype_clean``: ``"het"`` or ``"WT"``.
      - ``risk_clean``: ``"high_risk"`` or ``"low_risk"`` for exp-1 rows;
        ``None`` otherwise.
      - ``treatment_clean``: ``"FFA"`` or ``"Vehicle"`` for exp-2 rows;
        ``None`` otherwise.
      - ``age_clean``: integer age in days (19 or 22).
      - ``sudep_status``: ``"survivor"`` / ``"sudep"`` / ``None``.

    SUDEP exclusions from :data:`plethysmography.core.metadata.EXCLUSIONS` are
    applied unless ``apply_sudep_exclusions=False``.
    """
    from ..data_loading.data_log import (
        COL_FILENAME, COL_GENOTYPE, COL_AGE, COL_CONDITION, COL_SUDEP,
    )

    log = data_log_df.copy()
    log.columns = log.columns.str.strip()

    log = log.rename(columns={COL_FILENAME: "file_basename"})
    if "file_basename" in log.columns:
        log["file_basename"] = log["file_basename"].astype(str).str.strip()

    merged = breathing_df.merge(log, on="file_basename", how="inner")
    if merged.empty:
        return merged

    if apply_sudep_exclusions:
        keep_mask = merged.apply(
            lambda r: not is_excluded(r["file_basename"], r["period"]), axis=1,
        )
        merged = merged[keep_mask].reset_index(drop=True)

    merged["mouse_id"] = merged["file_basename"].apply(_extract_mouse_id)

    merged["genotype_clean"] = merged[COL_GENOTYPE].map(_clean_genotype)
    merged["risk_clean"] = merged[COL_CONDITION].map(_clean_risk)
    merged["treatment_clean"] = merged[COL_CONDITION].map(_clean_treatment)
    merged["age_clean"] = merged[COL_AGE].map(_clean_age)
    merged["sudep_status"] = merged[COL_SUDEP].map(_clean_sudep)

    return merged


def _extract_mouse_id(basename: object) -> Optional[str]:
    if not isinstance(basename, str):
        return None
    parts = basename.split()
    return parts[1] if len(parts) >= 2 else None


def _clean_genotype(raw: object) -> Optional[str]:
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if s.lower() == "het":
        return "het"
    if s.upper() == "WT":
        return "WT"
    return None


def _clean_risk(raw: object) -> Optional[str]:
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    if s == "high risk":
        return "high_risk"
    if s == "low risk":
        return "low_risk"
    return None


def _clean_treatment(raw: object) -> Optional[str]:
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    if s.startswith("ffa"):
        return "FFA"
    if s.startswith("veh"):
        return "Vehicle"
    return None


def _clean_age(raw: object) -> Optional[int]:
    if pd.isna(raw):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _clean_sudep(raw: object) -> Optional[str]:
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    if s in {"sudep", "survivor"}:
        return s
    return None
