"""
Read pleth data log.xlsx and build :class:`Recording` objects.

Public entry points:
  - :func:`load_data_log` — return a cleaned DataFrame.
  - :func:`load_recordings_for_experiment` — list[Recording] for experiment 1 or 2.
  - :func:`load_exp4_cohort` — list[Recording] for experiment 4 (P19 het survivors + SUDEP).
  - :func:`get_experiment_registry` — folder/column mapping for an experiment id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.data_models import Recording


# ----------------------------------------------------------------------------
# Column constants — derived from the actual schema of pleth data log.xlsx
# ----------------------------------------------------------------------------
COL_FILENAME = "file name"
COL_GENOTYPE = "genotype (het vs WT)"
COL_EXPERIMENT = "experiment"
COL_CONDITION = "condition"
COL_AGE = "age (P19 vs P22)"
COL_OFFSET = "Seizure offset or equivalent time (sec from lid closure) -- all trials"
COL_SUDEP = "Survivors vs eventual SUDEP (P19 trace)"
COL_RACINE = "Maximum racine score"

DEFAULT_DATA_LOG_PATH = Path("docs") / "pleth data log.xlsx"


# ----------------------------------------------------------------------------
# Experiment registry: maps experiment_id -> folder names + which `experiment`
# column values to filter on.
# ----------------------------------------------------------------------------
_EXPERIMENT_REGISTRY: Dict[int, Dict[str, Any]] = {
    1: {
        "experiment_values": frozenset({"HR vs LR"}),
        "cohort_folder": "experiment 1 - LR vs HR comparison",
        "raw_subfolder": "experiment 1 - raw data",
        "preprocessed_subfolder": "experiment 1 - preprocessed data",
        "results_folder": "experiment 1 - LR vs HR comparison",
    },
    2: {
        "experiment_values": frozenset({"FFA vs vehicle - chronic"}),
        "cohort_folder": "experiment 2 - chronic FFA vs vehicle",
        "raw_subfolder": "experiment 2 - raw data",
        "preprocessed_subfolder": "experiment 2 - preprocessed data",
        "results_folder": "experiment 2 - chronic FFA vs vehicle",
    },
    4: {
        # No single cohort folder; recordings are pooled from experiments 1 + 2.
        "experiment_values": frozenset(),
        "cohort_folder": None,
        "raw_subfolder": None,
        "preprocessed_subfolder": None,
        "results_folder": "experiment 4 - survivors vs SUDEP",
    },
}


def get_experiment_registry(experiment_id: int) -> Dict[str, Any]:
    if experiment_id not in _EXPERIMENT_REGISTRY:
        raise ValueError(
            f"No registry entry for experiment {experiment_id}; "
            f"valid ids: {sorted(_EXPERIMENT_REGISTRY)}"
        )
    return _EXPERIMENT_REGISTRY[experiment_id]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _strip_if_str(value: object) -> object:
    if isinstance(value, str):
        return value.strip()
    return value


def load_data_log(data_log_path: str | Path = DEFAULT_DATA_LOG_PATH) -> pd.DataFrame:
    """Read the data log, strip whitespace from columns and key string values,
    and return the cleaned DataFrame.
    """
    df = pd.read_excel(data_log_path)
    df.columns = df.columns.str.strip()

    string_cols_to_clean = [
        COL_FILENAME, COL_GENOTYPE, COL_EXPERIMENT, COL_CONDITION, COL_SUDEP,
    ]
    for col in string_cols_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(_strip_if_str)
    return df


# ----------------------------------------------------------------------------
# Cell parsers
# ----------------------------------------------------------------------------
def _parse_genotype(raw: object) -> Optional[str]:
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if s.lower() == "het":
        return "het"
    if s.upper() == "WT":
        return "WT"
    return None


def _parse_age(raw: object) -> Optional[str]:
    if pd.isna(raw):
        return None
    try:
        return f"P{int(raw)}"
    except (TypeError, ValueError):
        return None


def _parse_risk(raw_condition: object) -> Optional[str]:
    if pd.isna(raw_condition):
        return None
    s = str(raw_condition).strip().lower()
    if s == "high risk":
        return "HR"
    if s == "low risk":
        return "LR"
    return None


def _parse_treatment(raw_condition: object) -> Optional[str]:
    if pd.isna(raw_condition):
        return None
    s = str(raw_condition).strip().lower()
    if s.startswith("ffa"):
        return "FFA"
    if s.startswith("veh"):
        return "Vehicle"
    return None


def _parse_offset(raw: object) -> Optional[float]:
    """Parse the offset column. Returns None for NaN / 'n/a' / 'none' /
    'none (sudep)' style entries; otherwise a float."""
    if pd.isna(raw):
        return None
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"", "n/a", "na", "none", "none (sudep)", "nan"}:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _parse_racine(raw: object) -> Optional[int]:
    if pd.isna(raw):
        return None
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, (float, np.floating)) and not np.isnan(raw):
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"", "lost", "n/a", "na"}:
            return None
        try:
            return int(float(s))
        except ValueError:
            return None
    return None


def _parse_sudep(raw: object) -> Tuple[bool, bool]:
    """Returns (is_sudep, is_survivor). Both False if outside experiment 4 cohort."""
    if pd.isna(raw):
        return (False, False)
    s = str(raw).strip().lower()
    if s == "sudep":
        return (True, False)
    if s == "survivor":
        return (False, True)
    return (False, False)


# ----------------------------------------------------------------------------
# Recording construction
# ----------------------------------------------------------------------------
def _resolve_cohort_folder(experiment_value: object) -> Tuple[str, str]:
    """Map the data log 'experiment' value to (cohort_folder, raw_subfolder)."""
    if pd.isna(experiment_value):
        return ("", "")
    s = str(experiment_value).strip()
    for entry in _EXPERIMENT_REGISTRY.values():
        if s in entry["experiment_values"]:
            return (entry["cohort_folder"], entry["raw_subfolder"])
    return ("", "")


def _build_recording(row: pd.Series, data_root: Path) -> Optional[Recording]:
    file_basename = row.get(COL_FILENAME)
    if pd.isna(file_basename):
        return None
    file_basename = str(file_basename).strip()
    if not file_basename:
        return None

    genotype = _parse_genotype(row.get(COL_GENOTYPE))
    age = _parse_age(row.get(COL_AGE))
    if genotype is None or age is None:
        return None

    parts = file_basename.split()
    mouse_id = parts[1] if len(parts) >= 2 else ""

    cohort_folder, raw_subfolder = _resolve_cohort_folder(row.get(COL_EXPERIMENT))
    if cohort_folder and raw_subfolder:
        edf_path = data_root / cohort_folder / raw_subfolder / f"{file_basename}.EDF"
    else:
        edf_path = data_root / f"{file_basename}.EDF"

    is_sudep, is_survivor = _parse_sudep(row.get(COL_SUDEP))

    return Recording(
        file_basename=file_basename,
        edf_path=edf_path,
        mouse_id=mouse_id,
        age=age,
        genotype=genotype,
        cohort=cohort_folder,
        risk=_parse_risk(row.get(COL_CONDITION)),
        treatment=_parse_treatment(row.get(COL_CONDITION)),
        seizure_offset_s=_parse_offset(row.get(COL_OFFSET)),
        is_sudep=is_sudep,
        is_survivor=is_survivor,
        racine_max=_parse_racine(row.get(COL_RACINE)),
    )


# ----------------------------------------------------------------------------
# Public loaders
# ----------------------------------------------------------------------------
def load_recordings_for_experiment(
    experiment_id: int,
    data_log_path: str | Path = DEFAULT_DATA_LOG_PATH,
    data_root: str | Path = ".",
) -> List[Recording]:
    """Load all recordings whose 'experiment' column matches the registry for
    ``experiment_id`` (must be 1 or 2; experiment 4 uses :func:`load_exp4_cohort`).
    """
    if experiment_id not in (1, 2):
        raise ValueError(
            f"load_recordings_for_experiment supports ids 1 and 2, not {experiment_id}. "
            f"For experiment 4 use load_exp4_cohort."
        )
    registry = get_experiment_registry(experiment_id)
    experiment_values = registry["experiment_values"]

    df = load_data_log(data_log_path)
    matching = df[df[COL_EXPERIMENT].isin(experiment_values)]

    data_root_path = Path(data_root)
    out: List[Recording] = []
    for _, row in matching.iterrows():
        rec = _build_recording(row, data_root_path)
        if rec is not None:
            out.append(rec)
    return out


def load_exp4_cohort(
    data_log_path: str | Path = DEFAULT_DATA_LOG_PATH,
    data_root: str | Path = ".",
) -> List[Recording]:
    """Load the experiment 4 cohort: P19 het rows whose SUDEP column is
    'survivor' or 'SUDEP'. Recordings are pooled from experiments 1 and 2 raw
    data folders (each Recording.cohort tells you which one).
    """
    df = load_data_log(data_log_path)
    sudep_norm = df[COL_SUDEP].apply(_strip_if_str).astype("string").str.lower()
    cohort_mask = sudep_norm.isin({"survivor", "sudep"})
    matching = df[cohort_mask]

    data_root_path = Path(data_root)
    out: List[Recording] = []
    for _, row in matching.iterrows():
        rec = _build_recording(row, data_root_path)
        if rec is None:
            continue
        # Sanity: experiment 4 = P19 het only
        if rec.age != "P19" or rec.genotype != "het":
            continue
        out.append(rec)
    return out
