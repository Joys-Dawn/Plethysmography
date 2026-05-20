"""
Read pleth data log.xlsx and build :class:`Recording` objects.

Public entry points:
  - :func:`load_data_log` — return a cleaned DataFrame.
  - :func:`load_recordings_for_experiment` — list[Recording] for experiment 1 or 2.
  - :func:`load_exp4_cohort` — list[Recording] for experiment 4 (P19 het survivors + SUDEP).
  - :func:`get_experiment_registry` — folder/column mapping for an experiment id.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

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
# Column G — authoritative file-level population filter (int64; 1 = include,
# 0 = exclude from every population-level stat and plot). Distinct from
# core.metadata.EXCLUSIONS, which only governs preprocess-skip + period-level.
COL_INCLUDE = "include for population analysis"
COL_OFFSET = "Seizure offset or equivalent time (sec from lid closure) -- all trials"
COL_SUDEP = "Survivors vs eventual SUDEP (P19 trace)"
# Column J — free-text fatal-seizure annotation carrying an embedded
# ``(MM.SS-MM.SS)`` window (minutes.seconds, relative to the overall
# recording start). Drives Item H's SUDEP fatal-seizure pipeline; the
# H cohort is built directly from this column and intentionally does
# **not** route through Column G (near-SUDEP ``250304 4056 p22`` is
# G=1 yet must be included).
COL_FATAL_SEIZURE = (
    "SUDEP captured on pleth (start & end times relative to overall "
    "recording time)"
)
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
    # Experiment 3: acute FFA vs vehicle. Same design shape as experiment 2
    # (genotype × treatment factorial at P22, four cells WT/het × Vehicle/FFA),
    # so it reuses the same condition_col="treatment_clean" stats + plot path.
    # The visual differentiator is ACUTE_FFA_PALETTE (pink FFA family) wired
    # through generate_publication_plots' palette kwarg at the driver level.
    3: {
        "experiment_values": frozenset({"FFA vs vehicle - acute"}),
        "cohort_folder": "experiment 3 - acute FFA vs vehicle",
        "raw_subfolder": "experiment 3 - raw data",
        "preprocessed_subfolder": "experiment 3 - preprocessed data",
        "results_folder": "experiment 3 - acute FFA vs vehicle",
    },
    # Experiment 1b (Item G): a 2-group developmental slice of experiment 1 —
    # HR Scn1a+/- P19 vs LR Scn1a+/- P22. It reuses exp1's raw / preprocessed
    # / breathing-CSV artifacts (no re-preprocess), so its cohort/raw/
    # preprocessed folders are exp1's. ``experiment_values`` is intentionally
    # EMPTY so this entry never participates in ``_resolve_cohort_folder``
    # (the "HR vs LR" data-log value must keep resolving to experiment 1);
    # exp1b recordings are derived by filtering exp1's loader instead.
    "1b": {
        "experiment_values": frozenset(),
        "cohort_folder": "experiment 1 - LR vs HR comparison",
        "raw_subfolder": "experiment 1 - raw data",
        "preprocessed_subfolder": "experiment 1 - preprocessed data",
        "results_folder": "experiment 1b - HR P19 vs LR P22",
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


def get_experiment_registry(experiment_id) -> Dict[str, Any]:
    if experiment_id not in _EXPERIMENT_REGISTRY:
        # Keys are mixed int (1, 2, 4) and str ("1b"); stringify before
        # sorting so the error message can't itself raise a TypeError.
        raise ValueError(
            f"No registry entry for experiment {experiment_id}; "
            f"valid ids: {sorted(map(str, _EXPERIMENT_REGISTRY))}"
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


# Column J carries free text with an embedded ``(MM.SS-MM.SS)`` window,
# e.g. ``"evoked SUDEP (58.21-58.53)"``. MM = minutes, SS = seconds
# (corroborated against the seizure-offset column: parsed window
# durations 32/40/46 s match offsets 32/41/46 for the three evoked
# cases). The window is searched for anywhere in the cell, not anchored.
_FATAL_WINDOW_RE = re.compile(
    r"\(\s*(\d{1,3})\.(\d{1,2})\s*-\s*(\d{1,3})\.(\d{1,2})\s*\)"
)


def parse_fatal_seizure_window(cell: object) -> Optional[Tuple[float, float]]:
    """Parse a Column J cell into ``(start_s, end_s)`` seconds, or ``None``.

    Returns ``None`` only when the cell is genuinely empty (NaN or
    whitespace-only — e.g. the WT ``250325 4173 p19`` row, which has a
    bare ``' '``). For any non-empty cell that does **not** contain a
    well-formed ``(MM.SS-MM.SS)`` window — or whose seconds component is
    out of range, or whose end is not strictly after its start — this
    raises ``ValueError`` rather than guessing (the plan mandates a
    loud failure on unparseable cells).
    """
    if pd.isna(cell):
        return None
    text = str(cell).strip()
    if not text:
        return None
    m = _FATAL_WINDOW_RE.search(text)
    if m is None:
        raise ValueError(
            f"Column J cell {cell!r} has no parseable (MM.SS-MM.SS) window"
        )
    mm1, ss1, mm2, ss2 = (int(g) for g in m.groups())
    if not (0 <= ss1 < 60 and 0 <= ss2 < 60):
        raise ValueError(
            f"Column J cell {cell!r}: seconds component must be 0-59"
        )
    start_s = float(mm1 * 60 + ss1)
    end_s = float(mm2 * 60 + ss2)
    if end_s <= start_s:
        raise ValueError(
            f"Column J cell {cell!r}: window end ({end_s}s) is not after "
            f"start ({start_s}s)"
        )
    return (start_s, end_s)


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
    ``experiment_id`` (must be 1, 2, or 3; experiment 4 uses
    :func:`load_exp4_cohort`).
    """
    if experiment_id not in (1, 2, 3):
        raise ValueError(
            f"load_recordings_for_experiment supports ids 1, 2, and 3, not {experiment_id}. "
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


def population_included_basenames(
    data_log_path: str | Path = DEFAULT_DATA_LOG_PATH,
) -> set[str]:
    """Return the set of file basenames whose Column G
    ``include for population analysis`` == 1.

    This is Item A's authoritative file-level population filter, exposed for
    population-level outputs that **bypass**
    ``stats.helpers.prepare_breathing_data`` — notably the Item F population
    ictal histograms, which are accumulated over :class:`Recording` objects
    inside ``analysis.pipeline`` and would otherwise pool the 5 G=0 files'
    ictal breaths. Mirrors the coercion used by the stats-layer G filter
    (numeric-coerce, keep ``== 1``) so the two gates can never diverge.

    Raises ``KeyError`` if Column G is absent (fail fast, never silently
    pass everything through).
    """
    df = load_data_log(data_log_path)
    if COL_INCLUDE not in df.columns:
        raise KeyError(
            f"data log is missing the required Column G {COL_INCLUDE!r}; "
            "cannot apply the population filter."
        )
    include_num = pd.to_numeric(df[COL_INCLUDE], errors="coerce")
    kept = df.loc[include_num == 1, COL_FILENAME].apply(_strip_if_str)
    return {str(b) for b in kept if isinstance(b, str) and b}


def load_recordings_for_experiment1b(
    data_log_path: str | Path = DEFAULT_DATA_LOG_PATH,
    data_root: str | Path = ".",
) -> List[Recording]:
    """Load the experiment-1b developmental cohort (Item G): the two
    Scn1a+/- developmental groups **HR P19** and **LR P22**, derived by
    filtering experiment 1's recordings (so EDF / preprocessed paths point
    at exp1's folders — exp1b never re-preprocesses).

    Cohort rule: ``genotype == het`` and either ``(age P19 and risk HR)``
    or ``(age P22 and risk LR)``.
    """
    out: List[Recording] = []
    for rec in load_recordings_for_experiment(1, data_log_path, data_root):
        if rec.genotype != "het":
            continue
        if (rec.age == "P19" and rec.risk == "HR") or (
            rec.age == "P22" and rec.risk == "LR"
        ):
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


# ----------------------------------------------------------------------------
# Item H — SUDEP fatal-seizure cohort (built directly from Column J)
# ----------------------------------------------------------------------------
class SudepEvent(NamedTuple):
    """One fatal (or near-fatal) seizure window from Column J.

    ``edf_path`` is ``None`` when the raw EDF cannot be located in either
    the experiment-1 or experiment-2 raw folder — the pipeline logs and
    skips that mouse rather than crashing the whole run.
    """

    file_basename: str
    mouse_id: str
    genotype: str
    age: str
    start_s: float
    end_s: float
    edf_path: Optional[Path]


def resolve_raw_edf(
    file_basename: str,
    data_root: str | Path = ".",
) -> Optional[Path]:
    """Locate a recording's raw EDF across the experiment-1 and -2 raw
    folders (the SUDEP cohort is pooled from both). Returns the first
    existing ``<cohort>/<raw>/<basename>.EDF`` path, or ``None`` if it is
    absent in both.
    """
    root = Path(data_root)
    for exp_id in (1, 2):
        registry = get_experiment_registry(exp_id)
        candidate = (
            root
            / registry["cohort_folder"]
            / registry["raw_subfolder"]
            / f"{file_basename}.EDF"
        )
        if candidate.exists():
            return candidate
    return None


def load_sudep_event_cohort(
    data_log_path: str | Path = DEFAULT_DATA_LOG_PATH,
    data_root: str | Path = ".",
) -> List[SudepEvent]:
    """Build the Item H fatal-seizure cohort directly from Column J.

    This deliberately **bypasses** the Column G population filter: every
    row with a parseable ``(MM.SS-MM.SS)`` window becomes an event,
    including the near-SUDEP ``250304 4056 p22`` (G=1) and the four
    G=0 SUDEP recordings. Rows with an empty Column J (NaN / the WT
    ``' '`` cell) are skipped; a non-empty but unparseable cell raises
    via :func:`parse_fatal_seizure_window` (fail loud, never guess).

    Raises ``KeyError`` if Column J is absent (fail fast).
    """
    df = load_data_log(data_log_path)
    if COL_FATAL_SEIZURE not in df.columns:
        raise KeyError(
            f"data log is missing the required Column J "
            f"{COL_FATAL_SEIZURE!r}; cannot build the SUDEP cohort."
        )

    root = Path(data_root)
    out: List[SudepEvent] = []
    for _, row in df.iterrows():
        window = parse_fatal_seizure_window(row.get(COL_FATAL_SEIZURE))
        if window is None:
            continue
        basename = row.get(COL_FILENAME)
        if pd.isna(basename):
            continue
        basename = str(basename).strip()
        if not basename:
            continue
        start_s, end_s = window
        parts = basename.split()
        mouse_id = parts[1] if len(parts) >= 2 else ""
        out.append(
            SudepEvent(
                file_basename=basename,
                mouse_id=mouse_id,
                genotype=_parse_genotype(row.get(COL_GENOTYPE)) or "",
                age=_parse_age(row.get(COL_AGE)) or "",
                start_s=start_s,
                end_s=end_s,
                edf_path=resolve_raw_edf(basename, root),
            )
        )
    return out
