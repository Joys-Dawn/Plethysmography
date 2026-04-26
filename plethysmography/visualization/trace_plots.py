"""
Per-recording verification plots for visual QC of preprocessing.

Mirrors ``old_code/pleth_preprocessing.py``:

  - :func:`plot_lid_spikes`     -> ``plot_lid_open_and_close``
  - :func:`plot_periods_overlay` -> ``plot_periods``

Both write a PNG named after the recording. Use these to confirm the
3-pass lid detector picked the right events and that the period slicing
landed in the right place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from ..core.data_models import LidEvents, Period


_PERIOD_COLORS: Dict[str, str] = {
    "Habituation": "lightgreen",
    "Baseline": "gray",
    "Ictal": "lightcoral",
    "Immediate Postictal": "thistle",
    "Recovery": "moccasin",
}


def plot_lid_spikes(
    signal: np.ndarray,
    time_s: np.ndarray,
    lid_events: LidEvents,
    file_basename: str,
    output_dir: Path,
    *,
    channel_label: str = "VF1",
) -> Path:
    """Plot the full raw signal with red dashed verticals at every adjusted
    lid spike. Filename: ``<file_basename>_spikes.png``.

    Mirrors ``plot_lid_open_and_close`` in the old code (figsize 16x6,
    label='Signal', red dashed verticals at spike times).
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 6))
    plt.plot(time_s, signal, label="Signal")
    for st in lid_events.adjusted_spike_times_s:
        plt.axvline(x=float(st), color="red", linestyle="--", linewidth=1)
    plt.title(f"Channel: {channel_label} - File: {file_basename}.EDF")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    out_path = output_dir / f"{file_basename}_spikes.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_periods_overlay(
    periods: Sequence[Period],
    file_basename: str,
    output_dir: Path,
) -> Path:
    """Plot each period's filtered signal as a colored segment, overlaid in
    absolute time. Filename: ``<file_basename>_periods.png``.

    Mirrors ``plot_periods`` in the old code (figsize 18x7, dashed grid,
    legend in upper right).
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 7))
    legend_handles = []
    for period in periods:
        if period.signal.size == 0 or period.time_s.size == 0:
            continue
        color = _PERIOD_COLORS.get(period.name, "gray")
        line, = plt.plot(period.time_s, period.signal, color=color,
                         alpha=0.9, linewidth=1.2, label=period.name)
        legend_handles.append(line)

    plt.title(f"Filtered Signal Periods - File: {file_basename}.EDF")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    if legend_handles:
        plt.legend(handles=legend_handles, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.7)

    out_path = output_dir / f"{file_basename}_periods.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
