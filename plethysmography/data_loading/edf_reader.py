"""Thin pyedflib wrapper for reading plethysmography EDF channel 0."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pyedflib


def read_edf_signal(
    path: str | Path,
    channel: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Read a single EDF channel and return (signal, time_s, fs).

    Parameters
    ----------
    path : path to the .EDF file.
    channel : 0-based channel index (default 0 — VF1 in the pleth rig).

    Returns
    -------
    signal : 1D float array of samples.
    time_s : 1D float array of timestamps in seconds (np.arange(N)/fs).
    fs : sampling frequency in Hz.

    Raises
    ------
    OSError if the file cannot be opened (corrupt or non-EDF-compliant). The
    caller is responsible for handling skipped files.
    """
    reader = pyedflib.EdfReader(str(path))
    try:
        signal = reader.readSignal(channel)
        fs = float(reader.getSampleFrequency(channel))
    finally:
        reader.close()
    time_s = np.arange(len(signal)) / fs
    return signal, time_s, fs
