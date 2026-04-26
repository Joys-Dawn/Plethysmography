"""±artifact_sigma * std outlier replacement with linear interpolation.

Per-period: identify samples whose absolute deviation from the period mean
exceeds ``artifact_sigma`` standard deviations, set them to NaN, then linearly
interpolate using ``np.interp`` from the surviving samples.

Mirrors old_code/pleth_preprocessing.py:352-400 (remove_artifacts).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core.config import FilterConfig
from ..core.data_models import Period


def remove_artifacts_from_period(period: Period, config: FilterConfig) -> Period:
    """Return a new Period with samples beyond ±artifact_sigma·std replaced by
    linear interpolation. If interpolation is not feasible (fewer than 2 valid
    samples or no unique time points), the artifact samples are restored to
    their original values rather than left as NaN."""
    if period.signal.size == 0:
        return period

    cleaned = period.signal.astype(float, copy=True)
    mean_val = float(np.nanmean(cleaned))
    std_val = float(np.nanstd(cleaned))
    upper = mean_val + config.artifact_sigma * std_val
    lower = mean_val - config.artifact_sigma * std_val

    artifact_idx = np.where((cleaned > upper) | (cleaned < lower))[0]
    if artifact_idx.size == 0:
        return period

    cleaned[artifact_idx] = np.nan
    non_nan_mask = ~np.isnan(cleaned)

    if non_nan_mask.sum() >= 2:
        xp = period.time_s[non_nan_mask]
        fp = cleaned[non_nan_mask]
        if np.unique(xp).size >= 2:
            interp_vals = np.interp(period.time_s[artifact_idx], xp, fp)
            cleaned[artifact_idx] = interp_vals
        else:
            cleaned[artifact_idx] = period.signal[artifact_idx]
    else:
        cleaned[artifact_idx] = period.signal[artifact_idx]

    return replace(period, signal=cleaned)
