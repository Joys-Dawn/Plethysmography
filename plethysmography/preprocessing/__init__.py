"""Preprocessing: period slicing, filtering, artifact removal, and the orchestrator."""

from .periods import slice_periods
from .filtering import filter_period
from .artifacts import remove_artifacts_from_period
from .pipeline import preprocess_recording, save_period_csv
