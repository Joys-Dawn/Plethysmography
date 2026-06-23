"""Fatal pipeline errors — these must abort processing, not fall back silently."""


class PreprocessingError(Exception):
    """Preprocessing produced an unusable recording; downstream metrics are invalid."""


class MissingBaselineError(PreprocessingError):
    """No Baseline period could be sliced or loaded.

    Without Baseline, sigh/apnea thresholds and all cross-period comparisons
    are meaningless. The pipeline must stop rather than emit partial results.
    """
