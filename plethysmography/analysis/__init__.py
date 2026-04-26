"""Breath segmentation, sigh / apnea detection, baseline cache, and the two-pass
analysis driver."""

from .breath_segmentation import (
    Breath,
    segment_breaths,
)
from .breath_metrics import compute_breath_metrics
from .sigh_detection import (
    compute_sigh_threshold,
    classify_sighs,
)
from .apnea_detection import (
    compute_apnea_threshold,
    detect_apneas,
)
from .baseline_cache import (
    build_baseline_cache,
    cache_from_breaths,
)
from .pipeline import (
    PeriodAnalysisResult,
    analyze_period,
    analyze_recording,
    analyze_experiment,
)
