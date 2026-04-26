"""Configuration, data containers, and project-wide metadata."""

from .config import (
    PlethConfig,
    PeriodConfig,
    FilterConfig,
    LidDetectionConfig,
    BreathConfig,
    SighConfig,
    ApneaConfig,
    StatsConfig,
)
from .data_models import (
    PERIOD_NAMES,
    HABITUATION,
    BASELINE,
    ICTAL,
    IMMEDIATE_POSTICTAL,
    RECOVERY,
    Recording,
    LidEvents,
    Period,
    BreathMetrics,
    ApneaEvent,
    BaselineCache,
)
from .metadata import (
    EXCLUSIONS,
    PER_FILE_LID_OVERRIDES,
    PER_FILE_PREPROCESS_OVERRIDES,
    PER_FILE_ANALYSIS_OVERRIDES,
    is_excluded,
    should_skip_preprocess,
    get_lid_override,
    get_preprocess_override,
    get_analysis_override,
)
