"""
Correction Module

This module contains processors for correcting EEG artifacts.

Author: FACETpy Team
Date: 2025-01-12
"""

from .flex import (
    MOTION_METADATA_KEY,
    AveragingMatrixBuilder,
    CandidateScoringMode,
    CandidateScoringPolicy,
    CorrelationMode,
    CorrelationPolicy,
    DirectionalQuota,
    Flex,
    FlexCorrection,
    MatrixDecisionError,
    MatrixDecisions,
    MatrixMetadata,
    MotionEligibility,
    MotionEpochMetadata,
    SamplingMode,
    SamplingPolicy,
    TargetPolicy,
    TemplateSizeMode,
    TemplateSizePolicy,
    TemporalDistanceUnit,
    WeightingBasis,
    WeightingKernel,
    WeightingPolicy,
)
from .grid_search import CorrectionGridSearch, CorrectionGridSearchResult
from .legacy_adapters import (
    AASCorrection,
    AveragedArtifactSubtraction,
    AvgArtWghtCorrespondingSliceCorrection,
    AvgArtWghtMoosmannCorrection,
    AvgArtWghtSliceTriggerCorrection,
    AvgArtWghtVolumeTriggerCorrection,
    CorrespondingSliceCorrection,
    FARMArtifactCorrection,
    FARMCorrection,
    MoosmannCorrection,
    SliceTriggerCorrection,
    VolumeTriggerCorrection,
)
from .presets import (
    AAS_PER_TARGET,
    CLI_CORRECTION_PRESETS,
    CORRESPONDING_SLICE,
    FARM_PER_TARGET_K10,
    FLEX_DEFAULT,
    LEGACY_RESEMBLANCE,
    MOOSMANN_COST,
    PRESET_BUILDERS,
    STRUCTURAL_SLICE,
    STRUCTURAL_VOLUME,
    build_flex_preset,
)
from .volume import RemoveVolumeArtifactCorrection, VolumeArtifactCorrection

__all__ = [
    # Shared template-matrix engine
    "Flex",
    "FlexCorrection",
    "AveragingMatrixBuilder",
    "CandidateScoringMode",
    "CandidateScoringPolicy",
    "DirectionalQuota",
    "SamplingMode",
    "SamplingPolicy",
    "MotionEligibility",
    "TargetPolicy",
    "TemplateSizeMode",
    "TemplateSizePolicy",
    "CorrelationMode",
    "CorrelationPolicy",
    "WeightingBasis",
    "TemporalDistanceUnit",
    "WeightingKernel",
    "WeightingPolicy",
    "MatrixDecisions",
    "MatrixMetadata",
    "MotionEpochMetadata",
    "MatrixDecisionError",
    "MOTION_METADATA_KEY",
    # Named Flex recipes
    "FLEX_DEFAULT",
    "AAS_PER_TARGET",
    "FARM_PER_TARGET_K10",
    "STRUCTURAL_VOLUME",
    "STRUCTURAL_SLICE",
    "CORRESPONDING_SLICE",
    "MOOSMANN_COST",
    "CLI_CORRECTION_PRESETS",
    "LEGACY_RESEMBLANCE",
    "PRESET_BUILDERS",
    "build_flex_preset",
    # Flex-backed compatibility names
    "AASCorrection",
    "AveragedArtifactSubtraction",
    "FARMCorrection",
    "FARMArtifactCorrection",
    # Parameter search
    "CorrectionGridSearch",
    "CorrectionGridSearchResult",
    # "CORRELATION_THRESHOLD_GRID",
    # Volume transitions
    "VolumeArtifactCorrection",
    "RemoveVolumeArtifactCorrection",
    # Structural and motion compatibility names
    "CorrespondingSliceCorrection",
    "VolumeTriggerCorrection",
    "SliceTriggerCorrection",
    "MoosmannCorrection",
    "AvgArtWghtCorrespondingSliceCorrection",
    "AvgArtWghtVolumeTriggerCorrection",
    "AvgArtWghtSliceTriggerCorrection",
    "AvgArtWghtMoosmannCorrection",
]

# Import ANC if available
try:
    from .anc import AdaptiveNoiseCancellation, ANCCorrection  # noqa: F401

    __all__.extend(["AdaptiveNoiseCancellation", "ANCCorrection"])
except ImportError:
    pass

# Import PCA if available
try:
    from .pca import PCACorrection  # noqa: F401

    __all__.append("PCACorrection")
except ImportError:
    pass
