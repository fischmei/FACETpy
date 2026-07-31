"""
Correction Module

This module contains processors for correcting EEG artifacts.

Author: FACETpy Team
Date: 2025-01-12
"""

from .aas import AASCorrection, AveragedArtifactSubtraction
from .farm import FARMArtifactCorrection, FARMCorrection
from .flex import Flex, FlexCorrection
from .grid_search import CorrectionGridSearch, CorrectionGridSearchResult
from .volume import RemoveVolumeArtifactCorrection, VolumeArtifactCorrection
from .weighted import (
    AvgArtWghtCorrespondingSliceCorrection,
    AvgArtWghtMoosmannCorrection,
    AvgArtWghtSliceTriggerCorrection,
    AvgArtWghtVolumeTriggerCorrection,
    CorrespondingSliceCorrection,
    MoosmannCorrection,
    SliceTriggerCorrection,
    VolumeTriggerCorrection,
)

__all__ = [
    # Shared template-matrix engine
    "Flex",
    "FlexCorrection",
    # Legacy matrix strategies
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
    # Legacy structural and motion-weighted strategies
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
