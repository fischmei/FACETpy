"""
Preprocessing Module

This module contains processors for preprocessing EEG data.

Author: FACETpy Team
Date: 2025-01-12
"""

from .acquisition import CutAcquisitionWindow, PasteAcquisitionWindow
from .alignment import SliceAligner, SubsampleAligner, TriggerAligner
from .diagnostics import AnalyzeDataReport, CheckDataReport
from .filtering import BandPassFilter, Filter, HighPassFilter, LowPassFilter, NotchFilter
from .prefilter import MATLABPreFilter
from .resampling import DownSample, Resample, UpSample
from .transforms import (
    ChannelStandardizer,
    Crop,
    DropChannels,
    DropChannelsMatching,
    MagicErasor,
    PickChannels,
    PrintMetric,
    RawTransform,
)
from .trigger_detection import (
    MissingTriggerCompleter,
    MissingTriggerDetector,
    QRSTriggerDetector,
    SliceTriggerGenerator,
    TriggerDetector,
)
from .trigger_explorer import InteractiveTriggerExplorer, TriggerExplorer

__all__ = [
    # Filtering
    "HighPassFilter",
    "LowPassFilter",
    "BandPassFilter",
    "NotchFilter",
    "Filter",
    # Resampling
    "UpSample",
    "DownSample",
    "Resample",
    # Trigger Detection
    "TriggerDetector",
    "QRSTriggerDetector",
    "MissingTriggerDetector",
    "MissingTriggerCompleter",
    "SliceTriggerGenerator",
    "TriggerExplorer",
    "InteractiveTriggerExplorer",
    # Alignment
    "TriggerAligner",
    "SliceAligner",
    "SubsampleAligner",
    # Acquisition window
    "CutAcquisitionWindow",
    "PasteAcquisitionWindow",
    # Transforms
    "Crop",
    "MagicErasor",
    "RawTransform",
    "PickChannels",
    "DropChannels",
    "DropChannelsMatching",
    "ChannelStandardizer",
    "PrintMetric",
    # MATLAB parity tools
    "MATLABPreFilter",
    "AnalyzeDataReport",
    "CheckDataReport",
]
