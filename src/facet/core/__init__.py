"""
Core Module

This module contains the core infrastructure for FACETpy pipelines.

Author: FACETpy Team
Date: 2025-01-12
"""

from .channel_sequential import ChannelSequentialExecutor
from .context import ProcessingContext, ProcessingMetadata, ProcessingStep
from .parallel import ParallelExecutor
from .pipeline import (
    BatchResult,
    ChunkedPipelineResult,
    ChunkSpec,
    Pipeline,
    PipelineBuilder,
    PipelineError,
    PipelineResult,
)
from .processor import (
    ConditionalProcessor,
    LambdaProcessor,
    NoOpProcessor,
    Processor,
    ProcessorError,
    ProcessorValidationError,
    SequenceProcessor,
    SwitchProcessor,
)
from .registry import ProcessorRegistry, get_processor, list_processors, register_processor

__all__ = [
    # Context
    "ProcessingContext",
    "ProcessingMetadata",
    "ProcessingStep",
    # Processor
    "Processor",
    "ProcessorError",
    "ProcessorValidationError",
    "SequenceProcessor",
    "ConditionalProcessor",
    "SwitchProcessor",
    "NoOpProcessor",
    "LambdaProcessor",
    # Pipeline
    "Pipeline",
    "PipelineBuilder",
    "PipelineResult",
    "ChunkSpec",
    "ChunkedPipelineResult",
    "PipelineError",
    "BatchResult",
    # Registry
    "ProcessorRegistry",
    "register_processor",
    "get_processor",
    "list_processors",
    # Executors
    "ParallelExecutor",
    "ChannelSequentialExecutor",
]
