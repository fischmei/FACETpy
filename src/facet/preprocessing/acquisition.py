"""Acquisition Window Processors

Processors that manage the acquisition window metadata used by the
Cut/Paste correction steps.

These processors do not mutate the raw signal directly; instead they record
window bounds and trigger offsets in the processing metadata so downstream
processors (alignment, averaging, etc.) can operate on consistent segments.
"""

from dataclasses import asdict

from loguru import logger

from ..core import (
    ProcessingContext,
    ProcessingMetadata,
    Processor,
    ProcessorValidationError,
    register_processor,
)


def _derive_pre_post_samples(
    metadata: ProcessingMetadata,
    sfreq: float,
    artifact_length: int,
    pre_override: int | None = None,
    post_override: int | None = None,
) -> tuple[int, int]:
    """Derive pre- and post-trigger sample counts from current metadata.

    Parameters
    ----------
    metadata : ProcessingMetadata
        Processing metadata carrying existing window hints.
    sfreq : float
        Sampling frequency in Hz.
    artifact_length : int
        Artifact window length in samples.
    pre_override : int, optional
        Explicit number of samples before the trigger.
    post_override : int, optional
        Explicit number of samples after the trigger.

    Returns
    -------
    tuple of (int, int)
        ``(pre_samples, post_samples)`` clamped to ``[0, artifact_length]``.

    Raises
    ------
    ProcessorValidationError
        If ``artifact_length`` is not a positive integer.
    """
    if artifact_length is None or artifact_length <= 0:
        raise ProcessorValidationError("Artifact length must be a positive integer")

    if pre_override is not None:
        pre_samples = int(max(0, min(pre_override, artifact_length)))
    elif metadata.pre_trigger_samples is not None:
        pre_samples = int(max(0, min(metadata.pre_trigger_samples, artifact_length)))
    else:
        offset_samples = int(round(metadata.artifact_to_trigger_offset * sfreq))
        pre_samples = int(max(0, min(-offset_samples, artifact_length))) if offset_samples < 0 else 0

    remaining = artifact_length - pre_samples

    if post_override is not None:
        post_samples = int(max(0, min(post_override, artifact_length)))
    elif metadata.post_trigger_samples is not None:
        post_samples = int(max(0, min(metadata.post_trigger_samples, artifact_length)))
    else:
        post_samples = int(max(remaining, 0))

    if pre_samples + post_samples == 0:
        post_samples = artifact_length

    if pre_samples + post_samples < artifact_length:
        post_samples = artifact_length - pre_samples

    return pre_samples, post_samples


@register_processor
class CutAcquisitionWindow(Processor):
    """Derive acquisition window bounds similarly to MATLAB's RACut step.

    Records the acquisition start/end sample indices and the per-trigger
    pre/post sample counts in metadata so that downstream processors can
    operate on consistent artifact windows without accessing raw signal data
    directly.

    Parameters
    ----------
    pre_padding_samples : int, optional
        Explicit number of samples before each trigger. When ``None`` the
        value is derived from ``metadata.artifact_to_trigger_offset``.
    post_padding_samples : int, optional
        Explicit number of samples after each trigger. When ``None`` the
        remaining artifact window is used.
    """

    name = "cut_acquisition_window"
    description = "Record acquisition window boundaries for artifact processing"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        pre_padding_samples: int | None = None,
        post_padding_samples: int | None = None,
    ) -> None:
        self.pre_padding_samples = pre_padding_samples
        self.post_padding_samples = post_padding_samples
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError("Artifact length not set. Run TriggerDetector before CutAcquisitionWindow.")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info["sfreq"]

        # --- LOG ---
        logger.info(
            "Computing acquisition window for {} triggers (artifact_length={})",
            len(triggers),
            artifact_length,
        )

        # --- COMPUTE ---
        pre_samples, post_samples = _derive_pre_post_samples(
            metadata=context.metadata,
            sfreq=sfreq,
            artifact_length=artifact_length,
            pre_override=self.pre_padding_samples,
            post_override=self.post_padding_samples,
        )

        acq_start = int(max(0, triggers[0] - pre_samples))
        acq_end = int(min(raw.n_times, triggers[-1] + post_samples))

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        new_metadata.acq_start_sample = acq_start
        new_metadata.acq_end_sample = acq_end
        new_metadata.pre_trigger_samples = pre_samples
        new_metadata.post_trigger_samples = post_samples

        acquisition_info = new_metadata.custom.setdefault("acquisition", {})
        acquisition_info.update(
            {
                "pre_trigger_samples": pre_samples,
                "post_trigger_samples": post_samples,
                "acq_start_sample": acq_start,
                "acq_end_sample": acq_end,
                "upsampling_factor": new_metadata.upsampling_factor,
            }
        )

        logger.debug(
            "Acquisition window: pre={}, post={}, start={}, end={}",
            pre_samples,
            post_samples,
            acq_start,
            acq_end,
        )

        # Cache window info for downstream steps (non-serialized)
        context.cache_set("acquisition_window", (acq_start, acq_end))

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )


@register_processor
class PasteAcquisitionWindow(Processor):
    """Finalize acquisition metadata and clear cached window segments.

    FACETpy keeps the full-length raw data throughout the pipeline, so this
    processor ensures acquisition metadata is present and clears any cached
    segments set by :class:`CutAcquisitionWindow`.
    """

    name = "paste_acquisition_window"
    description = "Finalize acquisition metadata and clear cached window segments"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = False
    modifies_raw = False
    parallel_safe = False

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        metadata = context.metadata.copy()

        # --- LOG ---
        logger.info("Finalizing acquisition window metadata")

        # --- COMPUTE ---
        if metadata.acq_start_sample is None or metadata.acq_end_sample is None:
            logger.debug("PasteAcquisitionWindow found no acquisition bounds; nothing to do")
            return context

        # --- BUILD RESULT ---
        acquisition_info = metadata.custom.setdefault("acquisition", {})
        acquisition_info["paste_applied"] = True

        logger.debug("PasteAcquisitionWindow metadata: {}", asdict(metadata))

        context.cache_clear()

        # --- RETURN ---
        return context.with_metadata(
            metadata,
            copy_estimated_noise=False,
        )
