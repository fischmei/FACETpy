"""
Resampling Processors Module

This module contains processors for resampling EEG data.

Author: FACETpy Team
Date: 2025-01-12
"""

import mne
import numpy as np
from loguru import logger

from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor
from ..logging_config import suppress_stdout


@register_processor
class Resample(Processor):
    """Resample EEG data to a fixed target sampling frequency.

    Wraps :meth:`mne.io.Raw.resample`.  Trigger positions are scaled
    proportionally after resampling.  If a noise estimate is present in
    the context, it is resampled with identical parameters so that
    downstream evaluation steps remain consistent.

    Parameters
    ----------
    sfreq : float
        Target sampling frequency in Hz.
    npad : str, optional
        Amount to pad the start and end of the data (default: ``'auto'``).
    window : str, optional
        Window to use for resampling (default: ``'boxcar'``).
    n_jobs : int, optional
        Number of parallel jobs for resampling (default: ``1``).
    verbose : bool, optional
        MNE verbosity flag passed to ``raw.resample()`` (default: ``False``).
    """

    name = "resample"
    description = "Resample EEG data to a new sampling frequency"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True
    channel_wise = True

    def __init__(
        self, sfreq: float, npad: str = "auto", window: str = "boxcar", n_jobs: int = 1, verbose: bool = False
    ):
        self.sfreq = sfreq
        self.npad = npad
        self.window = window
        self.n_jobs = n_jobs
        self.verbose = verbose
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        old_sfreq = context.get_sfreq()

        # --- LOG ---
        logger.info("Resampling from {} Hz to {} Hz", old_sfreq, self.sfreq)

        # --- COMPUTE + NOISE + RETURN ---
        return self._do_resample(context, self.sfreq)

    def _do_resample(self, context: ProcessingContext, target_sfreq: float) -> ProcessingContext:
        """Resample raw and noise to *target_sfreq* and update triggers.

        Parameters
        ----------
        context : ProcessingContext
            Input context (not mutated).
        target_sfreq : float
            Target sampling frequency in Hz.

        Returns
        -------
        ProcessingContext
            New context with resampled raw, updated triggers, and propagated
            noise estimate.
        """
        raw = context.get_raw().copy()
        old_sfreq = raw.info["sfreq"]
        # Capture info at original sfreq before resampling — needed for noise propagation.
        pre_resample_info = raw.info.copy()

        raw.resample(sfreq=target_sfreq, npad=self.npad, window=self.window, n_jobs=self.n_jobs, verbose=self.verbose)

        # --- NOISE ---
        estimated_noise = None
        if context.has_estimated_noise():
            noise = context.get_estimated_noise().copy()
            with suppress_stdout():
                # Use the pre-resample info so noise_raw starts at old_sfreq.
                # Using raw.info here (which already has target_sfreq) would make
                # the subsequent resample() call a no-op, leaving noise at the
                # wrong sample count.
                noise_raw = mne.io.RawArray(noise, pre_resample_info)
            noise_raw.resample(
                sfreq=target_sfreq, npad=self.npad, window=self.window, n_jobs=self.n_jobs, verbose=False
            )
            estimated_noise = noise_raw._data
        else:
            logger.debug("No noise estimate present — skipping noise propagation")

        # --- BUILD RESULT ---
        new_ctx = context.with_raw(
            raw,
            estimated_noise=estimated_noise,
            copy_estimated_noise=False,
        )
        scale_factor = target_sfreq / old_sfreq

        if context.has_triggers():
            triggers = context.get_triggers()
            new_triggers = (triggers * scale_factor).astype(int)
            logger.debug("Updated {} trigger positions", len(new_triggers))
            metadata = new_ctx.metadata.copy()
            metadata.triggers = new_triggers
            new_ctx = new_ctx.with_metadata(
                metadata,
                copy_estimated_noise=False,
            )

        # Keep sample-based metadata (artifact/window bounds) consistent with
        # the new sampling frequency.
        new_ctx = self._scale_sample_metadata(new_ctx, scale_factor)

        # --- RETURN ---
        return new_ctx

    @staticmethod
    def _round_half_up(value: float) -> int:
        """Round positive sample indices with half-up semantics."""
        return int(np.floor(value + 0.5))

    def _scale_sample_metadata(self, context: ProcessingContext, scale_factor: float) -> ProcessingContext:
        """Scale sample-based metadata fields after resampling.

        Parameters
        ----------
        context : ProcessingContext
            Context after raw/triggers have been resampled.
        scale_factor : float
            ``target_sfreq / old_sfreq``.

        Returns
        -------
        ProcessingContext
            Context with scaled metadata.
        """
        if scale_factor <= 0:
            return context

        metadata = context.metadata.copy()
        changed = False

        def _scale_opt(value: int | None, min_value: int = 0) -> int | None:
            if value is None:
                return None
            scaled = self._round_half_up(float(value) * scale_factor)
            return max(min_value, scaled)

        old_art_len = metadata.artifact_length
        new_art_len = _scale_opt(old_art_len, min_value=1) if old_art_len is not None else None
        if new_art_len is not None and new_art_len != old_art_len:
            metadata.artifact_length = new_art_len
            changed = True

        for field in ("pre_trigger_samples", "post_trigger_samples", "acq_start_sample", "acq_end_sample"):
            old_val = getattr(metadata, field)
            new_val = _scale_opt(old_val, min_value=0)
            if new_val != old_val:
                setattr(metadata, field, new_val)
                changed = True

        # Keep custom acquisition mirror in sync when present.
        acquisition = metadata.custom.get("acquisition")
        if isinstance(acquisition, dict):
            for key in ("pre_trigger_samples", "post_trigger_samples", "acq_start_sample", "acq_end_sample"):
                if key in acquisition and isinstance(acquisition[key], (int, np.integer)):
                    old_val = int(acquisition[key])
                    new_val = _scale_opt(old_val, min_value=0)
                    if new_val != old_val:
                        acquisition[key] = new_val
                        changed = True

        if changed:
            logger.debug(
                "Scaled sample metadata by factor {:.6f} (artifact_length={} -> {})",
                scale_factor,
                old_art_len,
                metadata.artifact_length,
            )
            return context.with_metadata(
                metadata,
                copy_estimated_noise=False,
            )

        return context


@register_processor
class UpSample(Resample):
    """Upsample EEG data by a fixed integer factor.

    Increases the sampling frequency by multiplying it by *factor*.  Trigger
    positions and any noise estimate are scaled accordingly.

    Parameters
    ----------
    factor : int, optional
        Upsampling factor, e.g. ``10`` increases the sampling frequency
        ten-fold (default: ``10``).
    npad : str, optional
        Amount to pad the start and end of the data (default: ``'auto'``).
    window : str, optional
        Window to use for resampling (default: ``'boxcar'``).
    n_jobs : int, optional
        Number of parallel jobs for resampling (default: ``1``).
    verbose : bool, optional
        MNE verbosity flag passed to ``raw.resample()`` (default: ``False``).
    """

    name = "upsample"
    description = "Upsample EEG data by a factor"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(
        self, factor: int = 10, npad: str = "auto", window: str = "boxcar", n_jobs: int = 1, verbose: bool = False
    ):
        self.factor = factor
        self.npad = npad
        self.window = window
        self.n_jobs = n_jobs
        self.verbose = verbose
        # Bypass Resample.__init__ — no fixed target sfreq at construction time.
        Processor.__init__(self)

    def _get_parameters(self):
        """Expose factor instead of sfreq for history/serialization."""
        return {"factor": self.factor, "npad": self.npad, "window": self.window, "n_jobs": self.n_jobs}

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        old_sfreq = context.get_sfreq()
        target_sfreq = old_sfreq * self.factor

        # --- LOG ---
        logger.info("Upsampling by factor {} ({} Hz -> {} Hz)", self.factor, old_sfreq, target_sfreq)

        # --- COMPUTE + NOISE + RETURN ---
        return self._do_resample(context, target_sfreq)


@register_processor
class DownSample(Resample):
    """Downsample EEG data by a fixed integer factor.

    Decreases the sampling frequency by dividing it by *factor*.  Trigger
    positions and any noise estimate are scaled accordingly.

    Parameters
    ----------
    factor : int, optional
        Downsampling factor, e.g. ``10`` reduces the sampling frequency
        ten-fold (default: ``10``).
    npad : str, optional
        Amount to pad the start and end of the data (default: ``'auto'``).
    window : str, optional
        Window to use for resampling (default: ``'boxcar'``).
    n_jobs : int, optional
        Number of parallel jobs for resampling (default: ``1``).
    verbose : bool, optional
        MNE verbosity flag passed to ``raw.resample()`` (default: ``False``).
    """

    name = "downsample"
    description = "Downsample EEG data by a factor"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(
        self, factor: int = 10, npad: str = "auto", window: str = "boxcar", n_jobs: int = 1, verbose: bool = False
    ):
        self.factor = factor
        self.npad = npad
        self.window = window
        self.n_jobs = n_jobs
        self.verbose = verbose
        # Bypass Resample.__init__ — no fixed target sfreq at construction time.
        Processor.__init__(self)

    def _get_parameters(self):
        """Expose factor instead of sfreq for history/serialization."""
        return {"factor": self.factor, "npad": self.npad, "window": self.window, "n_jobs": self.n_jobs}

    def validate(self, context: ProcessingContext) -> None:
        """Check that the resulting sampling frequency is at least 1 Hz."""
        super().validate(context)
        target_sfreq = context.get_sfreq() / self.factor
        if target_sfreq < 1:
            raise ProcessorValidationError(
                f"Downsampling factor {self.factor} would result in sampling frequency < 1 Hz "
                f"(current sfreq={context.get_sfreq()} Hz)"
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        old_sfreq = context.get_sfreq()
        target_sfreq = old_sfreq / self.factor

        # --- LOG ---
        logger.info("Downsampling by factor {} ({} Hz -> {} Hz)", self.factor, old_sfreq, target_sfreq)

        # --- COMPUTE + NOISE + RETURN ---
        return self._do_resample(context, target_sfreq)
