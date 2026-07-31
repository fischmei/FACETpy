"""Trigger Alignment Processors Module

Processors for aligning trigger positions to artifact onsets in EEG-fMRI
recordings using cross-correlation techniques.
"""

import mne
import numpy as np
from loguru import logger

from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor
from ..helpers.crosscorr import crosscorrelation
from .resampling import UpSample


def _get_pre_post_samples(
    context: ProcessingContext,
    artifact_length: int,
) -> tuple[int, int]:
    """Derive pre/post trigger sample lengths using metadata.

    Parameters
    ----------
    context : ProcessingContext
        Current processing context carrying metadata window hints.
    artifact_length : int
        Total artifact window length in samples.

    Returns
    -------
    tuple of (int, int)
        ``(pre_samples, post_samples)`` clamped to ``[0, artifact_length]``.

    Raises
    ------
    ProcessorValidationError
        If ``artifact_length`` is not positive.
    """
    sfreq = context.get_raw().info["sfreq"]
    metadata = context.metadata

    if artifact_length is None or artifact_length <= 0:
        raise ProcessorValidationError("Artifact length must be positive for alignment")

    if metadata.pre_trigger_samples is not None:
        pre = int(max(0, min(metadata.pre_trigger_samples, artifact_length)))
    else:
        offset_samples = int(round(metadata.artifact_to_trigger_offset * sfreq))
        pre = int(max(0, min(-offset_samples, artifact_length))) if offset_samples < 0 else 0

    if metadata.post_trigger_samples is not None:
        post = int(max(0, min(metadata.post_trigger_samples, artifact_length)))
    else:
        post = int(max(artifact_length - pre, 0))

    if pre + post == 0:
        post = artifact_length
    if pre + post < artifact_length:
        post = artifact_length - pre

    return pre, post


def _extract_epoch_with_padding(
    data: np.ndarray,
    start_idx: int,
    length: int,
    total_length: int,
) -> np.ndarray:
    """Extract a window from data, padding with edge values at boundaries.

    Parameters
    ----------
    data : np.ndarray
        1-D signal array.
    start_idx : int
        First sample index of the desired window (may be negative).
    length : int
        Number of samples to extract.
    total_length : int
        Total length of ``data`` in samples.

    Returns
    -------
    np.ndarray
        Extracted (and possibly padded) segment of shape ``(length,)``.
    """
    end_idx = start_idx + length

    pad_left = max(0, -start_idx)
    pad_right = max(0, end_idx - total_length)

    valid_start = max(0, start_idx)
    valid_end = min(total_length, end_idx)

    segment = data[valid_start:valid_end]
    if pad_left or pad_right:
        segment = np.pad(segment, (pad_left, pad_right), mode="edge")

    if len(segment) != length:
        segment = np.zeros(length, dtype=data.dtype) if len(segment) == 0 else np.resize(segment, length)

    return segment


@register_processor
class TriggerAligner(Processor):
    """Align triggers to artifact positions using cross-correlation.

    Refines trigger positions by finding the lag that maximises cross-
    correlation with a reference artifact epoch. The search can optionally
    operate on temporarily upsampled data for sub-sample precision.

    Parameters
    ----------
    ref_trigger_index : int, optional
        Index of the reference trigger to use as template (default: 0).
    ref_channel : int, optional
        Reference channel index. Uses the first EEG channel when ``None``
        (default: None).
    search_window : int, optional
        Search window in samples. Derived from the upsampling factor when
        ``None`` (default: None).
    save_to_annotations : bool, optional
        If ``True``, save aligned triggers as raw annotations (default: False).
    upsample_for_alignment : bool, optional
        Temporarily upsample data before alignment for sub-sample accuracy
        (default: True).
    """

    name = "trigger_aligner"
    description = "Align triggers using cross-correlation"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = True
    channel_wise = True
    run_once = True

    def __init__(
        self,
        ref_trigger_index: int = 0,
        ref_channel: int | None = None,
        search_window: int | None = None,
        save_to_annotations: bool = False,
        upsample_for_alignment: bool = True,
    ) -> None:
        self.ref_trigger_index = ref_trigger_index
        self.ref_channel = ref_channel
        self.search_window = search_window
        self.save_to_annotations = save_to_annotations
        self.upsample_for_alignment = upsample_for_alignment
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError("Artifact length not set. Run TriggerDetector first.")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers().copy()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info["sfreq"]
        upsampling_factor = context.metadata.upsampling_factor

        # --- LOG ---
        logger.info("Aligning {} triggers using cross-correlation", len(triggers))

        # --- COMPUTE ---
        ref_channel = self._pick_ref_channel(raw)
        search_window = self.search_window if self.search_window is not None else 3 * upsampling_factor

        working_raw, working_triggers, did_upsample = self._prepare_working_data(
            context, raw, triggers, upsampling_factor, sfreq
        )
        tmin, tmax = self._compute_epoch_bounds(context, working_raw, artifact_length, upsampling_factor, did_upsample)
        aligned_triggers = self._align_all_triggers(
            working_raw, working_triggers, tmin, tmax, ref_channel, search_window
        )

        if did_upsample:
            aligned_triggers = (aligned_triggers / upsampling_factor).astype(int)

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        new_metadata.triggers = aligned_triggers
        if len(aligned_triggers) > 1:
            new_metadata.artifact_length = self._recalc_artifact_length(aligned_triggers, new_metadata.volume_gaps)

        if self.save_to_annotations:
            raw_copy = raw.copy()
            raw_copy.set_annotations(
                mne.Annotations(
                    onset=aligned_triggers / sfreq,
                    duration=np.zeros(len(aligned_triggers)),
                    description=["Aligned_Trigger"] * len(aligned_triggers),
                )
            )
            return context.with_raw(
                raw_copy,
                copy_estimated_noise=False,
            ).with_metadata(
                new_metadata,
                copy_estimated_noise=False,
            )

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )

    def _pick_ref_channel(self, raw: mne.io.Raw) -> int:
        if self.ref_channel is not None:
            return self.ref_channel
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        return int(eeg_channels[0]) if len(eeg_channels) > 0 else 0

    def _prepare_working_data(
        self,
        context: ProcessingContext,
        raw: mne.io.Raw,
        triggers: np.ndarray,
        upsampling_factor: int,
        sfreq: float,
    ) -> tuple[mne.io.Raw, np.ndarray, bool]:
        """Return working raw and triggers, upsampling temporarily if needed.

        Parameters
        ----------
        context : ProcessingContext
            Current processing context.
        raw : mne.io.Raw
            Current raw object.
        triggers : np.ndarray
            Current trigger positions.
        upsampling_factor : int
            Factor to use when upsampling.
        sfreq : float
            Current sampling frequency.

        Returns
        -------
        tuple of (mne.io.Raw, np.ndarray, bool)
            Working raw, working triggers, and whether upsampling was applied.
        """
        if self.upsample_for_alignment and sfreq == context.get_raw_original().info["sfreq"]:
            logger.debug("Temporarily upsampling for alignment")
            temp_context = UpSample(factor=upsampling_factor).execute(context)
            return temp_context.get_raw(), temp_context.get_triggers(), True
        return raw, triggers, False

    def _compute_epoch_bounds(
        self,
        context: ProcessingContext,
        working_raw: mne.io.Raw,
        artifact_length: int,
        upsampling_factor: int,
        did_upsample: bool,
    ) -> tuple[int, int]:
        """Compute tmin/tmax epoch bounds in working-raw sample space.

        Parameters
        ----------
        context : ProcessingContext
            Current processing context.
        working_raw : mne.io.Raw
            Raw object in which alignment is performed.
        artifact_length : int
            Artifact length in original-sample space.
        upsampling_factor : int
            Applied upsampling factor.
        did_upsample : bool
            Whether working_raw is upsampled.

        Returns
        -------
        tuple of (int, int)
            ``(tmin, tmax)`` offsets relative to each trigger sample.
        """
        tmin = int(context.metadata.artifact_to_trigger_offset * working_raw.info["sfreq"])
        tmax = tmin + (artifact_length * upsampling_factor if did_upsample else artifact_length)
        return tmin, tmax

    def _align_all_triggers(
        self,
        working_raw: mne.io.Raw,
        working_triggers: np.ndarray,
        tmin: int,
        tmax: int,
        ref_channel: int,
        search_window: int,
    ) -> np.ndarray:
        """Compute aligned trigger positions for all triggers.

        Parameters
        ----------
        working_raw : mne.io.Raw
            Raw object used for cross-correlation.
        working_triggers : np.ndarray
            Trigger positions in working-raw sample space.
        tmin : int
            Start offset from trigger to artifact onset.
        tmax : int
            End offset from trigger to artifact end.
        ref_channel : int
            Channel index used for cross-correlation.
        search_window : int
            Search radius in samples.

        Returns
        -------
        np.ndarray
            Aligned trigger positions.
        """
        ref_data = working_raw.get_data(picks=[ref_channel])[0]
        ref_trigger = working_triggers[self.ref_trigger_index]
        ref_artifact = ref_data[ref_trigger + tmin : ref_trigger + tmax]

        logger.debug("Using trigger {} as reference", self.ref_trigger_index)
        logger.debug("Reference artifact shape: {}", ref_artifact.shape)

        aligned_triggers = working_triggers.copy()
        for i, trigger in enumerate(working_triggers):
            if i == self.ref_trigger_index:
                continue
            current_artifact = ref_data[trigger + tmin : trigger + tmax + search_window]
            corr = crosscorrelation(current_artifact, ref_artifact, search_window)
            shift = int(np.argmax(corr)) - search_window
            aligned_triggers[i] = trigger + shift

        logger.info("Aligned {} triggers", len(aligned_triggers))
        return aligned_triggers

    def _recalc_artifact_length(self, aligned_triggers: np.ndarray, volume_gaps: bool) -> int:
        """Recalculate artifact length from aligned trigger spacings.

        Parameters
        ----------
        aligned_triggers : np.ndarray
            Aligned trigger positions.
        volume_gaps : bool
            Whether volume-level gaps are present in the trigger sequence.

        Returns
        -------
        int
            Updated artifact length estimate in samples.
        """
        trigger_diffs = np.diff(aligned_triggers)
        if volume_gaps:
            mean_val = np.mean([np.median(trigger_diffs), np.max(trigger_diffs)])
            slice_diffs = trigger_diffs[trigger_diffs < mean_val]
            return int(np.max(slice_diffs))
        return int(np.max(trigger_diffs))


@register_processor
class SliceAligner(TriggerAligner):
    """Align slice triggers on already-upsampled data.

    Identical to :class:`TriggerAligner` but skips the internal temporary
    upsampling step, assuming the data has already been upsampled upstream.

    Parameters
    ----------
    ref_trigger_index : int, optional
        Index of the reference trigger (default: 0).
    ref_channel : int, optional
        Reference channel index (default: None).
    search_window : int, optional
        Search window in samples (default: None).
    save_to_annotations : bool, optional
        Save aligned triggers as annotations (default: False).
    """

    name = "slice_aligner"
    description = "Align slice triggers on upsampled data"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = True
    channel_wise = True
    run_once = True

    def __init__(
        self,
        ref_trigger_index: int = 0,
        ref_channel: int | None = None,
        search_window: int | None = None,
        save_to_annotations: bool = False,
    ) -> None:
        super().__init__(
            ref_trigger_index=ref_trigger_index,
            ref_channel=ref_channel,
            search_window=search_window,
            save_to_annotations=save_to_annotations,
            upsample_for_alignment=False,
        )


@register_processor
class SubsampleAligner(Processor):
    """Refine trigger positions at subsample precision after upsampling.

    Mirrors the intent of the MATLAB RAAlignSubSample step. For each trigger
    a search segment is extracted, cross-correlated against a reference epoch,
    and the trigger is shifted by the lag that maximises the correlation.

    When ``apply_to_raw=True`` the corresponding raw data segments are also
    rolled by the computed shifts; otherwise only the trigger positions in
    metadata are updated.

    Parameters
    ----------
    ref_trigger_index : int, optional
        Index of the reference trigger used as the alignment template
        (default: 0).
    ref_channel : int, optional
        Reference channel index. Uses the first EEG channel when ``None``
        (default: None).
    search_window : int, optional
        Search radius in samples. Defaults to twice the upsampling factor.
    apply_to_raw : bool, optional
        If ``True``, roll raw data segments by the computed shifts
        (default: False).
    """

    name = "subsample_aligner"
    description = "Refine trigger alignment at subsample precision"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = True
    channel_wise = True
    run_once = True

    def __init__(
        self,
        ref_trigger_index: int = 0,
        ref_channel: int | None = None,
        search_window: int | None = None,
        apply_to_raw: bool = False,
    ) -> None:
        self.ref_trigger_index = ref_trigger_index
        self.ref_channel = ref_channel
        self.search_window = search_window
        self.apply_to_raw = apply_to_raw
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError("Artifact length not set. Run TriggerDetector before SubsampleAligner.")
        if self.ref_trigger_index >= len(context.get_triggers()):
            raise ProcessorValidationError(
                f"Reference trigger index {self.ref_trigger_index} is out of range "
                f"for {len(context.get_triggers())} triggers"
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers().copy()
        artifact_length = context.get_artifact_length()
        n_samples = raw.n_times
        upsampling_factor = max(1, context.metadata.upsampling_factor)

        # --- LOG ---
        logger.info("Refining trigger alignment with subsample precision")

        # --- COMPUTE ---
        pre_samples, post_samples = _get_pre_post_samples(context, artifact_length)
        window_length = pre_samples + post_samples
        search_radius = self._resolve_search_radius(upsampling_factor)
        ref_channel = self._pick_ref_channel(raw)
        ref_signal = raw.get_data(picks=[ref_channel])[0]

        ref_epoch = _extract_epoch_with_padding(
            ref_signal,
            triggers[self.ref_trigger_index] - pre_samples,
            window_length,
            n_samples,
        )
        shifts = self._compute_shifts(
            ref_signal, triggers, ref_epoch, pre_samples, window_length, search_radius, n_samples
        )
        aligned_triggers = np.clip(triggers + shifts, 0, n_samples - 1).astype(int)

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        new_metadata.triggers = aligned_triggers
        new_metadata.custom.setdefault("subsample_alignment", {}).update(
            {
                "shifts": shifts.tolist(),
                "ref_trigger_index": self.ref_trigger_index,
                "search_window": search_radius,
            }
        )
        if new_metadata.pre_trigger_samples is None:
            new_metadata.pre_trigger_samples = pre_samples
        if new_metadata.post_trigger_samples is None:
            new_metadata.post_trigger_samples = post_samples

        if self.apply_to_raw and np.any(shifts):
            logger.debug("Applying subsample shifts to raw data segments")
            raw_copy = self._apply_shifts_to_raw(raw, triggers, shifts, pre_samples, post_samples, n_samples)
            return context.with_raw(raw_copy).with_metadata(
                new_metadata,
                copy_estimated_noise=False,
            )

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )

    def _resolve_search_radius(self, upsampling_factor: int) -> int:
        """Return the effective search radius in samples.

        Parameters
        ----------
        upsampling_factor : int
            Current upsampling factor.

        Returns
        -------
        int
            Search radius (≥ 1).
        """
        if self.search_window is not None:
            return int(max(1, self.search_window))
        return int(max(1, upsampling_factor * 2))

    def _pick_ref_channel(self, raw: mne.io.Raw) -> int:
        """Return the reference channel index for cross-correlation.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object whose info is used for channel picking.

        Returns
        -------
        int
            Channel index.
        """
        if self.ref_channel is not None:
            return self.ref_channel
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        return int(eeg_channels[0]) if len(eeg_channels) else 0

    def _compute_shifts(
        self,
        ref_signal: np.ndarray,
        triggers: np.ndarray,
        ref_epoch: np.ndarray,
        pre_samples: int,
        window_length: int,
        search_radius: int,
        n_samples: int,
    ) -> np.ndarray:
        """Compute per-trigger subsample alignment shifts.

        Parameters
        ----------
        ref_signal : np.ndarray
            1-D reference channel signal.
        triggers : np.ndarray
            Original trigger positions.
        ref_epoch : np.ndarray
            Reference epoch used as the cross-correlation template.
        pre_samples : int
            Samples before the trigger forming the epoch start.
        window_length : int
            Total epoch length (pre + post) in samples.
        search_radius : int
            Search radius in samples.
        n_samples : int
            Total number of samples in the recording.

        Returns
        -------
        np.ndarray
            Integer shift for each trigger, shape ``(n_triggers,)``.
        """
        shifts = np.zeros(len(triggers), dtype=int)
        for idx, trigger in enumerate(triggers):
            if idx == self.ref_trigger_index:
                continue
            segment = _extract_epoch_with_padding(
                ref_signal,
                trigger - pre_samples - search_radius,
                window_length + 2 * search_radius,
                n_samples,
            )
            corr = np.nan_to_num(crosscorrelation(segment, ref_epoch, search_radius))
            shifts[idx] = int(np.argmax(corr) - search_radius)
        return shifts

    def _apply_shifts_to_raw(
        self,
        raw: mne.io.Raw,
        triggers: np.ndarray,
        shifts: np.ndarray,
        pre_samples: int,
        post_samples: int,
        n_samples: int,
    ) -> mne.io.Raw:
        """Apply computed shifts to raw data segments by rolling them.

        Parameters
        ----------
        raw : mne.io.Raw
            Source raw object (will be copied internally).
        triggers : np.ndarray
            Original trigger positions.
        shifts : np.ndarray
            Per-trigger shift values.
        pre_samples : int
            Samples before trigger forming each window start.
        post_samples : int
            Samples after trigger forming each window end.
        n_samples : int
            Total recording length in samples.

        Returns
        -------
        mne.io.Raw
            New raw object with rolled segments.
        """
        raw_copy = raw.copy()
        # Direct _data access avoids a redundant full-array copy on large datasets
        data = raw_copy.get_data()
        for idx, shift in enumerate(shifts):
            if shift == 0:
                continue
            trigger = triggers[idx]
            window_start = max(0, trigger - pre_samples)
            window_end = min(n_samples, trigger + post_samples)
            segment = data[:, window_start:window_end]
            data[:, window_start:window_end] = np.roll(segment, shift, axis=1)
        raw_copy._data[:] = data
        return raw_copy
