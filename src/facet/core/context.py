"""
Processing Context Module

This module defines the ProcessingContext class which wraps MNE Raw objects
and provides metadata storage, history tracking, and immutable operations.

Author: FACETpy Team
Date: 2025-01-12
"""

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import mne
import numpy as np
from loguru import logger

from facet.logging_config import suppress_stdout


class _UnspecifiedEstimatedNoise:
    """Sentinel type used when a context operation should preserve noise."""


_UNSPECIFIED_ESTIMATED_NOISE = _UnspecifiedEstimatedNoise()


@dataclass
class ProcessingMetadata:
    """Metadata associated with processing context."""

    triggers: np.ndarray | None = None
    trigger_regex: str | None = None
    artifact_to_trigger_offset: float = 0.0
    acq_start_sample: int | None = None
    acq_end_sample: int | None = None
    pre_trigger_samples: int | None = None
    post_trigger_samples: int | None = None
    upsampling_factor: int = 10
    artifact_length: int | None = None
    slices_per_volume: int | None = None
    volume_gaps: bool = False
    custom: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "ProcessingMetadata":
        """Create a deep copy of metadata."""
        return ProcessingMetadata(
            triggers=self.triggers.copy() if self.triggers is not None else None,
            trigger_regex=self.trigger_regex,
            artifact_to_trigger_offset=self.artifact_to_trigger_offset,
            acq_start_sample=self.acq_start_sample,
            acq_end_sample=self.acq_end_sample,
            pre_trigger_samples=self.pre_trigger_samples,
            post_trigger_samples=self.post_trigger_samples,
            upsampling_factor=self.upsampling_factor,
            artifact_length=self.artifact_length,
            slices_per_volume=self.slices_per_volume,
            volume_gaps=self.volume_gaps,
            custom=deepcopy(self.custom),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "triggers": self.triggers.tolist() if self.triggers is not None else None,
            "trigger_regex": self.trigger_regex,
            "artifact_to_trigger_offset": self.artifact_to_trigger_offset,
            "acq_start_sample": self.acq_start_sample,
            "acq_end_sample": self.acq_end_sample,
            "pre_trigger_samples": self.pre_trigger_samples,
            "post_trigger_samples": self.post_trigger_samples,
            "upsampling_factor": self.upsampling_factor,
            "artifact_length": self.artifact_length,
            "slices_per_volume": self.slices_per_volume,
            "volume_gaps": self.volume_gaps,
            "custom": deepcopy(self.custom),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingMetadata":
        """Deserialize metadata from dictionary."""
        triggers = data.get("triggers")
        if triggers is not None:
            triggers = np.array(triggers)
        return cls(
            triggers=triggers,
            trigger_regex=data.get("trigger_regex"),
            artifact_to_trigger_offset=data.get("artifact_to_trigger_offset", 0.0),
            acq_start_sample=data.get("acq_start_sample"),
            acq_end_sample=data.get("acq_end_sample"),
            pre_trigger_samples=data.get("pre_trigger_samples"),
            post_trigger_samples=data.get("post_trigger_samples"),
            upsampling_factor=data.get("upsampling_factor", 10),
            artifact_length=data.get("artifact_length"),
            slices_per_volume=data.get("slices_per_volume"),
            volume_gaps=data.get("volume_gaps", False),
            custom=deepcopy(data.get("custom", {})),
        )


@dataclass
class ProcessingStep:
    """Record of a processing step."""

    name: str
    processor_type: str
    parameters: dict[str, Any]
    timestamp: float


class ProcessingContext:
    """
    Processing context that wraps MNE Raw objects and provides metadata.

    This class serves as the primary data container passed between processors.
    It provides:
    - Access to current and original MNE Raw objects
    - Metadata storage (triggers, parameters, etc.)
    - Processing history tracking
    - Estimated noise tracking
    - Immutable operations (processors return new contexts)

    Attributes:
        _raw: Current processed MNE Raw object
        _raw_original: Original unprocessed MNE Raw object
        _metadata: Processing metadata
        _history: List of processing steps
        _estimated_noise: Accumulated noise estimates
        _cache: Cache for computed values
    """

    def __init__(
        self, raw: mne.io.Raw, raw_original: mne.io.Raw | None = None, metadata: ProcessingMetadata | None = None
    ):
        """
        Initialize processing context.

        Args:
            raw: MNE Raw object
            raw_original: Original Raw object (if None, copies raw)
            metadata: Processing metadata (if None, creates empty)
        """
        self._raw = raw
        self._raw_original = raw_original if raw_original is not None else raw
        self._metadata = metadata if metadata is not None else ProcessingMetadata()
        self._history: list[ProcessingStep] = []
        self._estimated_noise: np.ndarray | None = None
        self._cache: dict[str, Any] = {}

    # =========================================================================
    # Raw Data Access
    # =========================================================================

    def get_raw(self) -> mne.io.Raw:
        """Get current processed Raw object."""
        return self._raw

    def get_raw_original(self) -> mne.io.Raw:
        """Get original unprocessed Raw object."""
        return self._raw_original

    def has_raw(self) -> bool:
        """Check if Raw object exists."""
        return self._raw is not None

    def _call_raw_get_data(self, picks: Any | None = None, **kwargs) -> np.ndarray:
        """
        Call Raw.get_data while gracefully handling removed keywords.

        Some lightweight MNE containers (e.g., RawArray created during chunk
        execution) do not accept the ``copy`` keyword. We optimistically pass
        it to avoid extra allocations and fall back to the default behaviour if
        the object rejects it.
        """
        try:
            return self._raw.get_data(picks=picks, **kwargs)
        except TypeError as exc:
            if "copy" in kwargs and "unexpected keyword argument 'copy'" in str(exc):
                logger.debug("Raw.get_data() rejected 'copy'; retrying without it")
                safe_kwargs = dict(kwargs)
                safe_kwargs.pop("copy", None)
                return self._raw.get_data(picks=picks, **safe_kwargs)
            raise

    def get_data(self, picks: Any | None = None, **kwargs) -> np.ndarray:
        """
        Get data from Raw object.

        Args:
            picks: Channel picks (MNE format)
            **kwargs: Additional arguments for Raw.get_data(copy=False)

        Returns:
            Data array
        """
        return self._call_raw_get_data(picks=picks, **kwargs)

    # =========================================================================
    # Metadata Access
    # =========================================================================

    @property
    def metadata(self) -> ProcessingMetadata:
        """Get processing metadata."""
        return self._metadata

    def get_triggers(self) -> np.ndarray | None:
        """Get trigger positions."""
        return self._metadata.triggers

    def has_triggers(self) -> bool:
        """Check if triggers exist."""
        return self._metadata.triggers is not None and len(self._metadata.triggers) > 0

    def get_artifact_length(self) -> int | None:
        """Get artifact length in samples."""
        return self._metadata.artifact_length

    def get_sfreq(self) -> float:
        """Get sampling frequency."""
        return self._raw.info["sfreq"]

    def get_n_channels(self) -> int:
        """Get number of channels."""
        return len(self._raw.ch_names)

    def get_channel_names(self) -> list[str]:
        """Get channel names."""
        return self._raw.ch_names

    def get_metric(self, name: str, default=None):
        """
        Return a single evaluation metric stored in the context.

        Shortcut for the common pattern::

            ctx.metadata.custom.get('metrics', {}).get(name, default)

        Typically used inside a :class:`~facet.core.ConditionalProcessor`
        condition function after an evaluation step has run.

        Args:
            name: Metric name (e.g. ``'snr'``, ``'rms_ratio'``).
            default: Value returned when the metric is absent.

        Example::

            def needs_extra_correction(ctx):
                return ctx.get_metric('snr', float('inf')) < 10
        """
        return self._metadata.custom.get("metrics", {}).get(name, default)

    # =========================================================================
    # Noise Tracking
    # =========================================================================

    def get_estimated_noise(self) -> np.ndarray | None:
        """Get accumulated noise estimates."""
        return self._estimated_noise

    def has_estimated_noise(self) -> bool:
        """Check if noise estimates exist."""
        return self._estimated_noise is not None

    def set_estimated_noise(self, noise: np.ndarray, *, copy: bool = False) -> None:
        """Set the estimated noise.

        Args:
            noise: Noise array whose shape matches the processed Raw data.
            copy: Copy ``noise`` before storing it. The default preserves the
                historical ownership-transfer behaviour and avoids an
                allocation. Pass ``True`` when the caller will continue
                mutating its array.
        """
        self._estimated_noise = noise.copy() if copy else noise

    def accumulate_noise(self, noise: np.ndarray, *, copy: bool = True) -> None:
        """Add to accumulated noise estimates.

        Args:
            noise: Noise contribution to accumulate.
            copy: When this is the first contribution, copy it before storing.
                The default preserves the historical behaviour. Callers may
                pass ``False`` to transfer ownership of a newly allocated
                array and avoid retaining two identical full-size buffers.
        """
        if self._estimated_noise is None:
            self._estimated_noise = noise.copy() if copy else noise
        else:
            self._estimated_noise += noise

    # =========================================================================
    # History Tracking
    # =========================================================================

    def get_history(self) -> list[ProcessingStep]:
        """Get processing history."""
        return self._history.copy()

    def add_history_entry(
        self,
        name: str | None = None,
        processor_type: str = "",
        parameters: dict[str, Any] | None = None,
        *,
        processor_name: str | None = None,
    ) -> None:
        """Add entry to processing history."""
        import time

        resolved_name = processor_name if processor_name is not None else (name or "")
        step = ProcessingStep(
            name=resolved_name, processor_type=processor_type, parameters=parameters or {}, timestamp=time.time()
        )
        self._history.append(step)
        logger.debug(f"Added processing step: {resolved_name}")

    # =========================================================================
    # Cache Management
    # =========================================================================

    def cache_get(self, key: str) -> Any | None:
        """Get value from cache."""
        return self._cache.get(key)

    def cache_set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[key] = value

    def cache_has(self, key: str) -> bool:
        """Check if cache has key."""
        return key in self._cache

    def cache_clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    def release_signal_data(self) -> None:
        """Release signal-sized state while retaining metadata and history.

        This operation is intended for completed batch and chunk results whose
        metrics and processing history still need to be inspected. Raw data,
        original Raw data, estimated noise, and cached derived values become
        unavailable after the call.
        """
        self._raw = None
        self._raw_original = None
        self._estimated_noise = None
        self.cache_clear()

    # =========================================================================
    # Immutable Operations (Return New Context)
    # =========================================================================

    @staticmethod
    def _prepare_estimated_noise(
        noise: np.ndarray | None,
        *,
        copy_estimated_noise: bool,
    ) -> np.ndarray | None:
        """Copy or transfer an estimated-noise array for a new context."""
        if noise is None or not copy_estimated_noise:
            return noise
        return noise.copy()

    def with_raw(
        self,
        raw: mne.io.Raw,
        copy_metadata: bool = True,
        *,
        estimated_noise: np.ndarray | None | _UnspecifiedEstimatedNoise = _UNSPECIFIED_ESTIMATED_NOISE,
        copy_estimated_noise: bool = True,
    ) -> "ProcessingContext":
        """
        Create new context with updated Raw object.

        Args:
            raw: New Raw object
            copy_metadata: Whether to copy metadata from current context
            estimated_noise: Replacement estimated-noise array. When omitted,
                the current estimate is preserved. Pass ``None`` explicitly to
                create the new context without an estimate.
            copy_estimated_noise: Copy the preserved or replacement estimate.
                This defaults to ``True`` so ordinary immutable context
                operations keep independent arrays. Set it to ``False`` only
                when deliberately transferring ownership or sharing a
                read-only estimate.

        Returns:
            New ProcessingContext
        """
        metadata = self._metadata.copy() if copy_metadata else ProcessingMetadata()
        noise = self._estimated_noise if isinstance(estimated_noise, _UnspecifiedEstimatedNoise) else estimated_noise

        new_ctx = ProcessingContext(raw=raw, raw_original=self._raw_original, metadata=metadata)
        new_ctx._history = self._history.copy()
        new_ctx._estimated_noise = self._prepare_estimated_noise(
            noise,
            copy_estimated_noise=copy_estimated_noise,
        )
        # Don't copy cache (it may be invalidated)
        return new_ctx

    def with_metadata(
        self,
        metadata: ProcessingMetadata,
        *,
        estimated_noise: np.ndarray | None | _UnspecifiedEstimatedNoise = _UNSPECIFIED_ESTIMATED_NOISE,
        copy_estimated_noise: bool = True,
    ) -> "ProcessingContext":
        """
        Create new context with updated metadata.

        Args:
            metadata: New metadata
            estimated_noise: Replacement estimated-noise array. When omitted,
                the current estimate is preserved. Pass ``None`` explicitly to
                create the new context without an estimate.
            copy_estimated_noise: Copy the preserved or replacement estimate.
                This defaults to ``True`` to preserve immutable-operation
                behaviour. Set it to ``False`` only for an intentional
                ownership transfer or read-only share.

        Returns:
            New ProcessingContext
        """
        noise = self._estimated_noise if isinstance(estimated_noise, _UnspecifiedEstimatedNoise) else estimated_noise

        new_ctx = ProcessingContext(raw=self._raw, raw_original=self._raw_original, metadata=metadata)
        new_ctx._history = self._history.copy()
        new_ctx._estimated_noise = self._prepare_estimated_noise(
            noise,
            copy_estimated_noise=copy_estimated_noise,
        )
        return new_ctx

    def with_triggers(self, triggers: np.ndarray) -> "ProcessingContext":
        """
        Create new context with updated triggers.

        Args:
            triggers: New trigger array

        Returns:
            New ProcessingContext
        """
        new_metadata = self._metadata.copy()
        new_metadata.triggers = triggers
        return self.with_metadata(new_metadata)

    @staticmethod
    def _coerce_integer_array(values: Any, *, name: str) -> np.ndarray:
        """Convert scalar/array-like input to a 1D integer numpy array."""
        arr = np.asarray(values)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        arr = np.ravel(arr)

        if arr.size == 0:
            return arr.astype(np.int64)

        if np.issubdtype(arr.dtype, np.bool_):
            raise ValueError(f"{name} must contain integer sample positions, not booleans")

        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(np.int64, copy=False)

        if np.issubdtype(arr.dtype, np.floating):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite numeric values")
            rounded = np.rint(arr)
            if not np.allclose(arr, rounded):
                raise ValueError(f"{name} must contain integer-valued sample positions")
            return rounded.astype(np.int64)

        converted: list[int] = []
        for value in arr.tolist():
            if isinstance(value, bool):
                raise ValueError(f"{name} must contain integer sample positions, not booleans")
            if isinstance(value, Integral):
                converted.append(int(value))
                continue
            if isinstance(value, Real):
                float_value = float(value)
                if not np.isfinite(float_value) or not float_value.is_integer():
                    raise ValueError(f"{name} must contain integer-valued sample positions")
                converted.append(int(float_value))
                continue
            raise ValueError(f"{name} must contain integer sample positions")
        return np.asarray(converted, dtype=np.int64)

    def with_trigger_samples(
        self,
        triggers: np.ndarray | list[int],
        *,
        artifact_length: int | None = None,
        tr_seconds: float | None = None,
        trigger_regex: str | None = None,
        custom: dict[str, Any] | None = None,
        samples_are_absolute: bool = False,
    ) -> "ProcessingContext":
        """
        Return a new context with trigger samples and optional artifact metadata.

        Parameters
        ----------
        triggers : array-like of int
            Trigger sample positions.
        artifact_length : int, optional
            Artifact window length in samples. If omitted (and ``tr_seconds``
            is also omitted), FACETpy infers it as the median spacing between
            trigger samples when at least two triggers are available.
        tr_seconds : float, optional
            Artifact window length in seconds. Converted using ``raw.info['sfreq']``.
        trigger_regex : str, optional
            Label/pattern stored in metadata for provenance.
        custom : dict, optional
            Additional ``metadata.custom`` entries to merge in.
        samples_are_absolute : bool, optional
            If ``True``, trigger samples are interpreted as absolute MNE sample
            indices and converted to context-relative indices by subtracting
            ``raw.first_samp``.
        """
        trigger_array = self._coerce_integer_array(triggers, name="triggers")

        if samples_are_absolute:
            trigger_array = trigger_array - int(self._raw.first_samp)

        if trigger_array.size > 0 and (trigger_array.min() < 0 or trigger_array.max() >= self._raw.n_times):
            raise ValueError(
                f"Trigger samples must fall within [0, {self._raw.n_times - 1}] "
                f"for this context, got min={int(trigger_array.min())}, max={int(trigger_array.max())}"
            )

        if tr_seconds is not None and artifact_length is not None:
            raise ValueError("Pass either artifact_length or tr_seconds, not both")
        if tr_seconds is not None:
            if not isinstance(tr_seconds, Real) or isinstance(tr_seconds, bool) or tr_seconds <= 0:
                raise ValueError("tr_seconds must be a positive number")
            artifact_length = int(round(float(tr_seconds) * self.get_sfreq()))
        if artifact_length is not None:
            if not isinstance(artifact_length, Integral) or isinstance(artifact_length, bool):
                raise ValueError("artifact_length must be an integer number of samples")
            artifact_length = int(artifact_length)
            if artifact_length <= 0:
                raise ValueError("artifact_length must be > 0")
        elif self._metadata.artifact_length is None and trigger_array.size > 1:
            # Default for trigger-based workflows: infer artifact length from
            # trigger spacing when none is provided.
            sorted_triggers = np.sort(trigger_array)
            positive_diffs = np.diff(sorted_triggers)
            positive_diffs = positive_diffs[positive_diffs > 0]
            if positive_diffs.size > 0:
                artifact_length = int(round(float(np.median(positive_diffs))))

        new_metadata = self._metadata.copy()
        new_metadata.triggers = trigger_array.astype(np.int32, copy=False)

        if artifact_length is not None:
            new_metadata.artifact_length = artifact_length
        if trigger_regex is not None:
            new_metadata.trigger_regex = trigger_regex
        if custom:
            new_metadata.custom.update(deepcopy(custom))

        return self.with_metadata(new_metadata)

    def with_mne_events(
        self,
        events: np.ndarray,
        *,
        event: int | str | None = None,
        event_id: dict[str, int] | None = None,
        artifact_length: int | None = None,
        tr_seconds: float | None = None,
        store_event_id: bool = True,
    ) -> "ProcessingContext":
        """
        Build trigger metadata from MNE events and return a new context.

        Parameters
        ----------
        events : np.ndarray
            MNE-style event array with shape ``(n_events, 3)``.
        event : int or str, optional
            Event code or event name to select. If ``None``, events must
            contain exactly one unique code.
        event_id : dict, optional
            Mapping from event names to integer codes (from
            ``mne.events_from_annotations``). Required when ``event`` is a str.
        artifact_length : int, optional
            Artifact length in samples.
        tr_seconds : float, optional
            Artifact length in seconds. Takes precedence over default
            inference, but cannot be combined with ``artifact_length``.
        store_event_id : bool, optional
            If ``True`` and ``event_id`` is provided, stores it in
            ``metadata.custom['event_id']``.
        """
        event_arr = np.asarray(events)
        if event_arr.ndim != 2 or event_arr.shape[1] < 3:
            raise ValueError("events must be an array with shape (n_events, 3)")
        if event_arr.shape[0] == 0:
            raise ValueError("events is empty")

        sample_positions = self._coerce_integer_array(event_arr[:, 0], name="events[:, 0]")
        event_codes = self._coerce_integer_array(event_arr[:, 2], name="events[:, 2]")
        unique_codes = np.unique(event_codes)

        selected_name: str | None = None
        if event is None:
            if unique_codes.size != 1:
                raise ValueError(
                    "events contains multiple event codes; pass `event=<code>` or `event=<name>` to select one"
                )
            selected_code = int(unique_codes[0])
        elif isinstance(event, str):
            if event_id is None:
                raise ValueError("event_id is required when event is a string")
            if event not in event_id:
                available = ", ".join(sorted(event_id)) if event_id else "<none>"
                raise ValueError(f"Unknown event name '{event}'. Available names: {available}")
            selected_code = int(event_id[event])
            selected_name = event
        elif isinstance(event, Integral) and not isinstance(event, bool):
            selected_code = int(event)
        else:
            raise ValueError("event must be an integer code, an event name, or None")

        selected_samples = sample_positions[event_codes == selected_code]
        if selected_samples.size == 0:
            raise ValueError(f"No samples found for event code {selected_code}")

        custom: dict[str, Any] = {"selected_event_code": selected_code}
        if selected_name is not None:
            custom["selected_event_name"] = selected_name
        if store_event_id and event_id is not None:
            custom["event_id"] = deepcopy(event_id)

        trigger_label = f"MNE:{selected_name}" if selected_name is not None else f"MNE:{selected_code}"
        return self.with_trigger_samples(
            selected_samples,
            artifact_length=artifact_length,
            tr_seconds=tr_seconds,
            trigger_regex=trigger_label,
            custom=custom,
            samples_are_absolute=True,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def copy(self, deep: bool = False) -> "ProcessingContext":
        """
        Create a copy of the context.

        Args:
            deep: If True, deep copy Raw object (expensive)

        Returns:
            Copied ProcessingContext
        """
        raw_copy = self._raw.copy() if deep else self._raw
        return self.with_raw(raw_copy, copy_metadata=True)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize context to dictionary (for multiprocessing).

        Returns:
            Dictionary representation
        """
        raw_data = getattr(self._raw, "_data", None)
        if not self._raw.preload or raw_data is None:
            raw_data = self._call_raw_get_data(copy=False)

        return {
            # Reuse the preloaded buffer when possible. MNE versions without
            # ``Raw.get_data(copy=False)`` otherwise allocate a redundant
            # signal-sized array before multiprocessing pickles the payload.
            "raw_data": raw_data,
            "raw_info": self._raw.info,
            "metadata": self._metadata.to_dict(),
            "history": [
                {
                    "name": step.name,
                    "processor_type": step.processor_type,
                    "parameters": step.parameters,
                    "timestamp": step.timestamp,
                }
                for step in self._history
            ],
            "estimated_noise": self._estimated_noise,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingContext":
        """
        Deserialize context from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ProcessingContext instance
        """
        # suppress_stdout: RawArray creation prints verbose MNE messages
        with suppress_stdout():
            raw = mne.io.RawArray(data["raw_data"], data["raw_info"])

        metadata = ProcessingMetadata.from_dict(data["metadata"])

        ctx = cls(raw=raw, metadata=metadata)
        ctx._estimated_noise = data["estimated_noise"]
        return ctx

    def __or__(self, other) -> "ProcessingContext":
        """
        Apply a processor or callable to this context using the pipe operator.

        Enables a clean chaining syntax outside of a ``Pipeline``::

            ctx = ProcessingContext(raw)
            result = (
                ctx
                | HighPassFilter(1.0)
                | UpSample(10)
                | TriggerDetector(r"\\b1\\b")
                | AASCorrection()
            )
            filtered_raw = result.get_raw()

        Args:
            other: A :class:`~facet.core.Processor` instance **or** any
                ``Callable[[ProcessingContext], ProcessingContext]``.

        Returns:
            New :class:`~facet.core.context.ProcessingContext` produced by applying *other*.
        """
        from .processor import Processor

        if isinstance(other, Processor):
            return other.execute(self)
        elif callable(other):
            return other(self)
        return NotImplemented

    def __repr__(self) -> str:
        """String representation."""
        n_channels = len(self._raw.ch_names)
        n_times = self._raw.n_times
        sfreq = self._raw.info["sfreq"]
        n_triggers = len(self._metadata.triggers) if self._metadata.triggers is not None else 0
        return (
            f"ProcessingContext(n_channels={n_channels}, n_times={n_times}, "
            f"sfreq={sfreq}Hz, n_triggers={n_triggers}, "
            f"n_steps={len(self._history)})"
        )
