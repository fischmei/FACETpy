"""Trigger Detection Processors Module

Processors for detecting triggers and events in EEG data recorded during
simultaneous EEG-fMRI acquisition.
"""

import re

import mne
import numpy as np
from loguru import logger

from ..core import (
    ProcessingContext,
    Processor,
    ProcessorError,
    ProcessorValidationError,
    register_processor,
)
from ..helpers.crosscorr import crosscorrelation


@register_processor
class TriggerDetector(Processor):
    """Detect triggers from annotations, STIM, or discrete trigger channels.

    Searches the raw data for events whose description (annotation) or integer
    value (STIM channel) matches the supplied regular expression. If no typed
    STIM/annotation events match, it can also detect a zero-baseline discrete
    trigger channel that was loaded as regular data, such as an EEGLAB
    ``Status`` channel. Detected trigger sample positions are stored in
    ``context.metadata.triggers``.

    The artifact length is estimated from the median inter-trigger interval;
    volume-level gaps in slice-triggered acquisitions are detected
    automatically.

    Parameters
    ----------
    regex : str
        Regular expression pattern to match trigger values.
    save_to_annotations : bool, optional
        If ``True``, write detected triggers back to the raw annotations
        (default: False).
    """

    name = "trigger_detector"
    description = "Detect triggers using regex pattern"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, regex: str, save_to_annotations: bool = False) -> None:
        self.regex = regex
        self.save_to_annotations = save_to_annotations
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        sfreq = raw.info["sfreq"]

        # --- LOG ---
        logger.info("Detecting triggers with pattern: {}", self.regex)

        # --- COMPUTE ---
        filtered_events = self._find_events(raw)

        if len(filtered_events) == 0:
            logger.warning("No triggers found!")
            return context

        triggers = np.array([event[0] for event in filtered_events])
        # MNE returns absolute sample indices (onset * sfreq + first_samp).
        # Normalize to 0-indexed positions relative to the current raw start
        # so that triggers can be used directly as indices into raw._data.
        triggers = triggers - raw.first_samp
        logger.info("Found {} triggers", len(triggers))

        artifact_meta = self._compute_artifact_metadata(triggers)

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        new_metadata.triggers = triggers
        new_metadata.trigger_regex = self.regex
        new_metadata.artifact_length = artifact_meta["artifact_length"]
        new_metadata.volume_gaps = artifact_meta["volume_gaps"]
        if artifact_meta.get("slices_per_volume") is not None:
            new_metadata.slices_per_volume = artifact_meta["slices_per_volume"]

        logger.debug("Artifact length: {} samples", new_metadata.artifact_length)
        logger.debug("Volume gaps: {}", new_metadata.volume_gaps)

        if self.save_to_annotations:
            raw_copy = raw.copy()
            raw_copy.set_annotations(
                mne.Annotations(
                    onset=triggers / sfreq,
                    duration=np.zeros(len(triggers)),
                    description=["Trigger"] * len(triggers),
                )
            )
            return context.with_raw(raw_copy).with_metadata(new_metadata)

        # --- RETURN ---
        return context.with_metadata(new_metadata)

    def _find_events(self, raw: mne.io.Raw) -> list:
        """Search stim channels then annotations for matching events.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object to search.

        Returns
        -------
        list
            List of MNE-style event rows ``[sample, prev_id, event_id]``.
        """
        pattern = re.compile(self.regex)
        stim_events = self._find_stim_events(raw, pattern)
        if len(stim_events) > 0:
            return stim_events

        logger.debug("No matching STIM events, searching annotations")
        events_obj = mne.events_from_annotations(raw, verbose=False)
        logger.debug("Event types: {}", events_obj[1])
        annotation_events = list(mne.events_from_annotations(raw, regexp=self.regex, verbose=False)[0])
        if len(annotation_events) > 0:
            logger.info("Using annotations with {} detected trigger(s)", len(annotation_events))
            return annotation_events

        return self._find_discrete_channel_events(raw, pattern)

    def _find_stim_events(self, raw: mne.io.Raw, pattern: re.Pattern) -> list:
        """Find matching events in all channels already marked as STIM."""
        stim_channels = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)
        if len(stim_channels) == 0:
            return []

        logger.debug("Found {} stim channels", len(stim_channels))
        matched_events = []

        for channel_idx in stim_channels:
            channel_name = raw.ch_names[channel_idx]
            events = mne.find_events(
                raw,
                stim_channel=channel_name,
                initial_event=True,
                verbose=False,
            )
            if len(events) == 0:
                continue

            if self._allows_positive_trigger_fallback(pattern):
                channel_matches = [event for event in events if int(event[2]) > 0]
            else:
                channel_matches = [event for event in events if pattern.search(str(event[2]))]
            if channel_matches:
                logger.info(
                    "Using STIM trigger channel '{}' with {} detected trigger(s)",
                    channel_name,
                    len(channel_matches),
                )
                matched_events.extend(channel_matches)
                continue

        if matched_events:
            return self._sort_events(matched_events)
        return []

    def _find_discrete_channel_events(self, raw: mne.io.Raw, pattern: re.Pattern) -> list:
        """Find a trigger pulse train stored in a non-STIM data channel.

        Some loaders, notably EEGLAB, can expose trigger/status channels as
        ordinary EEG channels. We only accept channels that look discrete and
        contain zero-baseline positive pulses, which avoids mistaking normal EEG
        fluctuations for triggers.
        """
        stim_channels = set(mne.pick_types(raw.info, meg=False, eeg=False, stim=True))
        candidates = []

        for channel_idx, channel_name in enumerate(raw.ch_names):
            if channel_idx in stim_channels:
                continue

            data = raw.get_data(picks=[channel_idx])[0]
            candidate = self._events_from_discrete_channel(raw, channel_idx, data, pattern)
            if candidate is None:
                continue

            logger.debug(
                "Trigger candidate {}: {} events from values {}",
                channel_name,
                len(candidate["events"]),
                candidate["values"],
            )
            candidates.append(candidate)

        if len(candidates) == 0:
            return []

        candidates.sort(key=lambda item: item["score"], reverse=True)
        best = candidates[0]
        if len(candidates) > 1:
            logger.warning(
                "Multiple discrete trigger-like channels found ({}); using {}",
                ", ".join(candidate["channel_name"] for candidate in candidates),
                best["channel_name"],
            )

        logger.info(
            "Using trigger channel '{}' with {} detected trigger(s)",
            best["channel_name"],
            len(best["events"]),
        )
        return self._sort_events(best["events"])

    def _events_from_discrete_channel(
        self,
        raw: mne.io.Raw,
        channel_idx: int,
        data: np.ndarray,
        pattern: re.Pattern,
    ) -> dict | None:
        """Build MNE-style events from one discrete zero-to-positive channel."""
        if data.size == 0 or not np.all(np.isfinite(data)):
            return None

        quantized = self._quantize_channel(data)
        unique_values = np.unique(quantized)
        zero_tolerance = self._zero_tolerance(unique_values)
        nonzero_values = [value for value in unique_values if abs(value) > zero_tolerance]
        if len(nonzero_values) == 0 or not self._is_discrete_trigger_channel(unique_values, zero_tolerance):
            return None

        if self._allows_positive_trigger_fallback(pattern):
            matched_values = [value for value in nonzero_values if value > zero_tolerance]
            active = quantized > zero_tolerance
            event_value = self._event_id_from_values(matched_values)
        else:
            matched_values = [value for value in nonzero_values if pattern.search(self._format_trigger_value(value))]
            if not matched_values:
                return None
            active = np.isin(quantized, matched_values)
            event_value = self._event_id_from_values(matched_values)

        if len(matched_values) == 0:
            return None

        rising_edges = self._rising_edges(active)
        if len(rising_edges) == 0:
            return None

        events = [[int(sample + raw.first_samp), 0, event_value] for sample in rising_edges]
        channel_name = raw.ch_names[channel_idx]
        score = (
            int(self._has_trigger_name_hint(channel_name)) * 1000
            + int(bool(matched_values)) * 100
            - len(unique_values)
        )
        return {
            "channel_name": channel_name,
            "events": events,
            "score": score,
            "values": [self._format_trigger_value(value) for value in matched_values],
        }

    @staticmethod
    def _sort_events(events: list) -> list:
        """Return events sorted by sample index."""
        return sorted(events, key=lambda event: int(event[0]))

    @staticmethod
    def _quantize_channel(data: np.ndarray) -> np.ndarray:
        """Round tiny floating-point noise without changing trigger codes."""
        return np.round(data.astype(float), decimals=12)

    @staticmethod
    def _zero_tolerance(values: np.ndarray) -> float:
        """Compute a scale-aware tolerance for deciding whether a code is zero."""
        scale = float(np.max(np.abs(values))) if values.size else 0.0
        return max(np.finfo(float).eps, scale * 1e-9)

    @staticmethod
    def _is_discrete_trigger_channel(values: np.ndarray, zero_tolerance: float) -> bool:
        """Check whether channel values look like trigger codes, not EEG."""
        has_zero = np.any(np.abs(values) <= zero_tolerance)
        has_positive = np.any(values > zero_tolerance)
        return has_zero and has_positive and len(values) <= 128

    @staticmethod
    def _rising_edges(active: np.ndarray) -> np.ndarray:
        """Return the first sample of each active pulse."""
        if active.size == 0 or np.all(active) or not np.any(active):
            return np.array([], dtype=np.int64)

        edges = np.flatnonzero(np.diff(active.astype(np.int8)) == 1) + 1
        if active[0]:
            edges = np.r_[0, edges]
        return edges.astype(np.int64)

    @staticmethod
    def _format_trigger_value(value: float) -> str:
        """Format a numeric trigger value for regex matching and logging."""
        rounded = round(float(value))
        if np.isclose(value, rounded):
            return str(int(rounded))
        return f"{value:g}"

    @staticmethod
    def _event_id_from_values(values: list[float]) -> int:
        """Choose a stable positive event id for MNE-style event rows."""
        positive_values = [value for value in values if value > 0]
        if not positive_values:
            return 1
        return max(1, int(round(float(max(positive_values)))))

    @staticmethod
    def _has_trigger_name_hint(channel_name: str) -> bool:
        """Return whether a channel name commonly denotes triggers."""
        normalized = channel_name.lower().replace(" ", "").replace("_", "")
        hints = ("status", "stim", "trigger", "trig", "marker", "event", "ttl", "sync", "trev")
        return any(hint in normalized for hint in hints)

    @staticmethod
    def _allows_positive_trigger_fallback(pattern: re.Pattern) -> bool:
        """Treat non-zero pulses as triggers for the common trigger value regex."""
        return pattern.search("1") is not None

    def _compute_artifact_metadata(self, triggers: np.ndarray) -> dict:
        """Estimate artifact length and detect volume gaps from trigger spacing.

        Parameters
        ----------
        triggers : np.ndarray
            Detected trigger sample positions.

        Returns
        -------
        dict
            Keys: ``artifact_length``, ``volume_gaps``, optionally
            ``slices_per_volume``.
        """
        if len(triggers) <= 1:
            return {"artifact_length": None, "volume_gaps": False}

        trigger_diffs = np.diff(triggers)
        ptp = np.ptp(trigger_diffs)

        if ptp > 3:
            return self._compute_slice_volume_metadata(triggers, trigger_diffs)

        return {
            "artifact_length": int(np.max(trigger_diffs)),
            "volume_gaps": False,
        }

    def _compute_slice_volume_metadata(self, triggers: np.ndarray, trigger_diffs: np.ndarray) -> dict:
        """Compute metadata when volume-level gaps are present.

        Parameters
        ----------
        triggers : np.ndarray
            All trigger sample positions.
        trigger_diffs : np.ndarray
            Differences between consecutive triggers.

        Returns
        -------
        dict
            Keys: ``artifact_length``, ``volume_gaps``, ``slices_per_volume``.
        """
        mean_val = np.mean([np.median(trigger_diffs), np.max(trigger_diffs)])
        slice_diffs = trigger_diffs[trigger_diffs < mean_val]
        artifact_length = int(np.max(slice_diffs))

        gap_indices = np.where(trigger_diffs >= mean_val)[0]
        slices_per_volume = None

        if len(gap_indices) > 0:
            slice_counts = []
            last_idx = -1
            for idx in gap_indices:
                slice_counts.append(idx - last_idx)
                last_idx = idx
            if last_idx < len(triggers) - 1:
                slice_counts.append(len(triggers) - 1 - last_idx)
            if slice_counts:
                slices_per_volume = int(np.median(slice_counts))
                logger.info("Estimated slices per volume: {}", slices_per_volume)

        return {
            "artifact_length": artifact_length,
            "volume_gaps": True,
            "slices_per_volume": slices_per_volume,
        }


@register_processor
class QRSTriggerDetector(Processor):
    """Detect QRS complexes (heartbeats) for BCG artifact correction.

    Uses the FMRIB QRS detector from ``facet.helpers.bcg_detector``, which
    requires the ``neurokit2`` package (available via ``pip install
    facetpy[all]``).

    The artifact length is set to half the median RR interval, centred on each
    detected R-peak.

    Parameters
    ----------
    save_to_annotations : bool, optional
        If ``True``, write detected QRS peaks back to the raw annotations
        (default: False).
    """

    name = "qrs_trigger_detector"
    description = "Detect QRS complexes for BCG correction"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, save_to_annotations: bool = False) -> None:
        self.save_to_annotations = save_to_annotations
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        sfreq = raw.info["sfreq"]

        # --- LOG ---
        logger.info("Detecting QRS complexes")

        # --- COMPUTE ---
        try:
            from ..helpers import bcg_detector
        except ImportError as err:
            raise ProcessorError(
                "neurokit2 is required for QRSTriggerDetector. Install with: pip install facetpy[all]"
            ) from err

        peaks = bcg_detector.fmrib_qrsdetect(raw)
        triggers = np.array(peaks, dtype=np.int32)
        logger.info("Found {} QRS peaks", len(triggers))

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        new_metadata.triggers = triggers
        new_metadata.trigger_regex = "QRS"
        new_metadata.volume_gaps = True  # QRS peaks have variable spacing

        if len(triggers) > 1:
            rr_intervals = np.diff(triggers)
            median_rr = int(np.median(rr_intervals))
            new_metadata.artifact_length = median_rr // 2
            new_metadata.artifact_to_trigger_offset = -new_metadata.artifact_length / (2 * sfreq)

        if self.save_to_annotations:
            raw_copy = raw.copy()
            raw_copy.set_annotations(
                mne.Annotations(
                    onset=triggers / sfreq,
                    duration=np.zeros(len(triggers)),
                    description=["QRS"] * len(triggers),
                )
            )
            return context.with_raw(raw_copy).with_metadata(new_metadata)

        # --- RETURN ---
        return context.with_metadata(new_metadata)


@register_processor
class MissingTriggerDetector(Processor):
    """Detect and insert missing triggers by template matching.

    Scans the trigger sequence for gaps larger than 1.9× the artifact length
    and attempts to locate missing artifact epochs by cross-correlating a
    reference template against the signal. Optionally extends the search one
    step before the first trigger and one step after the last.

    Parameters
    ----------
    add_to_context : bool, optional
        If ``True``, insert found triggers into metadata and annotations
        (default: True).
    correlation_threshold : float, optional
        Minimum absolute Pearson correlation to accept a candidate trigger
        (default: 0.9).
    search_window_factor : float, optional
        Search window as a fraction of artifact length (default: 0.1).
    ref_channel : int, optional
        Reference channel index for template matching (default: 0).
    """

    name = "missing_trigger_detector"
    description = "Detect and add missing triggers"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        add_to_context: bool = True,
        correlation_threshold: float = 0.9,
        search_window_factor: float = 0.1,
        ref_channel: int = 0,
    ) -> None:
        self.add_to_context = add_to_context
        self.correlation_threshold = correlation_threshold
        self.search_window_factor = search_window_factor
        self.ref_channel = ref_channel
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
        tmin = int(context.metadata.artifact_to_trigger_offset * sfreq)
        tmax = tmin + artifact_length

        # --- LOG ---
        logger.info("Searching for missing triggers")

        # --- COMPUTE ---
        ref_channel_data = raw.get_data(picks=[self.ref_channel])[0]
        template = self._build_template(ref_channel_data, triggers, tmin, tmax)
        search_window = int(self.search_window_factor * artifact_length)

        missing_triggers = self._find_missing_triggers(
            ref_channel_data, triggers, template, artifact_length, search_window, tmin, tmax
        )
        logger.info("Found {} missing triggers", len(missing_triggers))

        if len(missing_triggers) == 0:
            return context

        # --- BUILD RESULT ---
        if self.add_to_context:
            return self._build_context_with_missing(context, raw, triggers, missing_triggers, sfreq)

        new_metadata = context.metadata.copy()
        new_metadata.custom["missing_triggers"] = missing_triggers

        # --- RETURN ---
        return context.with_metadata(new_metadata)

    def _build_template(
        self,
        ref_data: np.ndarray,
        triggers: np.ndarray,
        tmin: int,
        tmax: int,
    ) -> np.ndarray:
        """Build a reference artifact template from the first few triggers.

        Parameters
        ----------
        ref_data : np.ndarray
            1-D reference channel signal.
        triggers : np.ndarray
            Trigger sample positions.
        tmin : int
            Start offset from trigger to artifact onset.
        tmax : int
            End offset from trigger to artifact end.

        Returns
        -------
        np.ndarray
            Averaged template epoch.
        """
        n_template = min(5, len(triggers))
        template_epochs = []
        for i in range(n_template):
            start = triggers[i] + tmin
            end = triggers[i] + tmax
            if end <= len(ref_data):
                template_epochs.append(ref_data[start:end])
        return np.mean(template_epochs, axis=0)

    def _find_missing_triggers(
        self,
        ref_data: np.ndarray,
        triggers: np.ndarray,
        template: np.ndarray,
        artifact_length: int,
        search_window: int,
        tmin: int,
        tmax: int,
    ) -> list:
        """Search for missing triggers before, within, and after the sequence.

        Parameters
        ----------
        ref_data : np.ndarray
            1-D reference channel signal.
        triggers : np.ndarray
            Known trigger positions.
        template : np.ndarray
            Reference artifact template.
        artifact_length : int
            Expected artifact length in samples.
        search_window : int
            Cross-correlation search radius.
        tmin : int
            Artifact onset offset from trigger.
        tmax : int
            Artifact end offset from trigger.

        Returns
        -------
        list of int
            Positions of detected missing triggers.
        """
        missing_triggers = []

        for i in range(len(triggers) - 1):
            gap = triggers[i + 1] - triggers[i]
            if gap > artifact_length * 1.9:
                search_pos = triggers[i] + artifact_length
                candidate = self._align_to_template(ref_data, template, search_pos, search_window, tmin, tmax)
                if self._is_artifact(ref_data, template, candidate, tmin, artifact_length):
                    missing_triggers.append(candidate)

        search_pos = triggers[0] - artifact_length
        if search_pos > 0:
            candidate = self._align_to_template(ref_data, template, search_pos, search_window, tmin, tmax)
            if self._is_artifact(ref_data, template, candidate, tmin, artifact_length):
                missing_triggers.insert(0, candidate)

        search_pos = triggers[-1] + artifact_length
        if search_pos + tmax < len(ref_data):
            candidate = self._align_to_template(ref_data, template, search_pos, search_window, tmin, tmax)
            if self._is_artifact(ref_data, template, candidate, tmin, artifact_length):
                missing_triggers.append(candidate)

        return missing_triggers

    def _build_context_with_missing(
        self,
        context: ProcessingContext,
        raw: mne.io.Raw,
        triggers: np.ndarray,
        missing_triggers: list,
        sfreq: float,
    ) -> ProcessingContext:
        """Merge missing triggers into the context and add annotations.

        Parameters
        ----------
        context : ProcessingContext
            Current processing context.
        raw : mne.io.Raw
            Current raw object.
        triggers : np.ndarray
            Original trigger positions.
        missing_triggers : list of int
            Newly detected missing trigger positions.
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        ProcessingContext
            Updated context with merged triggers and annotations.
        """
        all_triggers = np.sort(np.concatenate([triggers, missing_triggers]))
        new_metadata = context.metadata.copy()
        new_metadata.triggers = all_triggers

        raw_copy = raw.copy()
        existing_annot = raw.annotations
        new_annot = mne.Annotations(
            onset=np.array(missing_triggers) / sfreq,
            duration=np.zeros(len(missing_triggers)),
            description=["missing_trigger"] * len(missing_triggers),
        )
        combined = mne.Annotations(
            onset=np.concatenate([existing_annot.onset, new_annot.onset]),
            duration=np.concatenate([existing_annot.duration, new_annot.duration]),
            description=list(existing_annot.description) + list(new_annot.description),
        )
        raw_copy.set_annotations(combined)
        return context.with_raw(raw_copy).with_metadata(new_metadata)

    def _align_to_template(
        self,
        data: np.ndarray,
        template: np.ndarray,
        position: int,
        search_window: int,
        tmin: int,
        tmax: int,
    ) -> int:
        """Find the best-matching position near ``position`` via cross-correlation.

        Parameters
        ----------
        data : np.ndarray
            1-D signal array.
        template : np.ndarray
            Reference artifact template.
        position : int
            Initial candidate trigger position.
        search_window : int
            Search radius in samples.
        tmin : int
            Artifact onset offset from trigger.
        tmax : int
            Artifact end offset from trigger.

        Returns
        -------
        int
            Refined trigger position.
        """
        segment = data[position + tmin : position + tmax + search_window]
        corr = crosscorrelation(segment, template, search_window)
        shift = int(np.argmax(corr)) - search_window
        return position + shift

    def _is_artifact(
        self,
        data: np.ndarray,
        template: np.ndarray,
        position: int,
        tmin: int,
        artifact_length: int,
    ) -> bool:
        """Return True if the segment at ``position`` correlates with the template.

        Parameters
        ----------
        data : np.ndarray
            1-D signal array.
        template : np.ndarray
            Reference artifact template.
        position : int
            Candidate trigger position.
        tmin : int
            Artifact onset offset from trigger.
        artifact_length : int
            Length of artifact in samples.

        Returns
        -------
        bool
            ``True`` if the Pearson correlation exceeds
            ``self.correlation_threshold``.
        """
        start = position + tmin
        end = start + min(artifact_length, len(template))

        if end > len(data):
            return False

        segment = data[start:end]
        template_segment = template[: len(segment)]

        if len(template_segment) < 3:
            return False

        try:
            corr = float(np.abs(np.corrcoef(segment, template_segment)[0, 1]))
            return corr > self.correlation_threshold
        except ValueError:
            return False


@register_processor
class MissingTriggerCompleter(Processor):
    """Deterministically complete missing triggers using volume/slice counts.

    MATLAB FACET parity processor for ``FindMissingTriggers(volumes, slices)``
    based on the ``CompleteTriggers`` reconstruction heuristic.

    Parameters
    ----------
    volumes : int
        Total number of fMRI volumes expected.
    slices : int
        Number of slices per volume.
    add_to_context : bool, optional
        If ``True``, replace ``metadata.triggers`` with the completed list
        (default: True).
    add_annotations : bool, optional
        If ``True`` and raw is available, add annotations for inserted
        triggers (default: True).
    strict : bool, optional
        If ``True``, mismatches raise ``ProcessorValidationError``. If
        ``False``, best-effort completion is used (default: True).
    """

    name = "missing_trigger_completer"
    description = "Deterministically complete missing triggers from volume/slice counts"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = False
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        volumes: int,
        slices: int,
        add_to_context: bool = True,
        add_annotations: bool = True,
        strict: bool = True,
    ) -> None:
        self.volumes = int(volumes)
        self.slices = int(slices)
        self.add_to_context = add_to_context
        self.add_annotations = add_annotations
        self.strict = strict
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if self.volumes < 1:
            raise ProcessorValidationError(f"volumes must be >= 1, got {self.volumes}")
        if self.slices < 1:
            raise ProcessorValidationError(f"slices must be >= 1, got {self.slices}")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        triggers = np.sort(np.asarray(context.get_triggers(), dtype=int))
        expected = int(self.volumes * self.slices)

        # --- LOG ---
        logger.info(
            "Completing triggers deterministically (expected={}, current={})",
            expected,
            len(triggers),
        )

        # --- COMPUTE ---
        completed = self._complete_triggers(triggers, expected, self.slices)
        inserted = np.setdiff1d(completed, triggers)
        logger.info("Inserted {} trigger(s)", len(inserted))

        # --- BUILD RESULT ---
        metadata = context.metadata.copy()
        metadata.custom["missing_trigger_completer"] = {
            "volumes": self.volumes,
            "slices": self.slices,
            "expected_triggers": expected,
            "original_triggers": int(len(triggers)),
            "completed_triggers": int(len(completed)),
            "inserted_triggers": inserted.tolist(),
        }

        if not self.add_to_context:
            return context.with_metadata(metadata)

        metadata.triggers = completed
        result = context.with_metadata(metadata)

        if self.add_annotations and context.has_raw() and len(inserted) > 0:
            raw = context.get_raw().copy()
            sfreq = raw.info["sfreq"]
            extra = mne.Annotations(
                onset=np.asarray(inserted, dtype=float) / sfreq,
                duration=np.zeros(len(inserted)),
                description=["completed_trigger"] * len(inserted),
            )
            existing = raw.annotations
            merged = mne.Annotations(
                onset=np.concatenate([existing.onset, extra.onset]),
                duration=np.concatenate([existing.duration, extra.duration]),
                description=list(existing.description) + list(extra.description),
            )
            raw.set_annotations(merged)
            return result.with_raw(raw)

        # --- RETURN ---
        return result

    def _complete_triggers(self, triggers: np.ndarray, expected: int, slices: int) -> np.ndarray:
        """Reconstruct missing triggers with MATLAB-style heuristics."""
        if len(triggers) == expected:
            return triggers
        if len(triggers) == 0:
            raise ProcessorValidationError("No triggers available for completion.")

        err_num = expected - len(triggers)
        if err_num < 0:
            raise ProcessorValidationError(f"Found more triggers ({len(triggers)}) than expected ({expected}).")

        diffs = np.diff(triggers)
        if len(diffs) == 0:
            raise ProcessorValidationError("Not enough triggers to infer missing positions.")

        trigs_per_section = 5 * slices
        sections = len(diffs) // trigs_per_section

        if sections > 0:
            diff_mat = diffs[: sections * trigs_per_section].reshape(sections, trigs_per_section)
            diff_max = int(np.min(np.max(diff_mat, axis=1)))
            diff_min = int(np.round(np.mean(np.min(diff_mat, axis=1))))
            diff_err = int(np.max(np.max(diff_mat, axis=1)))
        else:
            diff_min = int(np.round(np.median(diffs)))
            diff_max = int(np.max(diffs))
            diff_err = diff_max

        err_threshold = float(np.mean([diff_min, diff_err]))
        err_indices = np.where(diffs > err_threshold)[0]

        if len(err_indices) != err_num:
            if self.strict:
                raise ProcessorValidationError(
                    f"Cannot determine missing triggers: expected {err_num}, detected {len(err_indices)} gaps."
                )
            ranked = np.argsort(diffs)[::-1]
            err_indices = np.sort(ranked[:err_num])

        volume_trigs = np.where((diffs >= (diff_max - 2)) & (diffs <= (diff_max + 2)))[0]
        reconstructed = triggers.tolist()

        for err_idx in err_indices:
            post_volume_candidates = np.where((volume_trigs - err_idx) > 0)[0]
            if len(post_volume_candidates) == 0 or post_volume_candidates[0] == 0:
                interval = diff_min
            else:
                post_volume_index = int(post_volume_candidates[0])
                if (volume_trigs[post_volume_index] - volume_trigs[post_volume_index - 1]) > slices:
                    interval = diff_max
                else:
                    interval = diff_min
            reconstructed.append(int(triggers[err_idx] + interval))

        all_trigs = np.array(sorted(set(reconstructed)), dtype=int)

        # Best-effort fallback for ambiguous cases in non-strict mode.
        if not self.strict and len(all_trigs) < expected:
            while len(all_trigs) < expected:
                d = np.diff(all_trigs)
                idx = int(np.argmax(d))
                step = diff_max if d[idx] > (1.5 * diff_min) else diff_min
                candidate = int(all_trigs[idx] + step)
                all_trigs = np.array(sorted(set([*all_trigs.tolist(), candidate])), dtype=int)

        if len(all_trigs) != expected and self.strict:
            raise ProcessorValidationError(
                f"Trigger completion produced {len(all_trigs)} triggers, expected {expected}."
            )

        return all_trigs


@register_processor
class SliceTriggerGenerator(Processor):
    """Generate slice triggers from volume triggers.

    MATLAB FACET parity processor for ``GenerateSliceTriggers``.

    Parameters
    ----------
    slices : int
        Number of slices per volume.
    duration_samples : float, optional
        Slice duration in samples. If ``None``, derived from trigger spacing.
    relative_position : float
        Relative position of first slice trigger with respect to the volume
        trigger (default: 0.03).
    add_annotations : bool, optional
        If ``True`` and raw is available, generated triggers are added as
        annotations (default: False).
    """

    name = "slice_trigger_generator"
    description = "Generate slice triggers from volume triggers"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = False
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        slices: int,
        duration_samples: float | None = None,
        relative_position: float = 0.03,
        add_annotations: bool = False,
    ) -> None:
        self.slices = int(slices)
        self.duration_samples = duration_samples
        self.relative_position = relative_position
        self.add_annotations = add_annotations
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if self.slices < 1:
            raise ProcessorValidationError(f"slices must be >= 1, got {self.slices}")
        if self.duration_samples is not None and self.duration_samples <= 0:
            raise ProcessorValidationError(f"duration_samples must be positive when set, got {self.duration_samples}")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        volume_triggers = np.asarray(context.get_triggers(), dtype=float)
        if len(volume_triggers) == 0:
            return context

        # --- COMPUTE ---
        duration = self._resolve_slice_duration(context, volume_triggers)
        slice_offsets = np.round((np.arange(self.slices) - self.relative_position) * duration).astype(int)

        generated = np.zeros(len(volume_triggers) * self.slices, dtype=int)
        for i, trigger in enumerate(volume_triggers.astype(int)):
            generated[i * self.slices : (i + 1) * self.slices] = trigger + slice_offsets

        metadata = context.metadata.copy()
        metadata.triggers = generated
        metadata.slices_per_volume = self.slices

        if len(generated) > 1:
            diffs = np.diff(generated)
            if np.ptp(diffs) > 3:
                mean_val = np.mean([np.median(diffs), np.max(diffs)])
                slice_diffs = diffs[diffs < mean_val]
                metadata.artifact_length = int(np.max(slice_diffs)) if len(slice_diffs) > 0 else int(np.max(diffs))
                metadata.volume_gaps = True
            else:
                metadata.artifact_length = int(np.max(diffs))
                metadata.volume_gaps = False

        metadata.custom["slice_trigger_generator"] = {
            "slices": self.slices,
            "duration_samples": float(duration),
            "relative_position": float(self.relative_position),
            "num_generated_triggers": int(len(generated)),
        }

        result = context.with_metadata(metadata)

        if self.add_annotations and context.has_raw():
            raw = context.get_raw().copy()
            sfreq = raw.info["sfreq"]
            raw.set_annotations(
                mne.Annotations(
                    onset=generated.astype(float) / sfreq,
                    duration=np.zeros(len(generated)),
                    description=["generated_slice_trigger"] * len(generated),
                )
            )
            return result.with_raw(raw)

        return result

    def _resolve_slice_duration(self, context: ProcessingContext, volume_triggers: np.ndarray) -> float:
        """Resolve slice period in samples."""
        if self.duration_samples is not None:
            return float(self.duration_samples)

        if context.metadata.slices_per_volume and context.metadata.artifact_length:
            return float(context.metadata.artifact_length)

        if len(volume_triggers) > 1:
            vol_period = float(np.median(np.diff(volume_triggers)))
            return max(vol_period / float(self.slices), 1.0)

        if context.metadata.artifact_length:
            return float(context.metadata.artifact_length)

        raise ProcessorValidationError("Cannot infer slice duration; provide duration_samples explicitly.")
