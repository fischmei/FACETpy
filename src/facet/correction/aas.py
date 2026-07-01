"""
Averaged Artifact Subtraction (AAS) correction processor.
"""

import random

import mne
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from ..console import processor_progress
from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor
from ..helpers.crosscorr import crosscorrelation
from ..helpers.utils import split_vector

FULL_MATRIX_MAX_CELLS = 4096
SPARSE_MATRIX_MAX_NONZERO = 200_000
SPARSE_MATRIX_PREVIEW_ROWS = 50
ARTIFACT_TEMPLATE_PREVIEW_ROWS = 3
ARTIFACT_TEMPLATE_PREVIEW_COLUMNS = 12


@register_processor
class AASCorrection(Processor):
    """Remove fMRI gradient artifacts using Averaged Artifact Subtraction.

    For each trigger epoch, finds highly correlated epochs within a sliding
    window and computes a weighted average. The averaged template is then
    subtracted from the original epoch. The algorithm adapts to non-stationary
    artifacts through correlation-based epoch selection.

    References
    ----------
    Allen et al., 2000. "A method for removing imaging artifact from continuous
    EEG recorded during functional MRI." NeuroImage, 12(2), 230-239.

    Parameters
    ----------
    window_size : int
        Number of epochs to consider in the sliding window (default: 30).
    rel_window_position : float
        Relative position of the window center between -1 and 1, where 0 is
        centered on the current epoch (default: 0.0).
    correlation_threshold : float
        Minimum Pearson r required to include an epoch in the average
        (default: 0.975).
    plot_artifacts : bool
        If True, plots a randomly selected averaged artifact after computation
        (default: False).
    realign_after_averaging : bool
        If True, realigns trigger positions to the averaged artifact templates
        using cross-correlation (default: True).
    search_window_factor : float
        Multiplier of the upsampling factor used as the cross-correlation
        search window (default: 3.0).
    interpolate_volume_gaps : bool
        If ``True``, linearly interpolate estimated artifact/noise values in
        gaps between consecutive artifact windows (default: False).
    apply_epoch_alpha_scaling : bool
        If ``True``, scale each epoch template by a least-squares ``alpha``
        factor before subtraction, similar to MATLAB FACET ``CalcAvgArt``
        (default: False).
    """

    name = "aas_correction"
    description = "Averaged Artifact Subtraction for fMRI artifacts"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = True
    parallel_safe = True
    channel_wise = True

    def __init__(
        self,
        window_size: int = 30,
        rel_window_position: float = 0.0,
        correlation_threshold: float = 0.975,
        plot_artifacts: bool = False,
        realign_after_averaging: bool = True,
        search_window_factor: float = 3.0,
        interpolate_volume_gaps: bool = False,
        apply_epoch_alpha_scaling: bool = False,
    ) -> None:
        self.window_size = window_size
        self.rel_window_position = rel_window_position
        self.correlation_threshold = correlation_threshold
        self.plot_artifacts = plot_artifacts
        self.realign_after_averaging = realign_after_averaging
        self.search_window_factor = search_window_factor
        self.interpolate_volume_gaps = interpolate_volume_gaps
        self.apply_epoch_alpha_scaling = apply_epoch_alpha_scaling
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if self.window_size < 1:
            raise ProcessorValidationError(f"window_size must be >= 1, got {self.window_size}")
        if not (0 < self.correlation_threshold <= 1):
            raise ProcessorValidationError(f"correlation_threshold must be in (0, 1], got {self.correlation_threshold}")
        if not (-1.0 <= self.rel_window_position <= 1.0):
            raise ProcessorValidationError(f"rel_window_position must be in [-1, 1], got {self.rel_window_position}")
        if self.search_window_factor <= 0:
            raise ProcessorValidationError(f"search_window_factor must be positive, got {self.search_window_factor}")
        if context.get_artifact_length() is None:
            raise ProcessorValidationError("Artifact length not set. Run TriggerDetector first.")
        n_triggers = len(context.get_triggers())
        if n_triggers < self.window_size:
            logger.warning(
                "Number of triggers ({}) is less than window size ({}). Using smaller window.",
                n_triggers,
                self.window_size,
            )
        eeg_channels = mne.pick_types(
            context.get_raw().info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        if len(eeg_channels) == 0:
            raise ProcessorValidationError("No EEG channels found in raw data.")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        sfreq = context.get_sfreq()
        upsampling_factor = context.metadata.upsampling_factor
        artifact_offset = context.metadata.artifact_to_trigger_offset
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=True, exclude="bads")

        # --- LOG ---
        logger.info(
            "Applying AAS correction: {} channels, {} triggers, window={}",
            len(eeg_channels),
            len(triggers),
            self.window_size,
        )

        # --- COMPUTE ---
        averaging_matrices = self._compute_averaging_matrices(
            raw, eeg_channels, raw.ch_names, triggers, artifact_length, artifact_offset, sfreq
        )
        artifacts_per_channel = self._calc_averaged_artifacts(
            raw, averaging_matrices, triggers, artifact_length, artifact_offset, sfreq
        )

        if self.plot_artifacts and artifacts_per_channel:
            self._plot_artifact_debug(raw, averaging_matrices, artifacts_per_channel)

        aligned_triggers = self._get_aligned_triggers(
            raw,
            averaging_matrices,
            artifacts_per_channel,
            triggers,
            artifact_offset,
            artifact_length,
            sfreq,
            upsampling_factor,
        )
        artifact_offset_samples = int(artifact_offset * sfreq)
        estimated_artifacts = self._remove_artifacts(
            raw,
            averaging_matrices,
            artifacts_per_channel,
            aligned_triggers,
            artifact_offset_samples,
            artifact_length,
        )
        if self.interpolate_volume_gaps:
            self._interpolate_volume_gap_artifacts(
                raw=raw,
                estimated_artifacts=estimated_artifacts,
                aligned_triggers=aligned_triggers,
                artifact_offset_samples=artifact_offset_samples,
                artifact_length=artifact_length,
                channel_indices=list(averaging_matrices.keys()),
            )

        # --- NOISE ---
        new_ctx = context.with_raw(raw)
        new_ctx.accumulate_noise(estimated_artifacts)

        # --- BUILD RESULT ---
        if self.realign_after_averaging and not np.array_equal(aligned_triggers, triggers):
            logger.debug("Triggers realigned after AAS averaging")
            new_ctx = new_ctx.with_triggers(aligned_triggers)

        new_ctx = self._with_artifact_template_report(
            context=new_ctx,
            raw=raw,
            averaging_matrices=averaging_matrices,
            artifacts_per_channel=artifacts_per_channel,
            triggers=triggers,
            aligned_triggers=aligned_triggers,
            artifact_length=artifact_length,
            artifact_offset=artifact_offset,
            sfreq=sfreq,
        )

        # --- RETURN ---
        logger.info("AAS correction complete: {} artifacts, {} channels", len(triggers), len(eeg_channels))
        return new_ctx

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    def _with_artifact_template_report(
        self,
        context: ProcessingContext,
        raw: mne.io.Raw,
        averaging_matrices: dict[int, np.ndarray],
        artifacts_per_channel: list[np.ndarray],
        triggers: np.ndarray,
        aligned_triggers: np.ndarray,
        artifact_length: int,
        artifact_offset: float,
        sfreq: float,
    ) -> ProcessingContext:
        """Attach a JSON-friendly report of the template matrix calculation."""
        metadata = context.metadata.copy()
        reports = metadata.custom.setdefault("artifact_template_matrices", [])
        reports.append(
            self._build_artifact_template_report(
                raw=raw,
                averaging_matrices=averaging_matrices,
                artifacts_per_channel=artifacts_per_channel,
                triggers=triggers,
                aligned_triggers=aligned_triggers,
                artifact_length=artifact_length,
                artifact_offset=artifact_offset,
                sfreq=sfreq,
            )
        )
        return context.with_metadata(metadata)

    def _build_artifact_template_report(
        self,
        raw: mne.io.Raw,
        averaging_matrices: dict[int, np.ndarray],
        artifacts_per_channel: list[np.ndarray],
        triggers: np.ndarray,
        aligned_triggers: np.ndarray,
        artifact_length: int,
        artifact_offset: float,
        sfreq: float,
    ) -> dict:
        """Build a compact representation of ``N = A @ D`` for this correction."""
        channels = []
        for channel_list_index, (ch_idx, matrix) in enumerate(averaging_matrices.items()):
            artifacts = artifacts_per_channel[channel_list_index]
            channels.append(
                {
                    "channel_index": int(ch_idx),
                    "channel_name": raw.ch_names[ch_idx],
                    "data_matrix_D": {
                        "shape": [int(matrix.shape[0]), int(artifact_length)],
                        "description": "Rows are artifact epochs; columns are samples within each epoch.",
                    },
                    "averaging_matrix_A": self._serialize_averaging_matrix(matrix),
                    "artifact_template_matrix_N": self._serialize_artifact_template_matrix(artifacts),
                }
            )

        return {
            "processor_name": self.name,
            "processor_type": self.__class__.__name__,
            "description": self.description,
            "matrix_equation": {
                "equation": "N = A @ D",
                "D": "Artifact epoch data matrix with shape (n_epochs, artifact_length_samples).",
                "A": "Averaging matrix; each row defines which epochs are averaged for one target epoch.",
                "N": "Estimated artifact template matrix subtracted from the EEG.",
            },
            "parameters": self._get_parameters(),
            "num_triggers": int(len(triggers)),
            "num_aligned_triggers": int(len(aligned_triggers)),
            "artifact_length_samples": int(artifact_length),
            "artifact_offset_seconds": float(artifact_offset),
            "sampling_rate_hz": float(sfreq),
            "channels": channels,
        }

    def _serialize_averaging_matrix(self, matrix: np.ndarray) -> dict:
        """Return a dense or sparse JSON-friendly representation of ``A``."""
        matrix = np.asarray(matrix, dtype=float)
        n_rows, n_cols = matrix.shape
        nonzero_per_row = np.count_nonzero(matrix, axis=1)
        row_sums = np.sum(matrix, axis=1) if n_rows else np.array([], dtype=float)
        total_cells = int(matrix.size)
        total_nonzero = int(np.count_nonzero(matrix))

        payload = {
            "shape": [int(n_rows), int(n_cols)],
            "nonzero_weights": total_nonzero,
            "density": float(total_nonzero / total_cells) if total_cells else 0.0,
            "row_nonzero_count": {
                "min": int(np.min(nonzero_per_row)) if nonzero_per_row.size else 0,
                "mean": float(np.mean(nonzero_per_row)) if nonzero_per_row.size else 0.0,
                "max": int(np.max(nonzero_per_row)) if nonzero_per_row.size else 0,
            },
            "row_sum": {
                "min": float(np.min(row_sums)) if row_sums.size else 0.0,
                "mean": float(np.mean(row_sums)) if row_sums.size else 0.0,
                "max": float(np.max(row_sums)) if row_sums.size else 0.0,
            },
        }

        if total_cells <= FULL_MATRIX_MAX_CELLS:
            payload.update(
                {
                    "storage": "dense",
                    "truncated": False,
                    "matrix": matrix.tolist(),
                }
            )
            return payload

        include_all_sparse = total_nonzero <= SPARSE_MATRIX_MAX_NONZERO
        max_rows = n_rows if include_all_sparse else min(n_rows, SPARSE_MATRIX_PREVIEW_ROWS)
        payload.update(
            {
                "storage": "sparse_rows",
                "truncated": not include_all_sparse,
                "rows": self._sparse_rows(matrix, max_rows=max_rows),
            }
        )
        if not include_all_sparse:
            payload["note"] = (
                "Sparse row output was truncated to keep the JSON report bounded. "
                "Summary statistics above still describe the full matrix."
            )
        return payload

    @staticmethod
    def _sparse_rows(matrix: np.ndarray, max_rows: int) -> list[dict]:
        """Serialize non-zero entries row by row."""
        rows = []
        for row_idx in range(max_rows):
            columns = np.flatnonzero(matrix[row_idx])
            rows.append(
                {
                    "row": int(row_idx),
                    "columns": columns.astype(int).tolist(),
                    "weights": matrix[row_idx, columns].astype(float).tolist(),
                }
            )
        return rows

    @staticmethod
    def _serialize_artifact_template_matrix(artifacts: np.ndarray) -> dict:
        """Summarize ``N`` and include a small top-left preview."""
        artifacts = np.asarray(artifacts, dtype=float)
        preview_rows = min(int(artifacts.shape[0]), ARTIFACT_TEMPLATE_PREVIEW_ROWS) if artifacts.ndim == 2 else 0
        preview_cols = min(int(artifacts.shape[1]), ARTIFACT_TEMPLATE_PREVIEW_COLUMNS) if artifacts.ndim == 2 else 0
        preview = artifacts[:preview_rows, :preview_cols].tolist() if artifacts.ndim == 2 else []

        return {
            "shape": [int(dim) for dim in artifacts.shape],
            "preview_rows": preview_rows,
            "preview_columns": preview_cols,
            "preview": preview,
            "note": "Preview only; full artifact templates are represented in the corrected EEG/noise estimate.",
        }

    def _compute_averaging_matrices(
        self,
        raw: mne.io.Raw,
        eeg_channels: np.ndarray,
        ch_names: list[str],
        triggers: np.ndarray,
        artifact_length: int,
        artifact_offset: float,
        sfreq: float,
    ) -> dict[int, np.ndarray]:
        """Compute per-channel averaging matrices by slicing raw data directly.

        Epochs are extracted one channel at a time from ``raw._data`` so that
        peak memory stays at O(n_epochs × artifact_length) rather than
        O(n_channels × n_epochs × artifact_length).

        Parameters
        ----------
        raw : mne.io.Raw
            EEG data (already a copy; modified in-place by the caller).
        eeg_channels : np.ndarray
            Channel indices to process.
        ch_names : List[str]
            Full channel name list from raw (indexed by ch_idx).
        triggers : np.ndarray
            Trigger sample positions.
        artifact_length : int
            Length of each artifact in samples.
        artifact_offset : float
            Time offset of artifact relative to trigger, in seconds.
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        dict
            Mapping from channel index to averaging matrix (n_epochs, n_epochs).
        """
        logger.debug("Computing averaging matrices for {} channels", len(eeg_channels))
        averaging_matrices: dict[int, np.ndarray] = {}
        trigger_offset_samples = int(artifact_offset * sfreq)
        epoch_starts = triggers + trigger_offset_samples

        with processor_progress(
            total=len(eeg_channels) or None,
            message="Averaging matrices",
        ) as progress:
            for idx, ch_idx in enumerate(eeg_channels):
                ch_name = ch_names[ch_idx]
                channel_epochs = split_vector(raw._data[ch_idx], epoch_starts, artifact_length)
                avg_matrix = self._calc_averaging_matrix(
                    channel_epochs,
                    window_size=self.window_size,
                    rel_window_offset=self.rel_window_position,
                    correlation_threshold=self.correlation_threshold,
                )
                averaging_matrices[ch_idx] = avg_matrix
                progress.advance(
                    1,
                    message=f"{idx + 1}/{len(eeg_channels)} • {ch_name}",
                )

        return averaging_matrices

    def _get_aligned_triggers(
        self,
        raw: mne.io.Raw,
        averaging_matrices: dict[int, np.ndarray],
        artifacts_per_channel: list[np.ndarray],
        triggers: np.ndarray,
        artifact_offset: float,
        artifact_length: int,
        sfreq: float,
        upsampling_factor: int,
    ) -> np.ndarray:
        """Return realigned triggers or the original triggers unchanged.

        Parameters
        ----------
        raw : mne.io.Raw
            EEG data (used to read the first processed channel).
        averaging_matrices : dict
            Averaging matrices keyed by channel index.
        artifacts_per_channel : List[np.ndarray]
            Averaged artifact arrays, one per channel.
        triggers : np.ndarray
            Original trigger sample positions.
        artifact_offset : float
            Artifact-to-trigger offset in seconds.
        artifact_length : int
            Artifact length in samples.
        sfreq : float
            Sampling frequency in Hz.
        upsampling_factor : int
            Current upsampling factor (used to scale the search window).

        Returns
        -------
        np.ndarray
            Aligned or unchanged trigger positions.
        """
        if not self.realign_after_averaging:
            return triggers

        search_window = int(self.search_window_factor * upsampling_factor)
        first_ch_idx = list(averaging_matrices.keys())[0]
        # Direct _data access avoids a full array copy for this read-only use
        first_channel_data = raw._data[first_ch_idx]
        return self._align_triggers_to_artifacts(
            first_channel_data,
            artifacts_per_channel[0],
            triggers,
            int(artifact_offset * sfreq),
            artifact_length,
            search_window,
        )

    def _remove_artifacts(
        self,
        raw: mne.io.Raw,
        averaging_matrices: dict[int, np.ndarray],
        artifacts_per_channel: list[np.ndarray],
        aligned_triggers: np.ndarray,
        artifact_offset_samples: int,
        artifact_length: int,
    ) -> np.ndarray:
        """Subtract averaged artifacts from raw data in-place.

        Parameters
        ----------
        raw : mne.io.Raw
            EEG data modified in-place; must be a copy of the original.
        averaging_matrices : dict
            Averaging matrices keyed by channel index.
        artifacts_per_channel : List[np.ndarray]
            Averaged artifact arrays, one per channel.
        aligned_triggers : np.ndarray
            (Possibly realigned) trigger sample positions.
        artifact_offset_samples : int
            Artifact-to-trigger offset in samples.
        artifact_length : int
            Artifact length in samples.

        Returns
        -------
        np.ndarray
            Estimated artifact array, same shape as ``raw._data``.
        """
        smin = artifact_offset_samples
        smax = smin + artifact_length
        n_samples = raw._data.shape[1]
        # Direct _data access avoids a full array copy on large datasets
        estimated_artifacts = np.zeros(raw._data.shape)

        with processor_progress(
            total=len(averaging_matrices) or None,
            message="Removing artifacts",
        ) as progress:
            for ch_list_idx, ch_idx in enumerate(averaging_matrices.keys()):
                ch_name = raw.ch_names[ch_idx]
                artifacts = artifacts_per_channel[ch_list_idx]
                alpha_values = np.ones(len(aligned_triggers), dtype=float)
                # Keep a stable source signal for alpha estimation while raw is modified in-place.
                ch_data_zero_mean = None
                if self.apply_epoch_alpha_scaling:
                    ch_data_zero_mean = raw._data[ch_idx].copy() - np.mean(raw._data[ch_idx])

                for epoch_idx, trigger_pos in enumerate(aligned_triggers):
                    start = trigger_pos + smin
                    stop = min(trigger_pos + smax, n_samples)
                    if start < 0 or start >= n_samples:
                        continue
                    artifact_segment = artifacts[epoch_idx, : stop - start]
                    if self.apply_epoch_alpha_scaling:
                        data_segment = ch_data_zero_mean[start:stop]
                        denom = float(np.dot(artifact_segment, artifact_segment))
                        if denom > np.finfo(float).eps:
                            alpha = float(np.dot(data_segment, artifact_segment) / denom)
                            if np.isfinite(alpha):
                                alpha_values[epoch_idx] = alpha
                        artifact_segment = alpha_values[epoch_idx] * artifact_segment
                    raw._data[ch_idx, start:stop] -= artifact_segment
                    estimated_artifacts[ch_idx, start:stop] += artifact_segment

                if self.apply_epoch_alpha_scaling and alpha_values.size:
                    alpha_min = float(np.min(alpha_values))
                    alpha_mean = float(np.mean(alpha_values))
                    alpha_max = float(np.max(alpha_values))
                    if alpha_min < 0 or (alpha_mean > 0 and alpha_max > (2.0 * alpha_mean)):
                        logger.warning(
                            "[{}] AAS alpha scaling produced unusual values: min={:.3f}, mean={:.3f}, max={:.3f}",
                            ch_name,
                            alpha_min,
                            alpha_mean,
                            alpha_max,
                        )

                progress.advance(
                    1,
                    message=(f"{ch_name} cleaned ({ch_list_idx + 1}/{len(averaging_matrices)})"),
                )

        return estimated_artifacts

    def _interpolate_volume_gap_artifacts(
        self,
        raw: mne.io.Raw,
        estimated_artifacts: np.ndarray,
        aligned_triggers: np.ndarray,
        artifact_offset_samples: int,
        artifact_length: int,
        channel_indices: list[int],
    ) -> None:
        """Interpolate estimated artifacts in gaps between consecutive epochs.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw data modified in-place.
        estimated_artifacts : np.ndarray
            Estimated artifact signal, modified in-place.
        aligned_triggers : np.ndarray
            Trigger positions used for subtraction.
        artifact_offset_samples : int
            Artifact start offset relative to trigger.
        artifact_length : int
            Artifact length in samples.
        channel_indices : List[int]
            Processed channel indices.
        """
        if len(aligned_triggers) < 2 or artifact_length <= 0:
            return

        n_samples = raw._data.shape[1]
        smin = artifact_offset_samples
        smax = artifact_offset_samples + artifact_length - 1

        for i in range(1, len(aligned_triggers)):
            start_this = int(aligned_triggers[i] + smin)
            end_prev = int(aligned_triggers[i - 1] + smax)

            if start_this <= 0 or start_this >= n_samples:
                continue
            if end_prev < 0 or end_prev >= n_samples:
                continue

            gap_len = start_this - end_prev - 1
            if gap_len <= 0:
                continue

            for ch_idx in channel_indices:
                end_val = estimated_artifacts[ch_idx, end_prev]
                start_val = estimated_artifacts[ch_idx, start_this]
                diff = start_val - end_val
                gap = end_val + (np.arange(1, gap_len + 1) * (diff / (gap_len + 1)))

                gap_start = end_prev + 1
                gap_stop = start_this
                estimated_artifacts[ch_idx, gap_start:gap_stop] = gap
                raw._data[ch_idx, gap_start:gap_stop] -= gap

    def _plot_artifact_debug(
        self,
        raw: mne.io.Raw,
        averaging_matrices: dict[int, np.ndarray],
        artifacts_per_channel: list[np.ndarray],
    ) -> None:
        """Plot a randomly selected averaged artifact for visual debugging.

        Parameters
        ----------
        raw : mne.io.Raw
            EEG data (used for channel name lookup).
        averaging_matrices : dict
            Averaging matrices keyed by channel index.
        artifacts_per_channel : List[np.ndarray]
            Averaged artifact arrays, one per channel.
        """
        try:
            processed_channels = list(averaging_matrices.keys())
            random_ch_list_idx = random.randint(0, len(processed_channels) - 1)
            ch_idx = processed_channels[random_ch_list_idx]
            ch_name = raw.ch_names[ch_idx]
            artifacts = artifacts_per_channel[random_ch_list_idx]

            if len(artifacts) == 0:
                return

            random_epoch_idx = random.randint(0, len(artifacts) - 1)
            artifact_segment = artifacts[random_epoch_idx]
            logger.debug(
                "Plotting random artifact for channel {}, epoch {}",
                ch_name,
                random_epoch_idx,
            )
            plt.figure(figsize=(10, 4))
            plt.plot(artifact_segment)
            plt.title(f"Estimated Artifact: Channel {ch_name} (Epoch {random_epoch_idx})")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            logger.warning("Failed to plot random artifact: {}", exc)

    def _calc_averaging_matrix(
        self,
        epochs: np.ndarray,
        window_size: int,
        rel_window_offset: float,
        correlation_threshold: float,
    ) -> np.ndarray:
        """Calculate averaging matrix using correlation-based epoch selection.

        For each epoch, finds highly correlated epochs within a sliding window
        and creates a weighted average.

        Parameters
        ----------
        epochs : np.ndarray
            Epochs array of shape (n_epochs, n_times).
        window_size : int
            Size of the sliding window.
        rel_window_offset : float
            Relative offset of the window center.
        correlation_threshold : float
            Minimum Pearson r for an epoch to be included.

        Returns
        -------
        np.ndarray
            Averaging matrix of shape (n_epochs, n_epochs) where each row sums to 1.
        """
        n_epochs = len(epochs)
        averaging_matrix = np.zeros((n_epochs, n_epochs))
        window_offset = int(window_size * rel_window_offset)

        for idx in range(0, n_epochs, window_size):
            offset_idx = idx + window_offset
            reference_indices = np.arange(idx, min(idx + 5, n_epochs))
            candidates = np.arange(offset_idx, min(offset_idx + window_size, n_epochs))
            candidates = candidates[candidates >= 0]

            chosen = self._find_correlated_epochs(epochs, candidates, reference_indices, correlation_threshold)
            if len(chosen) == 0:
                chosen = reference_indices

            target_indices = np.arange(idx, min(idx + window_size, n_epochs))
            weight = 1.0 / len(chosen)
            averaging_matrix[np.ix_(target_indices, chosen)] = weight

        return averaging_matrix

    def _find_correlated_epochs(
        self,
        all_epochs: np.ndarray,
        candidate_indices: np.ndarray,
        reference_indices: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Find epochs that are highly correlated with the running average.

        Iteratively adds epochs whose Pearson r with the running average
        exceeds *threshold*.

        Parameters
        ----------
        all_epochs : np.ndarray
            All epochs of shape (n_epochs, n_times).
        candidate_indices : np.ndarray
            Indices of candidate epochs to check.
        reference_indices : np.ndarray
            Indices of seed epochs used to initialise the running average.
        threshold : float
            Minimum correlation to accept a candidate.

        Returns
        -------
        np.ndarray
            Indices of accepted epochs.
        """
        if len(reference_indices) == 0:
            return np.array([])

        sum_data = np.sum(all_epochs[reference_indices], axis=0)
        chosen = list(reference_indices)

        for idx in candidate_indices:
            if idx in chosen:
                continue
            avg_data = sum_data / len(chosen)
            corr = np.corrcoef(avg_data.squeeze(), all_epochs[idx].squeeze())[0, 1]
            if corr > threshold:
                sum_data += all_epochs[idx]
                chosen.append(idx)

        return np.array(chosen)

    def _calc_averaged_artifacts(
        self,
        raw: mne.io.Raw,
        averaging_matrices: dict[int, np.ndarray],
        triggers: np.ndarray,
        artifact_length: int,
        artifact_offset: float,
        sfreq: float,
    ) -> list[np.ndarray]:
        """Calculate averaged artifact templates for each channel.

        Parameters
        ----------
        raw : mne.io.Raw
            EEG data.
        averaging_matrices : dict
            Averaging matrices keyed by channel index.
        triggers : np.ndarray
            Trigger sample positions.
        artifact_length : int
            Artifact length in samples.
        artifact_offset : float
            Artifact-to-trigger offset in seconds.
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        List[np.ndarray]
            Averaged artifact arrays per channel; each element has shape
            (n_epochs, n_times).
        """
        artifacts_per_channel = []
        trigger_offset_samples = int(artifact_offset * sfreq)

        for ch_idx, avg_matrix in averaging_matrices.items():
            # Direct _data access avoids a full array copy on large datasets
            ch_data = raw._data[ch_idx]
            ch_data_zero_mean = ch_data - np.mean(ch_data)

            epoch_data = split_vector(
                ch_data_zero_mean,
                triggers + trigger_offset_samples,
                artifact_length,
            )

            while len(epoch_data) > len(avg_matrix):
                epoch_data = epoch_data[:-1]

            averaged_artifacts = np.dot(avg_matrix, epoch_data)

            if len(averaged_artifacts) < len(triggers):
                last_artifact = averaged_artifacts[-1].reshape(1, -1)
                padding_needed = len(triggers) - len(averaged_artifacts)
                padding = np.repeat(last_artifact, padding_needed, axis=0)
                averaged_artifacts = np.vstack([averaged_artifacts, padding])

            artifacts_per_channel.append(averaged_artifacts)

        return artifacts_per_channel

    def _align_triggers_to_artifacts(
        self,
        channel_data: np.ndarray,
        artifacts: np.ndarray,
        triggers: np.ndarray,
        smin: int,
        smax: int,
        search_window: int,
    ) -> np.ndarray:
        """Align triggers to averaged artifacts using cross-correlation.

        Parameters
        ----------
        channel_data : np.ndarray
            Single-channel time series.
        artifacts : np.ndarray
            Averaged artifacts of shape (n_epochs, n_times).
        triggers : np.ndarray
            Original trigger sample positions.
        smin : int
            Artifact start offset in samples relative to trigger.
        smax : int
            Artifact end offset in samples relative to trigger.
        search_window : int
            Half-width of the cross-correlation search window in samples.

        Returns
        -------
        np.ndarray
            Realigned trigger sample positions.
        """
        aligned_triggers = []
        for i, trigger in enumerate(triggers):
            start = trigger + smin
            stop = trigger + smax + search_window
            if stop > len(channel_data):
                aligned_triggers.append(trigger)
                continue
            segment = channel_data[start:stop]
            artifact = artifacts[i, :]
            corr = crosscorrelation(segment, artifact, search_window)
            best_shift = np.argmax(corr) - search_window
            aligned_triggers.append(trigger + best_shift)

        return np.array(aligned_triggers)


# Alias for backwards compatibility
AveragedArtifactSubtraction = AASCorrection
