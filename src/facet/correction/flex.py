"""Flexible template-matrix artifact correction processor."""

from __future__ import annotations

import random
from collections.abc import Iterator
from numbers import Integral
from pathlib import Path
from typing import Literal

import mne
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from ..console import processor_progress
from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor
from ..helpers.crosscorr import crosscorrelation
from ..helpers.utils import split_vector
from ..misc.plot_aas_matricies import plot_aas_matrices

DistributionName = Literal["equal", "normal"]

FULL_MATRIX_MAX_CELLS = 4096
SPARSE_MATRIX_MAX_NONZERO = 200_000
SPARSE_MATRIX_PREVIEW_ROWS = 50
ARTIFACT_TEMPLATE_PREVIEW_ROWS = 3
ARTIFACT_TEMPLATE_PREVIEW_COLUMNS = 12


@register_processor
class Flex(Processor):
    """Remove trigger-locked artifacts with configurable epoch averaging.

    Flex builds one artifact template for every target epoch. It searches the
    following ``window_size`` epochs first and, near the end of a recording,
    backfills the candidate window with the nearest preceding epochs. The
    target epoch is never allowed to contribute to its own template.

    Every candidate whose signed Pearson correlation meets ``threshold`` is
    accepted. If this produces fewer than ``min_accepted`` epochs, the
    strongest remaining correlations are added even though they are below the
    threshold. Selected epochs are then averaged either equally or with a
    Gaussian kernel that favors epochs temporally closer to the target.

    Flex owns the common template-correction lifecycle used by every
    ``N = A @ D`` correction in FACETpy. For each channel it extracts an epoch
    matrix ``D``, builds an averaging matrix ``A``, calculates artifact
    templates ``N``, subtracts each row of ``N`` from its target epoch, and
    records the subtraction as estimated noise. Legacy AAS, FARM, and
    structural weighting processors subclass Flex and replace only the rule
    that constructs ``A``.

    Parameters
    ----------
    window_size : int
        Maximum number of non-target epochs considered for each target
        (default: 30).
    threshold : float
        Minimum signed Pearson correlation required for automatic acceptance,
        in the interval ``(0, 1]`` (default: 0.975).
    min_accepted : int
        Minimum number of candidates used when enough non-target epochs are
        available. Below-threshold candidates are added in descending
        correlation order when necessary (default: 5).
    N_distribution : {"equal", "normal"}
        Weight distribution across accepted epochs. ``"equal"`` assigns the
        same weight to every accepted epoch. ``"normal"`` applies a Gaussian
        kernel to temporal distance from the target, with standard deviation
        ``max(effective_window_size / 3, 1)`` (default: ``"equal"``).
    plot_artifacts : bool
        If ``True``, plot a representative artifact template and its matrices
        (default: ``False``).
    realign_after_averaging : bool
        If ``True``, realign triggers to the computed templates before
        subtraction (default: ``True``).
    search_window_factor : float
        Multiplier of the upsampling factor used for trigger realignment
        (default: 3.0).
    interpolate_volume_gaps : bool
        If ``True``, interpolate the artifact estimate between consecutive
        artifact windows (default: ``False``).
    apply_epoch_alpha_scaling : bool
        If ``True``, least-squares scale each target template immediately
        before subtraction (default: ``False``).
    track_estimated_noise : bool
        If ``True``, retain the subtracted artifact as a full-length noise
        estimate for later ANC or inspection. Set to ``False`` when the
        corrected signal and metrics are the only outputs required
        (default: ``True``).
    """

    name = "flex_correction"
    description = "Flexible per-epoch correlation-based artifact subtraction"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = True
    # Flex operates on supported EEG/EOG picks within the full Raw object.
    # Generic channel splitting currently uses a broader, different channel
    # taxonomy and cannot preserve global realignment/noise/report semantics.
    # Keep both executor flags disabled; the internal D -> A -> N loop still
    # streams one supported channel at a time to bound matrix memory.
    parallel_safe = False
    channel_wise = False

    def __init__(
        self,
        window_size: int = 30,
        threshold: float = 0.975,
        min_accepted: int = 5,
        N_distribution: DistributionName = "equal",
        plot_artifacts: bool = False,
        realign_after_averaging: bool = True,
        search_window_factor: float = 3.0,
        interpolate_volume_gaps: bool = False,
        apply_epoch_alpha_scaling: bool = False,
        track_estimated_noise: bool = True,
    ) -> None:
        self.threshold = threshold
        self.min_accepted = min_accepted
        # Compatibility aliases are intentionally excluded from Flex's
        # constructor snapshot. Legacy strategies set their own values before
        # calling this initializer, so do not overwrite those public fields.
        if not hasattr(self, "correlation_threshold"):
            self.correlation_threshold = threshold
        if not hasattr(self, "rel_window_position"):
            self.rel_window_position = 0.0
        # Keep the requested public parameter name while storing a canonical
        # value so CLI/config inputs are case-insensitive and reproducible.
        self.N_distribution = str(N_distribution).strip().lower()

        self.window_size = window_size
        self.plot_artifacts = plot_artifacts
        self.realign_after_averaging = realign_after_averaging
        self.search_window_factor = search_window_factor
        self.interpolate_volume_gaps = interpolate_volume_gaps
        self.apply_epoch_alpha_scaling = apply_epoch_alpha_scaling
        self.track_estimated_noise = track_estimated_noise
        super().__init__()

    def _get_parameters(self) -> dict[str, object]:
        """Return constructor-compatible parameters for history and workers."""
        return {
            "window_size": self.window_size,
            "threshold": self.threshold,
            "min_accepted": self.min_accepted,
            "N_distribution": self.N_distribution,
            "plot_artifacts": self.plot_artifacts,
            "realign_after_averaging": self.realign_after_averaging,
            "search_window_factor": self.search_window_factor,
            "interpolate_volume_gaps": self.interpolate_volume_gaps,
            "apply_epoch_alpha_scaling": self.apply_epoch_alpha_scaling,
            "track_estimated_noise": self.track_estimated_noise,
        }

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)

        self._validate_common_template_configuration(context)
        self._validate_averaging_strategy(context)

    def _validate_common_template_configuration(self, context: ProcessingContext) -> None:
        """Validate requirements shared by every template-matrix strategy."""
        if isinstance(self.window_size, bool) or not isinstance(self.window_size, Integral):
            raise ProcessorValidationError(f"window_size must be an integer, got {self.window_size!r}")
        if self.window_size < 1:
            raise ProcessorValidationError(f"window_size must be >= 1, got {self.window_size}")
        if self.search_window_factor <= 0:
            raise ProcessorValidationError(f"search_window_factor must be positive, got {self.search_window_factor}")
        if not isinstance(self.track_estimated_noise, bool):
            raise ProcessorValidationError(
                f"track_estimated_noise must be a boolean, got {self.track_estimated_noise!r}"
            )
        if context.get_artifact_length() is None:
            raise ProcessorValidationError("Artifact length not set. Run TriggerDetector first.")

        n_triggers = len(context.get_triggers())
        if n_triggers < self.window_size:
            logger.warning(
                "Number of triggers ({}) is less than window size ({}). Using smaller window.",
                n_triggers,
                self.window_size,
            )

        channel_indices = self._pick_template_channels(context.get_raw())
        if len(channel_indices) == 0:
            raise ProcessorValidationError("No EEG or EOG channels found in raw data.")

    def _validate_averaging_strategy(self, context: ProcessingContext) -> None:
        """Validate the default flexible correlation-selection strategy.

        Legacy subclasses override this hook because their averaging matrices
        have different invariants, such as allowing self-weights or not using
        a minimum accepted-epoch count.
        """
        if not (0 < self.threshold <= 1):
            raise ProcessorValidationError(f"threshold must be in (0, 1], got {self.threshold}")
        if isinstance(self.min_accepted, bool) or not isinstance(self.min_accepted, Integral):
            raise ProcessorValidationError(f"min_accepted must be an integer, got {self.min_accepted!r}")
        if self.min_accepted < 1:
            raise ProcessorValidationError(f"min_accepted must be >= 1, got {self.min_accepted}")
        if self.min_accepted > self.window_size:
            raise ProcessorValidationError(
                "min_accepted cannot exceed window_size, "
                f"got min_accepted={self.min_accepted}, window_size={self.window_size}"
            )
        if self.N_distribution not in {"equal", "normal"}:
            raise ProcessorValidationError(
                f"N_distribution must be either 'equal' or 'normal', got {self.N_distribution!r}"
            )

        n_triggers = len(context.get_triggers())
        if n_triggers < 2:
            raise ProcessorValidationError(
                "Flex correction requires at least two triggers because the target epoch cannot average itself."
            )

        available_candidates = n_triggers - 1
        if available_candidates < self.min_accepted:
            logger.warning(
                "Only {} non-target epochs are available; Flex cannot reach min_accepted={} and will use all of them.",
                available_candidates,
                self.min_accepted,
            )

    def _matrix_rel_window_offset(self) -> float:
        """Return the shared matrix-hook offset for the Flex strategy."""
        return 0.0

    def _matrix_correlation_threshold(self) -> float:
        """Return the threshold passed to the averaging-matrix hook."""
        return float(self.threshold)

    @staticmethod
    def _pick_template_channels(raw: mne.io.Raw) -> np.ndarray:
        """Return good EEG and EOG channels supported by the Flex engine."""
        return mne.pick_types(
            raw.info,
            meg=False,
            eeg=True,
            stim=False,
            eog=True,
            exclude="bads",
        )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Build and subtract one artifact-template matrix per channel."""
        # --- EXTRACT ---
        raw = context.get_raw().copy()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        sfreq = context.get_sfreq()
        upsampling_factor = context.metadata.upsampling_factor
        artifact_offset = context.metadata.artifact_to_trigger_offset
        channel_indices = self._pick_template_channels(raw)

        # --- LOG ---
        logger.info(
            "Applying {}: {} channels, {} triggers, window={}",
            self.name,
            len(channel_indices),
            len(triggers),
            self.window_size,
        )

        # --- COMPUTE ---
        averaging_matrices, artifacts_per_channel = self._build_artifact_templates(
            context=context,
            raw=raw,
            channel_indices=channel_indices,
            triggers=triggers,
            artifact_length=artifact_length,
            artifact_offset=artifact_offset,
            sfreq=sfreq,
        )
        aligned_triggers, estimated_artifacts = self._subtract_artifact_templates(
            raw=raw,
            averaging_matrices=averaging_matrices,
            artifacts_per_channel=artifacts_per_channel,
            triggers=triggers,
            artifact_offset=artifact_offset,
            artifact_length=artifact_length,
            sfreq=sfreq,
            upsampling_factor=upsampling_factor,
        )

        # --- NOISE ---
        if self.track_estimated_noise:
            if estimated_artifacts is None:  # pragma: no cover - invariant
                raise RuntimeError("Tracked correction did not produce an artifact estimate")
            previous_noise = context.get_estimated_noise()
            if previous_noise is None:
                accumulated_noise = estimated_artifacts
            else:
                accumulated_noise = previous_noise.copy()
                accumulated_noise += estimated_artifacts
            new_context = context.with_raw(
                raw,
                estimated_noise=accumulated_noise,
                copy_estimated_noise=False,
            )
        else:
            # An incomplete estimate must never reach ANC accidentally.
            new_context = context.with_raw(
                raw,
                estimated_noise=None,
                copy_estimated_noise=False,
            )

        # --- BUILD RESULT ---
        if self.realign_after_averaging and not np.array_equal(aligned_triggers, triggers):
            logger.debug("Triggers realigned after {} template averaging", self.name)
            metadata = new_context.metadata.copy()
            metadata.triggers = aligned_triggers
            new_context = new_context.with_metadata(
                metadata,
                copy_estimated_noise=False,
            )

        new_context = self._with_artifact_template_report(
            context=new_context,
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
        logger.info("{} complete: {} artifacts, {} channels", self.name, len(triggers), len(channel_indices))
        return new_context

    def _build_artifact_templates(
        self,
        *,
        context: ProcessingContext,
        raw: mne.io.Raw,
        channel_indices: np.ndarray,
        triggers: np.ndarray,
        artifact_length: int,
        artifact_offset: float,
        sfreq: float,
    ) -> tuple[dict[int, np.ndarray], list[np.ndarray]]:
        """Build ``A`` and ``N`` while retaining at most one diagnostic ``D``.

        Each channel's epoch matrix is discarded as soon as its template
        matrix has been calculated. This keeps peak epoch-extraction memory
        proportional to one channel rather than to the number of channels.
        """
        averaging_matrices, artifacts_per_channel, diagnostic_epoch_matrices = self._compute_artifact_templates(
            raw,
            channel_indices,
            raw.ch_names,
            triggers,
            artifact_length,
            artifact_offset,
            sfreq,
        )

        self._maybe_save_artifact_matrix_plot(
            context=context,
            raw=raw,
            averaging_matrices=averaging_matrices,
            epoch_matrices_per_channel=diagnostic_epoch_matrices,
            sfreq=sfreq,
        )
        if self.plot_artifacts and artifacts_per_channel:
            self._plot_artifact_debug(raw, averaging_matrices, artifacts_per_channel)

        return averaging_matrices, artifacts_per_channel

    def _subtract_artifact_templates(
        self,
        *,
        raw: mne.io.Raw,
        averaging_matrices: dict[int, np.ndarray],
        artifacts_per_channel: list[np.ndarray],
        triggers: np.ndarray,
        artifact_offset: float,
        artifact_length: int,
        sfreq: float,
        upsampling_factor: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Realign triggers, subtract ``N``, and optionally fill epoch gaps."""
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
            track_estimated_noise=self.track_estimated_noise,
            interpolate_untracked=self.interpolate_volume_gaps,
        )
        if self.interpolate_volume_gaps and estimated_artifacts is not None:
            self._interpolate_volume_gap_artifacts(
                raw=raw,
                estimated_artifacts=estimated_artifacts,
                aligned_triggers=aligned_triggers,
                artifact_offset_samples=artifact_offset_samples,
                artifact_length=artifact_length,
                channel_indices=list(averaging_matrices),
            )

        return aligned_triggers, estimated_artifacts

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
        return context.with_metadata(
            metadata,
            copy_estimated_noise=False,
        )

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
        """Build a compact representation of ``N = A @ D``."""
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
        preview_columns = min(int(artifacts.shape[1]), ARTIFACT_TEMPLATE_PREVIEW_COLUMNS) if artifacts.ndim == 2 else 0
        preview = artifacts[:preview_rows, :preview_columns].tolist() if artifacts.ndim == 2 else []

        return {
            "shape": [int(dimension) for dimension in artifacts.shape],
            "preview_rows": preview_rows,
            "preview_columns": preview_columns,
            "preview": preview,
            "note": "Preview only; full artifact templates are represented in the corrected EEG/noise estimate.",
        }

    def _compute_artifact_templates(
        self,
        raw: mne.io.Raw,
        channel_indices: np.ndarray,
        channel_names: list[str],
        triggers: np.ndarray,
        artifact_length: int,
        artifact_offset: float,
        sfreq: float,
    ) -> tuple[dict[int, np.ndarray], list[np.ndarray], dict[int, np.ndarray]]:
        """Build channel matrices without retaining every extracted ``D``.

        The averaging matrices and final artifact templates are needed later
        for subtraction and reporting, so they remain in memory. Only the
        best diagnostic epoch matrix is retained for optional matrix plots.

        Returns
        -------
        tuple
            Averaging matrices ``A`` keyed by channel index, artifact-template
            matrices ``N`` in the same insertion order, and a dictionary
            containing at most one representative epoch matrix ``D``.
        """
        averaging_matrices: dict[int, np.ndarray] = {}
        artifacts_per_channel: list[np.ndarray] = []
        diagnostic_epoch_matrices: dict[int, np.ndarray] = {}
        diagnostic_score = float("-inf")

        for channel_index, averaging_matrix, channel_epochs in self._iter_channel_matrices(
            raw=raw,
            channel_indices=channel_indices,
            channel_names=channel_names,
            triggers=triggers,
            artifact_length=artifact_length,
            artifact_offset=artifact_offset,
            sfreq=sfreq,
        ):
            averaging_matrices[channel_index] = averaging_matrix
            artifacts_per_channel.append(self._calculate_channel_templates(averaging_matrix, channel_epochs))

            candidate_score = self._diagnostic_epoch_score(channel_epochs)
            if not diagnostic_epoch_matrices or candidate_score > diagnostic_score:
                diagnostic_epoch_matrices = {channel_index: channel_epochs}
                diagnostic_score = candidate_score

        return averaging_matrices, artifacts_per_channel, diagnostic_epoch_matrices

    def _compute_averaging_matrices(
        self,
        raw: mne.io.Raw,
        channel_indices: np.ndarray,
        channel_names: list[str],
        triggers: np.ndarray,
        artifact_length: int,
        artifact_offset: float,
        sfreq: float,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Return all extracted ``D`` and corresponding ``A`` matrices.

        This compatibility helper intentionally materializes every epoch
        matrix for callers that need both collections. Production correction
        uses :meth:`_compute_artifact_templates`, which discards each ``D`` as
        soon as ``N = A @ D`` has been calculated.
        """
        averaging_matrices: dict[int, np.ndarray] = {}
        epoch_matrices_per_channel: dict[int, np.ndarray] = {}
        for channel_index, averaging_matrix, channel_epochs in self._iter_channel_matrices(
            raw=raw,
            channel_indices=channel_indices,
            channel_names=channel_names,
            triggers=triggers,
            artifact_length=artifact_length,
            artifact_offset=artifact_offset,
            sfreq=sfreq,
        ):
            averaging_matrices[channel_index] = averaging_matrix
            epoch_matrices_per_channel[channel_index] = channel_epochs

        return averaging_matrices, epoch_matrices_per_channel

    def _iter_channel_matrices(
        self,
        *,
        raw: mne.io.Raw,
        channel_indices: np.ndarray,
        channel_names: list[str],
        triggers: np.ndarray,
        artifact_length: int,
        artifact_offset: float,
        sfreq: float,
    ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield ``(channel_index, A, D)`` for one channel at a time.

        Epochs are extracted one channel at a time from ``raw._data`` so peak
        memory remains proportional to one channel's epoch matrix instead of
        all channels at once. The source raw object is already a copy owned by
        :meth:`process`.

        Parameters
        ----------
        raw : mne.io.Raw
            Copied EEG data from which epochs are extracted.
        channel_indices : np.ndarray
            Channel indices processed by the correction.
        channel_names : list of str
            Full channel-name list indexed by channel number.
        triggers : np.ndarray
            Trigger positions in samples.
        artifact_length : int
            Artifact epoch length in samples.
        artifact_offset : float
            Artifact start offset from each trigger, in seconds.
        sfreq : float
            Sampling frequency in Hz.

        Yields
        -------
        tuple
            Channel index, averaging matrix ``A``, and epoch matrix ``D``.
        """
        logger.debug("Computing averaging matrices for {} channels", len(channel_indices))
        trigger_offset_samples = int(artifact_offset * sfreq)
        epoch_starts = triggers + trigger_offset_samples

        with processor_progress(
            total=len(channel_indices) or None,
            message="Averaging matrices",
        ) as progress:
            for index, channel_index in enumerate(channel_indices):
                channel_name = channel_names[channel_index]
                # Zero-meaning the complete channel preserves the historical
                # template/subtraction convention without another full copy.
                channel_data = raw._data[channel_index]
                zero_mean_channel = channel_data - np.mean(channel_data)
                channel_epochs = split_vector(zero_mean_channel, epoch_starts, artifact_length)
                self._require_finite_epoch_data(channel_epochs)
                averaging_matrix = self._calc_averaging_matrix(
                    channel_epochs,
                    window_size=self.window_size,
                    rel_window_offset=self._matrix_rel_window_offset(),
                    correlation_threshold=self._matrix_correlation_threshold(),
                )
                yield int(channel_index), averaging_matrix, channel_epochs
                progress.advance(
                    1,
                    message=f"{index + 1}/{len(channel_indices)} • {channel_name}",
                )

    def _require_finite_epoch_data(self, epochs: np.ndarray) -> None:
        """Reject non-finite samples before any matrix strategy sees ``D``."""
        finite_mask = np.isfinite(epochs)
        if np.all(finite_mask):
            return

        nonfinite_count = int(epochs.size - np.count_nonzero(finite_mask))
        raise ProcessorValidationError(
            f"{self.__class__.__name__} requires finite epoch data; found {nonfinite_count} non-finite sample(s)."
        )

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
        """Return realigned triggers or the original positions unchanged."""
        if not self.realign_after_averaging:
            return triggers

        search_window = int(self.search_window_factor * upsampling_factor)
        first_channel_index = next(iter(averaging_matrices))
        # Direct _data access avoids a full array copy for this read-only use.
        first_channel_data = raw._data[first_channel_index]
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
        *,
        track_estimated_noise: bool = True,
        interpolate_untracked: bool = False,
    ) -> np.ndarray | None:
        """Subtract artifact templates in-place from the copied raw data.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw data copied by :meth:`process` and modified in-place.
        averaging_matrices : dict
            Per-channel averaging matrices, used to retain stable channel
            ordering.
        artifacts_per_channel : list of np.ndarray
            Template matrix ``N`` for every processed channel.
        aligned_triggers : np.ndarray
            Trigger positions used for subtraction.
        artifact_offset_samples : int
            Artifact start offset from each trigger, in samples.
        artifact_length : int
            Artifact length in samples.

        Returns
        -------
        np.ndarray or None
            Full-length estimated artifact signal when tracking is enabled;
            otherwise ``None``.
        """
        start_offset = artifact_offset_samples
        stop_offset = start_offset + artifact_length
        n_samples = raw._data.shape[1]
        estimated_artifacts = np.zeros(raw._data.shape) if track_estimated_noise else None

        with processor_progress(
            total=len(averaging_matrices) or None,
            message="Removing artifacts",
        ) as progress:
            for channel_list_index, channel_index in enumerate(averaging_matrices):
                channel_name = raw.ch_names[channel_index]
                artifacts = artifacts_per_channel[channel_list_index]
                alpha_values = np.ones(len(aligned_triggers), dtype=float)
                channel_estimate = (
                    estimated_artifacts[channel_index]
                    if estimated_artifacts is not None
                    else (np.zeros(n_samples, dtype=raw._data.dtype) if interpolate_untracked else None)
                )

                # Alpha estimation must read a stable signal while the copied
                # raw channel is modified epoch by epoch.
                zero_mean_channel = None
                if self.apply_epoch_alpha_scaling:
                    zero_mean_channel = raw._data[channel_index].copy() - np.mean(raw._data[channel_index])

                for epoch_index, trigger_position in enumerate(aligned_triggers):
                    start = trigger_position + start_offset
                    stop = min(trigger_position + stop_offset, n_samples)
                    if start < 0 or start >= n_samples:
                        continue

                    artifact_segment = artifacts[epoch_index, : stop - start]
                    if self.apply_epoch_alpha_scaling:
                        data_segment = zero_mean_channel[start:stop]
                        denominator = float(np.dot(artifact_segment, artifact_segment))
                        if denominator > np.finfo(float).eps:
                            alpha = float(np.dot(data_segment, artifact_segment) / denominator)
                            if np.isfinite(alpha):
                                alpha_values[epoch_index] = alpha
                        artifact_segment = alpha_values[epoch_index] * artifact_segment

                    raw._data[channel_index, start:stop] -= artifact_segment
                    if channel_estimate is not None:
                        channel_estimate[start:stop] += artifact_segment

                if interpolate_untracked and estimated_artifacts is None:
                    self._interpolate_volume_gap_channel(
                        raw_channel=raw._data[channel_index],
                        estimated_channel=channel_estimate,
                        aligned_triggers=aligned_triggers,
                        artifact_offset_samples=artifact_offset_samples,
                        artifact_length=artifact_length,
                    )

                if self.apply_epoch_alpha_scaling and alpha_values.size:
                    self._warn_for_unusual_alpha_values(channel_name, alpha_values)

                progress.advance(
                    1,
                    message=(f"{channel_name} cleaned ({channel_list_index + 1}/{len(averaging_matrices)})"),
                )

        return estimated_artifacts

    @staticmethod
    def _warn_for_unusual_alpha_values(channel_name: str, alpha_values: np.ndarray) -> None:
        """Warn when per-epoch template scaling suggests an unstable fit."""
        alpha_minimum = float(np.min(alpha_values))
        alpha_mean = float(np.mean(alpha_values))
        alpha_maximum = float(np.max(alpha_values))
        has_large_maximum = alpha_mean > 0 and alpha_maximum > (2.0 * alpha_mean)
        if alpha_minimum < 0 or has_large_maximum:
            logger.warning(
                "[{}] template alpha scaling produced unusual values: min={:.3f}, mean={:.3f}, max={:.3f}",
                channel_name,
                alpha_minimum,
                alpha_mean,
                alpha_maximum,
            )

    def _interpolate_volume_gap_artifacts(
        self,
        raw: mne.io.Raw,
        estimated_artifacts: np.ndarray,
        aligned_triggers: np.ndarray,
        artifact_offset_samples: int,
        artifact_length: int,
        channel_indices: list[int],
    ) -> None:
        """Interpolate estimated artifacts between consecutive epochs."""
        for channel_index in channel_indices:
            self._interpolate_volume_gap_channel(
                raw_channel=raw._data[channel_index],
                estimated_channel=estimated_artifacts[channel_index],
                aligned_triggers=aligned_triggers,
                artifact_offset_samples=artifact_offset_samples,
                artifact_length=artifact_length,
            )

    @staticmethod
    def _interpolate_volume_gap_channel(
        *,
        raw_channel: np.ndarray,
        estimated_channel: np.ndarray,
        aligned_triggers: np.ndarray,
        artifact_offset_samples: int,
        artifact_length: int,
    ) -> None:
        """Interpolate one channel so untracked mode needs only one vector."""
        if len(aligned_triggers) < 2 or artifact_length <= 0:
            return

        n_samples = raw_channel.shape[0]
        start_offset = artifact_offset_samples
        stop_offset = artifact_offset_samples + artifact_length - 1

        for trigger_index in range(1, len(aligned_triggers)):
            current_start = int(aligned_triggers[trigger_index] + start_offset)
            previous_end = int(aligned_triggers[trigger_index - 1] + stop_offset)

            if current_start <= 0 or current_start >= n_samples:
                continue
            if previous_end < 0 or previous_end >= n_samples:
                continue

            gap_length = current_start - previous_end - 1
            if gap_length <= 0:
                continue

            end_value = estimated_channel[previous_end]
            start_value = estimated_channel[current_start]
            difference = start_value - end_value
            gap = end_value + (np.arange(1, gap_length + 1) * (difference / (gap_length + 1)))

            gap_start = previous_end + 1
            gap_stop = current_start
            estimated_channel[gap_start:gap_stop] = gap
            raw_channel[gap_start:gap_stop] -= gap

    def _maybe_save_artifact_matrix_plot(
        self,
        *,
        context: ProcessingContext,
        raw: mne.io.Raw,
        averaging_matrices: dict[int, np.ndarray],
        epoch_matrices_per_channel: dict[int, np.ndarray],
        sfreq: float,
    ) -> None:
        """Save a diagnostic plot using a representative EEG channel.

        The historical ``.aas_matrices.png`` suffix is retained for existing
        pipelines even though Flex now owns the common matrix engine.
        """
        if not epoch_matrices_per_channel:
            return

        output_path = None
        chunk_metadata = context.metadata.custom.get("chunk", {})
        if isinstance(chunk_metadata, dict):
            raw_output_path = chunk_metadata.get("output_path")
            if raw_output_path:
                output_path = Path(str(raw_output_path)).with_suffix(".aas_matrices.png")

        if output_path is None and not self.plot_artifacts:
            return

        try:
            processed_channels = list(epoch_matrices_per_channel)
            if not processed_channels:
                return

            channel_index = self._select_diagnostic_channel(
                processed_channels,
                epoch_matrices_per_channel,
            )
            epoch_data = np.asarray(epoch_matrices_per_channel[channel_index], dtype=float)

            if epoch_data.ndim != 2 or epoch_data.size == 0:
                logger.warning("Cannot create template-matrix diagnostic plot: selected epoch matrix is empty.")
                return

            data_minimum = float(np.nanmin(epoch_data))
            data_maximum = float(np.nanmax(epoch_data))
            data_standard_deviation = float(np.nanstd(epoch_data))
            nonzero_count = int(np.count_nonzero(epoch_data))

            logger.info(
                "Template diagnostic channel '{}': shape={}, min={:.6e}, max={:.6e}, std={:.6e}, nonzero={}/{}",
                raw.ch_names[channel_index],
                epoch_data.shape,
                data_minimum,
                data_maximum,
                data_standard_deviation,
                nonzero_count,
                epoch_data.size,
            )

            if not np.isfinite(data_standard_deviation) or data_standard_deviation <= 1e-15:
                logger.warning(
                    "Cannot create meaningful template-matrix diagnostic plot: all epochs for channel '{}' "
                    "are constant or effectively zero.",
                    raw.ch_names[channel_index],
                )
                return

            plot_aas_matrices(
                epoch_data=epoch_data,
                averaging_matrix=averaging_matrices[channel_index],
                sfreq=sfreq,
                threshold=self._matrix_correlation_threshold(),
                target_epoch=0,
                channel_name=raw.ch_names[channel_index],
                output_path=output_path,
                show_plot=False,
            )

            if output_path is not None:
                logger.info("Saved template-matrix diagnostic plot to {}", output_path)
        except Exception as exc:
            # Plotting is diagnostic-only and must never discard a completed
            # correction when a GUI/backend or malformed channel fails.
            logger.warning("Failed to save template-matrix diagnostic plot: {}", exc)

    @staticmethod
    def _select_diagnostic_channel(
        processed_channels: list[int],
        epoch_matrices_per_channel: dict[int, np.ndarray],
    ) -> int:
        """Choose the non-constant channel with the largest median epoch spread."""
        channel_scores = {
            channel_index: Flex._diagnostic_epoch_score(epoch_matrices_per_channel[channel_index])
            for channel_index in processed_channels
        }

        return max(channel_scores, key=channel_scores.get)

    @staticmethod
    def _diagnostic_epoch_score(epochs: np.ndarray) -> float:
        """Return a robust spread score for choosing one diagnostic channel."""
        candidate_epochs = np.asarray(epochs, dtype=float)
        if candidate_epochs.ndim != 2 or candidate_epochs.size == 0:
            return float("-inf")

        epoch_standard_deviations = np.nanstd(candidate_epochs, axis=1)
        finite_values = epoch_standard_deviations[np.isfinite(epoch_standard_deviations)]
        return float(np.median(finite_values)) if finite_values.size else float("-inf")

    def _plot_artifact_debug(
        self,
        raw: mne.io.Raw,
        averaging_matrices: dict[int, np.ndarray],
        artifacts_per_channel: list[np.ndarray],
    ) -> None:
        """Plot a randomly selected averaged artifact for visual debugging."""
        try:
            processed_channels = list(averaging_matrices)
            random_channel_list_index = random.randint(0, len(processed_channels) - 1)
            channel_index = processed_channels[random_channel_list_index]
            channel_name = raw.ch_names[channel_index]
            artifacts = artifacts_per_channel[random_channel_list_index]
            if len(artifacts) == 0:
                return

            random_epoch_index = random.randint(0, len(artifacts) - 1)
            artifact_segment = artifacts[random_epoch_index]
            logger.debug(
                "Plotting random artifact for channel {}, epoch {}",
                channel_name,
                random_epoch_index,
            )
            plt.figure(figsize=(10, 4))
            plt.plot(artifact_segment)
            plt.title(f"Estimated Artifact: Channel {channel_name} (Epoch {random_epoch_index})")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            logger.warning("Failed to plot random artifact: {}", exc)

    def _calc_averaged_artifacts(
        self,
        averaging_matrices: dict[int, np.ndarray],
        epoch_matrices_per_channel: dict[int, np.ndarray],
    ) -> list[np.ndarray]:
        """Calculate template matrices ``N = A @ D`` for every channel.

        Parameters
        ----------
        averaging_matrices : dict
            Averaging matrices ``A`` keyed by channel index.
        epoch_matrices_per_channel : dict
            Extracted epoch matrices ``D`` keyed by channel index.

        Returns
        -------
        list of np.ndarray
            Template matrix ``N`` for every channel, in dictionary insertion
            order.
        """
        artifacts_per_channel = []
        for channel_index, averaging_matrix in averaging_matrices.items():
            epoch_data = epoch_matrices_per_channel[channel_index]
            artifacts_per_channel.append(self._calculate_channel_templates(averaging_matrix, epoch_data))

        return artifacts_per_channel

    @staticmethod
    def _calculate_channel_templates(averaging_matrix: np.ndarray, epoch_data: np.ndarray) -> np.ndarray:
        """Calculate one channel's template matrix ``N = A @ D``."""
        # ``split_vector`` can retain a padded tail epoch in legacy inputs;
        # match the matrix row count before multiplication.
        if len(epoch_data) > len(averaging_matrix):
            epoch_data = epoch_data[: len(averaging_matrix)]

        return averaging_matrix @ epoch_data

    @staticmethod
    def _align_triggers_to_artifacts(
        channel_data: np.ndarray,
        artifacts: np.ndarray,
        triggers: np.ndarray,
        start_offset: int,
        artifact_length: int,
        search_window: int,
    ) -> np.ndarray:
        """Align triggers to template rows using cross-correlation.

        Parameters
        ----------
        channel_data : np.ndarray
            One complete EEG channel.
        artifacts : np.ndarray
            Per-trigger template matrix ``N``.
        triggers : np.ndarray
            Original trigger positions.
        start_offset : int
            Artifact start offset relative to a trigger, in samples.
        artifact_length : int
            Artifact length in samples.
        search_window : int
            Cross-correlation half-window in samples.

        Returns
        -------
        np.ndarray
            Realigned trigger positions.
        """
        aligned_triggers = []
        for epoch_index, trigger in enumerate(triggers):
            start = trigger + start_offset
            stop = trigger + artifact_length + search_window
            if stop > len(channel_data):
                aligned_triggers.append(trigger)
                continue

            segment = channel_data[start:stop]
            artifact = artifacts[epoch_index]
            correlations = crosscorrelation(segment, artifact, search_window)
            best_shift = np.argmax(correlations) - search_window
            aligned_triggers.append(trigger + best_shift)

        return np.asarray(aligned_triggers)

    def _calc_averaging_matrix(
        self,
        epochs: np.ndarray,
        window_size: int,
        rel_window_offset: float,
        correlation_threshold: float,
    ) -> np.ndarray:
        """Build the flexible epoch-averaging matrix for one channel.

        Parameters
        ----------
        epochs : np.ndarray
            Epoch data matrix with shape ``(n_epochs, n_samples)``.
        window_size : int
            Maximum number of non-target candidate epochs per row.
        rel_window_offset : float
            Unused shared matrix-hook argument. Flex always prioritizes future
            epochs and only backfills from preceding epochs at the recording
            end.
        correlation_threshold : float
            Signed Pearson correlation cutoff for automatic acceptance.

        Returns
        -------
        np.ndarray
            Square averaging matrix. Except for the zero- or one-epoch edge
            case, every row is finite, has a zero diagonal, and sums to one.
        """
        del rel_window_offset

        n_epochs = int(epochs.shape[0])
        averaging_matrix = np.zeros((n_epochs, n_epochs), dtype=float)
        self._require_finite_epoch_data(epochs)
        if n_epochs < 2 or window_size < 1:
            return averaging_matrix

        effective_window_size = min(window_size, n_epochs - 1)

        for target_idx in range(n_epochs):
            candidate_indices = self._candidate_indices(
                target_idx=target_idx,
                n_epochs=n_epochs,
                window_size=effective_window_size,
            )
            correlations = self._pearson_correlations(
                target_epoch=epochs[target_idx],
                candidate_epochs=epochs[candidate_indices],
            )
            selected_indices = self._select_epoch_indices(
                target_idx=target_idx,
                candidate_indices=candidate_indices,
                correlations=correlations,
                threshold=correlation_threshold,
            )
            weights = self._selection_weights(
                target_idx=target_idx,
                selected_indices=selected_indices,
                effective_window_size=effective_window_size,
            )
            averaging_matrix[target_idx, selected_indices] = weights

        return averaging_matrix

    @staticmethod
    def _candidate_indices(target_idx: int, n_epochs: int, window_size: int) -> np.ndarray:
        """Return a future-first window, backfilled with preceding epochs.

        The returned indices are chronological for stable diagnostics. Future
        epochs determine the window whenever enough are present; preceding
        epochs only fill positions that would otherwise be missing.
        """
        candidate_count = min(window_size, n_epochs - 1)
        forward_stop = min(n_epochs, target_idx + 1 + candidate_count)
        forward_indices = np.arange(target_idx + 1, forward_stop, dtype=int)

        backfill_count = candidate_count - len(forward_indices)
        backfill_start = max(0, target_idx - backfill_count)
        backfill_indices = np.arange(backfill_start, target_idx, dtype=int)

        return np.concatenate((backfill_indices, forward_indices))

    @staticmethod
    def _pearson_correlations(target_epoch: np.ndarray, candidate_epochs: np.ndarray) -> np.ndarray:
        """Calculate robust signed Pearson correlations against one target.

        Undefined correlations from constant or non-finite epochs are stored
        as negative infinity. They cannot pass the threshold, but remain
        deterministic last-resort candidates when ``min_accepted`` must be
        satisfied.
        """
        target_centered = target_epoch - np.mean(target_epoch)
        candidates_centered = candidate_epochs - np.mean(candidate_epochs, axis=1, keepdims=True)

        numerators = candidates_centered @ target_centered
        target_norm = np.linalg.norm(target_centered)
        candidate_norms = np.linalg.norm(candidates_centered, axis=1)
        denominators = candidate_norms * target_norm

        correlations = np.full(len(candidate_epochs), -np.inf, dtype=float)
        valid = np.isfinite(numerators) & np.isfinite(denominators) & (denominators > 0.0)
        np.divide(numerators, denominators, out=correlations, where=valid)

        finite = np.isfinite(correlations)
        correlations[finite] = np.clip(correlations[finite], -1.0, 1.0)
        return correlations

    def _select_epoch_indices(
        self,
        target_idx: int,
        candidate_indices: np.ndarray,
        correlations: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Select threshold matches and supplement them to the minimum count."""
        accepted = np.isfinite(correlations) & (correlations >= threshold)
        required_count = min(self.min_accepted, len(candidate_indices))
        accepted_count = int(np.count_nonzero(accepted))

        if accepted_count < required_count:
            # The first sort key is descending signed correlation. Distance and
            # absolute index make otherwise identical choices deterministic.
            distances = np.abs(candidate_indices - target_idx)
            ranked_positions = np.lexsort((candidate_indices, distances, -correlations))

            for candidate_position in ranked_positions:
                if accepted[candidate_position]:
                    continue
                accepted[candidate_position] = True
                accepted_count += 1
                if accepted_count >= required_count:
                    break

        return candidate_indices[accepted]

    def _selection_weights(
        self,
        target_idx: int,
        selected_indices: np.ndarray,
        effective_window_size: int,
    ) -> np.ndarray:
        """Return normalized equal or temporal-Gaussian selection weights."""
        if self.N_distribution == "equal":
            return np.full(len(selected_indices), 1.0 / len(selected_indices), dtype=float)

        standard_deviation = max(effective_window_size / 3.0, 1.0)
        temporal_distances = np.abs(selected_indices - target_idx).astype(float)
        weights = np.exp(-0.5 * (temporal_distances / standard_deviation) ** 2)
        return weights / np.sum(weights)


# Readable alias that follows the naming of the other correction processors.
FlexCorrection = Flex
