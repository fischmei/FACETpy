"""AAS variants with MATLAB FACET averaging-weight strategies."""

from __future__ import annotations

import numpy as np
from loguru import logger

from ..core import ProcessingContext, ProcessorValidationError, register_processor
from ..helpers.moosmann import calc_weighted_matrix_by_realignment_parameters_file
from .flex import Flex


@register_processor
class CorrespondingSliceCorrection(Flex):
    """AAS with corresponding-slice averaging across volumes.

    This implements MATLAB FACET's ``AvgArtWghtCorrespondingSlice`` rule:
    each slice epoch is averaged with the same slice position in neighboring
    volumes.

    Parameters
    ----------
    slices_per_volume : int, optional
        Number of slices per volume. If ``None``, the value is taken from
        ``context.metadata.slices_per_volume``.
    window_size : int
        Half-window in volumes (default: 30).
    plot_artifacts : bool
        If ``True``, plot one random averaged artifact (default: False).
    realign_after_averaging : bool
        If ``True``, realign triggers after averaging (default: True).
    search_window_factor : float
        Trigger realignment search-window factor (default: 3.0).
    apply_epoch_alpha_scaling : bool
        If ``True``, apply MATLAB-like per-epoch least-squares alpha scaling
        before subtraction (default: False).
    """

    name = "corresponding_slice_correction"
    description = "AAS with corresponding-slice averaging across volumes"
    version = "1.0.0"

    def __init__(
        self,
        slices_per_volume: int | None = None,
        window_size: int = 30,
        plot_artifacts: bool = False,
        realign_after_averaging: bool = True,
        search_window_factor: float = 3.0,
        apply_epoch_alpha_scaling: bool = False,
        track_estimated_noise: bool = True,
    ) -> None:
        self.slices_per_volume = slices_per_volume
        self._runtime_slices_per_volume: int | None = None
        super().__init__(
            window_size=window_size,
            threshold=0.975,
            min_accepted=1,
            N_distribution="equal",
            plot_artifacts=plot_artifacts,
            realign_after_averaging=realign_after_averaging,
            search_window_factor=search_window_factor,
            apply_epoch_alpha_scaling=apply_epoch_alpha_scaling,
            track_estimated_noise=track_estimated_noise,
        )

    def _get_parameters(self) -> dict[str, object]:
        """Return constructor-compatible corresponding-slice parameters."""
        return {
            "slices_per_volume": self.slices_per_volume,
            "window_size": self.window_size,
            "plot_artifacts": self.plot_artifacts,
            "realign_after_averaging": self.realign_after_averaging,
            "search_window_factor": self.search_window_factor,
            "apply_epoch_alpha_scaling": self.apply_epoch_alpha_scaling,
            "track_estimated_noise": self.track_estimated_noise,
        }

    def _validate_averaging_strategy(self, context: ProcessingContext) -> None:
        """Validate the slice-period input used by this matrix strategy."""
        runtime_spv = self._resolve_slices_per_volume(context)
        if runtime_spv < 1:
            raise ProcessorValidationError(f"slices_per_volume must be >= 1, got {runtime_spv}")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        self._runtime_slices_per_volume = self._resolve_slices_per_volume(context)
        try:
            return super().process(context)
        finally:
            self._runtime_slices_per_volume = None

    def _resolve_slices_per_volume(self, context: ProcessingContext) -> int:
        if self.slices_per_volume is not None:
            return int(self.slices_per_volume)
        if context.metadata.slices_per_volume is not None:
            return int(context.metadata.slices_per_volume)
        raise ProcessorValidationError(
            "slices_per_volume not available. Set it explicitly or run TriggerDetector on slice-trigger data."
        )

    def _calc_averaging_matrix(
        self,
        epochs: np.ndarray,
        window_size: int,
        rel_window_offset: float,
        correlation_threshold: float,
    ) -> np.ndarray:
        """Create corresponding-slice averaging matrix."""
        del rel_window_offset, correlation_threshold

        n_epochs = int(epochs.shape[0])
        matrix = np.zeros((n_epochs, n_epochs), dtype=float)
        if n_epochs == 0:
            return matrix

        period = int(self._runtime_slices_per_volume or 1)
        half_window = max(1, int(window_size))
        warning_shown = False

        for row0 in range(n_epochs):
            row = row0 + 1  # 1-based for parity with MATLAB equations
            i_start = row - (half_window * period)
            if i_start < 1:
                i_start = 1 + ((row - 1) % period)

            i_end = i_start + (2 * half_window * period)
            if i_end > n_epochs:
                i_end = self._nearest_multiple_before(row, period, n_epochs)
                i_start = i_end - (2 * half_window * period)

            if i_start < 1:
                i_start = 1 + ((row - 1) % period)
                if not warning_shown:
                    warning_shown = True
                    logger.warning("Not enough volumes for full corresponding-slice window. Using reduced slice sets.")

            indices_1based = np.arange(i_start, i_end + 1, period, dtype=int)
            indices_1based = indices_1based[(indices_1based >= 1) & (indices_1based <= n_epochs)]
            if indices_1based.size == 0:
                indices_1based = np.array([row], dtype=int)

            indices_0based = indices_1based - 1
            matrix[row0, indices_0based] = 1.0 / float(indices_0based.size)

        return matrix

    @staticmethod
    def _nearest_multiple_before(origin: int, period: int, limit: int) -> int:
        n = int(np.floor((limit - origin) / period))
        return origin + (n * period)


@register_processor
class VolumeTriggerCorrection(Flex):
    """AAS with MATLAB FACET volume-trigger weighting.

    Reproduces ``AvgArtWghtVolumeTrigger`` weighting for volume/section trigger
    workflows.

    Parameters
    ----------
    window_size : int
        Averaging window size in epochs (default: 30).
    plot_artifacts : bool
        If ``True``, plot one random averaged artifact (default: False).
    realign_after_averaging : bool
        If ``True``, realign triggers after averaging (default: True).
    search_window_factor : float
        Trigger realignment search-window factor (default: 3.0).
    apply_epoch_alpha_scaling : bool
        If ``True``, apply MATLAB-like per-epoch least-squares alpha scaling
        before subtraction (default: False).
    """

    name = "volume_trigger_correction"
    description = "AAS with MATLAB volume-trigger averaging weights"
    version = "1.0.0"

    def __init__(
        self,
        window_size: int = 30,
        plot_artifacts: bool = False,
        realign_after_averaging: bool = True,
        search_window_factor: float = 3.0,
        apply_epoch_alpha_scaling: bool = False,
        track_estimated_noise: bool = True,
    ) -> None:
        super().__init__(
            window_size=window_size,
            threshold=0.975,
            min_accepted=1,
            N_distribution="equal",
            plot_artifacts=plot_artifacts,
            realign_after_averaging=realign_after_averaging,
            search_window_factor=search_window_factor,
            apply_epoch_alpha_scaling=apply_epoch_alpha_scaling,
            track_estimated_noise=track_estimated_noise,
        )

    def _get_parameters(self) -> dict[str, object]:
        """Return constructor-compatible volume-trigger parameters."""
        return {
            "window_size": self.window_size,
            "plot_artifacts": self.plot_artifacts,
            "realign_after_averaging": self.realign_after_averaging,
            "search_window_factor": self.search_window_factor,
            "apply_epoch_alpha_scaling": self.apply_epoch_alpha_scaling,
            "track_estimated_noise": self.track_estimated_noise,
        }

    def _validate_averaging_strategy(self, context: ProcessingContext) -> None:
        """Skip correlation-selection checks unused by fixed volume weights."""
        del context

    def _calc_averaging_matrix(
        self,
        epochs: np.ndarray,
        window_size: int,
        rel_window_offset: float,
        correlation_threshold: float,
    ) -> np.ndarray:
        """Create volume-trigger averaging matrix."""
        del rel_window_offset, correlation_threshold

        n_epochs = int(epochs.shape[0])
        matrix = np.zeros((n_epochs, n_epochs), dtype=float)
        if n_epochs == 0:
            return matrix

        half_window = max(1, int(window_size // 2))
        i_start = 2  # MATLAB uses this initial value at borders

        for s0 in range(n_epochs):
            s = s0 + 1
            if s == 1:
                i_start = 2
            elif (s > (3 + half_window)) and (s <= (n_epochs - (half_window + 2))):
                i_start = s - half_window - 1

            indices_1based = np.arange(i_start, i_start + (2 * half_window) + 1, dtype=int)
            indices_1based = indices_1based[(indices_1based >= 1) & (indices_1based <= n_epochs)]
            if indices_1based.size == 0:
                indices_1based = np.array([s], dtype=int)

            indices_0based = indices_1based - 1
            matrix[s0, indices_0based] = 1.0 / float(indices_0based.size)

        return matrix


@register_processor
class SliceTriggerCorrection(Flex):
    """AAS with MATLAB FACET slice-trigger odd/even template weighting.

    Reproduces ``AvgArtWghtSliceTrigger`` behavior by constructing one
    averaging set from every second epoch (even offsets) and another from
    the complementary epochs (odd offsets), then alternating between them.

    Parameters
    ----------
    window_size : int
        Half-window size in epochs (default: 30).
    plot_artifacts : bool
        If ``True``, plot one random averaged artifact (default: False).
    realign_after_averaging : bool
        If ``True``, realign triggers after averaging (default: True).
    search_window_factor : float
        Trigger realignment search-window factor (default: 3.0).
    apply_epoch_alpha_scaling : bool
        If ``True``, apply MATLAB-like per-epoch least-squares alpha scaling
        before subtraction (default: False).
    """

    name = "slice_trigger_correction"
    description = "AAS with MATLAB slice-trigger odd/even averaging weights"
    version = "1.0.0"

    def __init__(
        self,
        window_size: int = 30,
        plot_artifacts: bool = False,
        realign_after_averaging: bool = True,
        search_window_factor: float = 3.0,
        apply_epoch_alpha_scaling: bool = False,
        track_estimated_noise: bool = True,
    ) -> None:
        super().__init__(
            window_size=window_size,
            threshold=0.975,
            min_accepted=1,
            N_distribution="equal",
            plot_artifacts=plot_artifacts,
            realign_after_averaging=realign_after_averaging,
            search_window_factor=search_window_factor,
            apply_epoch_alpha_scaling=apply_epoch_alpha_scaling,
            track_estimated_noise=track_estimated_noise,
        )

    def _get_parameters(self) -> dict[str, object]:
        """Return constructor-compatible slice-trigger parameters."""
        return {
            "window_size": self.window_size,
            "plot_artifacts": self.plot_artifacts,
            "realign_after_averaging": self.realign_after_averaging,
            "search_window_factor": self.search_window_factor,
            "apply_epoch_alpha_scaling": self.apply_epoch_alpha_scaling,
            "track_estimated_noise": self.track_estimated_noise,
        }

    def _validate_averaging_strategy(self, context: ProcessingContext) -> None:
        """Skip correlation-selection checks unused by odd/even weights."""
        del context

    def _calc_averaging_matrix(
        self,
        epochs: np.ndarray,
        window_size: int,
        rel_window_offset: float,
        correlation_threshold: float,
    ) -> np.ndarray:
        """Create slice-trigger odd/even averaging matrix."""
        del rel_window_offset, correlation_threshold

        n_epochs = int(epochs.shape[0])
        matrix = np.zeros((n_epochs, n_epochs), dtype=float)
        if n_epochs == 0:
            return matrix

        half_window = max(1, int(window_size))
        start_index_1based = 2

        for s0 in range(n_epochs):
            s = s0 + 1
            if s == 1:
                start_index_1based = 2
            elif s == 2:
                start_index_1based = 3
            elif (s > (3 + half_window)) and (s <= (n_epochs - (half_window + 2))):
                start_index_1based = s - half_window

            end_index_1based = start_index_1based + (2 * half_window)
            indices_1based = np.arange(start_index_1based, end_index_1based + 1, 2, dtype=int)
            indices_1based = indices_1based[(indices_1based >= 1) & (indices_1based <= n_epochs)]

            if indices_1based.size == 0:
                indices_1based = np.array([s], dtype=int)

            indices_0based = indices_1based - 1
            matrix[s0, indices_0based] = 1.0 / float(indices_0based.size)

        return matrix


@register_processor
class MoosmannCorrection(Flex):
    """AAS with motion-informed Moosmann averaging weights.

    Uses the realignment-parameter-informed weighting strategy from
    ``AvgArtWghtMoosmann``.

    Parameters
    ----------
    rp_file : str
        Path to SPM-style realignment parameter text file.
    window_size : int
        AAS base window size (default: 30).
    motion_threshold : float
        Motion threshold passed to the weighting routine (default: 5.0).
    motion_window_size : int, optional
        Number of neighboring epochs for motion weighting. If ``None``,
        uses ``2 * window_size``.
    plot_artifacts : bool
        If ``True``, plot one random averaged artifact (default: False).
    realign_after_averaging : bool
        If ``True``, realign triggers after averaging (default: True).
    search_window_factor : float
        Trigger realignment search-window factor (default: 3.0).
    apply_epoch_alpha_scaling : bool
        If ``True``, apply MATLAB-like per-epoch least-squares alpha scaling
        before subtraction (default: False).
    """

    name = "moosmann_correction"
    description = "AAS with motion-informed Moosmann template weighting"
    version = "1.0.0"

    def __init__(
        self,
        rp_file: str,
        window_size: int = 30,
        motion_threshold: float = 5.0,
        motion_window_size: int | None = None,
        plot_artifacts: bool = False,
        realign_after_averaging: bool = True,
        search_window_factor: float = 3.0,
        apply_epoch_alpha_scaling: bool = False,
        track_estimated_noise: bool = True,
    ) -> None:
        self.rp_file = rp_file
        self.motion_threshold = motion_threshold
        self.motion_window_size = motion_window_size
        self._matrix_cache: dict[int, np.ndarray] = {}
        self._last_motion_summary: dict | None = None
        super().__init__(
            window_size=window_size,
            threshold=0.975,
            min_accepted=1,
            N_distribution="equal",
            plot_artifacts=plot_artifacts,
            realign_after_averaging=realign_after_averaging,
            search_window_factor=search_window_factor,
            apply_epoch_alpha_scaling=apply_epoch_alpha_scaling,
            track_estimated_noise=track_estimated_noise,
        )

    def _get_parameters(self) -> dict[str, object]:
        """Return constructor-compatible motion-weighting parameters."""
        return {
            "rp_file": self.rp_file,
            "window_size": self.window_size,
            "motion_threshold": self.motion_threshold,
            "motion_window_size": self.motion_window_size,
            "plot_artifacts": self.plot_artifacts,
            "realign_after_averaging": self.realign_after_averaging,
            "search_window_factor": self.search_window_factor,
            "apply_epoch_alpha_scaling": self.apply_epoch_alpha_scaling,
            "track_estimated_noise": self.track_estimated_noise,
        }

    def _validate_averaging_strategy(self, context: ProcessingContext) -> None:
        """Validate motion-weighting settings without Flex-only constraints."""
        del context

        if not self.rp_file:
            raise ProcessorValidationError("rp_file must be provided for MoosmannCorrection.")
        if self.motion_threshold <= 0:
            raise ProcessorValidationError(f"motion_threshold must be positive, got {self.motion_threshold}")
        if self.motion_window_size is not None and self.motion_window_size < 1:
            raise ProcessorValidationError(f"motion_window_size must be >= 1 when set, got {self.motion_window_size}")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        result = super().process(context)
        if self._last_motion_summary is None:
            return result

        md = result.metadata.copy()
        md.custom["moosmann"] = self._last_motion_summary
        return result.with_metadata(
            md,
            copy_estimated_noise=False,
        )

    def _calc_averaging_matrix(
        self,
        epochs: np.ndarray,
        window_size: int,
        rel_window_offset: float,
        correlation_threshold: float,
    ) -> np.ndarray:
        """Create Moosmann weighting matrix from the RP file."""
        del rel_window_offset, correlation_threshold

        n_epochs = int(epochs.shape[0])
        if n_epochs in self._matrix_cache:
            return self._matrix_cache[n_epochs]

        motion_window = int(self.motion_window_size) if self.motion_window_size is not None else int(2 * window_size)
        motion_window = max(1, motion_window)

        motiondata, matrix = calc_weighted_matrix_by_realignment_parameters_file(
            rp_file=self.rp_file,
            n_fmri=n_epochs,
            k=motion_window,
            threshold=self.motion_threshold,
        )

        row_sums = np.sum(matrix, axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        matrix = matrix / row_sums

        motion_serialized = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in motiondata.items()}

        self._last_motion_summary = {
            "rp_file": self.rp_file,
            "motion_threshold": self.motion_threshold,
            "motion_window_size": motion_window,
            "num_epochs": n_epochs,
            "motion": motion_serialized,
        }

        self._matrix_cache[n_epochs] = matrix
        return matrix


# Aliases for backwards compatibility / readability
AvgArtWghtCorrespondingSliceCorrection = CorrespondingSliceCorrection
AvgArtWghtVolumeTriggerCorrection = VolumeTriggerCorrection
AvgArtWghtSliceTriggerCorrection = SliceTriggerCorrection
AvgArtWghtMoosmannCorrection = MoosmannCorrection
