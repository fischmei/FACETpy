"""FARM-based artifact correction processor."""

from __future__ import annotations

import numpy as np
from loguru import logger

from ..core import ProcessingContext, ProcessorValidationError, register_processor
from .flex import Flex


@register_processor
class FARMCorrection(Flex):
    """Remove fMRI artifacts using the MATLAB FACET FARM weighting strategy.

    This processor reuses the Flex template engine but replaces template
    selection with the FARM rule from MATLAB ``AvgArtWghtFARM``:

    - Compute epoch-to-epoch correlations for one channel.
    - For each epoch, search within a wide neighborhood.
    - Keep up to ``window_size`` most correlated epochs above threshold.
    - Average the selected epochs with equal weights.

    Parameters
    ----------
    window_size : int
        Number of similar epochs to average (default: 30).
    rel_window_position : float
        Shared AAS/FARM tuning parameter accepted for API parity. FARM keeps
        its row-wise candidate selection strategy, so this value is stored and
        validated like AAS but does not alter FARM weighting.
    correlation_threshold : float
        Minimum absolute correlation for candidate selection
        (default: 0.9, matching MATLAB FARM).
    search_half_window : int, optional
        Half-window in epochs used for candidate search. If ``None``, derived
        from ``search_half_window_factor * window_size``.
    search_half_window_factor : float
        # Legacy compatibility multiplier retained for API parity. FARM no
        # longer expands the candidate search span with this value; the search
        # neighborhood is capped by the averaging window size instead.
    plot_artifacts : bool
        If ``True``, plot one random averaged artifact (default: False).
    realign_after_averaging : bool
        If ``True``, realign triggers after template averaging (default: True).
    search_window_factor : float
        Trigger-realignment search-window factor (default: 3.0).
    interpolate_volume_gaps : bool
        If ``True``, interpolate estimated artifact/noise in inter-epoch gaps
        (default: False).
    apply_epoch_alpha_scaling : bool
        If ``True``, apply MATLAB-like per-epoch least-squares alpha scaling
        before subtraction (default: False).
    """

    name = "farm_correction"
    description = "AAS with MATLAB FACET FARM template weighting"
    version = "1.0.0"

    def __init__(
        self,
        window_size: int = 30,
        correlation_threshold: float = 0.9,
        search_half_window: int | None = None,
        search_half_window_factor: float = 3.0,
        plot_artifacts: bool = False,
        realign_after_averaging: bool = True,
        search_window_factor: float = 3.0,
        interpolate_volume_gaps: bool = False,
        apply_epoch_alpha_scaling: bool = False,
        track_estimated_noise: bool = True,
        *,
        rel_window_position: float = 0.0,
    ) -> None:
        # These legacy names remain public because existing pipelines, reports,
        # and the grid search configure FARM through the AAS-compatible API.
        # Flex consults the matrix hooks below instead of assuming its own
        # future-first strategy arguments.
        self.correlation_threshold = correlation_threshold
        self.rel_window_position = rel_window_position
        self.search_half_window = search_half_window
        self.search_half_window_factor = search_half_window_factor
        super().__init__(
            window_size=window_size,
            threshold=correlation_threshold,
            min_accepted=1,
            N_distribution="equal",
            plot_artifacts=plot_artifacts,
            realign_after_averaging=realign_after_averaging,
            search_window_factor=search_window_factor,
            interpolate_volume_gaps=interpolate_volume_gaps,
            apply_epoch_alpha_scaling=apply_epoch_alpha_scaling,
            track_estimated_noise=track_estimated_noise,
        )

    def _get_parameters(self) -> dict[str, object]:
        """Return only arguments accepted by the FARM constructor.

        Parallel workers recreate a processor with ``type(proc)(**params)``.
        Flex's strategy-only controls therefore must not leak into FARM's
        serialized public configuration.
        """
        return {
            "window_size": self.window_size,
            "correlation_threshold": self.correlation_threshold,
            "search_half_window": self.search_half_window,
            "search_half_window_factor": self.search_half_window_factor,
            "plot_artifacts": self.plot_artifacts,
            "realign_after_averaging": self.realign_after_averaging,
            "search_window_factor": self.search_window_factor,
            "interpolate_volume_gaps": self.interpolate_volume_gaps,
            "apply_epoch_alpha_scaling": self.apply_epoch_alpha_scaling,
            "track_estimated_noise": self.track_estimated_noise,
            "rel_window_position": self.rel_window_position,
        }

    def _validate_averaging_strategy(self, context: ProcessingContext) -> None:
        """Validate FARM-only settings without Flex selection constraints."""
        del context

        if not (0 < self.correlation_threshold <= 1):
            raise ProcessorValidationError(f"correlation_threshold must be in (0, 1], got {self.correlation_threshold}")
        if not (-1.0 <= self.rel_window_position <= 1.0):
            raise ProcessorValidationError(f"rel_window_position must be in [-1, 1], got {self.rel_window_position}")
        if self.search_half_window is not None and self.search_half_window < 1:
            raise ProcessorValidationError(f"search_half_window must be >= 1 when set, got {self.search_half_window}")
        if self.search_half_window_factor <= 0:
            raise ProcessorValidationError(
                f"search_half_window_factor must be positive, got {self.search_half_window_factor}"
            )

    def _matrix_rel_window_offset(self) -> float:
        """Return the legacy API value passed to the matrix strategy hook."""
        return float(self.rel_window_position)

    def _matrix_correlation_threshold(self) -> float:
        """Return FARM's absolute-correlation cutoff for matrix construction."""
        return float(self.correlation_threshold)

    def _calc_averaging_matrix(
        self,
        epochs: np.ndarray,
        window_size: int,
        rel_window_offset: float,
        correlation_threshold: float,
    ) -> np.ndarray:
        """Calculate FARM averaging weights for all epochs.

        Parameters
        ----------
        epochs : np.ndarray
            Epoch matrix with shape ``(n_epochs, n_times)``.
        window_size : int
            Maximum number of epochs to average per row.
        rel_window_offset : float
            Unused for FARM; accepted for API compatibility.
        correlation_threshold : float
            Minimum absolute Pearson correlation for inclusion.

        Returns
        -------
        np.ndarray
            Averaging matrix of shape ``(n_epochs, n_epochs)``.
        """
        del rel_window_offset  # Not used by FARM, kept for signature compatibility.

        n_epochs = int(epochs.shape[0])
        averaging_matrix = np.zeros((n_epochs, n_epochs), dtype=float)

        if n_epochs == 0:
            return averaging_matrix
        if n_epochs == 1:
            averaging_matrix[0, 0] = 1.0
            return averaging_matrix

        search_half_window = self._resolve_search_half_window(window_size)

        corr_mat = np.corrcoef(epochs)
        corr_mat = np.nan_to_num(corr_mat, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr_mat, 1.0)

        too_few_rows = 0
        for row in range(n_epochs):
            selected = self._select_row_indices(
                corr_mat=corr_mat,
                row=row,
                n_epochs=n_epochs,
                search_half_window=search_half_window,
                window_size=window_size,
                correlation_threshold=correlation_threshold,
            )
            if selected.size < window_size:
                too_few_rows += 1
            if selected.size == 0:
                # Keep the row valid so subtraction remains stable.
                selected = np.array([row], dtype=int)

            averaging_matrix[row, selected] = 1.0 / float(selected.size)

        if too_few_rows > 0:
            logger.warning(
                "FARM found fewer than {} similar epochs in {} rows; using reduced averages.",
                window_size,
                too_few_rows,
            )

        return averaging_matrix

    def _resolve_search_half_window(self, window_size: int) -> int:
        """Return FARM candidate search half-window in epochs."""
        if self.search_half_window is not None:
            return int(self.search_half_window)
        return max(1, int(round(self.search_half_window_factor * window_size)))

    def _select_row_indices(
        self,
        corr_mat: np.ndarray,
        row: int,
        n_epochs: int,
        search_half_window: int,
        window_size: int,
        correlation_threshold: float,
    ) -> np.ndarray:
        """Select averaged-epoch indices for one FARM matrix row."""
        left = max(0, row - search_half_window)
        right = min(left + (2 * search_half_window + 1), n_epochs)

        required_width = 2 * search_half_window + 1
        if (right - left) < required_width:
            left = max(0, right - required_width)

        local_indices = np.arange(left, right, dtype=int)
        local_corr = np.abs(corr_mat[row, left:right])

        order = np.argsort(local_corr)[::-1]
        ranked_indices = local_indices[order]
        ranked_corr = local_corr[order]

        not_self = ranked_indices != row
        ranked_indices = ranked_indices[not_self]
        ranked_corr = ranked_corr[not_self]

        selected = ranked_indices[ranked_corr >= correlation_threshold]
        if selected.size == 0:
            return np.array([], dtype=int)

        return selected[:window_size]


# Alias for backwards compatibility
FARMArtifactCorrection = FARMCorrection
