"""Legacy Averaged Artifact Subtraction matrix strategy."""

from __future__ import annotations

import numpy as np

from ..core import ProcessingContext, ProcessorValidationError, register_processor
from .flex import Flex


@register_processor
class AASCorrection(Flex):
    """Apply the legacy blockwise AAS averaging strategy through Flex.

    Flex owns epoch extraction, template construction, trigger realignment,
    subtraction, noise accumulation, and reporting. This compatibility class
    preserves the historical AAS rule for constructing averaging matrix
    ``A``: each target block shares an evolving correlation-selected template
    seeded by the first five epochs in that block.

    References
    ----------
    Allen et al., 2000. "A method for removing imaging artifact from continuous
    EEG recorded during functional MRI." NeuroImage, 12(2), 230-239.

    Parameters
    ----------
    window_size : int
        Number of epochs in each target/candidate block (default: 30).
    rel_window_position : float
        Relative candidate-window offset between -1 and 1 (default: 0.0).
    correlation_threshold : float
        Minimum signed Pearson correlation used by the evolving-average
        selection (default: 0.975).
    plot_artifacts : bool
        If ``True``, plot a representative artifact template (default:
        ``False``).
    realign_after_averaging : bool
        If ``True``, realign triggers to the templates before subtraction
        (default: ``True``).
    search_window_factor : float
        Trigger-realignment search-window multiplier (default: 3.0).
    interpolate_volume_gaps : bool
        If ``True``, interpolate the estimated artifact between adjacent
        artifact windows (default: ``False``).
    apply_epoch_alpha_scaling : bool
        If ``True``, least-squares scale every template immediately before
        subtraction (default: ``False``).
    """

    name = "aas_correction"
    description = "Averaged Artifact Subtraction for fMRI artifacts"
    version = "1.0.0"

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
        track_estimated_noise: bool = True,
    ) -> None:
        # These aliases are part of the established AAS public API and are
        # consumed by the matrix-parameter hooks below.
        self.rel_window_position = rel_window_position
        self.correlation_threshold = correlation_threshold

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
        """Return the exact AAS constructor arguments for worker rebuilding."""
        return {
            "window_size": self.window_size,
            "rel_window_position": self.rel_window_position,
            "correlation_threshold": self.correlation_threshold,
            "plot_artifacts": self.plot_artifacts,
            "realign_after_averaging": self.realign_after_averaging,
            "search_window_factor": self.search_window_factor,
            "interpolate_volume_gaps": self.interpolate_volume_gaps,
            "apply_epoch_alpha_scaling": self.apply_epoch_alpha_scaling,
            "track_estimated_noise": self.track_estimated_noise,
        }

    def _validate_averaging_strategy(self, context: ProcessingContext) -> None:
        """Validate parameters unique to the legacy AAS matrix rule."""
        del context
        if not (0 < self.correlation_threshold <= 1):
            raise ProcessorValidationError(f"correlation_threshold must be in (0, 1], got {self.correlation_threshold}")
        if not (-1.0 <= self.rel_window_position <= 1.0):
            raise ProcessorValidationError(f"rel_window_position must be in [-1, 1], got {self.rel_window_position}")

    def _matrix_rel_window_offset(self) -> float:
        """Return AAS's configured candidate-window offset."""
        return float(self.rel_window_position)

    def _matrix_correlation_threshold(self) -> float:
        """Return AAS's evolving-average correlation cutoff."""
        return float(self.correlation_threshold)

    def _calc_averaging_matrix(
        self,
        epochs: np.ndarray,
        window_size: int,
        rel_window_offset: float,
        correlation_threshold: float,
    ) -> np.ndarray:
        """Construct the historical blockwise AAS averaging matrix.

        For every ``window_size`` target block, the first five epochs seed a
        running average. Candidates whose signed correlation with the current
        average exceeds the cutoff are added iteratively. Every target row in
        the block then receives the same equal-weight selection.

        Parameters
        ----------
        epochs : np.ndarray
            Epoch matrix ``D`` with shape ``(n_epochs, n_samples)``.
        window_size : int
            Target-block and candidate-window size.
        rel_window_offset : float
            Relative offset of the candidate window.
        correlation_threshold : float
            Signed Pearson cutoff for candidate acceptance.

        Returns
        -------
        np.ndarray
            Square averaging matrix ``A``.
        """
        n_epochs = len(epochs)
        averaging_matrix = np.zeros((n_epochs, n_epochs))
        window_offset = int(window_size * rel_window_offset)

        for block_start in range(0, n_epochs, window_size):
            candidate_start = block_start + window_offset
            reference_indices = np.arange(block_start, min(block_start + 5, n_epochs))
            candidate_indices = np.arange(
                candidate_start,
                min(candidate_start + window_size, n_epochs),
            )
            candidate_indices = candidate_indices[candidate_indices >= 0]

            chosen_indices = self._find_correlated_epochs(
                epochs,
                candidate_indices,
                reference_indices,
                correlation_threshold,
            )
            if len(chosen_indices) == 0:
                chosen_indices = reference_indices

            target_indices = np.arange(block_start, min(block_start + window_size, n_epochs))
            averaging_matrix[np.ix_(target_indices, chosen_indices)] = 1.0 / len(chosen_indices)

        return averaging_matrix

    @staticmethod
    def _find_correlated_epochs(
        all_epochs: np.ndarray,
        candidate_indices: np.ndarray,
        reference_indices: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Add candidates correlated with the evolving reference average."""
        if len(reference_indices) == 0:
            return np.array([])

        summed_epochs = np.sum(all_epochs[reference_indices], axis=0)
        chosen_indices = list(reference_indices)

        for candidate_index in candidate_indices:
            if candidate_index in chosen_indices:
                continue

            running_average = summed_epochs / len(chosen_indices)
            correlation = np.corrcoef(
                running_average.squeeze(),
                all_epochs[candidate_index].squeeze(),
            )[0, 1]
            if correlation > threshold:
                summed_epochs += all_epochs[candidate_index]
                chosen_indices.append(candidate_index)

        return np.asarray(chosen_indices)


# Alias retained for backwards compatibility.
AveragedArtifactSubtraction = AASCorrection
