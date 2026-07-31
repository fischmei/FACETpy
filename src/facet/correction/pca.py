"""
PCA-based artifact correction processor.
"""

import mne
import numpy as np
from loguru import logger
from scipy.signal import butter, filtfilt

from ..console import processor_progress
from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor
from ..helpers.utils import split_vector


@register_processor
class PCACorrection(Processor):
    """Remove fMRI artifacts from EEG data using Principal Component Analysis.

    Splits the acquisition window into trigger-aligned epochs, applies PCA to
    each epoch, reconstructs the data using the retained components, and
    subtracts the reconstruction from the original signal.  The subtracted
    portion is treated as the artifact estimate.

    The number of retained components can be controlled precisely:

    - An integer keeps exactly that many components.
    - A float in (0, 1) retains enough components to explain that fraction of
      the total variance (e.g. 0.95 → 95 %).
    - ``"auto"`` uses MATLAB FACET OBS heuristics.
    - 0 skips PCA for all channels.

    Parameters
    ----------
    n_components : int or float or str
        Number of PCA components to retain (int) or variance fraction to
        retain (float in (0, 1)). Use ``"auto"`` for MATLAB-like OBS auto
        selection. Default: 0.95.
    hp_freq : float, optional
        High-pass cutoff frequency in Hz applied before PCA. None skips
        filtering (default: None).
    hp_filter_weights : np.ndarray, optional
        Pre-computed filter weights; overrides ``hp_freq`` when provided.
    exclude_channels : list, optional
        Channel indices to skip during PCA (default: empty list).
    track_estimated_noise : bool, optional
        Retain PCA residuals as a full-length noise estimate for downstream
        ANC or inspection (default: True).
    """

    name = "pca_correction"
    description = "PCA-based artifact removal"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = True
    parallel_safe = True
    channel_wise = True

    TH_SLOPE = 2.0
    TH_CUMVAR = 80.0
    TH_VAREXP = 5.0

    def __init__(
        self,
        n_components: int | float | str = 0.95,
        hp_freq: float | None = None,
        hp_filter_weights: np.ndarray | None = None,
        exclude_channels: list | None = None,
        track_estimated_noise: bool = True,
    ) -> None:
        self.n_components = n_components
        self.hp_freq = hp_freq
        self.hp_filter_weights = hp_filter_weights
        self.exclude_channels = exclude_channels or []
        self.track_estimated_noise = track_estimated_noise
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError("Artifact length not set. Run TriggerDetector first.")
        if not isinstance(self.track_estimated_noise, bool):
            raise ProcessorValidationError(
                f"track_estimated_noise must be a boolean, got {self.track_estimated_noise!r}"
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

        # --- LOG ---
        logger.info("Applying PCA artifact correction to {} channels", len(eeg_channels))

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found, skipping PCA")
            return context

        # --- COMPUTE ---
        hp_weights = self._resolve_hp_weights(raw.info["sfreq"])
        s_acq_start, s_acq_end = self._get_acquisition_window(context)
        # Direct _data access avoids a full array copy on large datasets
        estimated_artifacts = np.zeros(raw._data.shape) if self.track_estimated_noise else None

        with processor_progress(
            total=len(eeg_channels) or None,
            message="PCA artifact correction",
        ) as progress:
            for idx, ch_idx in enumerate(eeg_channels):
                ch_name = raw.ch_names[ch_idx]
                status_prefix = f"{idx + 1}/{len(eeg_channels)} • {ch_name}"

                if ch_idx in self.exclude_channels:
                    progress.advance(1, message=f"{status_prefix} (excluded)")
                    continue

                if self.n_components == 0:
                    progress.advance(1, message=f"{status_prefix} (disabled)")
                    continue

                try:
                    residuals = self._calc_pca_residuals(
                        raw._data[ch_idx],
                        triggers,
                        artifact_length,
                        s_acq_start,
                        s_acq_end,
                        hp_weights,
                    )
                    raw._data[ch_idx][s_acq_start:s_acq_end] -= residuals
                    if estimated_artifacts is not None:
                        estimated_artifacts[ch_idx][s_acq_start:s_acq_end] += residuals
                    progress.advance(1, message=status_prefix)
                except Exception as exc:
                    logger.error("PCA failed for channel {}: {}", ch_name, exc)
                    progress.advance(1, message=f"{status_prefix} (error)")

        # --- NOISE ---
        if estimated_artifacts is None:
            new_ctx = context.with_raw(
                raw,
                estimated_noise=None,
                copy_estimated_noise=False,
            )
        else:
            previous_noise = context.get_estimated_noise()
            if previous_noise is None:
                accumulated_noise = estimated_artifacts
            else:
                accumulated_noise = previous_noise.copy()
                accumulated_noise += estimated_artifacts
            new_ctx = context.with_raw(
                raw,
                estimated_noise=accumulated_noise,
                copy_estimated_noise=False,
            )

        # --- RETURN ---
        logger.info("PCA correction completed")
        return new_ctx

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    def _resolve_hp_weights(self, sfreq: float) -> np.ndarray | None:
        """Return the appropriate high-pass filter weights.

        Parameters
        ----------
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        np.ndarray or None
            Filter weights for use with ``scipy.signal.filtfilt``, or None
            when high-pass filtering is disabled.
        """
        if self.hp_filter_weights is not None:
            return self.hp_filter_weights
        if self.hp_freq is not None and self.hp_freq > 0:
            return self._create_hp_filter(sfreq)
        return None

    def _calc_pca_residuals(
        self,
        ch_data: np.ndarray,
        triggers: np.ndarray,
        artifact_length: int,
        s_acq_start: int,
        s_acq_end: int,
        hp_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Calculate PCA-based artifact residuals for a single channel.

        Parameters
        ----------
        ch_data : np.ndarray
            Full time series for one channel.
        triggers : np.ndarray
            Trigger sample positions.
        artifact_length : int
            Artifact length in samples.
        s_acq_start : int
            Acquisition window start sample.
        s_acq_end : int
            Acquisition window end sample.
        hp_weights : np.ndarray or None
            High-pass filter weights (or None to skip filtering).

        Returns
        -------
        np.ndarray
            Residual (artifact) signal of length ``s_acq_end - s_acq_start``.
        """
        ch_data_acq = ch_data[s_acq_start:s_acq_end]

        ch_data_filtered = filtfilt(hp_weights, 1, ch_data_acq) if hp_weights is not None else ch_data_acq

        adjusted_triggers = triggers - s_acq_start
        # Small offset prevents epoch boundaries from sitting exactly on the trigger
        offset = int(artifact_length * 0.1)

        epochs = split_vector(ch_data_filtered, adjusted_triggers + offset, artifact_length)
        residuals_per_epoch = self._calc_pca(epochs)

        fitted_res = np.zeros(len(ch_data_acq))
        for i, trigger in enumerate(adjusted_triggers):
            start_pos = trigger + offset
            end_pos = start_pos + artifact_length
            if start_pos < 0:
                continue
            if end_pos > len(ch_data_acq):
                epoch_length = len(ch_data_acq) - start_pos
                if epoch_length <= 0:
                    continue
                fitted_res[start_pos:] = residuals_per_epoch[i, :epoch_length]
            else:
                fitted_res[start_pos:end_pos] = residuals_per_epoch[i, :]

        return fitted_res

    def _calc_pca(self, epochs: np.ndarray) -> np.ndarray:
        """Apply PCA to epochs and return the artifact residuals.

        Parameters
        ----------
        epochs : np.ndarray
            Epoch matrix of shape (n_epochs, n_times).

        Returns
        -------
        np.ndarray
            Residual (artifact) matrix of shape (n_epochs, n_times).
        """
        epochs_t = epochs.T

        col_var = np.var(epochs_t, axis=0)
        variance_threshold = 1e-12
        valid_mask = col_var > variance_threshold

        if np.count_nonzero(valid_mask) < 2:
            return np.zeros_like(epochs)

        X_valid = epochs_t[:, valid_mask]
        mean_valid = np.mean(X_valid, axis=0)
        std_valid = np.std(X_valid, axis=0, ddof=0)
        std_valid = np.where(std_valid < variance_threshold, 1.0, std_valid)
        X_centered = (X_valid - mean_valid) / std_valid

        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            logger.warning("PCA SVD failed ({}); skipping channel", exc)
            return np.zeros_like(epochs)

        max_components = min(X_centered.shape[0], X_centered.shape[1])
        if max_components <= 1:
            return np.zeros_like(epochs)

        n_components = self._select_n_components(S, max_components, X_centered.shape[0])

        U_reduced = U[:, :n_components]
        S_reduced = S[:n_components]
        Vt_reduced = Vt[:n_components, :]
        X_reconstructed_valid = ((U_reduced @ np.diag(S_reduced) @ Vt_reduced) * std_valid) + mean_valid

        residuals_valid = X_valid - X_reconstructed_valid
        residuals_full = np.zeros_like(epochs_t)
        residuals_full[:, valid_mask] = residuals_valid

        return residuals_full.T

    def _select_n_components(self, singular_values: np.ndarray, max_components: int, n_samples: int) -> int:
        """Determine the number of PCA components to retain.

        Parameters
        ----------
        singular_values : np.ndarray
            Singular values from the SVD decomposition.
        max_components : int
            Upper bound on the number of components.
        n_samples : int
            Number of samples (time points) used in the SVD.

        Returns
        -------
        int
            Number of components to retain (≥ 1).
        """
        if isinstance(self.n_components, int):
            return max(1, min(self.n_components, max_components))

        if isinstance(self.n_components, str):
            if self.n_components.lower() != "auto":
                raise ValueError("n_components as string must be 'auto'.")

            explained_var = (singular_values**2) / (n_samples - 1)
            explained_pct = 100.0 * (explained_var / np.sum(explained_var))
            n = self._select_n_components_auto(explained_pct)
            return max(1, min(int(n), max_components))

        if not 0 < self.n_components < 1:
            raise ValueError("n_components as float must be between 0 and 1.")

        explained_var = (singular_values**2) / (n_samples - 1)
        explained_ratio = np.cumsum(explained_var) / np.sum(explained_var)
        n = np.searchsorted(explained_ratio, self.n_components) + 1
        return max(1, min(int(n), max_components))

    def _select_n_components_auto(self, explained_pct: np.ndarray) -> int:
        """Select component count using MATLAB FACET OBS auto heuristics."""
        if len(explained_pct) == 0:
            return 1

        d_oev = np.where(np.abs(np.diff(explained_pct)) < self.TH_SLOPE)[0] + 1
        slope_pc = 1
        if len(d_oev) > 0:
            if len(d_oev) >= 4:
                dd_oev = np.diff(d_oev)
                run_pos = None
                for i in range(len(dd_oev) - 2):
                    if dd_oev[i] == 1 and dd_oev[i + 1] == 1 and dd_oev[i + 2] == 1:
                        run_pos = i
                        break
                idx = run_pos if run_pos is not None else 0
                slope_pc = int(max(d_oev[idx] - 1, 1))
            else:
                slope_pc = int(max(d_oev[0] - 1, 1))

        cumvar = np.cumsum(explained_pct)
        tmp = np.where(cumvar > self.TH_CUMVAR)[0]
        cumvar_pc = int(tmp[0] + 1) if len(tmp) > 0 else len(explained_pct)

        tmp = np.where(explained_pct < self.TH_VAREXP)[0]
        varexp_pc = int(max(tmp[0], 1)) if len(tmp) > 0 else len(explained_pct)

        pcs = int(np.floor(np.mean([slope_pc, cumvar_pc, varexp_pc])))
        return max(1, pcs)

    def _create_hp_filter(self, sfreq: float) -> np.ndarray:
        """Create Butterworth high-pass filter weights.

        Parameters
        ----------
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        np.ndarray
            Filter weights for use with ``scipy.signal.filtfilt``.
        """
        nyq = 0.5 * sfreq
        normalized_cutoff = self.hp_freq / nyq
        b, _ = butter(5, normalized_cutoff, btype="high")
        return b

    def _get_acquisition_window(self, context: ProcessingContext) -> tuple:
        """Return the start and end sample indices of the acquisition window.

        Parameters
        ----------
        context : ProcessingContext
            Current processing context.

        Returns
        -------
        tuple
            ``(s_acq_start, s_acq_end)`` as integers.
        """
        raw = context.get_raw()
        triggers = context.get_triggers()

        if len(triggers) == 0:
            return 0, raw.n_times

        artifact_length = context.get_artifact_length()
        if artifact_length is None:
            return 0, raw.n_times

        trigger_min = int(np.min(triggers))
        trigger_max = int(np.max(triggers))
        s_acq_start = max(0, trigger_min - artifact_length)
        s_acq_end = min(raw.n_times, trigger_max + artifact_length)
        if s_acq_end <= s_acq_start:
            return 0, raw.n_times
        return s_acq_start, s_acq_end
