"""
Adaptive Noise Cancellation (ANC) correction processor.
"""

from typing import Any

import mne
import numpy as np
from loguru import logger
from scipy.signal import butter, filtfilt, firls

from ..console import processor_progress
from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor


@register_processor
class ANCCorrection(Processor):
    """Remove residual fMRI artifacts using Adaptive Noise Cancellation.

    Uses the estimated noise from a prior correction step (e.g. AAS) as a
    reference signal. The LMS adaptive filter iteratively minimises the
    residual between the EEG and a scaled, filtered copy of the reference,
    yielding a per-channel filtered-noise estimate that is subtracted from
    the EEG.

    The algorithm:

    1. High-pass filters EEG and reference to remove DC / low-frequency drift.
    2. Scales the reference to match EEG amplitude (Alpha factor).
    3. Adapts filter coefficients using the LMS algorithm.
    4. Subtracts the filtered noise from the EEG.

    Parameters
    ----------
    filter_order : int, optional
        Adaptive filter order. Defaults to artifact length derived from context.
    hp_freq : float, optional
        High-pass cutoff frequency in Hz. Auto-derived from trigger rate when
        not specified.
    hp_filter_weights : np.ndarray, optional
        Pre-computed FIR filter weights; overrides ``hp_freq`` when provided.
    use_c_extension : bool
        Use the optional fastranc C extension for speed (default: True).
        Falls back to the pure-Python LMS implementation automatically.
    mu_factor : float
        Learning-rate numerator; actual µ = mu_factor / (N × var(ref))
        (default: 0.05).
    max_gain : float
        Maximum allowed ratio of filtered-noise amplitude to EEG amplitude.
        Corrections exceeding this are discarded (default: 50.0).
    """

    name = "anc_correction"
    description = "Adaptive Noise Cancellation for residual artifacts"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = True
    parallel_safe = False
    channel_wise = True

    def __init__(
        self,
        filter_order: int | None = None,
        hp_freq: float | None = None,
        hp_filter_weights: np.ndarray | None = None,
        use_c_extension: bool = True,
        mu_factor: float = 0.05,
        max_gain: float = 50.0,
    ) -> None:
        self.filter_order_override = max(1, int(filter_order)) if filter_order is not None else None
        self.filter_order = self.filter_order_override
        self.hp_freq = hp_freq
        self.hp_filter_weights = hp_filter_weights
        self.use_c_extension = use_c_extension
        self.mu_factor = mu_factor
        self.max_gain = max_gain
        self._fastranc_available = None
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if not context.has_estimated_noise():
            raise ProcessorValidationError("Estimated noise not available. Run AAS or other correction first.")
        if not context.has_triggers():
            raise ProcessorValidationError("Triggers not set. Run TriggerDetector first.")
        artifact_length = context.get_artifact_length()
        if artifact_length is None or artifact_length <= 0:
            raise ProcessorValidationError("Artifact length not set. Run TriggerDetector before ANC.")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()
        estimated_noise = context.get_estimated_noise()
        sfreq = context.get_sfreq()
        artifact_length = context.get_artifact_length()
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

        # --- LOG ---
        logger.info("Applying ANC to {} channels", len(eeg_channels))

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found, skipping ANC")
            return context

        # --- COMPUTE ---
        hp_weights, hp_cutoff = self._resolve_hp_filter(context, artifact_length, sfreq)
        s_acq_start, s_acq_end = self._get_acquisition_window(context)
        filter_order = self._resolve_filter_order(artifact_length, s_acq_start, s_acq_end)

        if self.use_c_extension and self._fastranc_available is None:
            self._fastranc_available = self._check_fastranc()

        noise_updated = estimated_noise.copy()

        with processor_progress(
            total=len(eeg_channels) or None,
            message="Adaptive noise cancellation",
        ) as progress:
            for idx, ch_idx in enumerate(eeg_channels):
                ch_name = raw.ch_names[ch_idx]
                try:
                    corrected, filtered = self._anc_single_channel(
                        raw._data[ch_idx],
                        estimated_noise[ch_idx],
                        s_acq_start,
                        s_acq_end,
                        hp_weights,
                        filter_order,
                        ch_name,
                    )
                    raw._data[ch_idx] = corrected
                    noise_updated[ch_idx, s_acq_start:s_acq_end] += filtered
                    status = f"{idx + 1}/{len(eeg_channels)} • {ch_name}"
                except Exception as exc:
                    logger.error("ANC failed for channel {}: {}", ch_name, exc)
                    status = f"{idx + 1}/{len(eeg_channels)} • {ch_name} (skipped)"
                progress.advance(1, message=status)

        # --- NOISE ---
        new_ctx = context.with_raw(
            raw,
            estimated_noise=noise_updated,
            copy_estimated_noise=False,
        )

        # --- BUILD RESULT + RETURN ---
        logger.info("ANC correction completed")
        return self._with_anc_metadata(new_ctx, hp_cutoff, filter_order)

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    def _resolve_hp_filter(
        self,
        context: ProcessingContext,
        artifact_length: int,
        sfreq: float,
    ) -> tuple:
        """Select or derive the high-pass filter weights and cutoff.

        Parameters
        ----------
        context : ProcessingContext
            Current processing context.
        artifact_length : int
            Artifact length in samples (used when deriving parameters).
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        tuple
            ``(hp_weights, hp_cutoff_hz)`` where ``hp_weights`` is an
            np.ndarray or None, and ``hp_cutoff_hz`` is a float.
        """
        if self.hp_filter_weights is not None:
            derived_freq = self._derive_parameters(context, artifact_length, sfreq)["hp_freq"]
            hp_cutoff = self.hp_freq if self.hp_freq is not None else derived_freq
            return self.hp_filter_weights, hp_cutoff

        if self.hp_freq is not None and self.hp_freq > 0:
            return self._design_highpass(self.hp_freq, sfreq), self.hp_freq

        derived = self._derive_parameters(context, artifact_length, sfreq)
        return derived["hp_weights"], derived["hp_freq"]

    def _resolve_filter_order(self, artifact_length: int, s_acq_start: int, s_acq_end: int) -> int:
        """Determine the adaptive filter order, capped by the acquisition window.

        Parameters
        ----------
        artifact_length : int
            Default filter order when no override is provided.
        s_acq_start : int
            Acquisition window start sample.
        s_acq_end : int
            Acquisition window end sample.

        Returns
        -------
        int
            Effective adaptive filter order (≥ 1).
        """
        window_length = max(1, s_acq_end - s_acq_start)
        base_order = self.filter_order_override if self.filter_order_override is not None else artifact_length
        return max(1, min(int(base_order), window_length))

    def _with_anc_metadata(self, ctx: ProcessingContext, hp_cutoff: float, filter_order: int) -> ProcessingContext:
        """Return a new context with ANC diagnostics stored in custom metadata.

        Parameters
        ----------
        ctx : ProcessingContext
            Context to augment.
        hp_cutoff : float
            Effective high-pass cutoff used during processing (Hz).
        filter_order : int
            Effective adaptive filter order used during processing.

        Returns
        -------
        ProcessingContext
            New context with ``metadata.custom["anc"]`` populated.
        """
        new_metadata = ctx.metadata.copy()
        new_metadata.custom["anc"] = {
            "hp_frequency_hz": hp_cutoff,
            "filter_order": filter_order,
            "mu_factor": self.mu_factor,
            "max_gain": self.max_gain,
            "used_c_extension": bool(self._fastranc_available),
        }
        return ctx.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )

    def _anc_single_channel(
        self,
        eeg_data: np.ndarray,
        noise_data: np.ndarray,
        s_acq_start: int,
        s_acq_end: int,
        hp_weights: np.ndarray | None,
        filter_order: int,
        ch_name: str,
    ) -> tuple:
        """Apply ANC to a single channel.

        Parameters
        ----------
        eeg_data : np.ndarray
            Full EEG time series for one channel.
        noise_data : np.ndarray
            Estimated noise time series for the same channel.
        s_acq_start : int
            Acquisition window start sample.
        s_acq_end : int
            Acquisition window end sample.
        hp_weights : np.ndarray or None
            High-pass FIR filter weights (or None to skip filtering).
        filter_order : int
            Adaptive filter order.
        ch_name : str
            Channel name used in warning messages.

        Returns
        -------
        tuple
            ``(corrected_data, filtered_noise)`` both as np.ndarray.
        """
        reference = noise_data[s_acq_start:s_acq_end].astype(float, copy=True)
        segment_len = reference.size

        if segment_len == 0:
            logger.debug("[{}] ANC reference window is empty, skipping", ch_name)
            return eeg_data, np.zeros(0, dtype=float)
        if segment_len <= filter_order:
            logger.debug("[{}] ANC reference shorter than filter order, skipping", ch_name)
            return eeg_data, np.zeros(segment_len, dtype=float)

        if hp_weights is not None:
            data = filtfilt(hp_weights, 1, eeg_data, axis=0, padtype="odd")
            data = data[s_acq_start:s_acq_end].astype(float)
        else:
            data = eeg_data[s_acq_start:s_acq_end].astype(float)

        ref_energy = np.sum(reference * reference)
        if not np.isfinite(ref_energy) or ref_energy <= np.finfo(float).eps:
            logger.debug("[{}] Reference energy too small, skipping ANC", ch_name)
            return eeg_data, np.zeros(segment_len, dtype=float)

        alpha = np.sum(data * reference) / ref_energy
        if not np.isfinite(alpha):
            logger.debug("[{}] Alpha scaling not finite, skipping ANC", ch_name)
            return eeg_data, np.zeros(segment_len, dtype=float)

        reference = (alpha * reference).astype(float)

        var_ref = np.var(reference)
        if not np.isfinite(var_ref) or var_ref <= np.finfo(float).eps:
            logger.debug("[{}] Reference variance is zero, skipping ANC", ch_name)
            return eeg_data, np.zeros(segment_len, dtype=float)

        mu = float(self.mu_factor / (filter_order * var_ref))
        if not np.isfinite(mu) or mu <= 0:
            logger.debug("[{}] Computed ANC learning rate invalid, skipping", ch_name)
            return eeg_data, np.zeros(segment_len, dtype=float)

        if self._fastranc_available:
            filtered_noise = self._anc_fast(reference, data, mu, filter_order)
        else:
            filtered_noise = self._anc_python(reference, data, mu, filter_order)

        max_filtered = np.max(np.abs(filtered_noise))
        if not np.isfinite(max_filtered):
            logger.error("[{}] ANC produced invalid values (inf/nan), skipping", ch_name)
            return eeg_data, np.zeros(segment_len, dtype=float)

        eeg_segment = eeg_data[s_acq_start:s_acq_end]
        baseline = np.max(np.abs(eeg_segment)) if eeg_segment.size else 0.0
        gain = max_filtered / baseline if baseline > 0 else np.inf if max_filtered > 0 else 0.0

        if gain > self.max_gain:
            logger.error("[{}] ANC produced unstable gain ({:.2e}), skipping", ch_name, gain)
            return eeg_data, np.zeros(segment_len, dtype=float)

        corrected_data = eeg_data.copy()
        corrected_data[s_acq_start:s_acq_end] -= filtered_noise
        return corrected_data, filtered_noise

    def _anc_fast(
        self,
        reference: np.ndarray,
        data: np.ndarray,
        mu: float,
        filter_order: int,
    ) -> np.ndarray:
        """Apply ANC using the fastranc C extension.

        Parameters
        ----------
        reference : np.ndarray
            Scaled reference (noise) signal.
        data : np.ndarray
            EEG signal for the acquisition window.
        mu : float
            LMS learning rate.
        filter_order : int
            Adaptive filter order.

        Returns
        -------
        np.ndarray
            Filtered noise signal.
        """
        # Optional C extension — kept as lazy import so a missing build does not
        # prevent the module from being imported.
        from ..helpers.fastranc import fastr_anc

        _, filtered_noise = fastr_anc(reference, data, filter_order, mu)
        return filtered_noise

    def _anc_python(
        self,
        reference: np.ndarray,
        data: np.ndarray,
        mu: float,
        filter_order: int,
    ) -> np.ndarray:
        """Apply ANC using the pure-Python LMS fallback.

        Implements the standard Least Mean Squares (LMS) adaptive filter.

        Parameters
        ----------
        reference : np.ndarray
            Scaled reference (noise) signal.
        data : np.ndarray
            EEG signal for the acquisition window.
        mu : float
            LMS learning rate.
        filter_order : int
            Adaptive filter order.

        Returns
        -------
        np.ndarray
            Filtered noise signal.
        """
        N = max(1, int(filter_order))
        length = len(reference)
        w = np.zeros(N)
        y = np.zeros(length)

        for n in range(N, length):
            x = reference[n - N : n][::-1]
            y[n] = np.dot(w, x)
            e = data[n] - y[n]
            w += mu * e * x

        return y

    def _derive_parameters(
        self,
        context: ProcessingContext,
        artifact_length: int,
        sfreq: float,
    ) -> dict[str, Any]:
        """Derive ANC parameters from the trigger rate and sampling frequency.

        Parameters
        ----------
        context : ProcessingContext
            Current processing context (used to read triggers).
        artifact_length : int
            Artifact length in samples; used as default filter order.
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        dict
            Dictionary with keys ``hp_freq``, ``hp_weights``, ``filter_order``.
        """
        triggers = context.get_triggers()
        if triggers is None:
            triggers = np.array([], dtype=int)

        if len(triggers) >= 2:
            cutoff_samples = int(sfreq)
            count = 1
            while count < len(triggers):
                if triggers[count] - triggers[0] >= cutoff_samples:
                    break
                count += 1
            trigger_rate = max(count, 1)
        else:
            trigger_rate = 1

        hp_freq = max(0.75 * trigger_rate if trigger_rate >= 1 else 2.0, 0.5)
        hp_weights = self._design_highpass(hp_freq, sfreq)
        filter_order = max(artifact_length, 1)

        return {"hp_freq": hp_freq, "hp_weights": hp_weights, "filter_order": filter_order}

    def _design_highpass(self, cutoff_hz: float, sfreq: float) -> np.ndarray:
        """Design a high-pass FIR filter using ``firls``.

        Falls back to a 5th-order Butterworth filter if the FIR design fails.

        Parameters
        ----------
        cutoff_hz : float
            Desired high-pass cutoff in Hz.
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        np.ndarray
            Filter weights suitable for use with ``scipy.signal.filtfilt``.
        """
        nyq = 0.5 * sfreq
        cutoff_hz = min(max(cutoff_hz, 0.5), nyq * 0.95)
        trans = 0.15
        pass_edge = cutoff_hz / nyq
        stop_edge = min(max(pass_edge * (1 - trans), 0.0), pass_edge * 0.999)
        taps = max(int(round(1.2 * sfreq / (cutoff_hz * (1 - trans)))) | 1, 3)

        f = [0.0, stop_edge, pass_edge, 1.0]
        a = [0.0, 0.0, 1.0, 1.0]

        try:
            weights = firls(taps, f, a)
        except Exception as exc:
            logger.warning("FIR design failed ({}); falling back to Butterworth", exc)
            b, _ = butter(5, pass_edge, btype="high")
            weights = b

        return weights.astype(float)

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

    def _check_fastranc(self) -> bool:
        """Check whether the fastranc C extension is importable.

        Returns
        -------
        bool
            True if the extension is available, False otherwise.
        """
        try:
            from ..helpers.fastranc import fastranc

            if fastranc is not None:
                logger.debug("Using fastranc C extension for ANC")
                return True
            logger.info("fastranc C extension not available, using Python fallback")
            return False
        except Exception as exc:
            logger.info("fastranc C extension not available ({}), using Python fallback", exc)
            return False


# Alias for backwards compatibility
AdaptiveNoiseCancellation = ANCCorrection
