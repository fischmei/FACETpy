"""MATLAB-style prefilter processor."""

from __future__ import annotations

from collections.abc import Sequence

import mne
import numpy as np
from loguru import logger

from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor


@register_processor
class MATLABPreFilter(Processor):
    """Apply MATLAB FACET-style FFT prefiltering before artifact correction.

    Supports:
    - Piecewise-linear transfer-function filtering in frequency domain.
    - Gaussian high-pass filtering (DC-safe) in frequency domain.
    - Optional padded filtering inside the acquisition window.

    Parameters
    ----------
    lp_frequency : float or sequence of float, optional
        Low-pass cutoff in Hz (scalar or per-channel values). ``None`` means
        no LP from generated transfer functions.
    hp_frequency : float or sequence of float, optional
        High-pass cutoff in Hz (scalar or per-channel values). ``None`` means
        no HP from generated transfer functions.
    transfer_frequencies : np.ndarray or sequence of np.ndarray, optional
        Custom normalized frequency points in ``[0, 1]`` for transfer-function
        filtering. If set, overrides LP/HP synthesis.
    transfer_amplitudes : np.ndarray or sequence of np.ndarray, optional
        Amplitude points corresponding to ``transfer_frequencies``.
    gauss_hp_frequency : float or sequence of float, optional
        Gaussian high-pass cutoff in Hz (scalar or per-channel values).
    picks : str or sequence of str
        Channels to filter (default: ``"eeg"``).
    pad_acquisition : bool
        Use acquisition-window padding during filtering (default: True).
    """

    name = "matlab_prefilter"
    description = "Apply MATLAB FACET-style FFT/gaussian prefilter"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True
    channel_wise = True

    def __init__(
        self,
        lp_frequency: float | Sequence[float] | None = None,
        hp_frequency: float | Sequence[float] | None = None,
        transfer_frequencies: np.ndarray | Sequence[np.ndarray] | None = None,
        transfer_amplitudes: np.ndarray | Sequence[np.ndarray] | None = None,
        gauss_hp_frequency: float | Sequence[float] | None = None,
        picks: str | Sequence[str] = "eeg",
        pad_acquisition: bool = True,
    ) -> None:
        self.lp_frequency = lp_frequency
        self.hp_frequency = hp_frequency
        self.transfer_frequencies = transfer_frequencies
        self.transfer_amplitudes = transfer_amplitudes
        self.gauss_hp_frequency = gauss_hp_frequency
        self.picks = picks
        self.pad_acquisition = pad_acquisition
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if self.transfer_frequencies is not None and self.transfer_amplitudes is None:
            raise ProcessorValidationError("transfer_amplitudes must be provided with transfer_frequencies.")
        if self.transfer_amplitudes is not None and self.transfer_frequencies is None:
            raise ProcessorValidationError("transfer_frequencies must be provided with transfer_amplitudes.")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()
        sfreq = float(raw.info["sfreq"])
        target_indices = self._resolve_target_indices(raw)
        acq_start, acq_end = self._resolve_acquisition_bounds(context)

        # --- LOG ---
        logger.info(
            "Applying MATLABPreFilter to {} channels (pad_acquisition={})",
            len(target_indices),
            self.pad_acquisition,
        )

        # --- COMPUTE ---
        for i, ch_idx in enumerate(target_indices):
            ch_name = raw.ch_names[ch_idx]
            raw._data[ch_idx] = self._filter_channel(
                data=raw._data[ch_idx],
                sfreq=sfreq,
                channel_pos=i,
                context=context,
                acq_start=acq_start,
                acq_end=acq_end,
            )
            logger.debug("MATLABPreFilter processed channel {}", ch_name)

        # --- NOISE ---
        estimated_noise = None
        if context.has_estimated_noise():
            estimated_noise = context.get_estimated_noise().copy()
            for i, ch_idx in enumerate(target_indices):
                estimated_noise[ch_idx] = self._filter_channel(
                    data=estimated_noise[ch_idx],
                    sfreq=sfreq,
                    channel_pos=i,
                    context=context,
                    acq_start=acq_start,
                    acq_end=acq_end,
                )

        result = context.with_raw(
            raw,
            estimated_noise=estimated_noise,
            copy_estimated_noise=False,
        )

        # --- METADATA ---
        md = result.metadata.copy()
        md.custom["matlab_prefilter"] = {
            "pad_acquisition": self.pad_acquisition,
            "num_channels": len(target_indices),
            "acq_start_sample": int(acq_start),
            "acq_end_sample": int(acq_end),
        }
        return result.with_metadata(
            md,
            copy_estimated_noise=False,
        )

    def _resolve_target_indices(self, raw: mne.io.Raw) -> list[int]:
        """Resolve picks to channel indices."""
        try:
            from mne.io.pick import _picks_to_idx

            indices = _picks_to_idx(
                raw.info,
                self.picks,
                none="all",
                exclude=(),
            )
        except Exception as exc:
            raise ProcessorValidationError(f"Invalid picks '{self.picks}': {exc}") from exc
        return [int(index) for index in indices]

    def _resolve_acquisition_bounds(self, context: ProcessingContext) -> tuple[int, int]:
        """Resolve acquisition bounds for padded filtering."""
        raw = context.get_raw()
        md = context.metadata
        n_times = raw.n_times

        if md.acq_start_sample is not None and md.acq_end_sample is not None:
            return max(0, int(md.acq_start_sample)), min(n_times - 1, int(md.acq_end_sample))

        if context.has_triggers() and context.get_artifact_length():
            triggers = context.get_triggers()
            pre = md.pre_trigger_samples
            post = md.post_trigger_samples
            if pre is None:
                offset = int(round(md.artifact_to_trigger_offset * context.get_sfreq()))
                pre = max(0, -offset) if offset < 0 else 0
            if post is None:
                post = max(int(context.get_artifact_length()) - int(pre) - 1, 0)
            start = max(0, int(triggers[0] - pre))
            end = min(n_times - 1, int(triggers[-1] + post))
            return start, end

        return 0, n_times - 1

    def _filter_channel(
        self,
        data: np.ndarray,
        sfreq: float,
        channel_pos: int,
        context: ProcessingContext,
        acq_start: int,
        acq_end: int,
    ) -> np.ndarray:
        """Apply configured prefiltering to one channel."""
        out = data.astype(float, copy=True)

        if not self.pad_acquisition or acq_start >= acq_end:
            return self._apply_filters(out, sfreq, channel_pos)

        if acq_start > 0:
            out[:acq_start] = self._apply_filters(out[:acq_start], sfreq, channel_pos)
        out[acq_start : acq_end + 1] = self._apply_filters_padded(
            out[acq_start : acq_end + 1],
            sfreq,
            channel_pos,
            context,
        )
        if acq_end < len(out) - 1:
            out[acq_end + 1 :] = self._apply_filters(out[acq_end + 1 :], sfreq, channel_pos)

        return out

    def _apply_filters_padded(
        self,
        segment: np.ndarray,
        sfreq: float,
        channel_pos: int,
        context: ProcessingContext,
    ) -> np.ndarray:
        """Apply filters with repeated-edge padding during acquisition."""
        if len(segment) < 4:
            return self._apply_filters(segment, sfreq, channel_pos)

        triggers = context.get_triggers() if context.has_triggers() else None
        if triggers is None or len(triggers) < 2:
            return self._apply_filters(segment, sfreq, channel_pos)

        art_len_start = max(1, int(triggers[1] - triggers[0]))
        art_len_end = max(1, int(triggers[-1] - triggers[-2]))
        art_len_start = min(art_len_start, len(segment))
        art_len_end = min(art_len_end, len(segment))

        art_s = segment[:art_len_start]
        art_e = segment[-art_len_end:]
        num = int(np.ceil(sfreq / max(art_len_start, 1)) + 1)

        padded = np.concatenate([np.tile(art_s, num), segment, np.tile(art_e, num)])
        filtered = self._apply_filters(padded, sfreq, channel_pos)
        return filtered[(len(art_s) * num) : (len(filtered) - len(art_e) * num)]

    def _apply_filters(self, signal: np.ndarray, sfreq: float, channel_pos: int) -> np.ndarray:
        """Apply transfer-function and Gaussian HP filters to one signal."""
        if len(signal) < 2:
            return signal

        out = signal
        tf = self._resolve_transfer_function(channel_pos, sfreq)
        if tf is not None:
            f, a = tf
            out = self._fft_transfer_filter(out, f, a)

        gauss_hp = self._resolve_per_channel(self.gauss_hp_frequency, channel_pos)
        if gauss_hp is not None and gauss_hp > 0:
            out = self._fft_gaussian_highpass(out, float(gauss_hp), sfreq)

        return np.real(out)

    def _resolve_transfer_function(self, channel_pos: int, sfreq: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Return normalized transfer-function vectors for one channel."""
        if self.transfer_frequencies is not None:
            if isinstance(self.transfer_frequencies, np.ndarray) and self.transfer_frequencies.ndim == 1:
                freqs = self.transfer_frequencies
            elif (
                isinstance(self.transfer_frequencies, (list, tuple))
                and len(self.transfer_frequencies) > 0
                and isinstance(self.transfer_frequencies[0], (list, tuple, np.ndarray))
            ):
                freqs = self._resolve_per_channel(self.transfer_frequencies, channel_pos)
            else:
                freqs = self._resolve_per_channel(self.transfer_frequencies, channel_pos)

            if isinstance(self.transfer_amplitudes, np.ndarray) and self.transfer_amplitudes.ndim == 1:
                ampls = self.transfer_amplitudes
            elif (
                isinstance(self.transfer_amplitudes, (list, tuple))
                and len(self.transfer_amplitudes) > 0
                and isinstance(self.transfer_amplitudes[0], (list, tuple, np.ndarray))
            ):
                ampls = self._resolve_per_channel(self.transfer_amplitudes, channel_pos)
            else:
                ampls = self._resolve_per_channel(self.transfer_amplitudes, channel_pos)

            if freqs is None or ampls is None:
                return None
            f = np.asarray(freqs, dtype=float).ravel()
            a = np.asarray(ampls, dtype=float).ravel()
            self._validate_transfer_function(f, a)
            return f, a

        lp = self._resolve_per_channel(self.lp_frequency, channel_pos)
        hp = self._resolve_per_channel(self.hp_frequency, channel_pos)
        nyq = sfreq / 2.0

        lp = float(lp) if lp is not None else 0.0
        hp = float(hp) if hp is not None else 0.0
        if lp <= 0 and hp <= 0:
            return None

        if lp > 0 and hp > 0 and hp < lp:
            f = np.array([0, hp * 0.99, hp * 1.01, lp * 0.99, lp * 1.01, nyq], dtype=float) / nyq
            a = np.array([0, 0, 1, 1, 0, 0], dtype=float)
        elif lp > 0 and (hp <= 0 or hp >= lp):
            f = np.array([0, lp * 0.95, lp * 1.05, nyq], dtype=float) / nyq
            a = np.array([1, 1, 0, 0], dtype=float)
        else:
            f = np.array([0, hp * 0.95, hp * 1.05, nyq], dtype=float) / nyq
            a = np.array([0, 0, 1, 1], dtype=float)

        f = np.clip(f, 0.0, 1.0)
        if np.any(np.diff(f) < 0):
            f = np.maximum.accumulate(f)
        return f, a

    def _resolve_per_channel(self, value, channel_pos: int):
        """Resolve scalar/vector parameter values per channel position."""
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.ndim == 1 and np.issubdtype(value.dtype, np.number):
                if len(value) == 0:
                    return None
                if len(value) > channel_pos:
                    return value[channel_pos]
                return value[-1]
            if value.ndim == 1 and len(value) > 0 and isinstance(value[0], (np.ndarray, list, tuple)):
                return value[channel_pos] if len(value) > channel_pos else value[-1]
            return value
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            if isinstance(value[0], (list, tuple, np.ndarray)):
                return value[channel_pos] if len(value) > channel_pos else value[-1]
            return value[channel_pos] if len(value) > channel_pos else value[-1]
        return value

    @staticmethod
    def _validate_transfer_function(f: np.ndarray, a: np.ndarray) -> None:
        """Validate normalized transfer-function vectors."""
        if f.ndim != 1 or a.ndim != 1 or len(f) != len(a):
            raise ProcessorValidationError(
                "transfer_frequencies and transfer_amplitudes must be 1D vectors of same length."
            )
        if len(f) < 2:
            raise ProcessorValidationError("transfer_frequencies must contain at least 2 points.")
        if f[0] != 0.0 or f[-1] != 1.0:
            raise ProcessorValidationError("transfer_frequencies must start at 0.0 and end at 1.0.")
        if np.any(np.diff(f) < 0):
            raise ProcessorValidationError("transfer_frequencies must be sorted ascending.")
        if np.min(f) < 0.0 or np.max(f) > 1.0:
            raise ProcessorValidationError("transfer_frequencies must be in [0, 1].")

    @staticmethod
    def _fft_transfer_filter(signal: np.ndarray, f: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Apply piecewise-linear transfer function in the FFT domain."""
        n = len(signal)
        X = np.fft.rfft(signal)
        w = np.linspace(0.0, 1.0, len(X))
        H = np.interp(w, f, a)
        y = np.fft.irfft(X * H, n=n)
        return y

    @staticmethod
    def _fft_gaussian_highpass(signal: np.ndarray, cutoff_hz: float, sfreq: float) -> np.ndarray:
        """Apply MATLAB-style Gaussian high-pass in the FFT domain."""
        n = len(signal)
        X = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        sigma = cutoff_hz / (2 * np.sqrt(-np.log(1 - 1 / np.sqrt(2))))
        H = 1.0 - np.exp(-((freqs / (2 * sigma)) ** 2))
        y = np.fft.irfft(X * H, n=n)
        return y
