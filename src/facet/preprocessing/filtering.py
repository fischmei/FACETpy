"""
Filtering Processors Module

This module contains processors for filtering EEG data.

Author: FACETpy Team
Date: 2025-01-12
"""

import mne
from loguru import logger

from ..core import ProcessingContext, Processor, register_processor
from ..logging_config import suppress_stdout


@register_processor
class Filter(Processor):
    """Apply MNE's generic band/high/low-pass filter to EEG data.

    Wraps :meth:`mne.io.Raw.filter` with configurable FIR/IIR parameters.
    If a noise estimate is present in the context, the same filter is applied
    to the noise so that downstream evaluation steps remain consistent.

    Parameters
    ----------
    l_freq : float, optional
        Low cutoff frequency in Hz (``None`` for no highpass).
    h_freq : float, optional
        High cutoff frequency in Hz (``None`` for no lowpass).
    picks : str or list of str, optional
        Channels to filter.  ``None`` filters all channels.
    filter_length : str, optional
        Length of the FIR filter (default: ``'auto'``).
    method : str, optional
        Filter method, ``'fir'`` or ``'iir'`` (default: ``'fir'``).
    phase : str, optional
        Phase of the filter (default: ``'zero'``).
    fir_window : str, optional
        Window function for FIR filter (default: ``'hamming'``).
    verbose : bool, optional
        MNE verbosity flag passed to ``raw.filter()`` (default: ``False``).
    """

    name = "filter"
    description = "Apply generic filter to EEG data"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True
    channel_wise = True

    def __init__(
        self,
        l_freq: float | None = None,
        h_freq: float | None = None,
        picks: str | list[str] | None = None,
        filter_length: str = "auto",
        method: str = "fir",
        phase: str = "zero",
        fir_window: str = "hamming",
        verbose: bool = False,
    ):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.picks = picks
        self.filter_length = filter_length
        self.method = method
        self.phase = phase
        self.fir_window = fir_window
        self.verbose = verbose
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()

        # --- LOG ---
        if self.l_freq and self.h_freq:
            filter_type = f"bandpass ({self.l_freq}-{self.h_freq}Hz)"
        elif self.l_freq:
            filter_type = f"highpass ({self.l_freq}Hz)"
        elif self.h_freq:
            filter_type = f"lowpass ({self.h_freq}Hz)"
        else:
            filter_type = "no filter"
        logger.info("Applying {}", filter_type)

        # --- COMPUTE ---
        raw.filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            picks=self.picks,
            filter_length=self.filter_length,
            method=self.method,
            phase=self.phase,
            fir_window=self.fir_window,
            verbose=self.verbose,
        )

        # --- NOISE ---
        estimated_noise = None
        if context.has_estimated_noise():
            noise = context.get_estimated_noise().copy()
            with suppress_stdout():
                noise_raw = mne.io.RawArray(noise, raw.info)
            noise_raw.filter(
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                picks=self.picks,
                filter_length=self.filter_length,
                method=self.method,
                phase=self.phase,
                fir_window=self.fir_window,
                verbose=False,
            )
            estimated_noise = noise_raw._data
        else:
            logger.debug("No noise estimate present — skipping noise propagation")

        # --- RETURN ---
        return context.with_raw(
            raw,
            estimated_noise=estimated_noise,
            copy_estimated_noise=False,
        )


@register_processor
class HighPassFilter(Filter):
    """Apply a highpass filter to EEG data.

    Convenience subclass of :class:`Filter` that exposes a single ``freq``
    parameter instead of separate ``l_freq``/``h_freq``.

    Parameters
    ----------
    freq : float
        Highpass cutoff frequency in Hz.
    picks : str or list of str, optional
        Channels to filter.  ``None`` filters all channels.
    filter_length : str, optional
        Length of the FIR filter (default: ``'auto'``).
    method : str, optional
        Filter method, ``'fir'`` or ``'iir'`` (default: ``'fir'``).
    phase : str, optional
        Phase of the filter (default: ``'zero'``).
    fir_window : str, optional
        Window function for FIR filter (default: ``'hamming'``).
    """

    name = "highpass_filter"
    description = "Apply highpass filter to EEG data"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(
        self,
        freq: float,
        picks: str | list[str] | None = None,
        filter_length: str = "auto",
        method: str = "fir",
        phase: str = "zero",
        fir_window: str = "hamming",
    ):
        super().__init__(
            l_freq=freq,
            h_freq=None,
            picks=picks,
            filter_length=filter_length,
            method=method,
            phase=phase,
            fir_window=fir_window,
        )

    def _get_parameters(self):
        """Expose user-facing freq parameter for history/serialization."""
        return {
            "freq": self.l_freq,
            "picks": self.picks,
            "filter_length": self.filter_length,
            "method": self.method,
            "phase": self.phase,
            "fir_window": self.fir_window,
        }


@register_processor
class LowPassFilter(Filter):
    """Apply a lowpass filter to EEG data.

    Convenience subclass of :class:`Filter` that exposes a single ``freq``
    parameter instead of separate ``l_freq``/``h_freq``.

    Parameters
    ----------
    freq : float
        Lowpass cutoff frequency in Hz.
    picks : str or list of str, optional
        Channels to filter.  ``None`` filters all channels.
    filter_length : str, optional
        Length of the FIR filter (default: ``'auto'``).
    method : str, optional
        Filter method, ``'fir'`` or ``'iir'`` (default: ``'fir'``).
    phase : str, optional
        Phase of the filter (default: ``'zero'``).
    fir_window : str, optional
        Window function for FIR filter (default: ``'hamming'``).
    """

    name = "lowpass_filter"
    description = "Apply lowpass filter to EEG data"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(
        self,
        freq: float,
        picks: str | list[str] | None = None,
        filter_length: str = "auto",
        method: str = "fir",
        phase: str = "zero",
        fir_window: str = "hamming",
    ):
        super().__init__(
            l_freq=None,
            h_freq=freq,
            picks=picks,
            filter_length=filter_length,
            method=method,
            phase=phase,
            fir_window=fir_window,
        )

    def _get_parameters(self):
        """Expose user-facing freq parameter for history/serialization."""
        return {
            "freq": self.h_freq,
            "picks": self.picks,
            "filter_length": self.filter_length,
            "method": self.method,
            "phase": self.phase,
            "fir_window": self.fir_window,
        }


@register_processor
class BandPassFilter(Filter):
    """Apply a bandpass filter to EEG data.

    Convenience subclass of :class:`Filter` that requires both ``l_freq``
    and ``h_freq`` to be specified.

    Parameters
    ----------
    l_freq : float
        Low cutoff frequency in Hz.
    h_freq : float
        High cutoff frequency in Hz.
    picks : str or list of str, optional
        Channels to filter.  ``None`` filters all channels.
    filter_length : str, optional
        Length of the FIR filter (default: ``'auto'``).
    method : str, optional
        Filter method, ``'fir'`` or ``'iir'`` (default: ``'fir'``).
    phase : str, optional
        Phase of the filter (default: ``'zero'``).
    fir_window : str, optional
        Window function for FIR filter (default: ``'hamming'``).
    """

    name = "bandpass_filter"
    description = "Apply bandpass filter to EEG data"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(
        self,
        l_freq: float,
        h_freq: float,
        picks: str | list[str] | None = None,
        filter_length: str = "auto",
        method: str = "fir",
        phase: str = "zero",
        fir_window: str = "hamming",
    ):
        super().__init__(
            l_freq=l_freq,
            h_freq=h_freq,
            picks=picks,
            filter_length=filter_length,
            method=method,
            phase=phase,
            fir_window=fir_window,
        )


@register_processor
class NotchFilter(Processor):
    """Apply a notch filter to remove line noise from EEG data.

    Wraps :meth:`mne.io.Raw.notch_filter` with configurable FIR/IIR
    parameters.  Typical use is to remove 50 Hz or 60 Hz mains interference
    and their harmonics.  If a noise estimate is present in the context, the
    same filter is applied to keep evaluation results consistent.

    Parameters
    ----------
    freqs : float or list of float
        Frequencies to notch out in Hz (e.g., ``[50, 100]``).
    picks : str or list of str, optional
        Channels to filter.  ``None`` filters all channels.
    filter_length : str, optional
        Length of the FIR filter (default: ``'auto'``).
    notch_widths : float or list of float, optional
        Width of each notch filter.  ``None`` uses MNE defaults.
    method : str, optional
        Filter method, ``'fir'`` or ``'iir'`` (default: ``'fir'``).
    phase : str, optional
        Phase of the filter (default: ``'zero'``).
    fir_window : str, optional
        Window function for FIR filter (default: ``'hamming'``).
    """

    name = "notch_filter"
    description = "Apply notch filter to remove line noise"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True
    channel_wise = True

    def __init__(
        self,
        freqs: float | list[float],
        picks: str | list[str] | None = None,
        filter_length: str = "auto",
        notch_widths: float | list[float] | None = None,
        method: str = "fir",
        phase: str = "zero",
        fir_window: str = "hamming",
    ):
        self.freqs = freqs if isinstance(freqs, list) else [freqs]
        self.picks = picks
        self.filter_length = filter_length
        self.notch_widths = notch_widths
        self.method = method
        self.phase = phase
        self.fir_window = fir_window
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()

        # --- LOG ---
        logger.info("Applying notch filter at {} Hz", self.freqs)

        # --- COMPUTE ---
        raw.notch_filter(
            freqs=self.freqs,
            picks=self.picks,
            filter_length=self.filter_length,
            notch_widths=self.notch_widths,
            method=self.method,
            phase=self.phase,
            fir_window=self.fir_window,
            verbose=False,
        )

        # --- NOISE ---
        estimated_noise = None
        if context.has_estimated_noise():
            noise = context.get_estimated_noise().copy()
            with suppress_stdout():
                noise_raw = mne.io.RawArray(noise, raw.info)
            noise_raw.notch_filter(
                freqs=self.freqs,
                picks=self.picks,
                filter_length=self.filter_length,
                notch_widths=self.notch_widths,
                method=self.method,
                phase=self.phase,
                fir_window=self.fir_window,
                verbose=False,
            )
            estimated_noise = noise_raw._data
        else:
            logger.debug("No noise estimate present — skipping noise propagation")

        # --- RETURN ---
        return context.with_raw(
            raw,
            estimated_noise=estimated_noise,
            copy_estimated_noise=False,
        )
