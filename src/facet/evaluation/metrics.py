"""
Evaluation Metrics Processors Module

This module contains processors for evaluating correction quality.

Author: FACETpy Team
Date: 2025-01-12
"""

import contextlib
from collections.abc import Iterator, Sequence
from numbers import Integral
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.widgets import Button, SpanSelector
from scipy import signal

from ..console import get_console, report_metric, suspend_raw_mode
from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor
from ..helpers.plotting import show_matplotlib_figure

DEFAULT_CHANNEL_BLOCK_SIZE = 8


def _validate_channel_block_size(channel_block_size: int | None) -> int | None:
    """Validate and normalize a metric processor's channel block size."""
    if channel_block_size is None:
        return None
    if isinstance(channel_block_size, bool) or not isinstance(channel_block_size, Integral):
        raise ValueError("channel_block_size must be a positive integer or None")
    if channel_block_size < 1:
        raise ValueError("channel_block_size must be a positive integer or None")
    return int(channel_block_size)


def _iter_channel_blocks(
    picks: Sequence[int] | np.ndarray,
    channel_block_size: int | None,
) -> Iterator[np.ndarray]:
    """Yield stable channel-index blocks without copying signal samples."""
    pick_array = np.asarray(picks, dtype=int)
    if pick_array.size == 0:
        return

    block_size = len(pick_array) if channel_block_size is None else channel_block_size
    for start in range(0, len(pick_array), block_size):
        yield pick_array[start : start + block_size]


def _dist_summary(values: np.ndarray) -> str:
    """Return compact distribution summary for a 1D array."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size}, min={np.min(arr):.3g}, p25={np.percentile(arr, 25):.3g}, "
        f"median={np.median(arr):.3g}, p75={np.percentile(arr, 75):.3g}, max={np.max(arr):.3g}"
    )


def _signal_summary(data: np.ndarray) -> str:
    """Return compact summary for a 2D channel x time signal matrix."""
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return "empty"
    return (
        f"shape={arr.shape}, mean={np.mean(arr):.3g}, std={np.std(arr):.3g}, "
        f"rms={np.sqrt(np.mean(arr**2)):.3g}, p05={np.percentile(arr, 5):.3g}, p95={np.percentile(arr, 95):.3g}"
    )


def _top_channels(values: np.ndarray, channel_names: list[str], n: int = 3) -> tuple[str, str]:
    """Return formatted worst/best channel lists for a metric array."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return "[]", "[]"
    n = max(1, min(n, arr.size, len(channel_names)))
    order = np.argsort(arr)
    best_idx = order[:n]
    worst_idx = order[-n:][::-1]
    best = ", ".join(f"{channel_names[i]}={arr[i]:.3g}" for i in best_idx)
    worst = ", ".join(f"{channel_names[i]}={arr[i]:.3g}" for i in worst_idx)
    return worst, best


class ReferenceDataMixin:
    """Mixin for extracting reference data (outside acquisition)."""

    def get_eeg_channels(self, raw: mne.io.BaseRaw) -> np.ndarray:
        """Get EEG channel indices.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data object.

        Returns
        -------
        np.ndarray
            Array of EEG channel indices.
        """
        return mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

    def get_reference_data(
        self,
        raw: mne.io.BaseRaw,
        triggers: np.ndarray,
        artifact_length: int,
        time_buffer: float = 0.1,
        context: ProcessingContext | None = None,
        picks: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract reference data (outside acquisition window).

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data object.
        triggers : np.ndarray
            Trigger indices.
        artifact_length : int
            Length of one artifact in samples.
        time_buffer : float, optional
            Buffer in seconds to stay away from acquisition (default: 0.1).
        context : facet.core.ProcessingContext, optional
            Current processing context. If it contains a user-selected
            reference interval (set by ``ReferenceIntervalSelector``), that
            interval is used before falling back to automatic extraction.

        Returns
        -------
        np.ndarray
            Array of shape (n_channels, n_times) containing concatenated reference data.
        """
        sfreq = raw.info["sfreq"]
        eeg_picks = self.get_eeg_channels(raw) if picks is None else np.asarray(picks, dtype=int)

        if len(eeg_picks) == 0:
            logger.warning("No EEG channels found")
            return np.array([])

        # Prefer explicit user-selected interval when available.
        if context is not None:
            selected_interval = self._get_selected_reference_interval(context, raw)
            if selected_interval is not None:
                sel_tmin, sel_tmax = selected_interval
                try:
                    selected_data = raw.get_data(picks=eeg_picks, tmin=sel_tmin, tmax=sel_tmax)
                except Exception as exc:
                    logger.warning(
                        "Failed to load selected reference interval "
                        "[{:.3f}, {:.3f}] s: {}. Falling back to automatic selection.",
                        sel_tmin,
                        sel_tmax,
                        exc,
                    )
                else:
                    if selected_data.size > 0 and selected_data.shape[1] > 0:
                        return selected_data
                    logger.warning(
                        "Selected reference interval [{:.3f}, {:.3f}] s returned no data. "
                        "Falling back to automatic selection.",
                        sel_tmin,
                        sel_tmax,
                    )

        if triggers is None or len(triggers) == 0 or artifact_length is None:
            logger.warning("Cannot infer automatic reference interval without triggers and artifact length.")
            return np.array([]).reshape(len(eeg_picks), 0)

        # Acquisition is considered to span from first trigger to last trigger + artifact length.
        acq_start_sample = triggers[0]
        acq_end_sample = triggers[-1] + artifact_length

        acq_start_time = acq_start_sample / sfreq
        acq_end_time = acq_end_sample / sfreq

        ref_pre_tmax = acq_start_time - time_buffer
        ref_post_tmin = acq_end_time + time_buffer

        ref_segments = []

        if ref_pre_tmax > 0:
            try:
                data = raw.get_data(picks=eeg_picks, tmax=ref_pre_tmax)
                ref_segments.append(data)
            except Exception:
                pass

        if ref_post_tmin < raw.times[-1]:
            try:
                data = raw.get_data(picks=eeg_picks, tmin=ref_post_tmin)
                ref_segments.append(data)
            except Exception:
                pass

        if not ref_segments:
            logger.warning("No reference data found outside acquisition window")
            return np.array([]).reshape(len(eeg_picks), 0)

        return np.concatenate(ref_segments, axis=1)

    def _get_selected_reference_interval(
        self,
        context: ProcessingContext,
        raw: mne.io.BaseRaw,
    ) -> tuple[float, float] | None:
        """Return validated user-selected reference interval in seconds."""
        return self._get_selected_custom_interval(context, raw, "reference_interval")

    def _get_selected_evaluation_interval(
        self,
        context: ProcessingContext,
        raw: mne.io.BaseRaw,
    ) -> tuple[float, float] | None:
        """Return validated user-selected evaluation interval in seconds."""
        return self._get_selected_custom_interval(context, raw, "evaluation_interval")

    def _get_selected_custom_interval(
        self,
        context: ProcessingContext,
        raw: mne.io.BaseRaw,
        key: str,
    ) -> tuple[float, float] | None:
        """Return validated custom interval in seconds from metadata.custom[key]."""
        custom = context.metadata.custom
        interval = custom.get(key)
        if not isinstance(interval, dict):
            return None

        tmin = interval.get("tmin")
        tmax = interval.get("tmax")
        if tmin is None or tmax is None:
            return None

        try:
            tmin = float(tmin)
            tmax = float(tmax)
        except (TypeError, ValueError):
            logger.warning("Invalid {} metadata (non-numeric tmin/tmax)", key)
            return None

        raw_tmax = float(raw.times[-1]) if raw.n_times > 0 else 0.0
        tmin = max(0.0, min(tmin, raw_tmax))
        tmax = max(0.0, min(tmax, raw_tmax))

        if tmax <= tmin:
            logger.warning(
                "Invalid {} metadata (tmax <= tmin): [{:.3f}, {:.3f}]",
                key,
                tmin,
                tmax,
            )
            return None

        return tmin, tmax

    def _infer_acquisition_interval(
        self,
        raw: mne.io.BaseRaw,
        triggers: np.ndarray | None,
        artifact_length: int | None,
    ) -> tuple[float, float] | None:
        """Infer acquisition interval in seconds from triggers and artifact length."""
        if triggers is None or len(triggers) == 0 or artifact_length is None:
            return None

        sfreq = raw.info["sfreq"]
        if sfreq <= 0:
            return None

        acq_start = max(0, int(triggers[0]) - int(artifact_length * 0.5))
        acq_end = min(raw.n_times, int(triggers[-1]) + int(artifact_length * 1.5))

        if acq_end <= acq_start:
            return None

        tmin = acq_start / sfreq
        tmax = min(acq_end / sfreq, float(raw.times[-1]))
        if tmax <= tmin:
            return None
        return tmin, tmax

    def _resolve_configured_interval(
        self,
        raw: mne.io.BaseRaw,
        tmin: float | None,
        tmax: float | None,
        label: str,
    ) -> tuple[float, float] | None:
        """Resolve configured interval bounds against recording limits."""
        if tmin is None and tmax is None:
            return None

        raw_tmax = float(raw.times[-1]) if raw.n_times > 0 else 0.0

        try:
            left = 0.0 if tmin is None else float(tmin)
            right = raw_tmax if tmax is None else float(tmax)
        except (TypeError, ValueError) as exc:
            raise ProcessorValidationError(
                f"{label} interval bounds must be numeric, got tmin={tmin}, tmax={tmax}"
            ) from exc

        left = max(0.0, min(left, raw_tmax))
        right = max(0.0, min(right, raw_tmax))

        if right <= left:
            raise ProcessorValidationError(
                f"invalid {label} interval after clipping: tmin={left:.6f}, tmax={right:.6f}"
            )

        return left, right

    def get_acquisition_data(
        self,
        raw: mne.io.BaseRaw,
        triggers: np.ndarray,
        artifact_length: int,
        context: ProcessingContext | None = None,
        picks: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract data within the acquisition window.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data object.
        triggers : np.ndarray
            Trigger indices.
        artifact_length : int
            Length of one artifact in samples.
        context : facet.core.ProcessingContext, optional
            Current processing context. If it contains a user-selected
            evaluation interval (set by ``SignalIntervalSelector``), that
            interval is used before falling back to automatic extraction.

        Returns
        -------
        np.ndarray
            Array of shape (n_channels, n_times) from the acquisition window.
        """
        eeg_picks = self.get_eeg_channels(raw) if picks is None else np.asarray(picks, dtype=int)

        if len(eeg_picks) == 0:
            return np.array([])

        # Prefer explicit user-selected interval when available.
        if context is not None:
            selected_interval = self._get_selected_evaluation_interval(context, raw)
            if selected_interval is not None:
                sel_tmin, sel_tmax = selected_interval
                try:
                    selected_data = raw.get_data(picks=eeg_picks, tmin=sel_tmin, tmax=sel_tmax)
                except Exception as exc:
                    logger.warning(
                        "Failed to load selected evaluation interval "
                        "[{:.3f}, {:.3f}] s: {}. Falling back to automatic selection.",
                        sel_tmin,
                        sel_tmax,
                        exc,
                    )
                else:
                    if selected_data.size > 0 and selected_data.shape[1] > 0:
                        return selected_data
                    logger.warning(
                        "Selected evaluation interval [{:.3f}, {:.3f}] s returned no data. "
                        "Falling back to automatic selection.",
                        sel_tmin,
                        sel_tmax,
                    )

        inferred_interval = self._infer_acquisition_interval(raw, triggers, artifact_length)
        if inferred_interval is None:
            logger.warning("Cannot infer acquisition interval without triggers and artifact length.")
            return np.array([]).reshape(len(eeg_picks), 0)

        tmin, tmax = inferred_interval

        return raw.get_data(picks=eeg_picks, tmin=tmin, tmax=tmax)


@register_processor
class ReferenceIntervalSelector(Processor, ReferenceDataMixin):
    """Interactively select a clean reference interval from a signal plot.

    Opens a Matplotlib GUI window for one EEG channel and lets the user drag a
    time span. The selected region is highlighted in green and, after
    confirmation, stored in ``context.metadata.custom['reference_interval']``.
    Downstream metrics processors can use this interval as explicit reference
    data.

    Parameters
    ----------
    channel : str or int, optional
        Channel to display. If ``None`` (default), the first EEG channel is
        used.
    min_duration : float, optional
        Minimum selectable interval length in seconds (default: 0.5).
    tmin : float, optional
        Reference interval start in seconds. If provided together with
        ``tmax`` (or alone), the selector GUI is skipped.
    tmax : float, optional
        Reference interval end in seconds. If provided together with
        ``tmin`` (or alone), the selector GUI is skipped.
    """

    name = "reference_interval_selector"
    description = "Interactively select clean reference interval for metrics"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        channel: str | int | None = None,
        min_duration: float = 0.5,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> None:
        self.channel = channel
        self.min_duration = min_duration
        self.tmin = tmin
        self.tmax = tmax
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if self.min_duration <= 0:
            raise ProcessorValidationError(f"min_duration must be > 0, got {self.min_duration}")

        raw = context.get_raw()
        if raw.n_times < 2:
            raise ProcessorValidationError("Raw must contain at least 2 samples.")

        self._resolve_channel(raw)
        self._resolve_configured_interval(raw, self.tmin, self.tmax, label="reference")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        ch_idx = self._resolve_channel(raw)
        ch_name = raw.ch_names[ch_idx]
        configured_interval = self._resolve_configured_interval(raw, self.tmin, self.tmax, label="reference")

        # --- COMPUTE ---
        if configured_interval is not None:
            selected_interval = configured_interval
            logger.info(
                "Using configured reference interval: [{:.3f}, {:.3f}] s",
                selected_interval[0],
                selected_interval[1],
            )
        else:
            sfreq = raw.info["sfreq"]
            channel_data = raw.get_data(picks=[ch_idx])[0]
            time_axis = raw.times

            # --- LOG ---
            logger.info("Opening reference interval selector for channel {}", ch_name)

            selected_interval = self._show_selector(
                time_axis=time_axis,
                channel_data=channel_data,
                channel_name=ch_name,
                default_interval=self._get_default_interval(context, raw),
                min_duration=self.min_duration,
                sfreq=sfreq,
            )

        if selected_interval is None:
            logger.info("Reference interval selection cancelled; keeping previous settings.")
            return context

        # --- BUILD RESULT ---
        tmin, tmax = selected_interval
        new_metadata = context.metadata.copy()
        new_metadata.custom["reference_interval"] = {
            "tmin": float(tmin),
            "tmax": float(tmax),
            "channel": ch_name,
            "source": self.name,
        }

        logger.info(
            "Selected clean reference interval: [{:.3f}, {:.3f}] s ({:.3f} s)",
            tmin,
            tmax,
            tmax - tmin,
        )

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )

    def _resolve_channel(self, raw: mne.io.BaseRaw) -> int:
        """Resolve configured channel to an EEG channel index."""
        eeg_picks = self.get_eeg_channels(raw)
        if len(eeg_picks) == 0:
            raise ProcessorValidationError("No EEG channels found in raw data.")

        if self.channel is None:
            return int(eeg_picks[0])

        if isinstance(self.channel, int):
            if self.channel < 0 or self.channel >= len(raw.ch_names):
                raise ProcessorValidationError(f"channel index out of range: {self.channel}")
            return int(self.channel)

        if self.channel not in raw.ch_names:
            raise ProcessorValidationError(f"channel '{self.channel}' not found")
        return int(raw.ch_names.index(self.channel))

    def _get_default_interval(
        self,
        context: ProcessingContext,
        raw: mne.io.BaseRaw,
    ) -> tuple[float, float]:
        """Derive a sensible default interval for the selector."""
        existing = self._get_selected_reference_interval(context, raw)
        if existing is not None:
            return existing

        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        duration = float(raw.times[-1]) if raw.n_times > 0 else 0.0

        if triggers is not None and len(triggers) > 0 and artifact_length is not None and raw.info["sfreq"] > 0:
            sfreq = raw.info["sfreq"]
            acq_start = triggers[0] / sfreq
            acq_end = (triggers[-1] + artifact_length) / sfreq

            if acq_start > self.min_duration:
                return 0.0, acq_start
            if duration - acq_end > self.min_duration:
                return acq_end, duration

        default_len = min(max(self.min_duration, 1.0), duration)
        return 0.0, default_len

    def _show_selector(
        self,
        time_axis: np.ndarray,
        channel_data: np.ndarray,
        channel_name: str,
        default_interval: tuple[float, float],
        min_duration: float,
        sfreq: float,
    ) -> tuple[float, float] | None:
        """Show the interactive GUI and return selected interval."""
        backend = plt.get_backend().lower()
        if "agg" in backend:
            logger.warning("Matplotlib backend '{}' is non-interactive; skipping selector.", backend)
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.24)

        ax.plot(time_axis, channel_data, linewidth=0.8, alpha=0.9)
        ax.set_title(f"Select clean reference interval - {channel_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

        initial_tmin, initial_tmax = default_interval
        interval_state: dict[str, Any] = {
            "tmin": float(initial_tmin),
            "tmax": float(initial_tmax),
            "confirmed": False,
            "shade": None,
        }

        text_label = ax.text(
            0.02,
            0.96,
            "",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        def _refresh_overlay() -> None:
            if interval_state["shade"] is not None:
                interval_state["shade"].remove()
            interval_state["shade"] = ax.axvspan(
                interval_state["tmin"],
                interval_state["tmax"],
                facecolor="green",
                alpha=0.22,
                edgecolor="green",
                linewidth=1.0,
            )
            text_label.set_text(
                "Selected: {:.3f}s to {:.3f}s ({:.3f}s)".format(
                    interval_state["tmin"],
                    interval_state["tmax"],
                    interval_state["tmax"] - interval_state["tmin"],
                )
            )
            fig.canvas.draw_idle()

        def _on_select(xmin: float, xmax: float) -> None:
            if xmin is None or xmax is None:
                return
            left = max(0.0, min(float(xmin), float(xmax)))
            right = min(float(time_axis[-1]), max(float(xmin), float(xmax)))

            if right - left < min_duration:
                right = min(float(time_axis[-1]), left + min_duration)
                left = max(0.0, right - min_duration)

            # Snap to nearest sample boundaries.
            left = round(left * sfreq) / sfreq
            right = round(right * sfreq) / sfreq
            if right <= left:
                right = min(float(time_axis[-1]), left + (1.0 / sfreq))

            interval_state["tmin"] = left
            interval_state["tmax"] = right
            _refresh_overlay()

        span_props = dict(facecolor="green", edgecolor="green", alpha=0.25)
        span_selector = SpanSelector(
            ax,
            _on_select,
            "horizontal",
            useblit=True,
            interactive=True,
            drag_from_anywhere=True,
            props=span_props,
        )
        span_selector.set_active(True)

        confirm_ax = fig.add_axes([0.70, 0.06, 0.12, 0.07])
        confirm_btn = Button(confirm_ax, "Confirm")
        cancel_ax = fig.add_axes([0.84, 0.06, 0.12, 0.07])
        cancel_btn = Button(cancel_ax, "Cancel")

        def _close_fig() -> None:
            with contextlib.suppress(Exception):
                fig.canvas.manager.destroy()
            plt.close(fig)

        def _on_confirm(_) -> None:
            interval_state["confirmed"] = True
            _close_fig()

        def _on_cancel(_) -> None:
            _close_fig()

        confirm_btn.on_clicked(_on_confirm)
        cancel_btn.on_clicked(_on_cancel)
        _refresh_overlay()

        console = get_console()
        console.set_active_prompt("Drag to select clean reference interval, then click Confirm")
        try:
            with suspend_raw_mode():
                show_matplotlib_figure(fig)
        finally:
            plt.close(fig)
            console.clear_active_prompt()

        if not interval_state["confirmed"]:
            return None

        return float(interval_state["tmin"]), float(interval_state["tmax"])


@register_processor
class SignalIntervalSelector(Processor, ReferenceDataMixin):
    """Interactively select the evaluated signal interval from a signal plot.

    Opens a Matplotlib GUI window for one EEG channel and lets the user drag a
    time span that should be used as evaluated signal (typically acquisition).
    The selected region is highlighted in blue and, after confirmation, stored
    in ``context.metadata.custom['evaluation_interval']``. Downstream metrics
    processors can use this interval as explicit acquisition/evaluation data.

    The trigger-derived acquisition window is additionally shown as a weak
    orange background hint to help orient manual selection when boundaries are
    not obvious after correction.

    Parameters
    ----------
    channel : str or int, optional
        Channel to display. If ``None`` (default), the first EEG channel is
        used.
    min_duration : float, optional
        Minimum selectable interval length in seconds (default: 0.5).
    tmin : float, optional
        Evaluated interval start in seconds. If provided together with
        ``tmax`` (or alone), the selector GUI is skipped.
    tmax : float, optional
        Evaluated interval end in seconds. If provided together with
        ``tmin`` (or alone), the selector GUI is skipped.
    """

    name = "signal_interval_selector"
    description = "Interactively select evaluated signal interval for metrics"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        channel: str | int | None = None,
        min_duration: float = 0.5,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> None:
        self.channel = channel
        self.min_duration = min_duration
        self.tmin = tmin
        self.tmax = tmax
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if self.min_duration <= 0:
            raise ProcessorValidationError(f"min_duration must be > 0, got {self.min_duration}")

        raw = context.get_raw()
        if raw.n_times < 2:
            raise ProcessorValidationError("Raw must contain at least 2 samples.")

        self._resolve_channel(raw)
        self._resolve_configured_interval(raw, self.tmin, self.tmax, label="evaluation")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        ch_idx = self._resolve_channel(raw)
        ch_name = raw.ch_names[ch_idx]
        configured_interval = self._resolve_configured_interval(raw, self.tmin, self.tmax, label="evaluation")

        # --- COMPUTE ---
        if configured_interval is not None:
            selected_interval = configured_interval
            logger.info(
                "Using configured evaluated interval: [{:.3f}, {:.3f}] s",
                selected_interval[0],
                selected_interval[1],
            )
        else:
            sfreq = raw.info["sfreq"]
            channel_data = raw.get_data(picks=[ch_idx])[0]
            time_axis = raw.times
            acquisition_hint = self._infer_acquisition_interval(
                raw=raw,
                triggers=context.get_triggers(),
                artifact_length=context.get_artifact_length(),
            )

            # --- LOG ---
            logger.info("Opening evaluated-signal interval selector for channel {}", ch_name)

            selected_interval = self._show_selector(
                time_axis=time_axis,
                channel_data=channel_data,
                channel_name=ch_name,
                default_interval=self._get_default_interval(context, raw),
                acquisition_hint=acquisition_hint,
                min_duration=self.min_duration,
                sfreq=sfreq,
            )

        if selected_interval is None:
            logger.info("Signal interval selection cancelled; keeping previous settings.")
            return context

        # --- BUILD RESULT ---
        tmin, tmax = selected_interval
        new_metadata = context.metadata.copy()
        new_metadata.custom["evaluation_interval"] = {
            "tmin": float(tmin),
            "tmax": float(tmax),
            "channel": ch_name,
            "source": self.name,
        }

        logger.info(
            "Selected evaluated signal interval: [{:.3f}, {:.3f}] s ({:.3f} s)",
            tmin,
            tmax,
            tmax - tmin,
        )

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )

    def _resolve_channel(self, raw: mne.io.BaseRaw) -> int:
        """Resolve configured channel to an EEG channel index."""
        eeg_picks = self.get_eeg_channels(raw)
        if len(eeg_picks) == 0:
            raise ProcessorValidationError("No EEG channels found in raw data.")

        if self.channel is None:
            return int(eeg_picks[0])

        if isinstance(self.channel, int):
            if self.channel < 0 or self.channel >= len(raw.ch_names):
                raise ProcessorValidationError(f"channel index out of range: {self.channel}")
            return int(self.channel)

        if self.channel not in raw.ch_names:
            raise ProcessorValidationError(f"channel '{self.channel}' not found")
        return int(raw.ch_names.index(self.channel))

    def _get_default_interval(
        self,
        context: ProcessingContext,
        raw: mne.io.BaseRaw,
    ) -> tuple[float, float]:
        """Derive a sensible default interval for the selector."""
        existing = self._get_selected_evaluation_interval(context, raw)
        if existing is not None:
            return existing

        inferred = self._infer_acquisition_interval(
            raw=raw,
            triggers=context.get_triggers(),
            artifact_length=context.get_artifact_length(),
        )

        duration = float(raw.times[-1]) if raw.n_times > 0 else 0.0
        if inferred is not None:
            tmin, tmax = inferred
            if tmax - tmin >= self.min_duration:
                return tmin, tmax

            # Expand tiny intervals to satisfy minimum duration while staying in bounds.
            center = 0.5 * (tmin + tmax)
            left = max(0.0, center - 0.5 * self.min_duration)
            right = min(duration, left + self.min_duration)
            left = max(0.0, right - self.min_duration)
            if right > left:
                return left, right

        default_len = min(max(self.min_duration, 1.0), duration)
        return 0.0, default_len

    def _show_selector(
        self,
        time_axis: np.ndarray,
        channel_data: np.ndarray,
        channel_name: str,
        default_interval: tuple[float, float],
        acquisition_hint: tuple[float, float] | None,
        min_duration: float,
        sfreq: float,
    ) -> tuple[float, float] | None:
        """Show the interactive GUI and return selected interval."""
        backend = plt.get_backend().lower()
        if "agg" in backend:
            logger.warning("Matplotlib backend '{}' is non-interactive; skipping selector.", backend)
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.24)

        ax.plot(time_axis, channel_data, linewidth=0.8, alpha=0.9)
        ax.set_title(f"Select evaluated signal interval - {channel_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

        if acquisition_hint is not None:
            hint_tmin, hint_tmax = acquisition_hint
            ax.axvspan(
                hint_tmin,
                hint_tmax,
                facecolor="tab:orange",
                alpha=0.09,
                edgecolor="tab:orange",
                linewidth=0.8,
            )

        initial_tmin, initial_tmax = default_interval
        interval_state: dict[str, Any] = {
            "tmin": float(initial_tmin),
            "tmax": float(initial_tmax),
            "confirmed": False,
            "shade": None,
        }

        text_label = ax.text(
            0.02,
            0.96,
            "",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        def _refresh_overlay() -> None:
            if interval_state["shade"] is not None:
                interval_state["shade"].remove()
            interval_state["shade"] = ax.axvspan(
                interval_state["tmin"],
                interval_state["tmax"],
                facecolor="tab:blue",
                alpha=0.22,
                edgecolor="tab:blue",
                linewidth=1.0,
            )

            selected_text = "Selected: {:.3f}s to {:.3f}s ({:.3f}s)".format(
                interval_state["tmin"],
                interval_state["tmax"],
                interval_state["tmax"] - interval_state["tmin"],
            )
            if acquisition_hint is not None:
                hint_tmin, hint_tmax = acquisition_hint
                selected_text = f"Hint (orange): {hint_tmin:.3f}s to {hint_tmax:.3f}s\n{selected_text}"

            text_label.set_text(selected_text)
            fig.canvas.draw_idle()

        def _on_select(xmin: float, xmax: float) -> None:
            if xmin is None or xmax is None:
                return
            left = max(0.0, min(float(xmin), float(xmax)))
            right = min(float(time_axis[-1]), max(float(xmin), float(xmax)))

            if right - left < min_duration:
                right = min(float(time_axis[-1]), left + min_duration)
                left = max(0.0, right - min_duration)

            # Snap to nearest sample boundaries.
            left = round(left * sfreq) / sfreq
            right = round(right * sfreq) / sfreq
            if right <= left:
                right = min(float(time_axis[-1]), left + (1.0 / sfreq))

            interval_state["tmin"] = left
            interval_state["tmax"] = right
            _refresh_overlay()

        span_props = dict(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.25)
        span_selector = SpanSelector(
            ax,
            _on_select,
            "horizontal",
            useblit=True,
            interactive=True,
            drag_from_anywhere=True,
            props=span_props,
        )
        span_selector.set_active(True)

        confirm_ax = fig.add_axes([0.70, 0.06, 0.12, 0.07])
        confirm_btn = Button(confirm_ax, "Confirm")
        cancel_ax = fig.add_axes([0.84, 0.06, 0.12, 0.07])
        cancel_btn = Button(cancel_ax, "Cancel")

        def _close_fig() -> None:
            with contextlib.suppress(Exception):
                fig.canvas.manager.destroy()
            plt.close(fig)

        def _on_confirm(_) -> None:
            interval_state["confirmed"] = True
            _close_fig()

        def _on_cancel(_) -> None:
            _close_fig()

        confirm_btn.on_clicked(_on_confirm)
        cancel_btn.on_clicked(_on_cancel)
        _refresh_overlay()

        console = get_console()
        console.set_active_prompt("Drag to select evaluated interval (orange hint marks inferred acquisition)")
        try:
            with suspend_raw_mode():
                show_matplotlib_figure(fig)
        finally:
            plt.close(fig)
            console.clear_active_prompt()

        if not interval_state["confirmed"]:
            return None

        return float(interval_state["tmin"]), float(interval_state["tmax"])


@register_processor
class SNRCalculator(Processor, ReferenceDataMixin):
    """Calculate Signal-to-Noise Ratio (SNR).

    Compares corrected data to a clean reference (data outside acquisition
    window). Higher SNR indicates better correction.

    SNR = variance(reference) / variance(residual)

    Parameters
    ----------
    time_buffer : float, optional
        Time buffer around acquisition window in seconds (default: 0.1).
    channel_block_size : int or None, optional
        Number of EEG channels loaded at once. ``None`` restores the legacy
        all-channel allocation (default: 8).

    Examples
    --------
    ::

        snr = SNRCalculator()
        context = snr.execute(context)
        print(context.metadata.custom['snr'])
    """

    name = "snr_calculator"
    description = "Calculate Signal-to-Noise Ratio"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        time_buffer: float = 0.1,
        verbose: bool = False,
        channel_block_size: int | None = DEFAULT_CHANNEL_BLOCK_SIZE,
    ) -> None:
        self.time_buffer = time_buffer
        self.verbose = verbose
        self.channel_block_size = _validate_channel_block_size(channel_block_size)
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_raw_original() is None:
            raise ProcessorValidationError("Original raw data not available. Cannot calculate SNR.")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        eeg_picks = self.get_eeg_channels(raw)
        channel_names = [raw.ch_names[i] for i in eeg_picks]

        # --- LOG ---
        logger.info("Calculating Signal-to-Noise Ratio (SNR)")

        # --- COMPUTE ---
        corrected_variances: list[np.ndarray] = []
        reference_variances: list[np.ndarray] = []
        for block_index, block in enumerate(
            _iter_channel_blocks(eeg_picks, self.channel_block_size),
            start=1,
        ):
            ref_data = self.get_reference_data(
                raw,
                triggers,
                artifact_length,
                self.time_buffer,
                context=context,
                picks=block,
            )
            corrected_data = self.get_acquisition_data(
                raw,
                triggers,
                artifact_length,
                context=context,
                picks=block,
            )
            if ref_data.size == 0 or corrected_data.size == 0:
                logger.warning("Insufficient data for SNR calculation")
                return context
            if self.verbose:
                logger.info(
                    "SNR diagnostics block {}: reference {}; corrected {}",
                    block_index,
                    _signal_summary(ref_data),
                    _signal_summary(corrected_data),
                )
            reference_variances.append(np.var(ref_data, axis=1))
            corrected_variances.append(np.var(corrected_data, axis=1))

        if not reference_variances:
            logger.warning("Insufficient data for SNR calculation")
            return context

        var_corrected = np.concatenate(corrected_variances)
        var_reference = np.concatenate(reference_variances)

        # Residual variance is the difference; clamped to avoid division by zero.
        var_residual = np.maximum(var_corrected - var_reference, 1e-10)

        snr_per_channel = np.abs(var_reference / var_residual)
        snr_mean = np.mean(snr_per_channel)

        if self.verbose:
            logger.info("SNR diagnostics: var_reference {}", _dist_summary(var_reference))
            logger.info("SNR diagnostics: var_corrected {}", _dist_summary(var_corrected))
            logger.info("SNR diagnostics: var_residual {}", _dist_summary(var_residual))
            logger.info("SNR diagnostics: snr_per_channel {}", _dist_summary(snr_per_channel))
            worst, best = _top_channels(snr_per_channel, channel_names)
            logger.info("SNR diagnostics: lowest channels [{}]", best)
            logger.info("SNR diagnostics: highest channels [{}]", worst)

        report_metric("snr", float(snr_mean), label="SNR", display=f"{snr_mean:.2f}")

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault("metrics", {})
        metrics["snr"] = float(snr_mean)
        metrics["snr_per_channel"] = snr_per_channel.tolist()

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )


@register_processor
class RMSResidualCalculator(Processor, ReferenceDataMixin):
    """Calculate RMS Residual Ratio (corrected vs. reference).

    Compares the RMS of the corrected signal (during acquisition) to the RMS
    of the clean reference signal (outside acquisition).

    Ratio = RMS(corrected) / RMS(reference)

    A ratio of 1.0 is the target. Values below 1.0 suggest over-correction;
    values above 1.0 indicate residual artifacts. Corresponds to
    ``rms_residual`` in FACET MATLAB Edition.

    Parameters
    ----------
    time_buffer : float, optional
        Time buffer around acquisition window in seconds (default: 0.1).
    channel_block_size : int or None, optional
        Number of EEG channels loaded at once. ``None`` restores the legacy
        all-channel allocation (default: 8).
    """

    name = "rms_residual_calculator"
    description = "Calculate RMS ratio (corrected vs reference)"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        time_buffer: float = 0.1,
        verbose: bool = False,
        channel_block_size: int | None = DEFAULT_CHANNEL_BLOCK_SIZE,
    ) -> None:
        self.time_buffer = time_buffer
        self.verbose = verbose
        self.channel_block_size = _validate_channel_block_size(channel_block_size)
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        eeg_picks = self.get_eeg_channels(raw)
        channel_names = [raw.ch_names[i] for i in eeg_picks]

        # --- LOG ---
        logger.info("Calculating RMS Residual Ratio (corrected vs reference)")

        # --- COMPUTE ---
        corrected_stds: list[np.ndarray] = []
        reference_stds: list[np.ndarray] = []
        for block_index, block in enumerate(
            _iter_channel_blocks(eeg_picks, self.channel_block_size),
            start=1,
        ):
            ref_data = self.get_reference_data(
                raw,
                triggers,
                artifact_length,
                self.time_buffer,
                context=context,
                picks=block,
            )
            corrected_data = self.get_acquisition_data(
                raw,
                triggers,
                artifact_length,
                context=context,
                picks=block,
            )
            if ref_data.size == 0 or corrected_data.size == 0:
                logger.warning("Insufficient data for RMS Residual calculation")
                return context
            if self.verbose:
                logger.info(
                    "RMS Residual diagnostics block {}: reference {}; corrected {}",
                    block_index,
                    _signal_summary(ref_data),
                    _signal_summary(corrected_data),
                )
            reference_stds.append(np.std(ref_data, axis=1))
            corrected_stds.append(np.std(corrected_data, axis=1))

        if not reference_stds:
            logger.warning("Insufficient data for RMS Residual calculation")
            return context

        rms_corrected = np.concatenate(corrected_stds)
        # Clamp to avoid division by zero.
        rms_reference = np.maximum(np.concatenate(reference_stds), 1e-10)

        ratio_per_channel = rms_corrected / rms_reference
        ratio_mean = np.mean(ratio_per_channel)

        if self.verbose:
            logger.info("RMS Residual diagnostics: rms_reference {}", _dist_summary(rms_reference))
            logger.info("RMS Residual diagnostics: rms_corrected {}", _dist_summary(rms_corrected))
            logger.info("RMS Residual diagnostics: ratio_per_channel {}", _dist_summary(ratio_per_channel))
            worst, best = _top_channels(np.abs(ratio_per_channel - 1.0), channel_names)
            logger.info("RMS Residual diagnostics: closest-to-1 channels [{}]", best)
            logger.info("RMS Residual diagnostics: furthest-from-1 channels [{}]", worst)

        report_metric("rms_residual", float(ratio_mean), label="RMS Residual", display=f"{ratio_mean:.2f}")

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault("metrics", {})
        metrics["rms_residual"] = float(ratio_mean)
        metrics["rms_residual_per_channel"] = ratio_per_channel.tolist()

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )


@register_processor
class LegacySNRCalculator(Processor):
    """Calculate legacy-style Signal-to-Noise Ratio (SNR).

    Mirrors the original FACET implementation by comparing the variance of the
    corrected data to the variance of the uncorrected reference recording.

    ``channel_block_size`` controls how many EEG channels are materialized at
    once; it defaults to 8 and does not change the per-channel calculation.
    """

    name = "legacy_snr_calculator"
    description = "Legacy-style SNR using original raw as reference"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        verbose: bool = False,
        channel_block_size: int | None = DEFAULT_CHANNEL_BLOCK_SIZE,
    ) -> None:
        self.verbose = verbose
        self.channel_block_size = _validate_channel_block_size(channel_block_size)
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_raw_original() is context.get_raw():
            raise ProcessorValidationError("Original raw data not available. Cannot calculate legacy SNR.")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw_corrected = context.get_raw()
        raw_original = context.get_raw_original()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        sfreq = raw_corrected.info["sfreq"]

        # --- LOG ---
        logger.info("Calculating legacy SNR (corrected vs original)")

        # --- COMPUTE ---
        picks = mne.pick_types(raw_corrected.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        channel_names = [raw_corrected.ch_names[i] for i in picks]

        if len(picks) == 0:
            logger.warning("No EEG channels found for legacy SNR calculation")
            return context

        if triggers is None or len(triggers) == 0 or artifact_length is None:
            logger.warning("Legacy SNR requires triggers and artifact length; skipping")
            return context

        acq_start = max(0, triggers[0] - int(artifact_length * 0.5))
        acq_end = min(raw_corrected.n_times, triggers[-1] + int(artifact_length * 1.5))

        acq_tmin = acq_start / sfreq
        acq_tmax = min(acq_end / sfreq, raw_corrected.times[-1])

        corrected_variances: list[np.ndarray] = []
        reference_variances: list[np.ndarray] = []
        for block_index, block in enumerate(
            _iter_channel_blocks(picks, self.channel_block_size),
            start=1,
        ):
            corrected_data = raw_corrected.get_data(
                picks=block,
                tmin=acq_tmin,
                tmax=acq_tmax,
            )

            ref_segments = []
            if acq_tmin > 0:
                ref_segments.append(raw_original.get_data(picks=block, tmax=acq_tmin))
            if acq_tmax < raw_original.times[-1]:
                ref_segments.append(raw_original.get_data(picks=block, tmin=acq_tmax))
            reference_data = (
                np.concatenate(ref_segments, axis=1) if ref_segments else raw_original.get_data(picks=block)
            )

            if self.verbose:
                logger.info(
                    "Legacy SNR diagnostics block {}: corrected {}; reference {}",
                    block_index,
                    _signal_summary(corrected_data),
                    _signal_summary(reference_data),
                )
            corrected_variances.append(np.var(corrected_data, axis=1))
            reference_variances.append(np.var(reference_data, axis=1))

        var_corrected = np.concatenate(corrected_variances)
        var_reference = np.concatenate(reference_variances)

        var_residual = np.maximum(var_corrected - var_reference, 1e-10)

        snr_per_channel = np.abs(var_reference / var_residual)
        snr_mean = float(np.mean(snr_per_channel))

        if self.verbose:
            logger.info("Legacy SNR diagnostics: var_reference {}", _dist_summary(var_reference))
            logger.info("Legacy SNR diagnostics: var_corrected {}", _dist_summary(var_corrected))
            logger.info("Legacy SNR diagnostics: var_residual {}", _dist_summary(var_residual))
            logger.info("Legacy SNR diagnostics: snr_per_channel {}", _dist_summary(snr_per_channel))
            worst, best = _top_channels(snr_per_channel, channel_names)
            logger.info("Legacy SNR diagnostics: lowest channels [{}]", best)
            logger.info("Legacy SNR diagnostics: highest channels [{}]", worst)

        report_metric(
            "legacy_snr",
            snr_mean,
            label="Legacy SNR",
            display=f"{snr_mean:.2f}",
        )

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault("metrics", {})
        metrics["legacy_snr"] = snr_mean
        metrics["legacy_snr_per_channel"] = snr_per_channel.tolist()

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )


@register_processor
class RMSCalculator(Processor):
    """Calculate Root Mean Square (RMS) improvement ratio.

    Compares RMS of corrected data to uncorrected data. A higher ratio
    indicates better correction (more artifact removed).

    RMS_ratio = RMS(uncorrected) / RMS(corrected)

    Signal windows are loaded in blocks of ``channel_block_size`` channels
    (default: 8). Pass ``None`` to use one all-channel block.

    Examples
    --------
    ::

        rms = RMSCalculator()
        context = rms.execute(context)
        print(context.metadata.custom['rms_ratio'])
    """

    name = "rms_calculator"
    description = "Calculate RMS improvement ratio"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        verbose: bool = False,
        channel_block_size: int | None = DEFAULT_CHANNEL_BLOCK_SIZE,
    ) -> None:
        self.verbose = verbose
        self.channel_block_size = _validate_channel_block_size(channel_block_size)
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_raw_original() is None:
            raise ProcessorValidationError("Original raw data not available. Cannot calculate RMS ratio.")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        raw_orig = context.get_raw_original()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info["sfreq"]

        # --- LOG ---
        logger.info("Calculating RMS improvement ratio")

        # --- COMPUTE ---
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        channel_names = [raw.ch_names[i] for i in eeg_channels]

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found")
            return context

        acq_start = max(0, triggers[0] - int(artifact_length * 0.5))
        acq_end = min(raw.n_times, triggers[-1] + int(artifact_length * 1.5))

        acq_tmin = acq_start / sfreq
        acq_tmax = min(acq_end / sfreq, raw.times[-1])

        corrected_rms_blocks: list[np.ndarray] = []
        uncorrected_rms_blocks: list[np.ndarray] = []
        for block_index, block in enumerate(
            _iter_channel_blocks(eeg_channels, self.channel_block_size),
            start=1,
        ):
            data_corrected = raw.get_data(
                picks=block,
                tmin=acq_tmin,
                tmax=acq_tmax,
            )
            data_uncorrected = raw_orig.get_data(
                picks=block,
                tmin=acq_tmin,
                tmax=acq_tmax,
            )

            if data_corrected.shape[0] != data_uncorrected.shape[0]:
                min_channels = min(
                    data_corrected.shape[0],
                    data_uncorrected.shape[0],
                )
                data_corrected = data_corrected[:min_channels]
                data_uncorrected = data_uncorrected[:min_channels]

            if self.verbose:
                logger.info(
                    "RMS diagnostics block {}: uncorrected {}; corrected {}",
                    block_index,
                    _signal_summary(data_uncorrected),
                    _signal_summary(data_corrected),
                )
            uncorrected_rms_blocks.append(np.sqrt(np.mean(data_uncorrected**2, axis=1)))
            corrected_rms_blocks.append(np.sqrt(np.mean(data_corrected**2, axis=1)))

        rms_uncorrected = np.concatenate(uncorrected_rms_blocks)
        # Clamp corrected RMS to avoid division by zero.
        rms_corrected = np.maximum(
            np.concatenate(corrected_rms_blocks),
            1e-10,
        )

        rms_ratio_per_channel = rms_uncorrected / rms_corrected
        rms_ratio = np.median(rms_ratio_per_channel)

        if self.verbose:
            logger.info(
                "RMS diagnostics: triggers={}, artifact_length={}, acq=[{:.3f}, {:.3f}]s",
                len(triggers),
                artifact_length,
                acq_tmin,
                acq_tmax,
            )
            logger.info("RMS diagnostics: rms_uncorrected {}", _dist_summary(rms_uncorrected))
            logger.info("RMS diagnostics: rms_corrected {}", _dist_summary(rms_corrected))
            logger.info("RMS diagnostics: ratio_per_channel {}", _dist_summary(rms_ratio_per_channel))
            worst, best = _top_channels(rms_ratio_per_channel, channel_names)
            logger.info("RMS diagnostics: lowest-improvement channels [{}]", best)
            logger.info("RMS diagnostics: highest-improvement channels [{}]", worst)

        report_metric("rms_ratio", float(rms_ratio), label="RMS Ratio", display=f"{rms_ratio:.2f}")

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault("metrics", {})
        metrics["rms_ratio"] = float(rms_ratio)
        metrics["rms_ratio_per_channel"] = rms_ratio_per_channel.tolist()

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )


@register_processor
class MedianArtifactCalculator(Processor, ReferenceDataMixin):
    """Calculate median peak-to-peak artifact amplitude.

    Measures the median artifact amplitude across all epochs. Lower values
    indicate smaller artifacts (better correction).

    Also calculates the ratio to the median amplitude of the reference signal
    (outside acquisition), which should ideally be close to 1.0.

    Signal epochs are loaded in blocks of ``channel_block_size`` channels
    (default: 8), while the same global channel/epoch medians are retained.

    Examples
    --------
    ::

        median = MedianArtifactCalculator()
        context = median.execute(context)
        print(context.metadata.custom['median_artifact'])
    """

    name = "median_artifact_calculator"
    description = "Calculate median artifact amplitude"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        verbose: bool = False,
        channel_block_size: int | None = DEFAULT_CHANNEL_BLOCK_SIZE,
    ) -> None:
        self.verbose = verbose
        self.channel_block_size = _validate_channel_block_size(channel_block_size)
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError("Artifact length not set. Run TriggerDetector first.")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers()
        sfreq = raw.info["sfreq"]
        artifact_len = context.get_artifact_length()

        # --- LOG ---
        logger.info("Calculating median artifact amplitude")

        # --- COMPUTE ---
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found")
            return context

        median_artifact, median_ref, ratio = self._compute_median_metrics(
            raw, triggers, sfreq, artifact_len, eeg_channels, context
        )

        if self.verbose:
            logger.info(
                "Median Artifact diagnostics: triggers={}, artifact_length={}, offset={:.4f}s",
                0 if triggers is None else len(triggers),
                artifact_len,
                context.metadata.artifact_to_trigger_offset,
            )
            logger.info(
                "Median Artifact diagnostics: median_artifact={:.3g}, median_ref={}, ratio={}",
                median_artifact,
                "nan" if np.isnan(median_ref) else f"{median_ref:.3g}",
                "nan" if np.isnan(ratio) else f"{ratio:.3g}",
            )

        report_metric(
            "median_artifact",
            float(median_artifact),
            label="Median Artifact",
            display=f"{median_artifact:.2e}",
        )

        if not np.isnan(ratio):
            report_metric(
                "median_artifact_ratio",
                float(ratio),
                label="Median Ratio (Corr/Ref)",
                display=f"{ratio:.2f}",
            )

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault("metrics", {})
        metrics["median_artifact"] = float(median_artifact)
        if not np.isnan(median_ref):
            metrics["median_artifact_reference"] = float(median_ref)
        if not np.isnan(ratio):
            metrics["median_artifact_ratio"] = float(ratio)

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )

    def _compute_median_metrics(
        self,
        raw: mne.io.BaseRaw,
        triggers: np.ndarray,
        sfreq: float,
        artifact_len: int,
        eeg_channels: np.ndarray,
        context: ProcessingContext,
    ):
        """Compute median artifact amplitude for corrected and reference data.

        Returns
        -------
        tuple
            (median_artifact, median_ref, ratio) where nan indicates unavailable.
        """
        offset_samples = int(round(context.metadata.artifact_to_trigger_offset * sfreq))

        p2p_per_epoch: list[np.ndarray] = []
        for t in triggers:
            start = t + offset_samples
            end = start + artifact_len
            if start >= 0 and end <= raw.n_times:
                channel_values = []
                for block in _iter_channel_blocks(
                    eeg_channels,
                    self.channel_block_size,
                ):
                    epoch_data = raw.get_data(
                        picks=block,
                        start=start,
                        stop=end,
                    )
                    channel_values.append(np.ptp(epoch_data, axis=1))
                p2p_per_epoch.append(np.concatenate(channel_values))
        mean_p2p_per_epoch = [np.mean(epoch_p2p) for epoch_p2p in p2p_per_epoch]
        median_artifact = np.median(mean_p2p_per_epoch)

        median_ref = np.nan
        ratio = np.nan

        reference_p2p_blocks: list[np.ndarray] = []
        n_ref_epochs: int | None = None
        epoch_len = int(artifact_len)
        for block in _iter_channel_blocks(
            eeg_channels,
            self.channel_block_size,
        ):
            ref_data = self.get_reference_data(
                raw,
                triggers,
                artifact_len,
                context=context,
                picks=block,
            )
            if ref_data.size == 0:
                reference_p2p_blocks = []
                break

            block_ref_epochs = ref_data.shape[1] // epoch_len
            n_ref_epochs = block_ref_epochs if n_ref_epochs is None else min(n_ref_epochs, block_ref_epochs)
            if n_ref_epochs == 0:
                reference_p2p_blocks = []
                break

            truncated = ref_data[:, : n_ref_epochs * epoch_len]
            reshaped = truncated.reshape(
                len(block),
                n_ref_epochs,
                epoch_len,
            )
            reference_p2p_blocks.append(np.ptp(reshaped, axis=2).T)

        if reference_p2p_blocks and n_ref_epochs:
            # Keep only one peak-to-peak scalar per epoch/channel. Concatenating
            # these small arrays in channel order preserves the legacy median.
            p2p_ref = np.concatenate(reference_p2p_blocks, axis=1)
            mean_p2p_ref = np.mean(p2p_ref, axis=1)
            median_ref = np.median(mean_p2p_ref)

            if median_ref > 0:
                ratio = median_artifact / median_ref

        return median_artifact, median_ref, ratio


@register_processor
class FFTAllenCalculator(Processor, ReferenceDataMixin):
    """Calculate FFT Allen metric.

    Compares spectral power in specific frequency bands between corrected data
    and clean reference data. The metric is the median absolute percent
    difference per band.

    Bands: 0.8–4 Hz (delta), 4–8 Hz (theta), 8–12 Hz (alpha), 12–24 Hz (beta).

    Formula::

        metric = median(|Power_corr - Power_ref| / Power_ref) * 100

    Welch spectra are calculated in blocks of ``channel_block_size`` channels
    (default: 8) and reduced with the same global median.
    """

    name = "fft_allen_calculator"
    description = "Calculate spectral power difference (Allen)"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    BANDS = [(0.8, 4, "Delta"), (4, 8, "Theta"), (8, 12, "Alpha"), (12, 24, "Beta")]

    def __init__(
        self,
        verbose: bool = False,
        channel_block_size: int | None = DEFAULT_CHANNEL_BLOCK_SIZE,
    ) -> None:
        self.verbose = verbose
        self.channel_block_size = _validate_channel_block_size(channel_block_size)
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers()
        artifact_len = context.get_artifact_length()
        sfreq = raw.info["sfreq"]

        # --- LOG ---
        logger.info("Calculating FFT Allen metric")

        # --- COMPUTE ---
        n_fft = int(3.0 * sfreq)  # 3-second segments as in MATLAB
        eeg_picks = self.get_eeg_channels(raw)
        if len(eeg_picks) == 0:
            logger.warning("No EEG channels found for FFT Allen")
            return context

        band_values: dict[str, list[np.ndarray]] = {name: [] for _, _, name in self.BANDS}
        for block_index, block in enumerate(
            _iter_channel_blocks(eeg_picks, self.channel_block_size),
            start=1,
        ):
            ref_data = self.get_reference_data(
                raw,
                triggers,
                artifact_len,
                context=context,
                picks=block,
            )
            corr_data = self.get_acquisition_data(
                raw,
                triggers,
                artifact_len,
                context=context,
                picks=block,
            )
            if ref_data.size == 0 or corr_data.size == 0:
                logger.warning("Insufficient data for FFT Allen")
                return context

            nperseg = min(n_fft, ref_data.shape[1], corr_data.shape[1])
            if self.verbose:
                logger.info(
                    "FFT Allen diagnostics block {}: nperseg={}, reference {}; corrected {}",
                    block_index,
                    nperseg,
                    _signal_summary(ref_data),
                    _signal_summary(corr_data),
                )

            freqs_ref, psd_ref = signal.welch(
                ref_data,
                fs=sfreq,
                nperseg=nperseg,
                axis=1,
            )
            freqs_corr, psd_corr = signal.welch(
                corr_data,
                fs=sfreq,
                nperseg=nperseg,
                axis=1,
            )
            if not np.array_equal(freqs_ref, freqs_corr):
                logger.warning("Frequency mismatch in FFT Allen")
                return context

            for band_name, values in self._compute_band_difference_values(
                freqs_ref,
                psd_ref,
                psd_corr,
            ).items():
                band_values[band_name].append(values)

        results = {band_name: float(np.median(np.concatenate(values))) for band_name, values in band_values.items()}

        if self.verbose:
            for band_name, values in band_values.items():
                logger.info(
                    "FFT Allen diagnostics: {} diff_pct [{}]",
                    band_name,
                    _dist_summary(np.concatenate(values)),
                )

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        new_metadata.custom.setdefault("metrics", {})["fft_allen"] = results

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )

    def _compute_band_differences(
        self,
        freqs: np.ndarray,
        psd_ref: np.ndarray,
        psd_corr: np.ndarray,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Compute median absolute percent power difference per frequency band.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency array from Welch PSD.
        psd_ref : np.ndarray
            PSD of reference data, shape (n_channels, n_freqs).
        psd_corr : np.ndarray
            PSD of corrected data, shape (n_channels, n_freqs).

        Returns
        -------
        dict
            Band name → median percent difference.
        """
        values_by_band = self._compute_band_difference_values(
            freqs,
            psd_ref,
            psd_corr,
        )
        results = {band_name: float(np.median(diff_pct)) for band_name, diff_pct in values_by_band.items()}

        for fmin, fmax, band_name in self.BANDS:
            diff_pct = values_by_band[band_name]
            median_diff = results[band_name]
            if verbose:
                logger.info(
                    "FFT Allen diagnostics: {} ({}-{}Hz) diff_pct [{}]",
                    band_name,
                    fmin,
                    fmax,
                    _dist_summary(diff_pct),
                )
            logger.debug("FFT Allen {} ({}-{}Hz): {:.2f}%", band_name, fmin, fmax, median_diff)

        return results

    def _compute_band_difference_values(
        self,
        freqs: np.ndarray,
        psd_ref: np.ndarray,
        psd_corr: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return per-channel Allen differences for block-wise aggregation."""
        values: dict[str, np.ndarray] = {}
        for fmin, fmax, band_name in self.BANDS:
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            power_ref = np.mean(psd_ref[:, idx], axis=1)
            power_corr = np.mean(psd_corr[:, idx], axis=1)
            values[band_name] = np.abs(power_corr - power_ref) / (power_ref + 1e-10) * 100
        return values


@register_processor
class FFTNiazyCalculator(Processor, ReferenceDataMixin):
    """Calculate FFT Niazy metric.

    Analyzes residual artifacts at slice and volume frequencies by computing
    the power ratio (uncorrected / corrected) at these frequencies and their
    harmonics. Values are reported in dB.

    Welch spectra are calculated in blocks of ``channel_block_size`` channels
    (default: 8) and reduced with the same global channel medians.
    """

    name = "fft_niazy_calculator"
    description = "Calculate spectral power ratio at slice/volume frequencies"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        verbose: bool = False,
        channel_block_size: int | None = DEFAULT_CHANNEL_BLOCK_SIZE,
    ) -> None:
        self.verbose = verbose
        self.channel_block_size = _validate_channel_block_size(channel_block_size)
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        raw_orig = context.get_raw_original()
        triggers = context.get_triggers()
        artifact_len = context.get_artifact_length()
        sfreq = raw.info["sfreq"]

        # --- LOG ---
        logger.info("Calculating FFT Niazy metric")

        # --- COMPUTE ---
        if raw_orig is None:
            logger.warning("Original raw data missing for FFT Niazy")
            return context

        slice_freq, vol_freq = self._estimate_frequencies(triggers, sfreq, context)

        corrected_picks = self.get_eeg_channels(raw)
        original_picks = self.get_eeg_channels(raw_orig)
        n_channels = min(len(corrected_picks), len(original_picks))
        if n_channels == 0:
            logger.warning("No common EEG channels found for FFT Niazy")
            return context
        corrected_picks = corrected_picks[:n_channels]
        original_picks = original_picks[:n_channels]

        harmonic_powers: dict[
            str,
            dict[str, dict[str, list[np.ndarray]]],
        ] = {"slice": {}, "volume": {}}
        fundamentals = {"slice": slice_freq}
        if vol_freq is not None:
            fundamentals["volume"] = vol_freq
        for kind in fundamentals:
            harmonic_powers[kind] = {f"h{harmonic}": {"corrected": [], "original": []} for harmonic in range(1, 6)}

        block_size = n_channels if self.channel_block_size is None else self.channel_block_size
        for block_index, start in enumerate(
            range(0, n_channels, block_size),
            start=1,
        ):
            stop = min(start + block_size, n_channels)
            data_corr = self.get_acquisition_data(
                raw,
                triggers,
                artifact_len,
                context=context,
                picks=corrected_picks[start:stop],
            )
            data_orig = self.get_acquisition_data(
                raw_orig,
                triggers,
                artifact_len,
                context=context,
                picks=original_picks[start:stop],
            )
            min_ch = min(data_corr.shape[0], data_orig.shape[0])
            data_corr = data_corr[:min_ch]
            data_orig = data_orig[:min_ch]
            if data_corr.size == 0 or data_orig.size == 0:
                logger.warning("Insufficient data for FFT Niazy")
                return context

            nperseg = min(int(4 * sfreq), data_corr.shape[1])
            if self.verbose:
                logger.info(
                    "FFT Niazy diagnostics block {}: nperseg={}, corrected {}; original {}",
                    block_index,
                    nperseg,
                    _signal_summary(data_corr),
                    _signal_summary(data_orig),
                )

            freqs, psd_corr = signal.welch(
                data_corr,
                fs=sfreq,
                nperseg=nperseg,
                axis=1,
            )
            _, psd_orig = signal.welch(
                data_orig,
                fs=sfreq,
                nperseg=nperseg,
                axis=1,
            )
            self._append_harmonic_powers(
                harmonic_powers,
                freqs,
                psd_corr,
                psd_orig,
                fundamentals,
            )

        results = self._ratios_from_harmonic_powers(
            harmonic_powers,
            fundamentals,
            verbose=self.verbose,
        )

        slice_h1 = results["slice"].get("h1", float("nan"))
        if not np.isnan(slice_h1):
            logger.info("FFT Niazy Slice h1: {:.2f} dB", slice_h1)

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        new_metadata.custom.setdefault("metrics", {})["fft_niazy"] = results

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )

    def _estimate_frequencies(
        self,
        triggers: np.ndarray,
        sfreq: float,
        context: ProcessingContext,
    ):
        """Estimate slice and volume frequencies from triggers and context.

        Returns
        -------
        tuple
            (slice_freq, vol_freq) where vol_freq may be None.
        """
        mean_trigger_diff = np.mean(np.diff(triggers))
        slice_freq = sfreq / mean_trigger_diff

        slices_per_vol = getattr(context.metadata, "slices_per_volume", None)
        if not slices_per_vol:
            slices_per_vol = context.metadata.custom.get("slices_per_volume")

        if slices_per_vol:
            vol_freq = slice_freq / slices_per_vol
        else:
            vol_freq = None
            logger.warning("Slices per volume not found, skipping volume frequency analysis")

        return slice_freq, vol_freq

    def _compute_harmonic_ratios(
        self,
        freqs: np.ndarray,
        psd_corr: np.ndarray,
        psd_orig: np.ndarray,
        slice_freq: float,
        vol_freq: float | None,
        harmonics: int = 5,
        tolerance: float = 0.5,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Compute power ratios (orig/corr in dB) at harmonic frequencies.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency array.
        psd_corr : np.ndarray
            PSD of corrected data, shape (n_channels, n_freqs).
        psd_orig : np.ndarray
            PSD of original data, shape (n_channels, n_freqs).
        slice_freq : float
            Fundamental slice frequency in Hz.
        vol_freq : float or None
            Fundamental volume frequency in Hz, or None to skip.
        harmonics : int, optional
            Number of harmonics to analyze (default: 5).
        tolerance : float, optional
            Half-bandwidth around each harmonic in Hz (default: 0.5).

        Returns
        -------
        dict
            Nested dict with 'slice' and 'volume' harmonic ratios in dB.
        """
        results: dict[str, Any] = {"slice": {}, "volume": {}}

        def _ratio_db_at(f: float) -> float | None:
            idx = np.logical_and(freqs >= f - tolerance, freqs <= f + tolerance)
            if not np.any(idx):
                return None
            p_corr = np.median(np.mean(psd_corr[:, idx], axis=1))
            p_orig = np.median(np.mean(psd_orig[:, idx], axis=1))
            if p_corr <= 0:
                return None
            return float(10 * np.log10(p_orig / p_corr))

        for h in range(1, harmonics + 1):
            ratio_db = _ratio_db_at(slice_freq * h)
            if ratio_db is not None:
                results["slice"][f"h{h}"] = ratio_db
                if verbose:
                    logger.info(
                        "FFT Niazy diagnostics: slice h{} @ {:.3f}Hz ratio={:.2f} dB",
                        h,
                        slice_freq * h,
                        ratio_db,
                    )

        if vol_freq is not None:
            for h in range(1, harmonics + 1):
                ratio_db = _ratio_db_at(vol_freq * h)
                if ratio_db is not None:
                    results["volume"][f"h{h}"] = ratio_db
                    if verbose:
                        logger.info(
                            "FFT Niazy diagnostics: volume h{} @ {:.3f}Hz ratio={:.2f} dB",
                            h,
                            vol_freq * h,
                            ratio_db,
                        )

        return results

    @staticmethod
    def _append_harmonic_powers(
        target: dict[str, dict[str, dict[str, list[np.ndarray]]]],
        freqs: np.ndarray,
        psd_corr: np.ndarray,
        psd_orig: np.ndarray,
        fundamentals: dict[str, float],
        *,
        harmonics: int = 5,
        tolerance: float = 0.5,
    ) -> None:
        """Append per-channel harmonic powers from one channel block."""
        for kind, fundamental in fundamentals.items():
            for harmonic in range(1, harmonics + 1):
                frequency = fundamental * harmonic
                idx = np.logical_and(
                    freqs >= frequency - tolerance,
                    freqs <= frequency + tolerance,
                )
                if not np.any(idx):
                    continue
                key = f"h{harmonic}"
                target[kind][key]["corrected"].append(np.mean(psd_corr[:, idx], axis=1))
                target[kind][key]["original"].append(np.mean(psd_orig[:, idx], axis=1))

    @staticmethod
    def _ratios_from_harmonic_powers(
        powers: dict[str, dict[str, dict[str, list[np.ndarray]]]],
        fundamentals: dict[str, float],
        *,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Reduce block-wise per-channel powers using the legacy medians."""
        results: dict[str, Any] = {"slice": {}, "volume": {}}
        for kind, fundamental in fundamentals.items():
            for key, values in powers[kind].items():
                if not values["corrected"] or not values["original"]:
                    continue
                corrected = np.concatenate(values["corrected"])
                original = np.concatenate(values["original"])
                p_corr = np.median(corrected)
                if p_corr <= 0:
                    continue
                p_orig = np.median(original)
                ratio = float(10 * np.log10(p_orig / p_corr))
                results[kind][key] = ratio
                if verbose:
                    harmonic = int(key[1:])
                    logger.info(
                        "FFT Niazy diagnostics: {} {} @ {:.3f}Hz ratio={:.2f} dB",
                        kind,
                        key,
                        fundamental * harmonic,
                        ratio,
                    )
        return results


@register_processor
class MetricsReport(Processor):
    """Generate a summary report of all calculated metrics.

    Collects all metrics from context and logs a formatted summary. Can also
    store results in a shared dictionary for comparison and plotting.

    Parameters
    ----------
    name : str, optional
        Name of the result set (e.g., ``"Pipeline A"``). If ``None``, a
        default name is generated during processing.
    store : dict, optional
        Dictionary to accumulate results. Structure: ``{name: {metric: value}}``.

    Examples
    --------
    ::

        # Basic usage
        report = MetricsReport()
        context = report.execute(context)

        # Advanced usage (collecting results for comparison)
        results = {}
        report = MetricsReport(name="Pipeline A", store=results)
        context = report.execute(context)

        # Plot comparison
        MetricsReport.plot(results)

        # Plot specific metrics
        MetricsReport.plot(results, metrics=['snr', 'rms_ratio'])
    """

    name = "metrics_report"
    description = "Generate metrics summary report"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = False
    modifies_raw = False
    parallel_safe = False

    def __init__(self, name: str | None = None, store: dict | None = None) -> None:
        self.report_name = name
        self.store = store
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        metrics = context.metadata.custom.get("metrics", {})

        # --- COMPUTE ---
        if not metrics:
            logger.warning("No metrics available")
            return context

        self._log_metrics(metrics)

        if self.store is not None:
            result_name = self.report_name or f"Result_{len(self.store) + 1}"
            self.store[result_name] = self._flatten_metrics(metrics)
            logger.info("Stored metrics for '{}'", result_name)

        # --- RETURN ---
        return context

    def _log_metrics(self, metrics: dict[str, Any]) -> None:
        """Render all available metrics — rich panel when available, plain log fallback."""
        from ..console import get_console

        if get_console().enabled:
            self._plain_log_metrics(metrics)
            return
        try:
            self._rich_log_metrics(metrics)
        except Exception:
            self._plain_log_metrics(metrics)

    def _rich_log_metrics(self, metrics: dict[str, Any]) -> None:
        """Render metrics as a rich-formatted panel."""
        from rich import box
        from rich.console import Console as RichConsole
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = RichConsole(highlight=False)
        table = Table(
            box=None,
            show_header=True,
            padding=(0, 2),
            expand=True,
            show_edge=False,
        )
        table.add_column("Metric", style="bold", ratio=3)
        table.add_column("Value", style="white", ratio=2, justify="left")
        table.add_column("", style="dim italic", ratio=1)

        def _section(title: str) -> None:
            table.add_row("", "", "")
            table.add_row(Text(title, style="bold yellow underline"), "", "")

        # --- Core Metrics ---
        core_keys = ("snr", "rms_ratio", "rms_residual", "median_artifact", "legacy_snr")
        if any(k in metrics for k in core_keys):
            _section("Core Metrics")

            if "snr" in metrics:
                snr = metrics["snr"]
                color = "green" if snr > 10 else ("yellow" if snr > 3 else "red")
                table.add_row("SNR (Signal-to-Noise Ratio)", f"[{color}]{snr:.2f}[/]", "")

            if "rms_ratio" in metrics:
                table.add_row("RMS Ratio (improvement)", f"{metrics['rms_ratio']:.2f}", "×")

            if "rms_residual" in metrics:
                r = metrics["rms_residual"]
                color = "green" if abs(r - 1.0) < 0.1 else ("yellow" if abs(r - 1.0) < 0.3 else "red")
                table.add_row("RMS Residual Ratio", f"[{color}]{r:.2f}[/]", "target: 1.0")

            if "median_artifact" in metrics:
                table.add_row("Median Artifact Amplitude", f"{metrics['median_artifact']:.2e}", "")
                if "median_artifact_ratio" in metrics:
                    r = metrics["median_artifact_ratio"]
                    color = "green" if abs(r - 1.0) < 0.2 else ("yellow" if abs(r - 1.0) < 0.6 else "red")
                    table.add_row("Median Artifact Ratio", f"[{color}]{r:.2f}[/]", "target: 1.0")

            if "legacy_snr" in metrics:
                table.add_row("Legacy SNR", f"{metrics['legacy_snr']:.2f}", "")

        # --- FFT Allen ---
        if "fft_allen" in metrics:
            _section("FFT Allen — Spectral Diff to Reference")
            for band, val in metrics["fft_allen"].items():
                table.add_row(f"{band.capitalize()}", f"{val:.2f}%", "")

        # --- FFT Niazy ---
        if "fft_niazy" in metrics:
            _section("FFT Niazy — Power Ratio (Uncorr / Corr)")
            if "slice" in metrics["fft_niazy"]:
                harmonics = "  ".join(f"[cyan]{k}[/]: {v:.2f}" for k, v in metrics["fft_niazy"]["slice"].items())
                table.add_row("Slice Harmonics", harmonics, "dB")
            if "volume" in metrics["fft_niazy"]:
                harmonics = "  ".join(f"[cyan]{k}[/]: {v:.2f}" for k, v in metrics["fft_niazy"]["volume"].items())
                table.add_row("Volume Harmonics", harmonics, "dB")

        console.print()
        console.print(
            Panel(
                table,
                title="[bold white] Evaluation Metrics Report [/]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def _plain_log_metrics(self, metrics: dict[str, Any]) -> None:
        """Fallback plain loguru output for the interactive console / no-TTY case."""
        logger.info("=" * 60)
        logger.info("EVALUATION METRICS REPORT")
        logger.info("=" * 60)

        if "snr" in metrics:
            logger.info("SNR (Signal-to-Noise Ratio):     {:.2f}", metrics["snr"])

        if "rms_ratio" in metrics:
            logger.info("RMS Ratio (improvement):         {:.2f}", metrics["rms_ratio"])

        if "rms_residual" in metrics:
            logger.info(
                "RMS Residual Ratio (ref match):  {:.2f} (target: 1.0)",
                metrics["rms_residual"],
            )

        if "median_artifact" in metrics:
            logger.info("Median Artifact Amplitude:       {:.2e}", metrics["median_artifact"])
            if "median_artifact_ratio" in metrics:
                logger.info(
                    "Median Artifact Ratio (to ref):  {:.2f} (target: 1.0)",
                    metrics["median_artifact_ratio"],
                )

        if "legacy_snr" in metrics:
            logger.info("Legacy SNR:                      {:.2f}", metrics["legacy_snr"])

        if "fft_allen" in metrics:
            logger.info("FFT Allen (Diff to Ref):")
            for band, val in metrics["fft_allen"].items():
                logger.info("  - {}: {:.2f}%", band, val)

        if "fft_niazy" in metrics:
            logger.info("FFT Niazy (Power Ratio Uncorr/Corr dB):")
            if "slice" in metrics["fft_niazy"]:
                logger.info("  Slice Harmonics:")
                for k, v in metrics["fft_niazy"]["slice"].items():
                    logger.info("    - {}: {:.2f} dB", k, v)
            if "volume" in metrics["fft_niazy"]:
                logger.info("  Volume Harmonics:")
                for k, v in metrics["fft_niazy"]["volume"].items():
                    logger.info("    - {}: {:.2f} dB", k, v)

        logger.info("=" * 60)

    @staticmethod
    def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        """Flatten nested metric dicts to scalar values for plotting.

        Parameters
        ----------
        metrics : dict
            Raw metrics dict from context (may contain nested dicts).

        Returns
        -------
        dict
            Flat dict of scalar metric values.
        """
        scalar_metrics: dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.number)):
                scalar_metrics[k] = float(v)
            elif k == "fft_allen" and isinstance(v, dict):
                for band, val in v.items():
                    scalar_metrics[f"fft_allen_{band}"] = float(val)
            elif k == "fft_niazy" and isinstance(v, dict) and "slice" in v:
                scalar_metrics["fft_niazy_slice_h1"] = float(v["slice"].get("h1", 0))
        return scalar_metrics

    @staticmethod
    def compare(
        results: list | dict,
        labels: list[str] | None = None,
        title: str = "Metrics Comparison",
        save_path: str | None = None,
        show: bool = True,
        metrics: list[str] | None = None,
    ) -> None:
        """Compare metrics from a list of ``PipelineResult`` objects or a plain dict.

        Accepts either:

        - A list of ``PipelineResult`` instances (optionally named via *labels*).
        - The legacy ``{name: {metric: value}}`` dict format used by
          ``MetricsReport.plot``.

        Parameters
        ----------
        results : list or dict
            List of ``PipelineResult`` objects **or** legacy dict.
        labels : list of str, optional
            Names for each result when passing a list. Defaults to
            ``["Result 1", "Result 2", …]``.
        title : str, optional
            Plot title (default: "Metrics Comparison").
        save_path : str, optional
            If set, save the figure to this path.
        show : bool, optional
            Whether to display the figure interactively (default: True).
        metrics : list of str, optional
            Subset of metric keys to show. If ``None``, all metrics are plotted.

        Examples
        --------
        ::

            result_a = pipeline_aas.run()
            result_b = pipeline_aas_pca.run()

            MetricsReport.compare(
                [result_a, result_b],
                labels=["AAS only", "AAS + PCA"],
            )
        """
        if isinstance(results, dict):
            MetricsReport.plot(results, title=title, save_path=save_path, show=show, metrics=metrics)
            return

        if labels is None:
            labels = [f"Result {i + 1}" for i in range(len(results))]

        results_dict: dict[str, dict[str, float]] = {}
        for label, result in zip(labels, results, strict=False):
            raw_metrics = result.metrics if hasattr(result, "metrics") else {}
            results_dict[label] = MetricsReport._flatten_metrics(raw_metrics)

        MetricsReport.plot(results_dict, title=title, save_path=save_path, show=show, metrics=metrics)

    @staticmethod
    def plot(
        results: dict[str, dict[str, float]],
        title: str = "Metrics Comparison",
        save_path: str | None = None,
        show: bool = True,
        metrics: list[str] | None = None,
    ) -> None:
        """Plot comparison of metrics using Matplotlib.

        Parameters
        ----------
        results : dict
            Dictionary of results ``{name: {metric: value}}``.
        title : str, optional
            Plot title (default: "Metrics Comparison").
        save_path : str, optional
            Path to save the figure.
        show : bool, optional
            Whether to show the plot (default: True).
        metrics : list of str, optional
            Subset of metric keys to plot. If ``None``, all are shown.
        """
        if not results:
            logger.warning("No results to plot")
            return

        df = pd.DataFrame.from_dict(results, orient="index")

        if df.empty:
            logger.warning("Results DataFrame is empty")
            return

        if metrics:
            existing_metrics = [m for m in metrics if m in df.columns]
            if not existing_metrics:
                logger.warning("None of the requested metrics {} found in results.", metrics)
                return
            df = df[existing_metrics]

        metrics_list = df.columns.tolist()
        n_metrics = len(metrics_list)

        if n_metrics == 0:
            logger.warning("No metrics found in results")
            return

        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
        axes = axes.flatten()

        cmap = plt.get_cmap("viridis")
        colors = [cmap(i) for i in np.linspace(0.2, 0.8, len(df))]

        for i, metric in enumerate(metrics_list):
            ax = axes[i]
            values = df[metric]
            values.plot(kind="bar", ax=ax, color=colors, rot=45)

            ax.set_title(metric)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            for p in ax.patches:
                height = p.get_height()
                ax.annotate(
                    f"{height:.2g}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info("Saved metrics plot to {}", save_path)

        if show:
            plt.show()
        else:
            plt.close()
