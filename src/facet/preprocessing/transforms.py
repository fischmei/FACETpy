"""
Simple raw-transform processors.

Contains small, focused processors for common in-pipeline data manipulations
that don't fit neatly into filtering, resampling, or trigger handling.
"""

import contextlib
import re
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
from loguru import logger
from matplotlib.widgets import Button, RadioButtons, Slider, SpanSelector

from ..console import get_console, suspend_raw_mode
from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor
from ..helpers.plotting import show_matplotlib_figure
from ..misc import EEGGenerator

_STANDARD_1020_19_CHANNELS: list[str] = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]

_STANDARD_1020_32_CHANNELS: list[str] = [
    "Fp1",
    "Fpz",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "POz",
    "O1",
    "Oz",
    "O2",
    "Iz",
    "C1",
]

_CHANNEL_STANDARD_PRESETS: dict[str, list[str]] = {
    "standard_1020_19": _STANDARD_1020_19_CHANNELS,
    "standard_1020_32": _STANDARD_1020_32_CHANNELS,
}

_CHANNEL_STANDARD_ALIASES: dict[str, str] = {
    "10-20": "standard_1020_19",
    "10_20": "standard_1020_19",
    "1020": "standard_1020_19",
    "1020_19": "standard_1020_19",
    "standard_1020": "standard_1020_19",
    "standard_1020_19": "standard_1020_19",
    "32": "standard_1020_32",
    "1020_32": "standard_1020_32",
    "standard_1020_32": "standard_1020_32",
}

_CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {
    "T7": ("T3",),
    "T8": ("T4",),
    "P7": ("T5",),
    "P8": ("T6",),
    "M1": ("A1",),
    "M2": ("A2",),
}


def _normalize_channel_name(name: str) -> str:
    """Normalize channel names for robust matching."""
    return "".join(char for char in name.upper() if char.isalnum())


_NORMALIZED_CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {
    _normalize_channel_name(target): tuple(_normalize_channel_name(alias) for alias in aliases)
    for target, aliases in _CHANNEL_ALIASES.items()
}

_REVERSE_CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {}
for canonical_name, alias_names in _NORMALIZED_CHANNEL_ALIASES.items():
    for alias_name in alias_names:
        _REVERSE_CHANNEL_ALIASES.setdefault(alias_name, tuple())
        _REVERSE_CHANNEL_ALIASES[alias_name] = _REVERSE_CHANNEL_ALIASES[alias_name] + (canonical_name,)


@register_processor
class Crop(Processor):
    """Crop the Raw recording to a time interval.

    A concise alternative to ``LambdaProcessor`` for the common pattern of
    restricting the recording to a specific window before processing.

    Parameters
    ----------
    tmin : float, optional
        Start time in seconds.  ``None`` keeps the original start.
    tmax : float, optional
        End time in seconds.  ``None`` keeps the original end.

        If both ``tmin`` and ``tmax`` are ``None``, an interactive
        Matplotlib selector is opened to choose the crop window.

    Examples
    --------
    ::

        Crop(tmin=0, tmax=162)
        Crop()  # open interactive selector
    """

    name = "crop"
    description = "Crop Raw recording to a time interval"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = False

    def __init__(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
    ):
        self.tmin = tmin
        self.tmax = tmax
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)

        if self.tmin is not None and self.tmax is not None and self.tmax <= self.tmin:
            raise ProcessorValidationError(f"tmax must be greater than tmin, got tmin={self.tmin}, tmax={self.tmax}")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()
        resolved_tmin = self.tmin
        resolved_tmax = self.tmax

        # --- COMPUTE ---
        if self.tmin is None and self.tmax is None:
            logger.info("No crop boundaries provided; opening interactive crop selector.")
            selected_interval = self._show_interactive_crop_selector(raw)
            if selected_interval is not None:
                resolved_tmin, resolved_tmax = selected_interval
            else:
                logger.info("Interactive crop selection cancelled; keeping full recording.")

        kwargs = {}
        if resolved_tmin is not None:
            kwargs["tmin"] = resolved_tmin
        if resolved_tmax is not None:
            kwargs["tmax"] = resolved_tmax

        if resolved_tmin is not None and resolved_tmax is not None and resolved_tmax <= resolved_tmin:
            raise ProcessorValidationError(f"invalid crop interval: tmin={resolved_tmin}, tmax={resolved_tmax}")

        if kwargs:
            logger.info("Cropping raw: tmin={}, tmax={}", resolved_tmin, resolved_tmax)
            raw.crop(**kwargs)
        else:
            logger.info("Cropping skipped; no boundaries selected.")

        # --- RETURN ---
        return context.with_raw(raw)

    def _show_interactive_crop_selector(self, raw: mne.io.BaseRaw) -> tuple[float, float] | None:
        """Show interactive span selector and return selected crop bounds."""
        backend = plt.get_backend().lower()
        if "agg" in backend:
            logger.warning("Matplotlib backend '{}' is non-interactive; skipping crop selector.", backend)
            return None

        if raw.n_times < 2:
            return None

        sfreq = float(raw.info["sfreq"])
        if sfreq <= 0:
            return None

        eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        ch_idx = int(eeg_picks[0]) if len(eeg_picks) > 0 else 0
        ch_name = raw.ch_names[ch_idx]
        channel_data = raw.get_data(picks=[ch_idx])[0]
        time_axis = raw.times

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.24)

        ax.plot(time_axis, channel_data, linewidth=0.8, alpha=0.9)
        ax.set_title(f"Select crop interval - {ch_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

        interval_state: dict[str, Any] = {
            "tmin": float(time_axis[0]),
            "tmax": float(time_axis[-1]),
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

        min_duration = 1.0 / sfreq

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

            left = round(left * sfreq) / sfreq
            right = round(right * sfreq) / sfreq
            if right <= left:
                right = min(float(time_axis[-1]), left + min_duration)
                left = max(0.0, right - min_duration)

            interval_state["tmin"] = left
            interval_state["tmax"] = right
            _refresh_overlay()

        span_selector = SpanSelector(
            ax,
            _on_select,
            "horizontal",
            useblit=True,
            interactive=True,
            drag_from_anywhere=True,
            props=dict(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.25),
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
        console.set_active_prompt("Drag to select crop interval, then click Confirm")
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
class MagicErasor(Processor):
    """Interactively erase selected signal segments with configurable modes.

    Opens an interactive matplotlib editor for one preview channel and lets the
    user select time segments repeatedly. Each selected segment can be replaced
    using one of four modes:

    - ``zero``: set samples to zero.
    - ``mean``: set samples to the channel mean.
    - ``interpolate``: linearly interpolate between segment boundaries.
    - ``generated_eeg``: replace with synthetic EEG generated through
      ``EEGGenerator``, then adapt channel-wise mean and
      amplitude to the local surrounding signal.

    The editor stays open until the user clicks **Done**, enabling multiple
    edits in a single session.

    Parameters
    ----------
    picks : str or list[str], optional
        Channels to edit (default: ``"eeg"``).
    channel : str or int, optional
        Channel used for preview in the interactive window. When ``None``,
        the first edited EEG channel is used.
    default_mode : str, optional
        Initially selected editing mode (default: ``"zero"``).
    random_seed : int, optional
        Optional seed used when ``generated_eeg`` mode is applied.
    """

    name = "magic_erasor"
    description = "Interactively erase selected segments with multiple replacement modes"
    version = "1.1.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = False
    channel_wise = False

    _VALID_MODES = ("zero", "mean", "interpolate", "generated_eeg")

    def __init__(
        self,
        picks: str | list[str] = "eeg",
        channel: str | int | None = None,
        default_mode: str = "zero",
        random_seed: int | None = None,
    ) -> None:
        self.picks = picks
        self.channel = channel
        self.default_mode = default_mode
        self.random_seed = random_seed
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)

        raw = context.get_raw()
        if raw.n_times < 2:
            raise ProcessorValidationError("Raw must contain at least 2 samples for interactive editing.")

        if raw.info["sfreq"] <= 0:
            raise ProcessorValidationError("Sampling frequency must be positive.")

        if self.default_mode not in self._VALID_MODES:
            raise ProcessorValidationError(
                f"default_mode must be one of {self._VALID_MODES}, got '{self.default_mode}'"
            )

        target_picks = self._resolve_target_picks(raw)
        self._resolve_preview_channel(raw, target_picks)

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()
        sfreq = float(raw.info["sfreq"])
        target_picks = self._resolve_target_picks(raw)
        preview_channel = self._resolve_preview_channel(raw, target_picks)
        edited_data = raw.get_data().copy()

        # --- LOG ---
        logger.info(
            "Opening magic_erasor editor on {} channels (preview='{}')",
            len(target_picks),
            raw.ch_names[preview_channel],
        )

        # --- COMPUTE ---
        edits = self._show_interactive_editor(
            data=edited_data,
            sfreq=sfreq,
            target_picks=target_picks,
            preview_channel=preview_channel,
            channel_names=raw.ch_names,
        )
        if edits is None:
            logger.info("magic_erasor cancelled; returning context unchanged.")
            return context
        if not edits:
            logger.info("magic_erasor finished without edits; returning context unchanged.")
            return context

        raw._data[:] = edited_data
        result = context.with_raw(raw)

        # --- NOISE ---
        if context.has_estimated_noise():
            noise = context.get_estimated_noise().copy()
            self._apply_edits_to_noise(noise, target_picks, edits)
            result.set_estimated_noise(noise)
        else:
            logger.debug("No noise estimate present - skipping noise propagation in magic_erasor")

        # --- METADATA ---
        metadata = result.metadata.copy()
        metadata.custom["magic_erasor"] = {
            "channel": raw.ch_names[preview_channel],
            "picks": [raw.ch_names[idx] for idx in target_picks],
            "n_edits": len(edits),
            "edits": edits,
        }
        logger.info("magic_erasor applied {} edit(s).", len(edits))

        # --- RETURN ---
        return result.with_metadata(metadata)

    def _resolve_target_picks(self, raw: mne.io.BaseRaw) -> list[int]:
        """Resolve configured picks to channel indices."""
        try:
            picked_raw = raw.copy().pick(picks=self.picks, verbose=False)
        except Exception as exc:
            raise ProcessorValidationError(f"Invalid picks '{self.picks}': {exc}") from exc

        picks = [raw.ch_names.index(name) for name in picked_raw.ch_names]
        if len(picks) == 0:
            raise ProcessorValidationError(f"No channels selected by picks='{self.picks}'.")
        return picks

    def _resolve_preview_channel(self, raw: mne.io.BaseRaw, target_picks: list[int]) -> int:
        """Resolve configured preview channel."""
        if self.channel is None:
            eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
            for idx in eeg_picks:
                if int(idx) in target_picks:
                    return int(idx)
            return int(target_picks[0])

        if isinstance(self.channel, int):
            if self.channel < 0 or self.channel >= len(raw.ch_names):
                raise ProcessorValidationError(f"channel index out of range: {self.channel}")
            return int(self.channel)

        if self.channel not in raw.ch_names:
            raise ProcessorValidationError(f"channel '{self.channel}' not found")
        return int(raw.ch_names.index(self.channel))

    def _show_interactive_editor(
        self,
        data: np.ndarray,
        sfreq: float,
        target_picks: list[int],
        preview_channel: int,
        channel_names: list[str],
    ) -> list[dict[str, Any]] | None:
        """Show the interactive editing window and return applied edits."""
        backend = plt.get_backend().lower()
        if "agg" in backend:
            logger.warning("Matplotlib backend '{}' is non-interactive; skipping magic_erasor.", backend)
            return None

        n_times = data.shape[1]
        time_axis = np.arange(n_times) / sfreq

        fig, ax = plt.subplots(figsize=(14, 8))
        plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.30)

        (line,) = ax.plot(time_axis, data[preview_channel], linewidth=0.8, alpha=0.9)
        ax.set_title(f"Magic Erasor - {channel_names[preview_channel]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

        status_label = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        state: dict[str, Any] = {
            "selection": None,
            "mode": self.default_mode,
            "confirmed": False,
            "shade": None,
        }
        edits: list[dict[str, Any]] = []
        max_time = float(time_axis[-1])
        min_window = max(10.0 / sfreq, 0.05)
        default_window = min(max(max_time * 0.25, 1.0), max_time) if max_time > 0 else min_window
        default_center = max_time * 0.5
        default_y_zoom = 0.5

        def _close_fig() -> None:
            with contextlib.suppress(Exception):
                fig.canvas.manager.destroy()
            plt.close(fig)

        def _update_view_limits() -> None:
            center = float(view_center_slider.val)
            window = float(view_window_slider.val)
            y_zoom = float(y_zoom_slider.val)

            half_window = 0.5 * window
            left = max(0.0, center - half_window)
            right = min(max_time, center + half_window)
            if right - left < min_window:
                left = max(0.0, right - min_window)
                right = min(max_time, left + min_window)
            ax.set_xlim(left, right if right > left else left + min_window)

            left_idx = max(0, int(np.floor(left * sfreq)))
            right_idx = min(n_times, max(left_idx + 1, int(np.ceil(right * sfreq))))
            view_data = data[preview_channel, left_idx:right_idx]
            if view_data.size == 0:
                view_data = data[preview_channel]

            y_center = float(np.median(view_data))
            centered = view_data - y_center
            robust_span = float(np.percentile(np.abs(centered), 99))
            if not np.isfinite(robust_span) or robust_span <= 0.0:
                robust_span = float(np.max(np.abs(centered))) if centered.size > 0 else 1.0
            robust_span = max(robust_span, 1e-12)
            half_range = robust_span / max(y_zoom, 1e-3)
            ax.set_ylim(y_center - half_range, y_center + half_range)

        def _refresh_overlay() -> None:
            if state["shade"] is not None:
                state["shade"].remove()
                state["shade"] = None

            selection = state["selection"]
            if selection is not None:
                start_sample, end_sample = selection
                state["shade"] = ax.axvspan(
                    start_sample / sfreq,
                    end_sample / sfreq,
                    facecolor="tab:orange",
                    alpha=0.22,
                    edgecolor="tab:orange",
                    linewidth=1.0,
                )
                selected_text = (
                    f"Selection: {start_sample / sfreq:.3f}s to {end_sample / sfreq:.3f}s "
                    f"({(end_sample - start_sample) / sfreq:.3f}s)"
                )
            else:
                selected_text = "Selection: none"

            status_label.set_text(
                f"Mode: {state['mode']}\n{selected_text}\nApplied edits: {len(edits)}\n"
                "Drag to select. Adjust view sliders for precision. Click Done when satisfied."
            )
            _update_view_limits()
            fig.canvas.draw_idle()

        def _on_select(xmin: float, xmax: float) -> None:
            if xmin is None or xmax is None:
                return

            left = max(0.0, min(float(xmin), float(xmax)))
            right = min(float(time_axis[-1]), max(float(xmin), float(xmax)))

            start_sample = int(np.floor(left * sfreq))
            end_sample = int(np.ceil(right * sfreq))
            start_sample = max(0, min(start_sample, n_times - 1))
            end_sample = max(start_sample + 1, min(end_sample, n_times))

            state["selection"] = (start_sample, end_sample)
            _refresh_overlay()

        span_selector = SpanSelector(
            ax,
            _on_select,
            "horizontal",
            useblit=True,
            interactive=True,
            drag_from_anywhere=True,
            props=dict(facecolor="tab:orange", edgecolor="tab:orange", alpha=0.25),
        )
        span_selector.set_active(True)

        mode_ax = fig.add_axes([0.08, 0.05, 0.22, 0.18])
        mode_selector = RadioButtons(mode_ax, self._VALID_MODES, active=self._VALID_MODES.index(self.default_mode))
        mode_ax.set_title("Mode")

        def _on_mode_changed(label: str) -> None:
            state["mode"] = label
            _refresh_overlay()

        mode_selector.on_clicked(_on_mode_changed)

        # Keep generous left/right padding so slider labels and value readouts
        # do not overlap with mode/buttons panels.
        slider_left = 0.44
        slider_width = 0.30

        center_ax = fig.add_axes([slider_left, 0.18, slider_width, 0.03])
        view_center_slider = Slider(
            center_ax,
            "View Center (s)",
            0.0,
            max_time if max_time > 0 else min_window,
            valinit=default_center,
            valstep=1.0 / sfreq,
        )
        window_ax = fig.add_axes([slider_left, 0.13, slider_width, 0.03])
        view_window_slider = Slider(
            window_ax,
            "Window (s)",
            min_window,
            max(max_time, min_window),
            valinit=max(default_window, min_window),
            valstep=1.0 / sfreq,
        )
        y_zoom_ax = fig.add_axes([slider_left, 0.08, slider_width, 0.03])
        y_zoom_slider = Slider(
            y_zoom_ax,
            "Y Zoom",
            0.25,
            3.0,
            valinit=default_y_zoom,
        )

        def _on_view_change(_val: float) -> None:
            _update_view_limits()
            fig.canvas.draw_idle()

        view_center_slider.on_changed(_on_view_change)
        view_window_slider.on_changed(_on_view_change)
        y_zoom_slider.on_changed(_on_view_change)

        apply_ax = fig.add_axes([0.80, 0.17, 0.17, 0.06])
        apply_btn = Button(apply_ax, "Apply Edit")
        done_ax = fig.add_axes([0.80, 0.10, 0.17, 0.06])
        done_btn = Button(done_ax, "Done")
        cancel_ax = fig.add_axes([0.80, 0.03, 0.17, 0.06])
        cancel_btn = Button(cancel_ax, "Cancel")

        def _on_apply(_) -> None:
            selection = state["selection"]
            if selection is None:
                logger.warning("No segment selected; nothing to apply.")
                return

            start_sample, end_sample = selection
            mode = str(state["mode"])
            self._apply_edit(
                data=data,
                target_picks=target_picks,
                start_sample=start_sample,
                end_sample=end_sample,
                mode=mode,
                sfreq=sfreq,
                edit_index=len(edits),
            )
            edits.append(
                {
                    "mode": mode,
                    "start_sample": int(start_sample),
                    "end_sample": int(end_sample),
                    "start_time": float(start_sample / sfreq),
                    "end_time": float(end_sample / sfreq),
                }
            )

            line.set_ydata(data[preview_channel])
            ax.relim()
            ax.autoscale_view()
            _refresh_overlay()

        def _on_done(_) -> None:
            state["confirmed"] = True
            _close_fig()

        def _on_cancel(_) -> None:
            _close_fig()

        apply_btn.on_clicked(_on_apply)
        done_btn.on_clicked(_on_done)
        cancel_btn.on_clicked(_on_cancel)
        _refresh_overlay()

        console = get_console()
        console.set_active_prompt(
            "Magic Erasor: drag-select, set mode, tune View/Y sliders for precision, apply edits, click Done"
        )
        try:
            with suspend_raw_mode():
                show_matplotlib_figure(fig)
        finally:
            plt.close(fig)
            console.clear_active_prompt()

        if not state["confirmed"]:
            return None
        return edits

    def _apply_edit(
        self,
        data: np.ndarray,
        target_picks: list[int],
        start_sample: int,
        end_sample: int,
        mode: str,
        sfreq: float,
        edit_index: int,
    ) -> None:
        """Apply one editing operation in-place."""
        picks_array = np.asarray(target_picks, dtype=int)
        if mode == "zero":
            data[picks_array, start_sample:end_sample] = 0.0
            return

        if mode == "mean":
            channel_means = np.mean(data[picks_array], axis=1, keepdims=True)
            data[picks_array, start_sample:end_sample] = channel_means
            return

        if mode == "interpolate":
            self._apply_interpolation(data, picks_array, start_sample, end_sample)
            return

        if mode == "generated_eeg":
            generated = self._generate_segment(
                n_channels=len(picks_array),
                n_samples=end_sample - start_sample,
                sfreq=sfreq,
                edit_index=edit_index,
            )
            adapted = self._adapt_segment_to_environment(
                generated=generated,
                data=data,
                picks_array=picks_array,
                start_sample=start_sample,
                end_sample=end_sample,
            )
            data[picks_array, start_sample:end_sample] = adapted
            return

        raise ProcessorValidationError(f"Unsupported mode '{mode}'.")

    def _apply_interpolation(
        self,
        data: np.ndarray,
        picks_array: np.ndarray,
        start_sample: int,
        end_sample: int,
    ) -> None:
        """Linearly interpolate selected interval for each target channel."""
        n_times = data.shape[1]
        for ch_idx in picks_array:
            channel = data[ch_idx]
            has_left = start_sample > 0
            has_right = end_sample < n_times

            if has_left and has_right:
                left_val = float(channel[start_sample - 1])
                right_val = float(channel[end_sample])
                channel[start_sample:end_sample] = np.linspace(
                    left_val,
                    right_val,
                    num=(end_sample - start_sample) + 2,
                )[1:-1]
            elif has_left:
                channel[start_sample:end_sample] = float(channel[start_sample - 1])
            elif has_right:
                channel[start_sample:end_sample] = float(channel[end_sample])
            else:
                channel[start_sample:end_sample] = 0.0

    def _generate_segment(
        self,
        n_channels: int,
        n_samples: int,
        sfreq: float,
        edit_index: int,
    ) -> np.ndarray:
        """Generate synthetic EEG segment using EEGGenerator."""
        seed = None if self.random_seed is None else self.random_seed + edit_index
        generator = EEGGenerator(
            sampling_rate=sfreq,
            duration=n_samples / sfreq,
            channel_schema={
                "eeg_channels": n_channels,
                "eog_channels": 0,
                "ecg_channels": 0,
                "emg_channels": 0,
                "misc_channels": 0,
            },
            random_seed=seed,
        )
        generated_context = generator.process(None)
        generated = generated_context.get_raw().get_data(picks="eeg")
        return self._fit_segment_shape(generated, n_channels=n_channels, n_samples=n_samples)

    def _adapt_segment_to_environment(
        self,
        generated: np.ndarray,
        data: np.ndarray,
        picks_array: np.ndarray,
        start_sample: int,
        end_sample: int,
    ) -> np.ndarray:
        """Match generated segment statistics to surrounding channel data."""
        adapted = generated.copy()
        for out_idx, ch_idx in enumerate(picks_array):
            environment = self._extract_environment(
                channel_data=data[ch_idx],
                start_sample=start_sample,
                end_sample=end_sample,
            )
            target_mean, target_amplitude = self._compute_stats(environment)
            adapted[out_idx] = self._match_stats(
                segment=adapted[out_idx],
                target_mean=target_mean,
                target_amplitude=target_amplitude,
            )
        return adapted

    def _extract_environment(
        self,
        channel_data: np.ndarray,
        start_sample: int,
        end_sample: int,
    ) -> np.ndarray:
        """Extract neighboring samples around the edited interval."""
        n_times = channel_data.shape[0]
        window = max(1, end_sample - start_sample)

        left = channel_data[max(0, start_sample - window) : start_sample]
        right = channel_data[end_sample : min(n_times, end_sample + window)]

        if left.size + right.size > 0:
            return np.concatenate((left, right))

        fallback_left = channel_data[:start_sample]
        fallback_right = channel_data[end_sample:]
        if fallback_left.size + fallback_right.size > 0:
            return np.concatenate((fallback_left, fallback_right))

        return channel_data.copy()

    def _compute_stats(self, signal: np.ndarray) -> tuple[float, float]:
        """Compute mean and amplitude estimate for a signal."""
        if signal.size == 0:
            return 0.0, 1.0

        mean = float(np.mean(signal))
        centered = signal - mean
        amplitude = float(np.std(centered))
        if not np.isfinite(amplitude) or amplitude <= 1e-12:
            amplitude = float(np.max(np.abs(centered))) if centered.size > 0 else 1.0
        return mean, max(amplitude, 1e-12)

    def _match_stats(
        self,
        segment: np.ndarray,
        target_mean: float,
        target_amplitude: float,
    ) -> np.ndarray:
        """Shift and scale one segment to match target mean and amplitude."""
        source_mean = float(np.mean(segment))
        centered = segment - source_mean
        source_amplitude = float(np.std(centered))
        if not np.isfinite(source_amplitude) or source_amplitude <= 1e-12:
            return np.full(segment.shape, target_mean, dtype=float)

        scaled = centered * (target_amplitude / source_amplitude)
        return scaled + target_mean

    def _fit_segment_shape(self, segment: np.ndarray, n_channels: int, n_samples: int) -> np.ndarray:
        """Adapt generated segment to requested shape."""
        if segment.size == 0:
            return np.zeros((n_channels, n_samples))

        shaped = segment
        if shaped.shape[0] < n_channels:
            repeats = int(np.ceil(n_channels / shaped.shape[0]))
            shaped = np.tile(shaped, (repeats, 1))
        shaped = shaped[:n_channels]

        if shaped.shape[1] < n_samples:
            repeats = int(np.ceil(n_samples / shaped.shape[1]))
            shaped = np.tile(shaped, (1, repeats))
        shaped = shaped[:, :n_samples]
        return shaped

    def _apply_edits_to_noise(
        self,
        noise: np.ndarray,
        target_picks: list[int],
        edits: list[dict[str, Any]],
    ) -> None:
        """Apply compatible edits to estimated noise."""
        picks_array = np.asarray(target_picks, dtype=int)
        for edit in edits:
            start_sample = int(edit["start_sample"])
            end_sample = int(edit["end_sample"])
            mode = str(edit["mode"])
            if mode == "generated_eeg":
                noise[picks_array, start_sample:end_sample] = 0.0
                continue
            self._apply_edit(
                data=noise,
                target_picks=target_picks,
                start_sample=start_sample,
                end_sample=end_sample,
                mode=mode,
                sfreq=1.0,
                edit_index=0,
            )


@register_processor
class PickChannels(Processor):
    """Keep only the specified channels or channel types.

    A named, reusable alternative to the common ``lambda ctx: ctx.with_raw(
    ctx.get_raw().copy().pick(...))`` pattern.

    Parameters
    ----------
    picks : str or list of str
        Channel type (``"eeg"``, ``"stim"``, …) or list of channel
        names / types accepted by :meth:`mne.io.Raw.pick`.
    on_missing : str, optional
        Passed to MNE.  ``"ignore"`` (default) silently skips channels
        that are absent from the recording.

    Examples
    --------
    ::

        # Keep only EEG and stimulus channels
        PickChannels(picks=["eeg", "stim"])

        # Keep specific channels by name
        PickChannels(picks=["Fp1", "Fp2", "Fz"])
    """

    name = "pick_channels"
    description = "Keep only the specified channels or channel types"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(
        self,
        picks: str | list[str],
        on_missing: str = "ignore",
    ):
        self.picks = picks
        self.on_missing = on_missing
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- LOG ---
        logger.info("Picking channels: {}", self.picks)

        # --- COMPUTE + RETURN ---
        raw = context.get_raw().copy().pick(picks=self.picks, verbose=False)
        return context.with_raw(raw)


@register_processor
class DropChannels(Processor):
    """Remove named channels from the recording.

    A named, reusable alternative to the ``lambda ctx: ...drop_channels(...)``
    pattern commonly seen in inline pipeline steps.

    Parameters
    ----------
    channels : list of str
        List of channel names to remove.
    on_missing : str, optional
        ``"ignore"`` (default) skips absent names silently;
        ``"raise"`` raises an error when a channel is not found.

    Examples
    --------
    ::

        # Drop typical non-EEG channels that may be present in EDF files
        DropChannels(channels=["EKG", "EMG", "EOG", "ECG"])
    """

    name = "drop_channels"
    description = "Remove named channels from the recording"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(self, channels: list[str], on_missing: str = "ignore"):
        self.channels = channels
        self.on_missing = on_missing
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()

        # --- COMPUTE ---
        to_drop = [ch for ch in self.channels if ch in raw.ch_names] if self.on_missing == "ignore" else self.channels

        if to_drop:
            logger.info("Dropping channels: {}", to_drop)
            raw.drop_channels(to_drop)

        # --- RETURN ---
        return context.with_raw(raw)


@register_processor
class DropChannelsMatching(Processor):
    """Remove channels whose names match a regular expression.

    This is useful for vendor-specific numbered channels, for example EGI
    ``E1`` through ``E128`` channels, while keeping auxiliary channels such as
    trigger, ECG, or respiration channels.

    Parameters
    ----------
    regex : str
        Regular expression used to select channel names for removal.
    fullmatch : bool, optional
        If ``True`` (default), the whole channel name must match ``regex``.
        If ``False``, any substring match is enough.
    """

    name = "drop_channels_matching"
    description = "Remove channels matching a regular expression"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True
    chunk_probe_safe = True

    def __init__(self, regex: str, fullmatch: bool = True):
        self.regex = regex
        self.fullmatch = fullmatch
        self._pattern = re.compile(regex)
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()

        # --- COMPUTE ---
        if self.fullmatch:
            to_drop = [ch for ch in raw.ch_names if self._pattern.fullmatch(ch)]
        else:
            to_drop = [ch for ch in raw.ch_names if self._pattern.search(ch)]

        if not to_drop:
            logger.info("No channels matched regex: {}", self.regex)
            return context.with_raw(raw)

        logger.info("Dropping {} channel(s) matching {}: {}", len(to_drop), self.regex, to_drop)
        raw.drop_channels(to_drop)

        # --- RETURN ---
        return context.with_raw(raw)


@register_processor
class ChannelStandardizer(Processor):
    """Convert EEG channel layouts to predefined standards or custom subsets.

    Keeps EEG channels that match the requested standard in a deterministic
    order, optionally preserving non-EEG channels (e.g., trigger/stim channels).
    Legacy aliases such as ``T3 -> T7`` and ``T5 -> P7`` are resolved
    automatically. When alias channels are used, they can be renamed to the
    target standard labels.

    Parameters
    ----------
    standard : str or list[str]
        Target standard identifier or explicit ordered channel list.
        Supported built-in identifiers:
        - ``"10-20"`` / ``"standard_1020_19"``
        - ``"32"`` / ``"standard_1020_32"``
        For other strings, the processor tries to load an MNE standard montage
        via :func:`mne.channels.make_standard_montage` and uses its channel list.
    on_missing : str, optional
        Missing-channel behavior: ``"ignore"`` (default) keeps available
        matches only, ``"raise"`` fails when any requested channel is missing.
    keep_auxiliary : bool, optional
        If ``True`` (default), append non-EEG channels after the selected EEG
        subset.
    rename_aliases : bool, optional
        If ``True`` (default), rename selected alias channels to the requested
        target labels when possible (e.g., ``T3`` becomes ``T7``).
    """

    name = "channel_standardizer"
    description = "Convert EEG channels to standard subsets with optional alias renaming"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True
    channel_wise = False

    _VALID_ON_MISSING = ("ignore", "raise")

    def __init__(
        self,
        standard: str | list[str],
        on_missing: str = "ignore",
        keep_auxiliary: bool = True,
        rename_aliases: bool = True,
    ) -> None:
        self.standard = standard
        self.on_missing = on_missing
        self.keep_auxiliary = keep_auxiliary
        self.rename_aliases = rename_aliases
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if self.on_missing not in self._VALID_ON_MISSING:
            raise ProcessorValidationError(
                f"on_missing must be one of {self._VALID_ON_MISSING}, got '{self.on_missing}'"
            )

        target_channels = self._resolve_target_channels()
        if len(target_channels) == 0:
            raise ProcessorValidationError("standard must resolve to at least one channel")

        normalized_targets = [_normalize_channel_name(name) for name in target_channels]
        if len(normalized_targets) != len(set(normalized_targets)):
            raise ProcessorValidationError(
                f"standard contains duplicate channel names after normalization: {target_channels}"
            )

        channel_types = context.get_raw().get_channel_types()
        if "eeg" not in channel_types:
            raise ProcessorValidationError("Raw contains no EEG channels")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        input_raw = context.get_raw()
        raw = input_raw.copy()
        target_channels = self._resolve_target_channels()
        eeg_channels, auxiliary_channels = self._split_channels_by_type(input_raw)
        matched_channels, missing_channels = self._match_channels(
            target_channels=target_channels,
            available_eeg_channels=eeg_channels,
        )

        if self.on_missing == "raise" and missing_channels:
            raise ProcessorValidationError(
                "Missing channels for standard '{}': {}".format(
                    self._resolve_standard_label(),
                    ", ".join(missing_channels),
                )
            )

        if len(matched_channels) == 0:
            raise ProcessorValidationError(
                f"No requested EEG channels found for standard '{self._resolve_standard_label()}'"
            )

        selected_eeg_channels = [source_name for _, source_name in matched_channels]
        selected_channels = selected_eeg_channels.copy()
        if self.keep_auxiliary:
            selected_channels.extend(ch for ch in auxiliary_channels if ch not in selected_eeg_channels)

        # --- LOG ---
        logger.info(
            "Converting to channel standard '{}' (matched={}, missing={}, keep_auxiliary={})",
            self._resolve_standard_label(),
            len(matched_channels),
            len(missing_channels),
            self.keep_auxiliary,
        )

        # --- COMPUTE ---
        raw.pick(picks=selected_channels, verbose=False)
        rename_map = self._build_rename_map(
            selected_channels=raw.ch_names,
            matched_channels=matched_channels,
        )
        if rename_map:
            raw.rename_channels(rename_map)

        # --- BUILD RESULT ---
        new_ctx = context.with_raw(raw)

        # --- NOISE ---
        if context.has_estimated_noise():
            estimated_noise = context.get_estimated_noise()
            if estimated_noise is None or estimated_noise.shape[0] != len(input_raw.ch_names):
                raise ProcessorValidationError(
                    "Estimated noise shape does not match raw channels: "
                    f"noise_shape={None if estimated_noise is None else estimated_noise.shape}, "
                    f"n_channels={len(input_raw.ch_names)}"
                )

            index_by_name = {ch_name: idx for idx, ch_name in enumerate(input_raw.ch_names)}
            selected_indices = [index_by_name[ch_name] for ch_name in selected_channels]
            new_ctx.set_estimated_noise(estimated_noise[selected_indices, :].copy())
        else:
            logger.debug("No noise estimate present — skipping noise propagation")

        metadata = new_ctx.metadata.copy()
        metadata.custom["channel_standardizer"] = {
            "standard": self._resolve_standard_label(),
            "requested_eeg_channels": len(target_channels),
            "matched_eeg_channels": len(matched_channels),
            "missing_eeg_channels": missing_channels,
            "keep_auxiliary": self.keep_auxiliary,
            "rename_aliases": self.rename_aliases,
            "renamed_channels": rename_map,
        }

        # --- RETURN ---
        return new_ctx.with_metadata(metadata)

    def _resolve_target_channels(self) -> list[str]:
        """Resolve standard configuration to an ordered target channel list."""
        if isinstance(self.standard, list):
            return [str(name) for name in self.standard]

        normalized_standard = self.standard.strip().lower()
        resolved_standard = _CHANNEL_STANDARD_ALIASES.get(normalized_standard, normalized_standard)
        if resolved_standard in _CHANNEL_STANDARD_PRESETS:
            return _CHANNEL_STANDARD_PRESETS[resolved_standard].copy()

        try:
            montage = mne.channels.make_standard_montage(self.standard)
        except Exception as exc:
            raise ProcessorValidationError(
                f"Unknown channel standard '{self.standard}'. Use a built-in standard "
                "(10-20, standard_1020_19, 32, standard_1020_32) or a valid "
                "MNE montage name."
            ) from exc
        return [str(name) for name in montage.ch_names]

    def _resolve_standard_label(self) -> str:
        """Return human-readable standard label for logs/metadata."""
        if isinstance(self.standard, str):
            return self.standard
        return "custom"

    @staticmethod
    def _split_channels_by_type(raw: mne.io.BaseRaw) -> tuple[list[str], list[str]]:
        """Split raw channel names into EEG and non-EEG sets."""
        eeg_channels: list[str] = []
        auxiliary_channels: list[str] = []
        for ch_name, ch_type in zip(raw.ch_names, raw.get_channel_types(), strict=True):
            if ch_type == "eeg":
                eeg_channels.append(ch_name)
            else:
                auxiliary_channels.append(ch_name)
        return eeg_channels, auxiliary_channels

    def _match_channels(
        self,
        target_channels: list[str],
        available_eeg_channels: list[str],
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Match ordered target channels against available EEG channels."""
        lookup = {_normalize_channel_name(ch_name): ch_name for ch_name in available_eeg_channels}

        matched: list[tuple[str, str]] = []
        missing: list[str] = []
        used_sources: set[str] = set()

        for target_name in target_channels:
            source_name = self._resolve_source_channel(target_name, lookup, used_sources)
            if source_name is None:
                missing.append(target_name)
                continue

            matched.append((target_name, source_name))
            used_sources.add(source_name)

        return matched, missing

    def _resolve_source_channel(
        self,
        target_name: str,
        lookup: dict[str, str],
        used_sources: set[str],
    ) -> str | None:
        """Resolve one target channel to an available source channel."""
        candidate_names = self._candidate_aliases(target_name)

        for candidate_name in candidate_names:
            resolved_name = lookup.get(candidate_name)
            if resolved_name is None:
                continue
            if resolved_name in used_sources:
                continue
            return resolved_name
        return None

    def _candidate_aliases(self, target_name: str) -> list[str]:
        """Return normalized candidate names for one target channel."""
        normalized_target = _normalize_channel_name(target_name)
        candidates = [normalized_target]
        candidates.extend(_NORMALIZED_CHANNEL_ALIASES.get(normalized_target, tuple()))
        candidates.extend(_REVERSE_CHANNEL_ALIASES.get(normalized_target, tuple()))
        return candidates

    def _build_rename_map(
        self,
        selected_channels: list[str],
        matched_channels: list[tuple[str, str]],
    ) -> dict[str, str]:
        """Build source->target rename mapping for alias-normalization."""
        if not self.rename_aliases:
            return {}

        rename_map: dict[str, str] = {}
        normalized_to_current = {_normalize_channel_name(ch_name): ch_name for ch_name in selected_channels}

        for target_name, source_name in matched_channels:
            source_norm = _normalize_channel_name(source_name)
            target_norm = _normalize_channel_name(target_name)
            if source_norm == target_norm:
                continue

            existing_target_name = normalized_to_current.get(target_norm)
            if existing_target_name is not None and existing_target_name != source_name:
                continue

            rename_map[source_name] = target_name
            normalized_to_current[target_norm] = target_name

        return rename_map


@register_processor
class PrintMetric(Processor):
    """Print one or more evaluation metric values — useful for debugging pipelines.

    Inserts a transparent logging step that reads from the shared metrics dict
    populated by evaluation processors (e.g. :class:`~facet.evaluation.SNRCalculator`).
    The context is returned unchanged.

    Parameters
    ----------
    *metric_names : str
        One or more metric names to print (e.g. ``'snr'``, ``'rms_ratio'``).
    label : str, optional
        Optional prefix shown in brackets, e.g. ``"after PCA"``.

    Examples
    --------
    ::

        pipeline = Pipeline([
            ...,
            SNRCalculator(),
            PrintMetric("snr"),          # → "  snr=12.345"
            PCACorrection(...),
            SNRCalculator(),
            PrintMetric("snr", label="after PCA"),   # → "  [after PCA] snr=14.201"
        ])
    """

    name = "print_metric"
    description = "Print evaluation metric values for debugging"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = False
    modifies_raw = False
    parallel_safe = False

    def __init__(self, *metric_names: str, label: str | None = None):
        self._metric_names = metric_names
        self._label = label
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- COMPUTE ---
        parts = []
        for name in self._metric_names:
            val = context.get_metric(name)
            if isinstance(val, float):
                parts.append(f"{name}={val:.3f}")
            elif val is not None:
                parts.append(f"{name}={val}")
            else:
                parts.append(f"{name}=N/A")

        prefix = f"[{self._label}] " if self._label else ""
        message = "{}{}".format(prefix, ", ".join(parts))

        # --- LOG ---
        logger.info("{}", message)
        print(f"  {message}")

        # --- RETURN ---
        return context


@register_processor
class RawTransform(Processor):
    """Apply an arbitrary callable to the Raw object.

    A lighter-weight alternative to ``LambdaProcessor`` when only the Raw
    object needs to be modified.  The callable receives the **current** Raw
    object and must return a *new* (or modified copy of a) Raw object.

    Parameters
    ----------
    name : str
        Human-readable label shown in pipeline logs and progress.
    func : callable
        ``Callable[[mne.io.Raw], mne.io.Raw]`` — receives the current Raw
        object, must return a (possibly new) Raw object.

    Examples
    --------
    ::

        # Drop bad channels inline
        RawTransform("drop_bad", lambda raw: raw.copy().pick_channels(
            [ch for ch in raw.ch_names if ch not in ["EKG", "EMG"]]
        ))

        # Set average reference
        RawTransform("set_eeg_ref", lambda raw: raw.copy().set_eeg_reference("average"))
    """

    name = "raw_transform"
    description = "Apply a callable transform to the Raw object"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = False

    def __init__(self, name: str, func: Callable):
        self.name = name
        self._func = func
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- LOG ---
        logger.info("Applying raw transform: {}", self.name)

        # --- COMPUTE + RETURN ---
        new_raw = self._func(context.get_raw())
        return context.with_raw(new_raw)
