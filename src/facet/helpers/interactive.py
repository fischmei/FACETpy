"""
Interactive Helpers Module

Processors that facilitate interactive pipeline steps such as awaiting user
confirmation before continuing execution.

Author: FACETpy Team
Date: 2025-01-12
"""

import contextlib
import os
import sys
import time

import mne
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider

from ..console import get_console, suspend_raw_mode
from ..core import (
    ProcessingContext,
    Processor,
    ProcessorError,
    ProcessorValidationError,
    register_processor,
)
from .plotting import show_matplotlib_figure


@register_processor
class WaitForConfirmation(Processor):
    """
    Pause pipeline execution until the user confirms continuation.

    Designed for iterative, notebook-driven, or CLI debugging workflows where
    manual inspection is required between processing stages. When interactive
    input is unavailable, the processor automatically continues to avoid
    blocking headless runs.
    """

    name = "wait_for_confirmation"
    description = "Pause pipeline until user confirmation."
    modifies_raw = False

    def __init__(
        self,
        message: str = "Press Enter to continue...",
        auto_continue: bool = False,
        timeout: float | None = None,
        continue_on_timeout: bool = True,
    ):
        """
        Initialize the confirmation step.

        Args:
            message: Prompt presented to the user.
            auto_continue: Skip the pause entirely when True.
            timeout: Optional timeout in seconds before continuing automatically.
            continue_on_timeout: Whether to resume automatically after timeout
                expires. If False, raises a TimeoutError instead.
        """
        self.message = message
        self.auto_continue = auto_continue
        self.timeout = timeout
        self.continue_on_timeout = continue_on_timeout
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Wait for the user to confirm or continue automatically."""
        if self.auto_continue:
            logger.info("Auto-continue enabled; skipping confirmation step.")
            return context

        if not sys.stdin or not sys.stdin.isatty():
            logger.warning("Standard input is not interactive; continuing automatically.")
            return context

        # Print the (optionally Rich-markup) message through the log panel so it
        # lands inside the live display rather than escaping above it.
        self._print_message()

        # Derive a short single-line footer hint — strip any Rich markup tags so
        # they don't appear literally in the footer text.
        try:
            from rich.text import Text as _RichText

            def _strip(s):
                return _RichText.from_markup(s).plain
        except Exception:
            import re as _re

            def _strip(s):
                return _re.sub(r"\[/?[^\]]*\]", "", s)

        footer_hint = next(
            (_strip(line.strip()) for line in self.message.split("\n") if line.strip()),
            "Press Enter to continue...",
        )
        console = get_console()
        console.set_active_prompt(footer_hint)
        try:
            # Suspend raw terminal mode while waiting so input()/readline() work
            # correctly even when the ModernConsole keyboard listener is active.
            with suspend_raw_mode():
                try:
                    if self.timeout is None:
                        input("")
                    else:
                        self._prompt_with_timeout()
                except (EOFError, KeyboardInterrupt):
                    logger.info("User aborted confirmation step; continuing execution.")
                except TimeoutError as exc:
                    logger.warning(str(exc))
        finally:
            console.clear_active_prompt()
        return context

    def _print_message(self) -> None:
        """Render the message through the Rich console, auto-colouring plain text."""
        rich_console = get_console().get_rich_console()
        # If the message contains no Rich markup tags, apply a default colour scheme:
        # the first non-empty line is rendered bold+yellow, remaining lines dim.
        has_markup = "[" in self.message
        first = True
        for line in self.message.split("\n"):
            line = line.strip()
            if not line:
                continue
            if rich_console is None:
                logger.info(line)
            elif has_markup:
                rich_console.print(line)
            elif first:
                rich_console.print(f"[bold yellow]{line}[/bold yellow]")
            else:
                rich_console.print(f"[dim]{line}[/dim]")
            first = False

    def _prompt_with_timeout(self) -> None:
        """Wait for user confirmation with an optional timeout."""
        logger.info("Waiting for user confirmation (timeout=%.1fs)...", self.timeout)
        start_time = time.time()

        if os.name == "nt":
            self._wait_windows(start_time)
        else:
            self._wait_posix(start_time)

    def _wait_windows(self, start_time: float) -> None:
        """Handle confirmation on Windows platforms."""
        import msvcrt

        while True:
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in ("\n", "\r"):
                    return
            if self.timeout is not None and (time.time() - start_time) > self.timeout:
                self._handle_timeout()
                return
            time.sleep(0.05)

    def _wait_posix(self, start_time: float) -> None:
        """Handle confirmation on POSIX platforms using select."""
        import select

        ready, _, _ = select.select([sys.stdin], [], [], self.timeout)
        if ready:
            sys.stdin.readline()
        else:
            self._handle_timeout()

    def _handle_timeout(self) -> None:
        """Handle timeout conditions according to configuration."""
        if self.continue_on_timeout:
            logger.warning("Confirmation timeout reached; continuing automatically.")
        else:
            raise TimeoutError("Confirmation timeout reached and continue_on_timeout=False.")


@register_processor
class ArtifactOffsetFinder(Processor):
    """Interactively determine the artifact-to-trigger offset.

    Displays a matplotlib plot of overlaid EEG epochs centred on the trigger
    positions.  A slider and click-to-set interface let the user visually
    align an artifact window with the data.  The confirmed offset is written
    to ``context.metadata.artifact_to_trigger_offset``.

    The plot shows multiple overlaid epochs (mean-subtracted) so the
    repeating artifact pattern is clearly visible.  A blue dashed line marks
    the trigger; a red line and shaded region mark the current artifact
    window.  Drag the slider or left-click on the plot to reposition the
    offset, then press **Confirm**.

    Parameters
    ----------
    channel : str | int | None
        Channel to display.  Name (str), index (int), or ``None`` for the
        first EEG channel.
    n_epochs : int
        Number of artifact epochs to overlay (default: 5).
    initial_offset : float | None
        Starting offset in seconds.  When ``None`` (default), the current
        context value is used.
    """

    name = "artifact_offset_finder"
    description = "Interactively find artifact-to-trigger offset"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        channel: str | int | None = None,
        n_epochs: int = 5,
        initial_offset: float | None = None,
    ) -> None:
        self.channel = channel
        self.n_epochs = n_epochs
        self.initial_offset = initial_offset
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError("Artifact length not set. Run TriggerDetector first.")
        n_triggers = len(context.get_triggers())
        if n_triggers < 2:
            raise ProcessorValidationError(f"Need at least 2 triggers to determine offset, got {n_triggers}.")
        if self.n_epochs < 1:
            raise ProcessorValidationError(f"n_epochs must be >= 1, got {self.n_epochs}")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers()
        sfreq = context.get_sfreq()
        artifact_length = context.get_artifact_length()
        current_offset = context.metadata.artifact_to_trigger_offset

        # --- LOG ---
        logger.info(
            "Opening artifact offset finder (current offset={:.4f} s)",
            current_offset,
        )

        # --- COMPUTE ---
        ch_idx = self._resolve_channel(raw)
        artifact_duration = artifact_length / sfreq
        epochs_data, time_axis = self._extract_epochs(
            raw._data[ch_idx],
            triggers,
            sfreq,
            artifact_duration,
        )

        if self.initial_offset is not None:
            start_offset = self.initial_offset
        else:
            auto_offset = self._auto_detect_offset(
                epochs_data,
                time_axis,
                sfreq,
                artifact_duration,
            )
            if auto_offset is not None:
                start_offset = auto_offset
                logger.info(
                    "Auto-detected artifact onset at {:.4f} s ({:.2f} ms)",
                    auto_offset,
                    auto_offset * 1000,
                )
            else:
                start_offset = current_offset
                logger.debug("Auto-detection failed; using current offset")

        chosen_offset = self._show_interactive_plot(
            epochs_data,
            time_axis,
            raw.ch_names[ch_idx],
            artifact_duration,
            sfreq,
            start_offset,
        )

        # --- BUILD RESULT ---
        logger.info(
            "Artifact-to-trigger offset set to {:.6f} s ({:.3f} ms)",
            chosen_offset,
            chosen_offset * 1000,
        )
        new_metadata = context.metadata.copy()
        new_metadata.artifact_to_trigger_offset = chosen_offset

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )

    # -----------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------

    def _resolve_channel(self, raw: mne.io.Raw) -> int:
        """Determine the channel index to display."""
        if self.channel is None:
            eeg_channels = mne.pick_types(
                raw.info,
                eeg=True,
                exclude="bads",
            )
            if len(eeg_channels) == 0:
                raise ProcessorError("No EEG channels found in raw data.")
            return int(eeg_channels[0])

        if isinstance(self.channel, int):
            return self.channel
        return raw.ch_names.index(self.channel)

    def _extract_epochs(
        self,
        channel_data: np.ndarray,
        triggers: np.ndarray,
        sfreq: float,
        artifact_duration: float,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Extract mean-subtracted epochs centred on triggers.

        Returns
        -------
        epochs : list of np.ndarray
            Epoch arrays, one per trigger shown.
        time_axis : np.ndarray
            Time in seconds relative to trigger.
        """
        padding = artifact_duration * 0.25
        n_pre = int(padding * sfreq)
        n_post = int((artifact_duration + padding) * sfreq)
        n_total = n_pre + n_post
        time_axis = (np.arange(n_total) - n_pre) / sfreq

        n_show = min(self.n_epochs, len(triggers) - 1)
        epochs: list[np.ndarray] = []
        for i in range(n_show):
            start = int(triggers[i]) - n_pre
            end = start + n_total
            if start < 0 or end > len(channel_data):
                continue
            epoch = channel_data[start:end].copy()
            epoch -= np.mean(epoch)
            epochs.append(epoch)

        if not epochs:
            raise ProcessorError("Could not extract any epochs for display.")
        return epochs, time_axis

    def _auto_detect_offset(
        self,
        epochs_data: list[np.ndarray],
        time_axis: np.ndarray,
        sfreq: float,
        artifact_duration: float,
    ) -> float | None:
        """Estimate the artifact onset from the averaged epoch shape.

        Averages overlaid epochs to enhance the repeating artifact and
        suppress random EEG.  Because gradient artifacts are contiguous
        (each period immediately follows the previous one), the boundary
        between two periods shows up as a *minimum* in the smoothed
        amplitude envelope.  This method finds the deepest such minimum
        in a window around the trigger.

        Parameters
        ----------
        epochs_data : list of np.ndarray
            Mean-subtracted epochs extracted by :meth:`_extract_epochs`.
        time_axis : np.ndarray
            Time in seconds relative to the trigger.
        sfreq : float
            Sampling frequency in Hz.
        artifact_duration : float
            Duration of one artifact period in seconds.

        Returns
        -------
        float or None
            Estimated offset in seconds relative to the trigger, or
            ``None`` if detection fails.
        """
        if len(epochs_data) < 2:
            return None

        mean_epoch = np.mean(epochs_data, axis=0)

        # Smoothed amplitude envelope — kernel spans ~1 % of the
        # artifact period, enough to iron out sample-level noise
        # without smearing the inter-artifact dip.
        envelope = np.abs(mean_epoch)
        kernel_size = max(5, int(sfreq * artifact_duration * 0.01))
        kernel = np.ones(kernel_size) / kernel_size
        smooth_env = np.convolve(envelope, kernel, mode="same")

        # Search for the amplitude minimum near the trigger (±20 % of
        # artifact duration).  The dip between consecutive artifact
        # periods marks the epoch boundary.
        margin = artifact_duration * 0.2
        search_mask = (time_axis >= -margin) & (time_axis <= margin)
        if not np.any(search_mask):
            return None

        search_indices = np.where(search_mask)[0]
        min_idx = search_indices[np.argmin(smooth_env[search_indices])]
        return float(time_axis[min_idx])

    def _setup_plot_axes(
        self,
        epochs_data: list[np.ndarray],
        time_axis: np.ndarray,
        ch_name: str,
        offset: float,
        artifact_duration: float,
    ) -> tuple:
        """Create the figure with epoch overlay and offset markers.

        Returns
        -------
        tuple
            ``(fig, ax, offset_line, offset_text)`` — the figure, axes,
            moveable offset line, and offset label.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.28)

        for epoch in epochs_data:
            ax.plot(time_axis, epoch, alpha=0.5, linewidth=0.7)

        ax.axvline(0, color="blue", ls="--", lw=1.5, label="Trigger")
        offset_line = ax.axvline(
            offset,
            color="red",
            lw=2,
            label="Artifact start",
        )

        offset_text = ax.text(
            0.02,
            0.95,
            f"Offset: {offset * 1000:.2f} ms",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax.set_xlabel("Time relative to trigger (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Artifact Offset Finder \u2014 {ch_name}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        return fig, ax, offset_line, offset_text

    def _show_interactive_plot(
        self,
        epochs_data: list[np.ndarray],
        time_axis: np.ndarray,
        ch_name: str,
        artifact_duration: float,
        sfreq: float,
        start_offset: float,
    ) -> float:
        """Display the interactive offset finder and return the chosen offset.

        Adds a slider, click-to-set handler, and confirm/cancel buttons on
        top of the axes created by :meth:`_setup_plot_axes`.
        """
        fig, ax, offset_line, offset_text = self._setup_plot_axes(
            epochs_data,
            time_axis,
            ch_name,
            start_offset,
            artifact_duration,
        )
        state: dict = {"offset": start_offset, "confirmed": False, "shade": None}

        def _draw_shade() -> None:
            if state["shade"] is not None:
                state["shade"].remove()
            state["shade"] = ax.axvspan(
                state["offset"],
                state["offset"] + artifact_duration,
                alpha=0.12,
                color="red",
                label="_nolegend_",
            )

        _draw_shade()

        # --- Slider ---
        slider_ax = fig.add_axes([0.15, 0.12, 0.65, 0.04])
        max_range = artifact_duration * 0.5
        slider = Slider(
            slider_ax,
            "Offset (s)",
            -max_range,
            max_range,
            valinit=start_offset,
            valstep=1.0 / sfreq,
        )

        def _on_slider_changed(val: float) -> None:
            state["offset"] = val
            offset_line.set_xdata([val, val])
            _draw_shade()
            offset_text.set_text(f"Offset: {val * 1000:.2f} ms")
            fig.canvas.draw_idle()

        slider.on_changed(_on_slider_changed)

        # --- Click-to-set ---
        def _on_click(event) -> None:
            if event.inaxes == ax and event.button == 1:
                slider.set_val(event.xdata)

        fig.canvas.mpl_connect("button_press_event", _on_click)

        # --- Confirm / Cancel buttons ---
        confirm_ax = fig.add_axes([0.68, 0.03, 0.12, 0.05])
        confirm_btn = Button(confirm_ax, "Confirm")

        def _close_fig() -> None:
            """Destroy the native window while the event loop is active."""
            with contextlib.suppress(Exception):
                fig.canvas.manager.destroy()
            plt.close(fig)

        def _on_confirm(_) -> None:
            state["confirmed"] = True
            _close_fig()

        confirm_btn.on_clicked(_on_confirm)

        cancel_ax = fig.add_axes([0.82, 0.03, 0.12, 0.05])
        cancel_btn = Button(cancel_ax, "Cancel")
        cancel_btn.on_clicked(lambda _: _close_fig())

        console = get_console()
        console.set_active_prompt("Adjust offset in plot window, then click Confirm")
        try:
            with suspend_raw_mode():
                show_matplotlib_figure(fig)
        finally:
            plt.close(fig)
            console.clear_active_prompt()

        if not state["confirmed"]:
            logger.info("Offset selection cancelled; keeping current offset")
            return start_offset

        return state["offset"]
