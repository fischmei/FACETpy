"""Interactive Trigger Explorer Processor

Scans EEG data for available annotations and STIM channel events, presents
them to the user via a GUI preview window or terminal table, and lets them
interactively select the trigger source before proceeding with the pipeline.
"""

import re
from typing import Any

import mne
import numpy as np
from loguru import logger

from ..console import get_console, suspend_raw_mode
from ..core import (
    ProcessingContext,
    Processor,
    ProcessorError,
    ProcessorValidationError,
    register_processor,
)

_DISPLAY_DOWNSAMPLE_TARGET = 5000


@register_processor
class TriggerExplorer(Processor):
    """Interactively explore and select trigger events from annotations or STIM channels.

    Scans the loaded data for all available event sources (MNE annotations and
    STIM channels) and lets the user pick the correct trigger source.  Three
    interaction modes are supported:

    ``"gui"`` (default)
        Opens a matplotlib window with a downsampled preview of the first EEG
        channel.  Radio buttons list every discovered event type; selecting one
        highlights the corresponding trigger positions on the waveform.  A
        *Confirm* button finalises the choice.  Falls back to ``"terminal"``
        automatically when no GUI backend is available.

    ``"terminal"``
        Prints a Rich table and prompts for a selection number in the terminal.

    ``"auto"``
        Alias for ``"gui"`` — kept for backward compatibility.

    When ``auto_select`` is provided, all interactive prompts are skipped and
    the first event matching the regex is chosen automatically — useful for
    scripted / non-interactive pipelines.

    Parameters
    ----------
    mode : str, optional
        Interaction mode: ``"gui"`` (default), ``"terminal"``, or ``"auto"``.
    auto_select : str or None, optional
        If given, automatically select the event whose description matches
        this regex, bypassing any interactive prompt (default: None).
    save_to_annotations : bool, optional
        If ``True``, write detected triggers back to the raw annotations
        (default: False).
    """

    name = "trigger_explorer"
    description = "Interactively explore and select trigger events"
    version = "1.1.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        mode: str = "gui",
        auto_select: str | None = None,
        save_to_annotations: bool = False,
    ) -> None:
        self.mode = mode
        self.auto_select = auto_select
        self.save_to_annotations = save_to_annotations
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if self.mode not in ("gui", "terminal", "auto"):
            raise ProcessorValidationError(f"mode must be 'gui', 'terminal', or 'auto', got '{self.mode}'")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()

        # --- LOG ---
        sfreq = raw.info["sfreq"]
        t_start = raw.times[0]
        t_end = raw.times[-1]
        n_raw_annotations = len(raw.annotations)
        logger.info(
            "Exploring available trigger sources (mode={}) | "
            "data window: {:.2f}s – {:.2f}s ({:.1f}s) | "
            "raw.annotations: {}",
            self.mode,
            t_start,
            t_end,
            t_end - t_start,
            n_raw_annotations,
        )
        if n_raw_annotations > 0:
            onset_min = raw.annotations.onset.min()
            onset_max = raw.annotations.onset.max()
            logger.debug("raw.annotations onset range: {:.2f}s – {:.2f}s", onset_min, onset_max)

        # --- COMPUTE ---
        annotation_events = self._collect_annotation_events(raw)
        stim_events = self._collect_stim_events(raw)

        logger.debug(
            "Collected {} annotation event type(s) and {} STIM event type(s)",
            len(annotation_events),
            len(stim_events),
        )

        if len(annotation_events) == 0 and len(stim_events) == 0:
            hint = ""
            if n_raw_annotations == 0:
                hint = (
                    " Tip: raw.annotations is empty — if the file has triggers, "
                    "they may have been removed by a preceding Crop step "
                    "(check that tmin/tmax covers the trigger region)."
                )
            raise ProcessorError(
                f"No trigger sources found: data contains neither annotations nor STIM channels.{hint}"
            )

        event_table = self._build_event_table(annotation_events, stim_events)
        selected = self._select_event(event_table, raw)
        regex = self._regex_for_selection(selected)
        triggers = self._detect_triggers(raw, selected)

        if len(triggers) == 0:
            raise ProcessorError(f"Selection '{selected['description']}' matched 0 triggers.")

        logger.info(
            "Selected trigger '{}' → {} events detected",
            selected["description"],
            len(triggers),
        )

        artifact_meta = self._compute_artifact_metadata(triggers)

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        new_metadata.triggers = triggers
        new_metadata.trigger_regex = regex
        new_metadata.artifact_length = artifact_meta["artifact_length"]
        new_metadata.volume_gaps = artifact_meta["volume_gaps"]
        if artifact_meta.get("slices_per_volume") is not None:
            new_metadata.slices_per_volume = artifact_meta["slices_per_volume"]

        if self.save_to_annotations:
            sfreq = raw.info["sfreq"]
            raw_copy = raw.copy()
            raw_copy.set_annotations(
                mne.Annotations(
                    onset=triggers / sfreq,
                    duration=np.zeros(len(triggers)),
                    description=["Trigger"] * len(triggers),
                )
            )
            return context.with_raw(
                raw_copy,
                copy_estimated_noise=False,
            ).with_metadata(
                new_metadata,
                copy_estimated_noise=False,
            )

        # --- RETURN ---
        return context.with_metadata(
            new_metadata,
            copy_estimated_noise=False,
        )

    # ------------------------------------------------------------------
    # Event collection helpers
    # ------------------------------------------------------------------

    def _collect_annotation_events(self, raw: mne.io.Raw) -> list[dict[str, Any]]:
        """Gather unique annotation descriptions with counts and timing info.

        Uses ``mne.events_from_annotations`` rather than direct
        ``raw.annotations`` access to handle all EDF/EDF+ annotation
        formats reliably.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object to scan.

        Returns
        -------
        list of dict
            Each entry: ``{description, count, first_onset, last_onset}``.
        """
        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
        except (ValueError, RuntimeError) as exc:
            logger.debug("events_from_annotations raised {}: {}", type(exc).__name__, exc)
            return []

        logger.debug(
            "events_from_annotations: {} event(s), types: {}",
            len(events),
            list(event_id.keys()),
        )
        if len(events) == 0:
            return []

        id_to_desc = {v: k for k, v in event_id.items()}
        sfreq = raw.info["sfreq"]

        desc_map: dict[str, list[float]] = {}
        for event in events:
            desc = id_to_desc.get(int(event[2]), str(event[2]))
            onset = event[0] / sfreq
            desc_map.setdefault(desc, []).append(onset)

        results = []
        for desc, onsets in sorted(desc_map.items()):
            results.append(
                {
                    "description": desc,
                    "count": len(onsets),
                    "first_onset": min(onsets),
                    "last_onset": max(onsets),
                }
            )

        results = self._maybe_group_sequential_annotations(results)
        return results

    @staticmethod
    def _maybe_group_sequential_annotations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge annotations that are 'prefix + number' sequences into one entry.

        Many EDF files encode fMRI scan triggers as sequential annotations
        (e.g. ``"TR 1"``, ``"TR 2"``, …, ``"TR 346"``), each appearing exactly
        once.  Showing 346 separate one-occurrence rows in the GUI is unusable;
        this method collapses them into a single ``"TR"`` entry with
        ``count=346`` and the ``grouped_prefix=True`` flag so that downstream
        methods know to match the whole sequence rather than a single label.

        A group is formed whenever **two or more** annotation descriptions share
        the same text prefix and differ only in a trailing integer.

        Parameters
        ----------
        events : list of dict
            Raw per-description entries from ``_collect_annotation_events``.

        Returns
        -------
        list of dict
            Same format; sequential groups replaced by a single merged entry.
        """
        _NUMERIC_SUFFIX_RE = re.compile(r"^(.+?)\s+\d+$")

        prefix_groups: dict[str, list[dict[str, Any]]] = {}
        no_prefix: list[dict[str, Any]] = []

        for ev in events:
            m = _NUMERIC_SUFFIX_RE.match(ev["description"])
            if m:
                prefix_groups.setdefault(m.group(1), []).append(ev)
            else:
                no_prefix.append(ev)

        merged: list[dict[str, Any]] = list(no_prefix)
        for prefix, group in sorted(prefix_groups.items()):
            if len(group) < 2:
                merged.extend(group)
            else:
                merged.append(
                    {
                        "description": prefix,
                        "count": sum(e["count"] for e in group),
                        "first_onset": min(e["first_onset"] for e in group),
                        "last_onset": max(e["last_onset"] for e in group),
                        "grouped_prefix": True,
                    }
                )

        return merged

    def _collect_stim_events(self, raw: mne.io.Raw) -> list[dict[str, Any]]:
        """Gather unique STIM channel event values with counts and timing info.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object to scan.

        Returns
        -------
        list of dict
            Each entry: ``{description, count, first_sample, last_sample,
            channel_name}``.
        """
        stim_picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)
        if len(stim_picks) == 0:
            return []

        results = []
        for ch_idx in stim_picks:
            ch_name = raw.ch_names[ch_idx]
            events = mne.find_events(raw, stim_channel=ch_name, initial_event=True, verbose=False)
            if len(events) == 0:
                continue

            value_map: dict[int, list[int]] = {}
            for event in events:
                value_map.setdefault(int(event[2]), []).append(int(event[0]))

            for value, samples in sorted(value_map.items()):
                results.append(
                    {
                        "description": str(value),
                        "count": len(samples),
                        "first_sample": min(samples),
                        "last_sample": max(samples),
                        "channel_name": ch_name,
                    }
                )
        return results

    # ------------------------------------------------------------------
    # Event table building
    # ------------------------------------------------------------------

    def _build_event_table(
        self,
        annotation_events: list[dict[str, Any]],
        stim_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge annotation and stim events into a numbered table.

        Parameters
        ----------
        annotation_events : list of dict
            Events from annotations.
        stim_events : list of dict
            Events from STIM channels.

        Returns
        -------
        list of dict
            Unified table with ``index``, ``source``, ``description``,
            ``count``, and ``detail`` keys.
        """
        table: list[dict[str, Any]] = []
        idx = 1

        for ev in annotation_events:
            time_range = f"{ev['first_onset']:.2f}s – {ev['last_onset']:.2f}s"
            if ev.get("grouped_prefix"):
                detail = f"{time_range}  (sequential: '{ev['description']} 1' … '{ev['description']} N')"
            else:
                detail = time_range
            table.append(
                {
                    "index": idx,
                    "source": "annotation",
                    "description": ev["description"],
                    "count": ev["count"],
                    "detail": detail,
                    "grouped_prefix": ev.get("grouped_prefix", False),
                }
            )
            idx += 1

        for ev in stim_events:
            sfreq_label = ev.get("channel_name", "STIM")
            table.append(
                {
                    "index": idx,
                    "source": f"stim ({sfreq_label})",
                    "description": ev["description"],
                    "count": ev["count"],
                    "detail": (f"sample {ev['first_sample']} – {ev['last_sample']}"),
                }
            )
            idx += 1

        return table

    # ------------------------------------------------------------------
    # Selection dispatch
    # ------------------------------------------------------------------

    def _select_event(self, event_table: list[dict[str, Any]], raw: mne.io.Raw) -> dict[str, Any]:
        """Select a trigger event via the configured interaction mode.

        Parameters
        ----------
        event_table : list of dict
            Unified event table.
        raw : mne.io.Raw
            Raw object (needed for the GUI preview plot).

        Returns
        -------
        dict
            The selected row from the event table.
        """
        if self.auto_select is not None:
            return self._auto_select_event(event_table)

        if self.mode in ("gui", "auto"):
            if self._gui_backend_available():
                return self._gui_select_event(event_table, raw)
            logger.warning("No matplotlib GUI backend available — falling back to terminal mode")

        self._display_event_table(event_table)
        return self._terminal_select_event(event_table)

    @staticmethod
    def _gui_backend_available() -> bool:
        """Return True if matplotlib can open an interactive window."""
        try:
            import matplotlib

            backend = matplotlib.get_backend().lower()
            non_interactive = {"agg", "pdf", "svg", "ps", "cairo", "template"}
            return backend not in non_interactive
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # GUI selection mode
    # ------------------------------------------------------------------

    def _gui_select_event(self, event_table: list[dict[str, Any]], raw: mne.io.Raw) -> dict[str, Any]:
        """Open a matplotlib window with a waveform preview for interactive selection.

        Parameters
        ----------
        event_table : list of dict
            Unified event table.
        raw : mne.io.Raw
            Raw object used for the preview plot.

        Returns
        -------
        dict
            The confirmed row from the event table.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons

        sfreq = raw.info["sfreq"]
        ch_idx, ch_name = self._pick_preview_channel(raw)
        data, times = self._downsample_for_display(raw, ch_idx)

        trigger_times_map = {}
        for row in event_table:
            trigs = self._detect_triggers(raw, row)
            trigger_times_map[row["index"]] = trigs / sfreq

        fig, axes = self._create_gui_layout(len(event_table))
        ax_plot, ax_radio, ax_info, ax_btn = axes

        ax_plot.plot(times, data, linewidth=0.5, color="#2196F3", rasterized=True)
        ax_plot.set_xlabel("Time (s)")
        ax_plot.set_ylabel("Amplitude")
        ax_plot.margins(x=0)

        labels = [f"{r['description']}  ({r['count']})" for r in event_table]
        radio = RadioButtons(ax_radio, labels, active=0)
        for lbl in radio.labels:
            lbl.set_fontsize(9)

        state: dict[str, Any] = {
            "selected": event_table[0],
            "confirmed": False,
            "vlines": None,
        }

        def on_radio_change(label: str) -> None:
            idx = labels.index(label)
            state["selected"] = event_table[idx]
            self._update_gui_plot(
                ax_plot,
                state,
                trigger_times_map[event_table[idx]["index"]],
                ch_name,
                data,
            )
            self._update_gui_info(ax_info, event_table[idx])
            fig.canvas.draw_idle()

        radio.on_clicked(on_radio_change)
        on_radio_change(labels[0])

        btn = Button(ax_btn, "Confirm Selection", color="#4CAF50", hovercolor="#66BB6A")
        btn.label.set_fontweight("bold")
        btn.label.set_fontsize(11)

        def on_confirm(_event: Any) -> None:
            state["confirmed"] = True
            plt.close(fig)

        btn.on_clicked(on_confirm)

        logger.info("Waiting for trigger selection in GUI window…")
        plt.show(block=True)

        if not state["confirmed"]:
            raise ProcessorError("Trigger selection cancelled (window closed without confirming).")

        return state["selected"]

    @staticmethod
    def _pick_preview_channel(raw: mne.io.Raw) -> tuple:
        """Return ``(ch_idx, ch_name)`` for the first EEG channel.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object.

        Returns
        -------
        tuple of (int, str)
            Channel index and name.
        """
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(eeg_picks) > 0:
            return int(eeg_picks[0]), raw.ch_names[eeg_picks[0]]
        return 0, raw.ch_names[0]

    @staticmethod
    def _downsample_for_display(raw: mne.io.Raw, ch_idx: int) -> tuple:
        """Return a downsampled ``(data, times)`` pair for plotting.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object.
        ch_idx : int
            Channel index to extract.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            Downsampled amplitude and corresponding time arrays.
        """
        full_data = raw.get_data(picks=[ch_idx])[0]
        n_samples = len(full_data)
        sfreq = raw.info["sfreq"]

        if n_samples > _DISPLAY_DOWNSAMPLE_TARGET:
            step = n_samples // _DISPLAY_DOWNSAMPLE_TARGET
            data = full_data[::step]
            times = np.arange(len(data)) * (step / sfreq)
        else:
            data = full_data
            times = np.arange(n_samples) / sfreq

        return data, times

    @staticmethod
    def _create_gui_layout(n_events: int) -> tuple:
        """Build the matplotlib figure and axes for the explorer window.

        Parameters
        ----------
        n_events : int
            Number of event rows (controls radio-button area height).

        Returns
        -------
        tuple
            ``(fig, (ax_plot, ax_radio, ax_info, ax_btn))``.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(15, 7))
        fig.canvas.manager.set_window_title("FACETpy — Trigger Explorer")
        fig.patch.set_facecolor("#FAFAFA")

        radio_height = max(0.40, min(0.68, n_events * 0.065))

        ax_plot = fig.add_axes([0.06, 0.13, 0.56, 0.80])
        ax_radio = fig.add_axes(
            [0.67, 0.93 - radio_height, 0.30, radio_height],
            facecolor="#F5F5F5",
        )
        ax_info = fig.add_axes(
            [0.67, 0.93 - radio_height - 0.17, 0.30, 0.14],
            facecolor="#FAFAFA",
        )
        ax_btn = fig.add_axes([0.67, 0.04, 0.30, 0.08])

        ax_radio.set_title("Trigger Sources", fontsize=10, fontweight="bold", loc="left")
        ax_info.set_xticks([])
        ax_info.set_yticks([])
        for spine in ax_info.spines.values():
            spine.set_visible(False)

        return fig, (ax_plot, ax_radio, ax_info, ax_btn)

    @staticmethod
    def _update_gui_plot(
        ax: Any,
        state: dict[str, Any],
        trigger_times: np.ndarray,
        ch_name: str,
        data: np.ndarray,
    ) -> None:
        """Redraw trigger markers on the waveform axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The waveform axes.
        state : dict
            Mutable state dict holding the current ``vlines`` collection.
        trigger_times : np.ndarray
            Trigger onset times in seconds.
        ch_name : str
            Preview channel name (for the title).
        data : np.ndarray
            Downsampled amplitude array (for y-limits).
        """
        if state["vlines"] is not None:
            state["vlines"].remove()
            state["vlines"] = None

        if len(trigger_times) > 0:
            ymin, ymax = np.min(data), np.max(data)
            margin = (ymax - ymin) * 0.05
            state["vlines"] = ax.vlines(
                trigger_times,
                ymin - margin,
                ymax + margin,
                colors="#FF5722",
                alpha=0.45,
                linewidth=0.6,
                label="triggers",
            )

        n_shown = len(trigger_times)
        ax.set_title(
            f"{ch_name}  —  {n_shown} trigger{'s' if n_shown != 1 else ''} shown",
            fontsize=10,
        )

    @staticmethod
    def _update_gui_info(ax: Any, row: dict[str, Any]) -> None:
        """Update the info text box below the radio buttons.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Info axes (text only, no ticks).
        row : dict
            Currently selected event-table row.
        """
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        lines = [
            f"Source:  {row['source']}",
            f"Value:   {row['description']}",
            f"Count:  {row['count']}",
            f"Range:  {row['detail']}",
        ]
        ax.text(
            0.05,
            0.90,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
        )

    # ------------------------------------------------------------------
    # Terminal / Rich table display
    # ------------------------------------------------------------------

    def _display_event_table(self, event_table: list[dict[str, Any]]) -> None:
        """Render the event table to the console using Rich.

        Parameters
        ----------
        event_table : list of dict
            Unified event table produced by ``_build_event_table``.
        """
        try:
            from rich.console import Console as RichConsole
            from rich.table import Table
        except ImportError:
            self._display_event_table_plain(event_table)
            return

        console_obj = get_console()
        rich_console = console_obj.get_rich_console()
        if rich_console is None:
            rich_console = RichConsole()

        table = Table(
            title="Available Trigger Sources",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("Source", style="magenta", min_width=14)
        table.add_column("Description / Value", style="green", min_width=18)
        table.add_column("Count", justify="right", style="yellow", min_width=7)
        table.add_column("Range", style="dim", min_width=20)

        for row in event_table:
            table.add_row(
                str(row["index"]),
                row["source"],
                row["description"],
                str(row["count"]),
                row["detail"],
            )

        rich_console.print()
        rich_console.print(table)
        rich_console.print()

    @staticmethod
    def _display_event_table_plain(event_table: list[dict[str, Any]]) -> None:
        """Fallback plain-text display when Rich is unavailable.

        Parameters
        ----------
        event_table : list of dict
            Unified event table.
        """
        header = f"{'#':>4}  {'Source':<16} {'Description':<20} {'Count':>7}  {'Range'}"
        sep = "-" * len(header)
        lines = ["\nAvailable Trigger Sources", sep, header, sep]
        for row in event_table:
            lines.append(
                f"{row['index']:>4}  {row['source']:<16} {row['description']:<20} {row['count']:>7}  {row['detail']}"
            )
        lines.append(sep)
        print("\n".join(lines))

    # ------------------------------------------------------------------
    # Terminal selection
    # ------------------------------------------------------------------

    def _auto_select_event(self, event_table: list[dict[str, Any]]) -> dict[str, Any]:
        """Select the first event whose description matches ``auto_select``.

        Parameters
        ----------
        event_table : list of dict
            Unified event table.

        Returns
        -------
        dict
            Matched row.
        """
        pattern = re.compile(self.auto_select)  # type: ignore[arg-type]
        for row in event_table:
            if pattern.search(row["description"]):
                logger.info(
                    "Auto-selected trigger '{}' (matched '{}')",
                    row["description"],
                    self.auto_select,
                )
                return row

        descriptions = [r["description"] for r in event_table]
        raise ProcessorError(
            f"auto_select pattern '{self.auto_select}' did not match any event. Available: {descriptions}"
        )

    def _terminal_select_event(self, event_table: list[dict[str, Any]]) -> dict[str, Any]:
        """Prompt the user to pick a trigger event by number or description.

        Parameters
        ----------
        event_table : list of dict
            Unified event table.

        Returns
        -------
        dict
            Selected row.
        """
        console_obj = get_console()
        max_idx = len(event_table)

        with suspend_raw_mode():
            console_obj.set_active_prompt(f"Select trigger source [1-{max_idx}]: ")
            try:
                answer = input(f"Select trigger source [1-{max_idx}] or type a description: ").strip()
            finally:
                console_obj.clear_active_prompt()

        if not answer:
            raise ProcessorError("No trigger source selected (empty input).")

        if answer.isdigit():
            choice = int(answer)
            if 1 <= choice <= max_idx:
                return event_table[choice - 1]
            raise ProcessorError(f"Invalid selection '{choice}'. Must be between 1 and {max_idx}.")

        for row in event_table:
            if row["description"] == answer:
                return row

        pattern = re.compile(answer)
        matches = [r for r in event_table if pattern.search(r["description"])]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            descs = [m["description"] for m in matches]
            raise ProcessorError(
                f"Pattern '{answer}' matched multiple events: {descs}. Please be more specific or use the row number."
            )

        raise ProcessorError(f"'{answer}' does not match any available event description.")

    # ------------------------------------------------------------------
    # Trigger detection
    # ------------------------------------------------------------------

    @staticmethod
    def _regex_for_selection(selected: dict[str, Any]) -> str:
        """Build a regex pattern anchored to the selected event description.

        For grouped sequential annotations (e.g. ``"TR 1"``, ``"TR 2"``, …)
        the stored description is the bare prefix (``"TR"``).  The regex is
        expanded to ``^TR\\s+\\d+$`` so that downstream ``TriggerDetector``
        steps correctly match all members of the sequence.

        Parameters
        ----------
        selected : dict
            Selected row from the event table.

        Returns
        -------
        str
            Regex pattern suitable for ``TriggerDetector``.
        """
        desc = selected["description"]
        if selected.get("grouped_prefix"):
            return rf"^{re.escape(desc)}\s+\d+$"
        return rf"\b{re.escape(desc)}\b"

    def _detect_triggers(self, raw: mne.io.Raw, selected: dict[str, Any]) -> np.ndarray:
        """Detect trigger sample positions for the selected event.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object.
        selected : dict
            Selected row from the event table.

        Returns
        -------
        np.ndarray
            Trigger sample positions (int64).
        """
        desc = selected["description"]
        source = selected["source"]

        if source.startswith("stim"):
            pattern = re.compile(rf"\b{re.escape(desc)}\b")
            triggers = self._triggers_from_stim(raw, pattern, source)
        elif selected.get("grouped_prefix"):
            triggers = self._triggers_from_annotations_prefix(raw, desc)
        else:
            triggers = self._triggers_from_annotations(raw, desc)

        # MNE returns absolute sample indices (onset * sfreq + first_samp).
        # Normalize to 0-indexed positions relative to the current raw start
        # so that triggers can be used directly as indices into raw._data.
        return triggers - raw.first_samp

    @staticmethod
    def _triggers_from_annotations_prefix(raw: mne.io.Raw, prefix: str) -> np.ndarray:
        """Extract trigger positions for all annotations matching ``prefix N``.

        Used when the user selects a grouped sequential annotation (e.g. all
        ``"TR N"`` triggers).

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object.
        prefix : str
            The common text prefix (e.g. ``"TR"``).

        Returns
        -------
        np.ndarray
            Sorted trigger sample positions (int64).
        """
        pattern = rf"^{re.escape(prefix)}\s+\d+$"
        events, _ = mne.events_from_annotations(raw, regexp=pattern, verbose=False)
        if len(events) == 0:
            return np.array([], dtype=np.int64)
        return np.array(sorted(events[:, 0]), dtype=np.int64)

    @staticmethod
    def _triggers_from_stim(raw: mne.io.Raw, pattern: re.Pattern, source: str) -> np.ndarray:
        """Extract trigger positions from a STIM channel.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object.
        pattern : re.Pattern
            Compiled regex to match event values.
        source : str
            Source label containing the channel name in parentheses.

        Returns
        -------
        np.ndarray
            Trigger sample positions.
        """
        ch_match = re.search(r"\((.+)\)", source)
        stim_picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)
        ch_name = ch_match.group(1) if ch_match else raw.ch_names[stim_picks[0]]

        events = mne.find_events(raw, stim_channel=ch_name, initial_event=True, verbose=False)
        filtered = [ev for ev in events if pattern.search(str(ev[2]))]
        return np.array([ev[0] for ev in filtered], dtype=np.int64)

    @staticmethod
    def _triggers_from_annotations(raw: mne.io.Raw, description: str) -> np.ndarray:
        """Extract trigger positions from annotations matching ``description``.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object.
        description : str
            Exact annotation description to match.

        Returns
        -------
        np.ndarray
            Trigger sample positions.
        """
        regex = rf"\b{re.escape(description)}\b"
        events, _ = mne.events_from_annotations(raw, regexp=regex, verbose=False)
        if len(events) == 0:
            return np.array([], dtype=np.int64)
        return np.array([ev[0] for ev in events], dtype=np.int64)

    # ------------------------------------------------------------------
    # Artifact metadata (shared with TriggerDetector)
    # ------------------------------------------------------------------

    def _compute_artifact_metadata(self, triggers: np.ndarray) -> dict:
        """Estimate artifact length and detect volume gaps from trigger spacing.

        Parameters
        ----------
        triggers : np.ndarray
            Detected trigger sample positions.

        Returns
        -------
        dict
            Keys: ``artifact_length``, ``volume_gaps``, optionally
            ``slices_per_volume``.
        """
        if len(triggers) <= 1:
            return {"artifact_length": None, "volume_gaps": False}

        trigger_diffs = np.diff(triggers)
        ptp = np.ptp(trigger_diffs)

        if ptp > 3:
            return self._compute_slice_volume_metadata(triggers, trigger_diffs)

        return {
            "artifact_length": int(np.max(trigger_diffs)),
            "volume_gaps": False,
        }

    def _compute_slice_volume_metadata(self, triggers: np.ndarray, trigger_diffs: np.ndarray) -> dict:
        """Compute metadata when volume-level gaps are present.

        Parameters
        ----------
        triggers : np.ndarray
            All trigger sample positions.
        trigger_diffs : np.ndarray
            Differences between consecutive triggers.

        Returns
        -------
        dict
            Keys: ``artifact_length``, ``volume_gaps``, ``slices_per_volume``.
        """
        mean_val = np.mean([np.median(trigger_diffs), np.max(trigger_diffs)])
        slice_diffs = trigger_diffs[trigger_diffs < mean_val]
        artifact_length = int(np.max(slice_diffs))

        gap_indices = np.where(trigger_diffs >= mean_val)[0]
        slices_per_volume = None

        if len(gap_indices) > 0:
            slice_counts = []
            last_idx = -1
            for idx in gap_indices:
                slice_counts.append(idx - last_idx)
                last_idx = idx
            if last_idx < len(triggers) - 1:
                slice_counts.append(len(triggers) - 1 - last_idx)
            if slice_counts:
                slices_per_volume = int(np.median(slice_counts))
                logger.info("Estimated slices per volume: {}", slices_per_volume)

        return {
            "artifact_length": artifact_length,
            "volume_gaps": True,
            "slices_per_volume": slices_per_volume,
        }


InteractiveTriggerExplorer = TriggerExplorer
