"""
Channel-Sequential Execution Module

Provides memory-efficient channel-by-channel pipeline execution that is
completely independent of multiprocessing parallelism.
"""

import time

import mne
import numpy as np
from loguru import logger

from facet.console.manager import get_console
from facet.console.progress import get_current_step_index
from facet.logging_config import suppress_stdout

from .context import ProcessingContext
from .processor import Processor

MERGED_LIST_METADATA_KEYS = ("artifact_template_matrices",)


class ChannelSequentialExecutor:
    """
    Execute a sequence of processors one channel at a time.

    For each data channel the full processor sequence runs to completion
    before the next channel starts.  This ensures that large intermediate
    representations (e.g. 10x upsampled data) only ever exist for a single
    channel simultaneously.

    Processors with ``run_once = True`` execute only for the first channel;
    all subsequent channels skip them and inherit the metadata produced by
    that first run.

    Non-data channels (stim, misc, ...) are handled separately: copied
    unchanged when the sampling rate stays the same, or resampled via MNE
    when the batch changes the sampling rate.

    The live console (when active) receives fine-grained channel and
    processor progress updates via the modern console manager.

    Example::

        executor = ChannelSequentialExecutor()
        result = executor.execute(
            [HighPassFilter(1.0), UpSample(10), AASCorrection(), DownSample(10)],
            context,
        )
    """

    def execute(
        self,
        processors: list[Processor],
        context: ProcessingContext,
    ) -> ProcessingContext:
        """
        Run *processors* on *context* one channel at a time.

        Parameters
        ----------
        processors : list of Processor
            Processors to execute in order for every channel.
        context : ProcessingContext
            Input context containing the full multi-channel dataset.

        Returns
        -------
        ProcessingContext
            Merged output context with all channels processed.
        """
        if not processors:
            return context

        raw = context.get_raw()
        ch_names = raw.ch_names
        n_ch = len(ch_names)
        if n_ch == 0:
            return context

        data_idx, passthrough_idx = self._classify_channels(raw)

        if not data_idx:
            logger.warning("No data channels found; returning context unchanged")
            return context

        proc_names = " → ".join(p.name for p in processors)
        logger.info(
            "Channel-sequential [{}] ({} data channels)",
            proc_names,
            len(data_idx),
        )

        console = get_console()
        step_idx = get_current_step_index() or 0
        data_ch_names = [ch_names[i] for i in data_idx]

        console.start_channel_batch(
            processor_names=[p.name for p in processors],
            channel_names=data_ch_names,
            batch_step_offset=step_idx,
        )

        _run_once_executed: set[str] = set()
        metadata_states: list[object] = []
        merged_data: np.ndarray | None = None
        n_times_out = 0
        new_sfreq = raw.info["sfreq"]
        saved_metadata = None
        handle_noise = False
        noise_data: np.ndarray | None = None

        try:
            for k, ch_abs_idx in enumerate(data_idx):
                ch_start = time.time()
                logger.info(
                    "Channel-sequential channel {}/{}: {}",
                    k + 1,
                    len(data_idx),
                    data_ch_names[k],
                )
                console.channel_started(k, data_ch_names[k])

                ch_ctx = self._create_channel_context(context, ch_abs_idx)
                if k == 0:
                    metadata_states = [self._metadata_for_replay(ch_ctx.metadata)]
                for pi, proc in enumerate(processors):
                    if k > 0 and pi < len(metadata_states):
                        ch_ctx = ch_ctx.with_metadata(metadata_states[pi].copy())
                    before_counts = self._custom_list_lengths(ch_ctx, MERGED_LIST_METADATA_KEYS)
                    skipped = proc.run_once and proc.name in _run_once_executed
                    console.channel_processor_started(k, pi)
                    proc_start = time.time()
                    ch_ctx = self._run_proc(proc, ch_ctx, _run_once_executed)
                    if k > 0 and saved_metadata is not None:
                        self._merge_new_custom_list_items(
                            target_metadata=saved_metadata,
                            before_counts=before_counts,
                            source_context=ch_ctx,
                            keys=MERGED_LIST_METADATA_KEYS,
                        )
                    if k == 0:
                        metadata_states.append(self._metadata_for_replay(ch_ctx.metadata))
                    console.channel_processor_completed(
                        k,
                        pi,
                        time.time() - proc_start,
                        skipped=skipped,
                    )

                ch_data = ch_ctx.get_data(copy=False)

                if k == 0:
                    n_times_out = ch_data.shape[1]
                    new_sfreq = ch_ctx.get_raw().info["sfreq"]
                    saved_metadata = ch_ctx.metadata.copy()
                    merged_data = np.zeros((n_ch, n_times_out), dtype=ch_data.dtype)

                    handle_noise = ch_ctx.has_estimated_noise()
                    if handle_noise:
                        first_noise = ch_ctx.get_estimated_noise()
                        if first_noise is not None and first_noise.ndim == 2:
                            noise_data = np.zeros(
                                (n_ch, first_noise.shape[1]),
                                dtype=first_noise.dtype,
                            )

                merged_data[ch_abs_idx] = ch_data[0]

                if handle_noise and noise_data is not None and ch_ctx.has_estimated_noise():
                    ch_noise = ch_ctx.get_estimated_noise()
                    if ch_noise is not None and ch_noise.ndim == 2:
                        noise_data[ch_abs_idx] = ch_noise[0]

                del ch_ctx
                console.channel_completed(k, time.time() - ch_start)
        finally:
            console.end_channel_batch()

        # --- pass-through channels (stim, misc, ...) -------------------------
        if passthrough_idx:
            if n_times_out == raw.n_times:
                orig = raw.get_data()
                for i in passthrough_idx:
                    merged_data[i] = orig[i]
            else:
                picks = [ch_names[i] for i in passthrough_idx]
                pt_raw = raw.copy().pick(picks)
                with suppress_stdout():
                    pt_raw.resample(new_sfreq)
                for j, i in enumerate(passthrough_idx):
                    merged_data[i] = pt_raw.get_data()[j]
                del pt_raw

        # --- build merged output context -------------------------------------
        info = raw.info.copy()
        if hasattr(info, "_unlock"):
            with info._unlock():
                info["sfreq"] = new_sfreq
        else:
            info["sfreq"] = new_sfreq

        with suppress_stdout():
            new_raw = mne.io.RawArray(merged_data, info)

        result = context.with_raw(new_raw)
        result._metadata = saved_metadata
        if handle_noise and noise_data is not None:
            result.set_estimated_noise(noise_data)
        return result

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _run_proc(
        proc: Processor,
        ctx: ProcessingContext,
        run_once_executed: set[str],
    ) -> ProcessingContext:
        """Execute *proc* on *ctx*, honouring the ``run_once`` flag."""
        if proc.run_once and proc.name in run_once_executed:
            return ctx
        result = proc.execute(ctx)
        if proc.run_once:
            run_once_executed.add(proc.name)
        return result

    @staticmethod
    def _custom_list_lengths(context: ProcessingContext, keys: tuple[str, ...]) -> dict[str, int]:
        """Return lengths for mergeable custom-metadata list keys."""
        lengths: dict[str, int] = {}
        for key in keys:
            value = context.metadata.custom.get(key)
            lengths[key] = len(value) if isinstance(value, list) else 0
        return lengths

    @staticmethod
    def _merge_new_custom_list_items(
        target_metadata,
        before_counts: dict[str, int],
        source_context: ProcessingContext,
        keys: tuple[str, ...],
    ) -> None:
        """Merge list entries produced by one channel before metadata resets."""
        for key in keys:
            value = source_context.metadata.custom.get(key)
            if not isinstance(value, list):
                continue

            start = before_counts.get(key, 0)
            if len(value) <= start:
                continue

            target_metadata.custom.setdefault(key, []).extend(value[start:])

    @staticmethod
    def _metadata_for_replay(metadata):
        """Copy metadata used for later channels without bulky reports."""
        replay = metadata.copy()
        for key in MERGED_LIST_METADATA_KEYS:
            replay.custom.pop(key, None)
        return replay

    @staticmethod
    def _classify_channels(raw: mne.io.Raw):
        """Split channel indices into data channels and pass-through channels."""
        try:
            from mne.io.pick import _DATA_CH_TYPES_SPLIT

            data_idx = [i for i, t in enumerate(raw.get_channel_types()) if t in _DATA_CH_TYPES_SPLIT]
        except ImportError:
            data_idx = list(range(len(raw.ch_names)))

        passthrough_idx = [i for i in range(len(raw.ch_names)) if i not in set(data_idx)]
        return data_idx, passthrough_idx

    @staticmethod
    def _create_channel_context(
        context: ProcessingContext,
        ch_idx: int,
    ) -> ProcessingContext:
        """
        Create a single-channel subset context.

        Uses ``raw.get_data(picks=[name])`` to extract only the requested
        channel's data without duplicating the full array first.
        """
        raw = context.get_raw()
        ch_name = raw.ch_names[ch_idx]
        data = raw.get_data(picks=[ch_name])
        info = mne.pick_info(raw.info, [ch_idx])
        with suppress_stdout():
            subset_raw = mne.io.RawArray(data, info)
        subset_ctx = context.with_raw(subset_raw)

        # with_raw() copies the full noise matrix; keep only the active channel.
        subset_ctx._estimated_noise = None
        if context.has_estimated_noise():
            noise = context.get_estimated_noise()
            if noise is not None and noise.ndim == 2:
                if noise.shape[0] == 1:
                    subset_ctx.set_estimated_noise(noise.copy())
                elif ch_idx < noise.shape[0]:
                    subset_ctx.set_estimated_noise(noise[ch_idx : ch_idx + 1].copy())

        return subset_ctx
