"""
Parallel Execution Module

This module provides multiprocessing support for pipeline execution.

Author: FACETpy Team
Date: 2025-01-12
"""

import contextlib
import functools
import multiprocessing as mp
import sys
from collections.abc import Callable

import mne
import numpy as np
from loguru import logger

from facet.console.progress import processor_progress
from facet.logging_config import suppress_stdout

from .context import ProcessingContext
from .processor import Processor

# Use "spawn" so child processes start clean without inheriting threads
# (e.g. the Textual TUI thread).  "fork" is unsafe in multithreaded
# processes and can deadlock when the forked child inherits held locks.
# Workers already serialise everything via to_dict/from_dict, so spawn
# is a drop-in replacement.
if sys.platform != "win32":
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn", force=True)


def _worker_function(processor_config: dict, context_data: dict) -> dict:
    """
    Worker function for multiprocessing.

    This function runs in a separate process and must be picklable.

    Args:
        processor_config: Serialized processor configuration
        context_data: Serialized context data

    Returns:
        Serialized result context
    """
    processor_class = processor_config["class"]
    processor_params = processor_config["parameters"]
    processor = processor_class(**processor_params)

    context = ProcessingContext.from_dict(context_data)

    result = processor.execute(context)

    return result.to_dict()


class ParallelExecutor:
    """
    Executor for parallel processing of channels or epochs.

    This class handles multiprocessing for processors that support it,
    typically for channel-wise or epoch-wise operations.

    Example:
        executor = ParallelExecutor(n_jobs=4)
        result_context = executor.execute(processor, context)
    """

    # Multiprocessing has a meaningful fixed cost even before signal arrays
    # are deserialized. Keeping this estimate explicit makes the automatic
    # worker cap conservative and easy to adjust as the runtime evolves.
    _WORKER_BASE_OVERHEAD_BYTES = 64 * 1024**2
    _AVAILABLE_MEMORY_FRACTION = 0.5

    def __init__(self, n_jobs: int = -1, backend: str = "multiprocessing", verbose: bool = True):
        """
        Initialize parallel executor.

        Args:
            n_jobs: Number of parallel jobs (-1 for all CPUs, -2 for all but one)
            backend: Parallel backend ("multiprocessing", "threading", or "serial")
            verbose: Show progress messages
        """
        self.requested_n_jobs = n_jobs
        self.n_jobs = self._determine_n_jobs(n_jobs)
        self.backend = backend
        self.verbose = verbose

        if backend not in ["multiprocessing", "threading", "serial"]:
            raise ValueError(f"Invalid backend: {backend}. Choose from: multiprocessing, threading, serial")

    def _determine_n_jobs(self, n_jobs: int) -> int:
        """Determine actual number of jobs."""
        if n_jobs == -1:
            return mp.cpu_count()
        elif n_jobs == -2:
            return max(1, mp.cpu_count() - 1)
        elif n_jobs < -2:
            return max(1, mp.cpu_count() + n_jobs + 1)
        elif n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        else:
            return n_jobs

    def execute(self, processor: Processor, context: ProcessingContext) -> ProcessingContext:
        """
        Execute processor in parallel if possible.

        This method attempts to parallelize the processor execution.
        If parallelization is not applicable, falls back to serial execution.

        Args:
            processor: Processor to execute
            context: Input context

        Returns:
            Output context
        """
        if not processor.parallel_safe:
            logger.warning(f"Processor {processor.name} is not parallel-safe, executing serially")
            return processor.execute(context)

        if getattr(processor, "channel_wise", False):
            return self._execute_channel_wise(processor, context)

        if hasattr(processor, "parallelize_by_epochs") and processor.parallelize_by_epochs:
            return self._execute_epoch_wise(processor, context)

        # Fall back to serial execution
        logger.debug(f"No parallelization strategy found for {processor.name}, executing serially")
        return processor.execute(context)

    def _execute_channel_wise(self, processor: Processor, context: ProcessingContext) -> ProcessingContext:
        """
        Execute processor channel-wise in parallel.

        Args:
            processor: Processor to execute
            context: Input context

        Returns:
            Output context with processed channels
        """
        raw = context.get_raw()
        n_channels = len(raw.ch_names)

        if n_channels == 0:
            logger.warning("No channels available for parallel execution")
            return context

        worker_count = self._resolve_worker_count(
            processor=processor,
            context=context,
            task_count=n_channels,
        )
        logger.info(
            "Executing {} in parallel across {} job(s)",
            processor.name,
            worker_count,
        )

        channel_chunks = self._split_into_chunks(list(range(n_channels)), worker_count)

        progress_total = n_channels if n_channels > 0 else None
        with processor_progress(
            total=progress_total,
            message=f"{processor.name}: channels",
        ) as progress:

            def _update_progress(processed: int) -> None:
                if processed <= 0:
                    return
                next_value = progress.current + processed
                progress.advance(
                    processed,
                    message=(f"{int(next_value)}/{n_channels} channels" if n_channels else "channels"),
                )

            if self.backend == "multiprocessing":
                results = self._execute_multiprocessing(
                    processor,
                    context,
                    channel_chunks,
                    worker_count=worker_count,
                    progress_callback=_update_progress,
                )
            elif self.backend == "threading":
                results = self._execute_threading(
                    processor,
                    context,
                    channel_chunks,
                    worker_count=worker_count,
                    progress_callback=_update_progress,
                )
            else:  # serial
                results = self._execute_serial(
                    processor,
                    context,
                    channel_chunks,
                    progress_callback=_update_progress,
                )

        return self._merge_channel_results(context, results)

    def _execute_epoch_wise(self, processor: Processor, context: ProcessingContext) -> ProcessingContext:
        """
        Execute processor epoch-wise in parallel.

        Args:
            processor: Processor to execute
            context: Input context

        Returns:
            Output context with processed epochs
        """
        logger.info(f"Executing {processor.name} epoch-wise in parallel across {self.n_jobs} jobs")

        if not context.has_triggers():
            raise ValueError("Context has no triggers for epoch-wise processing")

        triggers = context.get_triggers()
        n_epochs = len(triggers)

        epoch_chunks = self._split_into_chunks(list(range(n_epochs)), self.n_jobs)

        if self.backend == "multiprocessing":
            results = self._execute_multiprocessing_epochs(processor, context, epoch_chunks)
        elif self.backend == "threading":
            results = self._execute_threading_epochs(processor, context, epoch_chunks)
        else:  # serial
            results = self._execute_serial_epochs(processor, context, epoch_chunks)

        return self._merge_epoch_results(context, results)

    def _execute_multiprocessing(
        self,
        processor: Processor,
        context: ProcessingContext,
        channel_chunks: list[list[int]],
        worker_count: int | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[ProcessingContext]:
        """Execute using multiprocessing without retaining every task payload."""
        processor_config = {"class": processor.__class__, "parameters": processor._parameters}
        processes = worker_count or min(self.n_jobs, max(1, len(channel_chunks)))

        def _serialized_contexts():
            # ``Pool.imap`` consumes this iterator incrementally. Each selected
            # channel payload can therefore be released by the parent as soon
            # as it has been sent to a worker instead of all payloads being
            # accumulated in a list before the pool starts.
            for chunk in channel_chunks:
                chunk_context = self._create_channel_subset_context(context, chunk)
                yield chunk_context.to_dict()

        worker = functools.partial(_worker_function, processor_config)
        contexts: list[ProcessingContext] = []
        with mp.Pool(processes=processes) as pool:
            for idx, result in enumerate(pool.imap(worker, _serialized_contexts(), chunksize=1)):
                contexts.append(ProcessingContext.from_dict(result))
                if progress_callback:
                    chunk_size = len(channel_chunks[idx])
                    progress_callback(chunk_size)

        return contexts

    def _execute_threading(
        self,
        processor: Processor,
        context: ProcessingContext,
        channel_chunks: list[list[int]],
        worker_count: int | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[ProcessingContext]:
        """Execute using threading (GIL-limited)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[ProcessingContext] = []
        max_workers = worker_count or min(self.n_jobs, max(1, len(channel_chunks)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for chunk in channel_chunks:
                chunk_context = self._create_channel_subset_context(context, chunk)
                future = executor.submit(processor.execute, chunk_context)
                futures[future] = len(chunk)

            for future in as_completed(futures):
                results.append(future.result())
                if progress_callback:
                    progress_callback(futures[future])

        return results

    def _execute_serial(
        self,
        processor: Processor,
        context: ProcessingContext,
        channel_chunks: list[list[int]],
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[ProcessingContext]:
        """Execute serially (for debugging/comparison)."""
        results = []
        for chunk in channel_chunks:
            chunk_context = self._create_channel_subset_context(context, chunk)
            result = processor.execute(chunk_context)
            results.append(result)
            if progress_callback:
                progress_callback(len(chunk))
        return results

    def _create_channel_subset_context(
        self, context: ProcessingContext, channel_indices: list[int]
    ) -> ProcessingContext:
        """Create a context containing only the requested channels.

        ``raw.copy().pick(...)`` first duplicates the complete recording and
        only then discards unselected channels. Direct extraction keeps the
        transient allocation proportional to the worker's actual subset while
        preserving picked channel info, the original sample offset, and
        annotations.
        """
        raw = context.get_raw()
        if not channel_indices:
            raise ValueError("channel_indices must not be empty")

        if raw.preload and getattr(raw, "_data", None) is not None:
            subset_data = raw._data[channel_indices, :]
        else:
            subset_data = raw.get_data(picks=channel_indices)

        subset_info = mne.pick_info(raw.info, channel_indices, copy=True)
        with suppress_stdout():
            subset_raw = mne.io.RawArray(
                data=subset_data,
                info=subset_info,
                first_samp=raw.first_samp,
                copy="auto",
            )
        if len(raw.annotations):
            annotations = raw.annotations.copy()
            if annotations.orig_time is None:
                # MNE stores no-orig-time annotation onsets with
                # ``raw.first_time`` already applied. ``set_annotations``
                # applies it again, so convert back to local onsets first.
                annotations.onset -= raw.first_time
            subset_raw.set_annotations(annotations)

        subset_noise = None
        copy_subset_noise = False
        if context.has_estimated_noise():
            noise = context.get_estimated_noise()
            if noise is not None and noise.ndim == 2:
                subset_noise = noise[channel_indices, :]
            else:
                # Estimated noise is expected to be channel x sample. Retain
                # historical behaviour for any non-standard representation.
                subset_noise = noise
                copy_subset_noise = True

        return context.with_raw(
            subset_raw,
            copy_metadata=True,
            estimated_noise=subset_noise,
            copy_estimated_noise=copy_subset_noise,
        )

    def _merge_channel_results(
        self, original_context: ProcessingContext, results: list[ProcessingContext]
    ) -> ProcessingContext:
        """Merge channel-wise results back into single context."""
        if not results:
            return original_context

        original_raw = original_context.get_raw()
        template_raw = results[0].get_raw()
        template_data = results[0].get_data(copy=False)

        new_sfreq = template_raw.info["sfreq"]
        n_times = template_data.shape[1]
        dtype = template_data.dtype

        merged_data = np.zeros((len(original_raw.ch_names), n_times), dtype=dtype)
        channel_index = {name: idx for idx, name in enumerate(original_raw.ch_names)}

        for result_ctx in results:
            result_raw = result_ctx.get_raw()
            result_data = result_ctx.get_data(copy=False)
            for j, ch_name in enumerate(result_raw.ch_names):
                ch_idx = channel_index[ch_name]
                merged_data[ch_idx] = result_data[j]

        # Build new RawArray at the upsampled rate to avoid mutating protected info
        info = original_raw.info.copy()
        if hasattr(info, "_unlock"):
            with info._unlock():
                info["sfreq"] = new_sfreq
        else:
            info["sfreq"] = new_sfreq

        with suppress_stdout():
            new_raw = mne.io.RawArray(data=merged_data, info=info)

        noise_data = None
        if any(result_ctx.has_estimated_noise() for result_ctx in results):
            # Estimated noise is stored channel-wise; merge similarly
            noise_data = np.zeros_like(merged_data)
            for result_ctx in results:
                if not result_ctx.has_estimated_noise():
                    continue
                result_noise = result_ctx.get_estimated_noise()
                result_raw = result_ctx.get_raw()
                for j, ch_name in enumerate(result_raw.ch_names):
                    ch_idx = channel_index[ch_name]
                    noise_data[ch_idx] = result_noise[j]

        # Supply the already merged estimate atomically so ``with_raw`` does
        # not copy the original full noise array only to replace it.
        merged_context = original_context.with_raw(
            new_raw,
            estimated_noise=noise_data,
            copy_estimated_noise=False,
        )
        merged_context._metadata = results[0].metadata.copy()

        return merged_context

    def _resolve_worker_count(
        self,
        processor: Processor,
        context: ProcessingContext,
        task_count: int,
    ) -> int:
        """Cap multiprocessing workers using a conservative RAM estimate."""
        requested = min(self.n_jobs, max(1, task_count))
        if self.backend != "multiprocessing" or requested <= 1:
            return requested

        available = self._available_memory_bytes()
        if available is None:
            return requested

        memory_limit = max(1, int(available * self._AVAILABLE_MEMORY_FRACTION))
        for worker_count in range(requested, 0, -1):
            estimated = self._estimate_parallel_memory(
                processor=processor,
                context=context,
                worker_count=worker_count,
            )
            if estimated <= memory_limit:
                if worker_count < requested:
                    logger.warning(
                        "Capping parallel jobs from {} to {}: estimated {:.2f} GiB "
                        "fits the {:.2f} GiB worker-memory allowance",
                        requested,
                        worker_count,
                        estimated / 1024**3,
                        memory_limit / 1024**3,
                    )
                return worker_count

        estimated_one = self._estimate_parallel_memory(
            processor=processor,
            context=context,
            worker_count=1,
        )
        logger.warning(
            "One worker is estimated to require {:.2f} GiB, above the {:.2f} GiB "
            "worker-memory allowance; continuing serially because the worker "
            "count cannot be reduced further",
            estimated_one / 1024**3,
            memory_limit / 1024**3,
        )
        return 1

    @classmethod
    def _estimate_parallel_memory(
        cls,
        processor: Processor,
        context: ProcessingContext,
        worker_count: int,
    ) -> int:
        """Estimate aggregate child-process memory for channel-wise work.

        The estimate accounts for selected-channel input/output payloads,
        common processor copies/intermediates, the largest sample-rate
        expansion of the current processor, and a fixed interpreter/library
        cost for every spawned worker.
        """
        raw = context.get_raw()
        n_channels = max(1, len(raw.ch_names))
        channels_per_worker = (n_channels + worker_count - 1) // worker_count

        signal_bytes = cls._raw_nbytes(raw)
        if context.has_estimated_noise():
            noise = context.get_estimated_noise()
            if noise is not None:
                signal_bytes += int(noise.nbytes)

        bytes_per_channel = (signal_bytes + n_channels - 1) // n_channels
        worker_input_bytes = bytes_per_channel * channels_per_worker
        expansion = cls._processor_sample_expansion(processor, context)

        # Input serialization/deserialization plus a processor-owned copy and
        # expanded output/intermediate storage. This intentionally errs on the
        # conservative side; numerical processing itself is unchanged.
        worker_signal_bytes = worker_input_bytes * (2.0 + 2.0 * expansion)
        per_worker = cls._WORKER_BASE_OVERHEAD_BYTES + int(worker_signal_bytes)
        return max(1, worker_count) * per_worker

    @staticmethod
    def _raw_nbytes(raw) -> int:
        """Return Raw signal bytes without materializing a lazy recording."""
        data = getattr(raw, "_data", None)
        if raw.preload and data is not None:
            return int(data.nbytes)

        dtype = getattr(raw, "_dtype", np.float64)
        try:
            sample_bytes = np.dtype(dtype).itemsize
        except TypeError:
            sample_bytes = np.dtype(np.float64).itemsize
        return int(len(raw.ch_names) * raw.n_times * sample_bytes)

    @staticmethod
    def _processor_sample_expansion(
        processor: Processor,
        context: ProcessingContext,
    ) -> float:
        """Estimate the processor's peak sample-count expansion."""
        if processor.name == "upsample" and hasattr(processor, "factor"):
            return max(1.0, float(processor.factor))
        if processor.name == "resample" and hasattr(processor, "sfreq"):
            return max(1.0, float(processor.sfreq) / context.get_sfreq())
        return 1.0

    @staticmethod
    def _available_memory_bytes() -> int | None:
        """Return available physical memory without requiring psutil."""
        import os

        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, OSError, ValueError):
            return None
        return max(1, int(pages * page_size))

    def _merge_epoch_results(
        self, original_context: ProcessingContext, results: list[ProcessingContext]
    ) -> ProcessingContext:
        """Merge epoch-wise results."""
        # Implementation depends on how epochs are stored
        # This is a placeholder
        logger.warning("Epoch-wise merging not fully implemented yet")
        return original_context

    def _split_into_chunks(self, items: list, n_chunks: int) -> list[list]:
        """Split list into approximately equal chunks."""
        chunk_size = len(items) // n_chunks
        remainder = len(items) % n_chunks

        chunks = []
        start = 0
        for i in range(n_chunks):
            # Distribute remainder across first chunks
            size = chunk_size + (1 if i < remainder else 0)
            if size > 0:
                chunks.append(items[start : start + size])
                start += size

        return chunks

    # Epoch-wise methods (placeholders for now)
    def _execute_multiprocessing_epochs(self, processor, context, epoch_chunks):
        """Execute epoch-wise using multiprocessing."""
        raise NotImplementedError("Epoch-wise multiprocessing not yet implemented")

    def _execute_threading_epochs(self, processor, context, epoch_chunks):
        """Execute epoch-wise using threading."""
        raise NotImplementedError("Epoch-wise threading not yet implemented")

    def _execute_serial_epochs(self, processor, context, epoch_chunks):
        """Execute epoch-wise serially."""
        raise NotImplementedError("Epoch-wise serial execution not yet implemented")
