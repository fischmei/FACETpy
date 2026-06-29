"""
Pipeline Module

This module defines the Pipeline class for executing sequences of processors.

Author: FACETpy Team
Date: 2025-01-12
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from ..console import get_console
from ..console.progress import set_current_step_index
from .channel_sequential import ChannelSequentialExecutor
from .context import ProcessingContext
from .parallel import ParallelExecutor
from .processor import Processor


class PipelineError(Exception):
    """Base exception for pipeline-related errors."""

    pass


class PipelineResult:
    """
    Result of pipeline execution.

    Contains the final context and metadata about execution.
    """

    def __init__(
        self,
        context: ProcessingContext | None,
        success: bool = True,
        error: Exception | None = None,
        execution_time: float = 0.0,
        failed_processor: str | None = None,
        failed_processor_index: int | None = None,
    ):
        """
        Initialize pipeline result.

        Args:
            context: Final processing context
            success: Whether pipeline completed successfully
            error: Exception if pipeline failed
            execution_time: Total execution time in seconds
        """
        self.context = context
        self.success = success
        self.error = error
        self.execution_time = execution_time
        self.failed_processor = failed_processor
        self.failed_processor_index = failed_processor_index

    def get_context(self) -> ProcessingContext:
        """Get final processing context."""
        return self.context

    def get_raw(self):
        """Get final raw data (convenience method)."""
        return self.context.get_raw()

    def get_history(self):
        """Get processing history."""
        return self.context.get_history()

    def was_successful(self) -> bool:
        """Check if pipeline succeeded."""
        return self.success

    @property
    def metrics(self) -> dict[str, Any]:
        """
        Shortcut to evaluation metrics stored in context.

        Returns the ``metrics`` dict from ``context.metadata.custom``, or an
        empty dict if no metrics have been calculated yet.

        Example::

            result = pipeline.run()
            print(result.metrics['snr'])
        """
        if self.context is None:
            return {}
        return self.context.metadata.custom.get("metrics", {})

    @property
    def metrics_df(self):
        """
        Return scalar evaluation metrics as a ``pandas.Series``.

        Nested dicts (e.g. ``fft_allen``) are flattened with ``_`` separators.
        Returns ``None`` if pandas is not available.

        Example::

            result = pipeline.run()
            print(result.metrics_df)
        """
        try:
            import pandas as pd
        except ImportError:
            return None

        flat: dict[str, Any] = {}
        for k, v in self.metrics.items():
            if isinstance(v, (int, float)):
                flat[k] = v
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float)):
                        flat[f"{k}_{sub_k}"] = sub_v
        return pd.Series(flat, name=self.context.metadata.custom.get("pipeline_name", "metrics"))

    def metric(self, name: str, default=None):
        """
        Return a single evaluation metric by name.

        Shortcut for ``result.metrics.get(name, default)`` that avoids having
        to remember the dict key and provides a clean default.

        Args:
            name: Metric name (e.g. ``'snr'``, ``'rms_ratio'``).
            default: Value returned when the metric is absent.

        Example::

            snr = result.metric('snr')
            print(f"SNR = {snr:.2f} dB")
        """
        return self.metrics.get(name, default)

    def print_metrics(self) -> None:
        """
        Print a formatted table of all evaluation metrics.

        Uses *rich* for colour and alignment when available.

        Example::

            result = pipeline.run()
            result.print_metrics()
        """
        import numpy as np
        from rich import box
        from rich.console import Console as RichConsole
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        metrics = self.metrics
        if not metrics:
            print("No metrics available — did you add evaluation processors?")
            return

        con = get_console().get_rich_console() or RichConsole(highlight=False)
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

        def _fmt_per_channel(val: list) -> str:
            arr = np.asarray(val, dtype=float)
            return f"mean {arr.mean():.3g}  ± {arr.std():.3g}  [dim](min {arr.min():.3g} – max {arr.max():.3g})[/]"

        def _color_snr(v: float) -> str:
            return "green" if v > 10 else ("yellow" if v > 3 else "red")

        def _color_ratio(v: float) -> str:
            return "green" if abs(v - 1.0) < 0.1 else ("yellow" if abs(v - 1.0) < 0.3 else "red")

        # --- Core scalar metrics ---
        core_keys = ("snr", "rms_ratio", "rms_residual", "median_artifact", "legacy_snr")
        if any(k in metrics for k in core_keys):
            _section("Core Metrics")
            if "snr" in metrics:
                snr = metrics["snr"]
                c = _color_snr(snr)
                table.add_row("SNR (Signal-to-Noise Ratio)", f"[{c}]{snr:.2f}[/]", "")
            if "rms_ratio" in metrics:
                table.add_row("RMS Ratio (improvement)", f"{metrics['rms_ratio']:.2f}", "×")
            if "rms_residual" in metrics:
                r = metrics["rms_residual"]
                c = _color_ratio(r)
                table.add_row("RMS Residual Ratio", f"[{c}]{r:.2f}[/]", "target: 1.0")
            if "median_artifact" in metrics:
                table.add_row("Median Artifact Amplitude", f"{metrics['median_artifact']:.2e}", "")
                if "median_artifact_ratio" in metrics:
                    r = metrics["median_artifact_ratio"]
                    c = "green" if abs(r - 1.0) < 0.2 else ("yellow" if abs(r - 1.0) < 0.6 else "red")
                    table.add_row("Median Artifact Ratio", f"[{c}]{r:.2f}[/]", "target: 1.0")
            if "legacy_snr" in metrics:
                table.add_row("Legacy SNR", f"{metrics['legacy_snr']:.2f}", "")

        # --- Per-channel breakdowns ---
        per_ch = {k: v for k, v in metrics.items() if k.endswith("_per_channel") and isinstance(v, list)}
        if per_ch:
            _section("Per-Channel Summary  (mean ± std,  min – max)")
            for key, val in per_ch.items():
                label = key.replace("_per_channel", "").replace("_", " ").title()
                table.add_row(label, _fmt_per_channel(val), "")

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

        # --- Other unknown keys ---
        known = (
            set(core_keys)
            | set(per_ch)
            | {"median_artifact_ratio", "median_artifact_reference", "fft_allen", "fft_niazy"}
        )
        extras = {k: v for k, v in metrics.items() if k not in known}
        if extras:
            _section("Other")
            for key, val in extras.items():
                label = key.replace("_", " ").title()
                if isinstance(val, float):
                    formatted = f"{val:.4g}"
                elif isinstance(val, dict):
                    formatted = "  ".join(
                        f"{k}: {v:.3g}" if isinstance(v, float) else f"{k}: {v}" for k, v in val.items()
                    )
                elif isinstance(val, list):
                    formatted = _fmt_per_channel(val) if val and isinstance(val[0], (int, float)) else str(val)
                else:
                    formatted = str(val)
                table.add_row(label, formatted, "")

        con.print()
        con.print(
            Panel(
                table,
                title="[bold white] Evaluation Metrics Report [/]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def print_summary(self) -> None:
        """
        Print a one-line summary of the pipeline result.

        Shows success/failure, execution time, and any key metrics (SNR, RMS
        ratio, RMS residual) that were calculated.

        Example::

            result = pipeline.run()
            result.print_summary()
        """
        from rich.console import Console as RichConsole

        con = get_console().get_rich_console() or RichConsole()
        if self.success:
            parts = [f"[green]Done[/green] in {self.execution_time:.2f}s"]
            for name in ("snr", "rms_ratio", "rms_residual", "median_artifact"):
                val = self.metrics.get(name)
                if val is not None:
                    parts.append(f"{name}={val:.3g}")
            con.print("  ".join(parts))
        else:
            con.print(f"[red]Failed[/red] after {self.execution_time:.2f}s — {self.error}")

    def plot(self, **kwargs):
        """
        Plot the corrected data using ``RawPlotter`` defaults.

        Accepts any keyword arguments supported by ``RawPlotter``.

        Example::

            result = pipeline.run()
            result.plot(channel="Fp1", start=5.0, duration=10.0)
        """
        from ..evaluation import RawPlotter

        plotter = RawPlotter(**kwargs)
        plotter.execute(self.context)

    def release_raw(self) -> None:
        """
        Release the Raw data held by the context to free memory.

        After calling this, ``get_raw()`` and ``plot()`` will no longer work,
        but :attr:`metrics` and ``execution_time`` remain accessible.
        Useful when running batch jobs where you only need summary statistics.
        """
        if self.context is not None:
            self.context._raw = None
            self.context._raw_original = None

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"PipelineResult({status}, time={self.execution_time:.2f}s)"


class BatchResult:
    """
    Result of ``Pipeline.map(...)`` - a list of
    :class:`~facet.core.pipeline.PipelineResult` objects
    with built-in helpers for quick inspection.

    It behaves like a regular list (iteration, indexing, ``len``), so existing
    code that iterates over the return value of ``Pipeline.map()`` continues to
    work without changes.

    Example::

        results = pipeline.map(files, loader_factory=lambda p: Loader(p))
        results.print_summary()
        df = results.summary_df
    """

    def __init__(
        self,
        results: list["PipelineResult"],
        labels: list[str] | None = None,
    ):
        self._results = results
        self._labels = labels or [f"input_{i}" for i in range(len(results))]

    # ------------------------------------------------------------------
    # List-like interface
    # ------------------------------------------------------------------

    def __iter__(self):
        return iter(self._results)

    def __getitem__(self, index):
        return self._results[index]

    def __len__(self):
        return len(self._results)

    def __repr__(self):
        n_ok = sum(1 for r in self._results if r.success)
        return f"BatchResult({n_ok}/{len(self._results)} succeeded)"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """
        Print a formatted table with one row per input file.

        Columns include the file label, success/failure status, execution time,
        and any scalar metrics that were computed.

        Example::

            results = pipeline.map(files, loader_factory=...)
            results.print_summary()
        """
        from rich import box
        from rich.console import Console as RichConsole
        from rich.table import Table

        con = get_console().get_rich_console() or RichConsole(highlight=False)
        table = Table(
            title="Batch Results",
            show_header=True,
            header_style="bold cyan",
            box=box.SIMPLE_HEAVY,
            padding=(0, 1),
        )
        table.add_column("File", style="bold", no_wrap=True)
        table.add_column("Status", justify="left")
        table.add_column("Time", justify="left")

        metric_names: list[str] = []
        for r in self._results:
            for k, v in r.metrics.items():
                if k not in metric_names and isinstance(v, (int, float)):
                    metric_names.append(k)

        for m in metric_names:
            table.add_column(m, justify="left")

        for label, result in zip(self._labels, self._results, strict=False):
            status = "[green]OK[/green]" if result.success else "[red]FAIL[/red]"
            time_str = f"{result.execution_time:.2f}s"
            row: list[str] = [label, status, time_str]
            for m in metric_names:
                if result.success:
                    val = result.metrics.get(m)
                    row.append(f"{val:.3f}" if isinstance(val, float) else (str(val) if val is not None else "—"))
                else:
                    row.append("—")
            table.add_row(*row)

        con.print(table)

    @property
    def summary_df(self):
        """
        Return a ``pandas.DataFrame`` with one row per input.

        Columns: ``file``, ``success``, ``execution_time``, plus one column per
        scalar metric.  Returns ``None`` when *pandas* is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            return None

        rows = []
        for label, result in zip(self._labels, self._results, strict=False):
            row: dict[str, Any] = {
                "file": label,
                "success": result.success,
                "execution_time": result.execution_time,
            }
            if result.success and result.metrics_df is not None:
                row.update(result.metrics_df.to_dict())
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class ChunkSpec:
    """Description of one fixed-length raw-data chunk."""

    index: int
    total: int
    start_sample: int
    stop_sample: int
    sfreq: float
    output_path: Path

    @property
    def duration_seconds(self) -> float:
        """Return chunk duration in seconds."""
        return (self.stop_sample - self.start_sample) / self.sfreq


class ChunkedPipelineResult:
    """Result of :meth:`Pipeline.run_chunked`.

    Behaves like a list of :class:`PipelineResult` objects and keeps the chunk
    windows and numbered output paths together for reporting.
    """

    def __init__(
        self,
        results: list[PipelineResult],
        chunks: list[ChunkSpec],
        source_path: str,
        output_dir: str,
        execution_time: float,
        manifest_path: Path | None = None,
    ) -> None:
        self._results = results
        self.chunks = chunks
        self.source_path = source_path
        self.output_dir = output_dir
        self.execution_time = execution_time
        self.manifest_path = manifest_path

    def __iter__(self):
        return iter(self._results)

    def __getitem__(self, index):
        return self._results[index]

    def __len__(self) -> int:
        return len(self._results)

    @property
    def output_paths(self) -> list[Path]:
        """Return numbered chunk output paths."""
        return [chunk.output_path for chunk in self.chunks]

    @property
    def success(self) -> bool:
        """Return ``True`` only when every chunk succeeded."""
        return bool(self._results) and all(result.success for result in self._results)

    def was_successful(self) -> bool:
        """Check if all chunk pipelines succeeded."""
        return self.success

    def print_summary(self) -> None:
        """Print a compact chunk-processing summary."""
        from rich import box
        from rich.console import Console as RichConsole
        from rich.table import Table

        con = get_console().get_rich_console() or RichConsole(highlight=False)
        table = Table(
            title="Chunked Pipeline Results",
            show_header=True,
            header_style="bold cyan",
            box=box.SIMPLE_HEAVY,
            padding=(0, 1),
        )
        table.add_column("Chunk", no_wrap=True)
        table.add_column("Samples")
        table.add_column("Duration")
        table.add_column("Status")
        table.add_column("Output")

        for chunk, result in zip(self.chunks, self._results, strict=False):
            status = "[green]OK[/green]" if result.success else "[red]FAIL[/red]"
            sample_range = f"{chunk.start_sample}:{chunk.stop_sample}"
            table.add_row(
                f"{chunk.index}/{chunk.total}",
                sample_range,
                f"{chunk.duration_seconds:.1f}s",
                status,
                str(chunk.output_path),
            )

        con.print(table)
        con.print(f"Done in {self.execution_time:.2f}s")
        if self.manifest_path is not None:
            con.print(f"Manifest: {self.manifest_path}")


class Pipeline:
    """
    Pipeline for executing sequences of processors.

    A pipeline orchestrates the execution of multiple processors in sequence,
    handles errors, provides progress tracking, and supports parallelization.

    Example::

        pipeline = Pipeline([
            Loader("data.edf"),
            HighPassFilter(freq=1.0),
            UpSample(factor=10),
            TriggerDetector(regex=r"\\btrigger\\b"),
            AASCorrection(),
            EDFExporter("output.edf")
        ])

        result = pipeline.run()
        if result.was_successful():
            print(f"Completed in {result.execution_time:.2f}s")

    Attributes:
        processors: List of processors to execute
        name: Optional pipeline name
    """

    def __init__(self, processors: list[Processor | Callable], name: str | None = None):
        """
        Initialize pipeline.

        Plain callables (``Callable[[ProcessingContext], ProcessingContext]``)
        are automatically wrapped in a :class:`~facet.core.LambdaProcessor` so
        they can be used as inline steps without ceremony::

            pipeline = Pipeline([
                Loader("data.edf"),
                HighPassFilter(1.0),
                lambda ctx: (print(ctx.get_sfreq()) or ctx),
                AASCorrection(),
            ])

        Args:
            processors: List of :class:`~facet.core.Processor` instances **or**
                plain callables to execute in order.
            name: Optional pipeline name (for logging)
        """
        self.processors = self._normalise_processors(processors)
        self.name = name or "Pipeline"

    @staticmethod
    def _normalise_processors(
        items: list[Processor | Callable],
        _index_offset: int = 0,
    ) -> list[Processor]:
        """
        Coerce each item to a :class:`Processor`.

        Plain callables are wrapped in a :class:`~facet.core.LambdaProcessor`.
        Anything else that is not a :class:`Processor` raises :exc:`TypeError`.
        """
        from .processor import LambdaProcessor

        result: list[Processor] = []
        for i, p in enumerate(items):
            if isinstance(p, Processor):
                result.append(p)
            elif callable(p):
                display_name = getattr(p, "__name__", None) or f"step_{_index_offset + i}"
                result.append(LambdaProcessor(name=display_name, func=p))
            else:
                raise TypeError(
                    f"Item at index {_index_offset + i} must be a Processor instance or a callable, got {type(p)}"
                )
        return result

    def _validate_pipeline(self) -> None:
        """No-op — validation now happens in _normalise_processors."""

    # ---------------------------------------------------------------------- #
    # Execution helpers                                                       #
    # ---------------------------------------------------------------------- #

    def _group_processors(
        self,
        parallel: bool,
        channel_sequential: bool,
    ) -> list[tuple[list[Processor], str]]:
        """
        Partition processors into execution groups.

        Returns a list of ``(processors, mode)`` tuples where *mode* is one of
        ``'channel_sequential'``, ``'parallel'``, or ``'serial'``.

        In channel_sequential mode consecutive processors with
        ``channel_wise = True`` (or ``run_once = True``) are merged into a
        single ``'channel_sequential'`` group.  This grouping is entirely
        independent of ``parallel_safe``.
        """
        groups: list[tuple[list[Processor], str]] = []
        i = 0
        while i < len(self.processors):
            proc = self.processors[i]
            ch_eligible = getattr(proc, "channel_wise", False) or getattr(proc, "run_once", False)
            if channel_sequential and ch_eligible:
                batch: list[Processor] = []
                while i < len(self.processors):
                    p = self.processors[i]
                    if getattr(p, "channel_wise", False) or getattr(p, "run_once", False):
                        batch.append(p)
                        i += 1
                    else:
                        break
                groups.append((batch, "channel_sequential"))
            elif parallel and proc.parallel_safe:
                groups.append(([proc], "parallel"))
                i += 1
            else:
                groups.append(([proc], "serial"))
                i += 1
        return groups

    def _dispatch_step(
        self,
        processors: list[Processor],
        mode: str,
        context: ProcessingContext,
        n_jobs: int,
    ) -> ProcessingContext:
        """Execute one group of processors according to *mode*."""
        if mode == "channel_sequential":
            return ChannelSequentialExecutor().execute(processors, context)
        if mode == "parallel":
            return ParallelExecutor(n_jobs=n_jobs).execute(processors[0], context)
        return processors[0].execute(context)

    # ---------------------------------------------------------------------- #
    # Public API                                                              #
    # ---------------------------------------------------------------------- #

    def run(
        self,
        initial_context: ProcessingContext | None = None,
        parallel: bool = False,
        n_jobs: int = -1,
        channel_sequential: bool = False,
        show_progress: bool = True,
    ) -> PipelineResult:
        """
        Execute the pipeline.

        Args:
            initial_context: Initial context (if None, first processor creates it)
            parallel: Enable parallel execution for compatible processors
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            channel_sequential: Run consecutive channel-wise processors
                (``channel_wise = True``) as a single per-channel pass.
                For each channel the full local sequence executes before
                the next channel starts::

                    for each channel:
                        channel → HP-filter → UpSample → AAS → DownSample → store

                The output array is pre-allocated at the final sfreq so the
                full high-sfreq intermediate data never exists for all
                channels at once.

                Processors with ``run_once = True`` (e.g. TriggerAligner)
                are included in the per-channel pass but only execute for
                the first channel; all subsequent channels skip them.

                This flag is independent of ``parallel_safe`` and has no
                relation to multiprocessing.  Takes precedence over
                *parallel* for eligible processors.
            show_progress: Show progress bar

        Returns:
            PipelineResult containing final context and metadata
        """
        import time

        start_time = time.time()
        console = get_console()
        n_procs = len(self.processors)

        execution_mode = "channel_sequential" if channel_sequential else "parallel" if parallel else "serial"
        console.set_pipeline_metadata(
            {
                "execution_mode": execution_mode,
                "n_jobs": "1" if channel_sequential else str(n_jobs),
            }
        )
        console.start_pipeline(
            self.name,
            n_procs,
            step_names=[p.name for p in self.processors],
        )
        logger.info(f"Starting pipeline: {self.name} ({n_procs} processors)")

        context = initial_context
        current_processor: tuple[int, Processor] | None = None

        try:
            step_offset = 0
            for processors, mode in self._group_processors(parallel, channel_sequential):
                current_processor = (step_offset, processors[0])

                label = " → ".join(p.name for p in processors)
                logger.info(f"[{step_offset + 1}/{n_procs}] {label}")
                for k, p in enumerate(processors):
                    console.step_started(step_offset + k, p.name)

                set_current_step_index(step_offset)
                step_start = time.time()
                try:
                    context = self._dispatch_step(processors, mode, context, n_jobs)
                finally:
                    set_current_step_index(None)

                duration = time.time() - step_start
                for k, p in enumerate(processors):
                    console.step_completed(
                        step_offset + k,
                        p.name,
                        duration / len(processors),
                        metrics={
                            "execution_mode": mode,
                            "last_duration": f"{duration:.2f}s",
                        },
                    )
                step_offset += len(processors)

            execution_time = time.time() - start_time
            logger.info(f"Pipeline completed in {execution_time:.2f}s")
            console.pipeline_complete(True, execution_time)
            return PipelineResult(context=context, success=True, execution_time=execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            if current_processor:
                failed_idx, failed_proc = current_processor
                logger.error(
                    f"Pipeline failed after {execution_time:.2f}s during "
                    f"{failed_proc.name} (step {failed_idx + 1}/{n_procs}): {e}"
                )
            else:
                logger.error(f"Pipeline failed after {execution_time:.2f}s: {e}")
            logger.opt(exception=e).debug("Exception details")

            console.pipeline_failed(
                execution_time,
                e,
                current_processor[0] if current_processor else None,
                current_processor[1].name if current_processor else None,
            )
            return PipelineResult(
                context=context,
                success=False,
                error=e,
                execution_time=execution_time,
                failed_processor=current_processor[1].name if current_processor else None,
                failed_processor_index=current_processor[0] if current_processor else None,
            )

    def add(self, processor: Processor | Callable) -> "Pipeline":
        """
        Add a processor or callable to the pipeline (fluent API).

        Args:
            processor: :class:`~facet.core.Processor` instance or callable.

        Returns:
            Self for chaining
        """
        [normalised] = self._normalise_processors([processor], _index_offset=len(self.processors))
        self.processors.append(normalised)
        return self

    def extend(self, processors: list[Processor | Callable]) -> "Pipeline":
        """
        Extend pipeline with multiple processors or callables.

        Args:
            processors: List of processors or callables to add.

        Returns:
            Self for chaining
        """
        self.processors.extend(self._normalise_processors(processors, _index_offset=len(self.processors)))
        return self

    def insert(self, index: int, processor: Processor | Callable) -> "Pipeline":
        """
        Insert a processor or callable at a specific position.

        Args:
            index: Position to insert
            processor: :class:`~facet.core.Processor` instance or callable.

        Returns:
            Self for chaining
        """
        [normalised] = self._normalise_processors([processor], _index_offset=index)
        self.processors.insert(index, normalised)
        return self

    def remove(self, index: int) -> "Pipeline":
        """
        Remove processor at index.

        Args:
            index: Index to remove

        Returns:
            Self for chaining
        """
        self.processors.pop(index)
        return self

    def validate_all(self, context: ProcessingContext) -> list[str]:
        """
        Validate all processors against a context.

        Useful for checking if a pipeline can run before actually running it.

        Args:
            context: Context to validate against

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        for i, processor in enumerate(self.processors):
            try:
                processor.validate(context)
            except Exception as e:
                errors.append(f"Processor {i} ({processor.name}): {str(e)}")
        return errors

    def describe(self) -> str:
        """
        Get human-readable pipeline description.

        Returns:
            Multi-line string describing pipeline
        """
        lines = [f"Pipeline: {self.name}", "=" * 50]

        for i, processor in enumerate(self.processors):
            lines.append(f"{i + 1}. {processor.name} ({processor.__class__.__name__})")
            if hasattr(processor, "_parameters"):
                for key, value in processor._parameters.items():
                    lines.append(f"   - {key}: {value}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize pipeline to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "processors": [
                {
                    "class": proc.__class__.__name__,
                    "name": proc.name,
                    "parameters": proc._parameters if hasattr(proc, "_parameters") else {},
                }
                for proc in self.processors
            ],
        }

    def map(
        self,
        inputs: list[str | ProcessingContext],
        loader_factory: Callable[[str], "Processor"] | None = None,
        parallel: bool = False,
        n_jobs: int = -1,
        on_error: str = "continue",
        keep_raw: bool = True,
    ) -> "BatchResult":
        """
        Run the pipeline on multiple inputs and return a result per input.

        Each input can be:

        - A ``ProcessingContext`` — passed directly as ``initial_context``.
        - A **file path string** — a fresh :class:`~facet.io.Loader` is created
          automatically for each path via *loader_factory*.

        .. note::
            Do **not** add a :class:`~facet.io.Loader` processor to the pipeline
            when using ``map()``.  Loading is handled outside the pipeline so
            that each file gets its own isolated loader instance.

        Args:
            inputs: List of file paths or ``ProcessingContext`` objects.
            loader_factory: ``Callable[[path], Processor]`` that creates a fresh
                loader for each path string.  Defaults to
                ``lambda p: Loader(path=p, preload=True)``.
            parallel: Whether to pass ``parallel=True`` to each ``pipeline.run()``.
            n_jobs: Passed through to ``pipeline.run()``.
            on_error: ``"continue"`` (default) — log failures and keep going;
                      ``"raise"`` — re-raise the first error encountered.
            keep_raw: If ``False``, the Raw data is released from each result
                after the pipeline run completes, keeping only metrics and
                history in memory.  Set to ``False`` when processing many files
                and you only need summary statistics.  Defaults to ``True``.

        Returns:
            :class:`~facet.core.pipeline.BatchResult` containing one
            :class:`~facet.core.pipeline.PipelineResult` per
            input, in the same order.  It behaves like a plain list but also
            offers ``BatchResult.print_summary()`` and ``BatchResult.summary_df``.

        Example::

            pipeline = Pipeline([
                TriggerDetector(regex=r"\\b1\\b"),
                UpSample(factor=10),
                AASCorrection(window_size=30),
                DownSample(factor=10),
                SNRCalculator(),
            ])

            results = pipeline.map(
                ["sub-01.edf", "sub-02.edf", "sub-03.edf"],
                keep_raw=False,
            )
            results.print_summary()
        """
        from ..io.loaders import Loader as _Loader

        if loader_factory is None:
            loader_factory = lambda p: _Loader(path=p, preload=True)  # noqa: E731

        for proc in self.processors:
            if isinstance(proc, _Loader):
                raise ValueError(
                    "A Loader processor was found inside the pipeline passed to map(). "
                    "map() handles loading automatically — remove the Loader from the "
                    "pipeline and pass file paths directly to map()."
                )

        results: list[PipelineResult] = []
        labels: list[str] = []

        for item in inputs:
            if isinstance(item, ProcessingContext):
                run_pipeline = self
                initial_ctx = item
                label = repr(item)
            else:
                label = str(item)
                initial_ctx = None

                loader = loader_factory(item)
                try:
                    initial_ctx = loader.execute(None)
                except Exception as exc:
                    logger.error(f"Loader failed for '{label}': {exc}")
                    result = PipelineResult(
                        context=None,
                        success=False,
                        error=exc,
                        failed_processor=getattr(loader, "name", "loader"),
                    )
                    results.append(result)
                    labels.append(label)
                    if on_error == "raise":
                        raise
                    continue
                run_pipeline = self

            logger.info(f"Pipeline.map: processing '{label}'")

            result = run_pipeline.run(
                initial_context=initial_ctx,
                parallel=parallel,
                n_jobs=n_jobs,
            )

            if not keep_raw and result.success:
                result.release_raw()

            results.append(result)
            labels.append(label)

            if not result.success and on_error == "raise":
                raise result.error

        return BatchResult(results, labels=labels)

    def run_chunked(
        self,
        input_path: str,
        output_dir: str,
        output_extension: str = ".edf",
        output_stem: str | None = None,
        loader_kwargs: dict[str, Any] | None = None,
        exporter_factory: Callable[[str], Processor] | None = None,
        min_chunks: int = 2,
        max_chunks: int = 128,
        memory_budget_mb: float | None = None,
        memory_fraction: float = 0.5,
        processing_memory_multiplier: float = 4.0,
        overwrite: bool = True,
        parallel: bool = False,
        n_jobs: int = -1,
        channel_sequential: bool = False,
        show_progress: bool = True,
        on_error: str = "raise",
        keep_raw: bool = False,
        chunk_by_trigger_sections: bool | None = None,
        trigger_section_padding_seconds: float = 10.0,
        trigger_section_min_triggers: int = 30,
        trigger_section_gap_seconds: float | None = None,
        # trigger_section_max_sections: int = 2,
        trigger_section_max_sections: int | None = None,
    ) -> ChunkedPipelineResult:
        """Run a processing pipeline over bounded chunks of one input file.

        The input is opened lazily to inspect its sampling rate and sample
        count. If the pipeline contains a ``TriggerDetector``, chunks are built
        around trigger sections by default so that every AAS chunk contains
        triggers. Otherwise the method falls back to fixed-length chunks,
        starting with ``min_chunks`` and increasing the division count until the
        estimated peak memory per chunk fits inside the requested or available
        memory budget.

        The pipeline passed to ``run_chunked`` should contain processing steps
        only. Loading and exporting are handled here so that every chunk gets a
        fresh bounded :class:`~facet.io.Loader` and a numbered output file.

        Parameters
        ----------
        input_path : str
            Source EEG file or MFF directory accepted by :class:`facet.io.Loader`.
        output_dir : str
            Folder that will receive numbered corrected chunk files.
        output_extension : str, optional
            Export extension used by :class:`facet.io.Exporter` (default:
            ``".edf"``).
        output_stem : str, optional
            Base name for chunk files. Defaults to the input file stem.
        loader_kwargs : dict, optional
            Extra keyword arguments forwarded to each chunk loader.
        exporter_factory : callable, optional
            Factory receiving an output path and returning an exporter
            processor. Defaults to ``Exporter(path=..., overwrite=...)``.
        min_chunks, max_chunks : int
            Search range for the automatic chunk count.
        memory_budget_mb : float, optional
            Explicit per-chunk memory budget. If omitted, a fraction of
            currently available system memory is used when detectable.
        memory_fraction : float
            Fraction of available memory to use when ``memory_budget_mb`` is
            not supplied.
        processing_memory_multiplier : float
            Safety multiplier for intermediate arrays and copies.
        overwrite : bool
            Passed to the default exporter.
        on_error : {"raise", "continue"}
            Whether to stop on the first failed chunk or return failed results.
        keep_raw : bool
            If ``False``, release each successful chunk's Raw object after
            export so only metadata/results remain in memory.
        chunk_by_trigger_sections : bool, optional
            If ``True``, build one output chunk per detected trigger section.
            If ``None`` (default), this is enabled automatically when the
            pipeline contains a ``TriggerDetector``.
        trigger_section_padding_seconds : float
            Seconds to include before the first trigger and after the last
            trigger in each detected section.
        trigger_section_min_triggers : int
            Minimum trigger count required for a trigger cluster to be treated
            as a section.
        trigger_section_gap_seconds : float, optional
            Explicit gap threshold for splitting trigger sections. If omitted,
            a robust threshold is estimated from the trigger spacing.
        trigger_section_max_sections : int
            Maximum number of trigger sections to export.
        """
        import gc
        import math
        import os
        import time

        import mne

        from ..io.exporters import _EXTENSION_EXPORTERS
        from ..io.exporters import Exporter as _Exporter
        from ..io.loaders import Loader as _Loader

        if min_chunks < 1:
            raise ValueError("min_chunks must be >= 1")
        if max_chunks < min_chunks:
            raise ValueError("max_chunks must be >= min_chunks")
        if on_error not in {"raise", "continue"}:
            raise ValueError("on_error must be 'raise' or 'continue'")
        if not (0 < memory_fraction <= 1):
            raise ValueError("memory_fraction must be in (0, 1]")
        if processing_memory_multiplier <= 0:
            raise ValueError("processing_memory_multiplier must be positive")
        if trigger_section_padding_seconds < 0:
            raise ValueError("trigger_section_padding_seconds must be >= 0")
        if trigger_section_min_triggers < 1:
            raise ValueError("trigger_section_min_triggers must be >= 1")
        if trigger_section_gap_seconds is not None and trigger_section_gap_seconds <= 0:
            raise ValueError("trigger_section_gap_seconds must be positive when provided")
        # if trigger_section_max_sections < 1:
        #     raise ValueError("trigger_section_max_sections must be >= 1")
        if ( trigger_section_max_sections is not None and trigger_section_max_sections < 1):
            raise ValueError("trigger_section_max_sections must be >= 1")

        exporter_types = tuple(_EXTENSION_EXPORTERS.values()) + (_Exporter,)
        for proc in self.processors:
            if isinstance(proc, _Loader):
                raise ValueError("run_chunked() handles loading; remove Loader from this pipeline")
            if isinstance(proc, exporter_types):
                raise ValueError("run_chunked() handles exporting; remove exporters from this pipeline")

        loader_kwargs = dict(loader_kwargs or {})
        for forbidden in ("start_sample", "stop_sample", "preload"):
            if forbidden in loader_kwargs:
                raise ValueError(f"loader_kwargs must not include {forbidden!r}; run_chunked controls it")

        start_time = time.time()
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        extension = self._normalise_chunk_extension(output_extension)
        stem = output_stem or self._source_stem(input_path)

        # Probe the file without loading signal samples, mirroring MNE's
        # fixed-length epoch workflow where epochs can be created with
        # preload=False.
        probe_context = _Loader(path=input_path, preload=False, **loader_kwargs).execute(None)
        probe_raw = probe_context.get_raw()
        n_times = int(probe_raw.n_times)
        sfreq = float(probe_raw.info["sfreq"])
        n_channels = len(probe_raw.ch_names)
        sample_bytes = self._raw_sample_bytes(probe_raw)
        peak_sample_multiplier = self._peak_sample_multiplier()

        memory_budget = self._memory_budget_bytes(memory_budget_mb, memory_fraction, os)
        trigger_detector = self._find_trigger_detector_for_chunking()
        if chunk_by_trigger_sections is None:
            use_trigger_sections = trigger_detector is not None
        else:
            use_trigger_sections = chunk_by_trigger_sections

        trigger_section_chunks: list[ChunkSpec] | None = None
        chunking_mode = "fixed_length"
        if use_trigger_sections:
            if trigger_detector is None:
                raise ValueError(
                    "chunk_by_trigger_sections=True requires a TriggerDetector in the pipeline"
                )

            trigger_context = self._detect_triggers_for_chunking(probe_context)
            triggers = trigger_context.get_triggers()
            if triggers is None or len(triggers) == 0:
                raise PipelineError("Cannot build trigger-section chunks: no triggers were detected")

            trigger_section_chunks = self._make_trigger_section_chunks(
                raw=probe_raw,
                triggers=triggers,
                output_dir=output_root,
                output_stem=stem,
                extension=extension,
                padding_seconds=trigger_section_padding_seconds,
                min_triggers=trigger_section_min_triggers,
                gap_seconds=trigger_section_gap_seconds,
                max_sections=trigger_section_max_sections,
            )
            chunking_mode = "trigger_sections"
            attempt_chunks = len(trigger_section_chunks)
            retry_limit = attempt_chunks

            logger.info(
                "Chunking {} into {} trigger-section window(s): {} samples, {} channels, {} Hz",
                input_path,
                len(trigger_section_chunks),
                n_times,
                n_channels,
                sfreq,
            )
        else:
            n_chunks = self._choose_chunk_count(
                n_times=n_times,
                n_channels=n_channels,
                sample_bytes=sample_bytes,
                peak_sample_multiplier=peak_sample_multiplier,
                processing_memory_multiplier=processing_memory_multiplier,
                memory_budget=memory_budget,
                min_chunks=min_chunks,
                max_chunks=max_chunks,
                math_module=math,
            )

            logger.info(
                "Chunking {} into {} fixed-length part(s): {} samples, {} channels, {} Hz",
                input_path,
                n_chunks,
                n_times,
                n_channels,
                sfreq,
            )

            attempt_chunks = n_chunks
            retry_limit = max_chunks

        while attempt_chunks <= retry_limit:
            if trigger_section_chunks is not None:
                chunks = trigger_section_chunks
            else:
                chunks = self._make_fixed_length_chunks(
                    raw=probe_raw,
                    n_chunks=attempt_chunks,
                    output_dir=output_root,
                    output_stem=stem,
                    extension=extension,
                    mne_module=mne,
                    math_module=math,
                )
            results: list[PipelineResult] = []

            try:
                for chunk in chunks:
                    logger.info(
                        "Processing chunk {}/{}: samples {}:{} -> {}",
                        chunk.index,
                        chunk.total,
                        chunk.start_sample,
                        chunk.stop_sample,
                        chunk.output_path,
                    )

                    chunk_loader = _Loader(
                        path=input_path,
                        preload=True,
                        start_sample=chunk.start_sample,
                        stop_sample=chunk.stop_sample,
                        **loader_kwargs,
                    )
                    chunk_context = chunk_loader.execute(None)
                    chunk_context.metadata.custom["chunk"] = {
                        "index": chunk.index,
                        "total": chunk.total,
                        "start_sample": chunk.start_sample,
                        "stop_sample": chunk.stop_sample,
                        "chunking_mode": chunking_mode,
                        "source_path": input_path,
                    }

                    exporter = (
                        exporter_factory(str(chunk.output_path))
                        if exporter_factory is not None
                        else _Exporter(path=str(chunk.output_path), overwrite=overwrite)
                    )
                    chunk_pipeline = Pipeline(
                        [*self.processors, exporter],
                        name=f"{self.name} chunk {chunk.index}/{chunk.total}",
                    )
                    result = chunk_pipeline.run(
                        initial_context=chunk_context,
                        parallel=parallel,
                        n_jobs=n_jobs,
                        channel_sequential=channel_sequential,
                        show_progress=show_progress,
                    )

                    if not result.success:
                        if self._is_memory_failure(result.error):
                            raise MemoryError(str(result.error)) from result.error
                        if on_error == "raise":
                            raise result.error

                    if result.success and not keep_raw:
                        result.release_raw()

                    results.append(result)
                    gc.collect()

                execution_time = time.time() - start_time
                manifest_path = self._write_chunk_manifest(
                    output_root,
                    source_path=input_path,
                    chunks=chunks,
                    results=results,
                    execution_time=execution_time,
                    memory_budget=memory_budget,
                    chunking_mode=chunking_mode,
                )
                return ChunkedPipelineResult(
                    results=results,
                    chunks=chunks,
                    source_path=input_path,
                    output_dir=str(output_root),
                    execution_time=execution_time,
                    manifest_path=manifest_path,
                )

            except MemoryError:
                if trigger_section_chunks is not None or attempt_chunks >= retry_limit:
                    raise
                attempt_chunks += 1
                logger.warning("Chunk run exceeded memory; retrying with {} chunks", attempt_chunks)
                gc.collect()

        raise PipelineError("Chunked processing failed before producing results")

    def _find_trigger_detector_for_chunking(self) -> Processor | None:
        """Return the first TriggerDetector-like processor in this pipeline."""
        for proc in self.processors:
            if proc.name == "trigger_detector" and hasattr(proc, "regex"):
                return proc
        return None

    def _detect_triggers_for_chunking(self, probe_context: ProcessingContext) -> ProcessingContext:
        """Run the safe pre-trigger steps and TriggerDetector on the lazy probe.

        Only processors that declare ``chunk_probe_safe`` are executed before
        trigger detection. This allows metadata-only channel removal, such as
        dropping EGI ``E`` channels, without accidentally filtering or
        resampling the full recording during the probe pass.
        """
        context = probe_context

        for proc in self.processors:
            if proc.name == "trigger_detector" and hasattr(proc, "regex"):
                logger.info("Detecting full-file triggers for trigger-section chunking")
                return proc.execute(context)

            if getattr(proc, "chunk_probe_safe", False):
                logger.info("Applying {} before trigger-section detection", proc.name)
                context = proc.execute(context)

        raise PipelineError("Cannot build trigger-section chunks: no TriggerDetector was found")

    @staticmethod
    def _normalise_chunk_extension(extension: str) -> str:
        """Return a normalized export extension with a leading dot."""
        if not extension:
            raise ValueError("output_extension must not be empty")
        return extension if extension.startswith(".") else f".{extension}"

    @staticmethod
    def _source_stem(input_path: str) -> str:
        """Return a clean file stem, preserving directory-style inputs."""
        path = Path(input_path)
        name = path.name
        for suffix in reversed(path.suffixes):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name or "chunked_output"

    @staticmethod
    def _raw_sample_bytes(raw) -> int:
        """Best-effort byte size for one sample in one channel."""
        import numpy as np

        dtype = getattr(raw, "_dtype", None)
        if dtype is None:
            return np.dtype(np.float64).itemsize
        try:
            return np.dtype(dtype).itemsize
        except TypeError:
            return np.dtype(np.float64).itemsize

    def _peak_sample_multiplier(self) -> float:
        """Estimate peak sample-count expansion from resampling steps."""
        current = 1.0
        peak = 1.0
        for proc in self.processors:
            if proc.name == "upsample" and hasattr(proc, "factor"):
                current *= float(proc.factor)
            elif proc.name == "downsample" and hasattr(proc, "factor"):
                current /= float(proc.factor)
            peak = max(peak, current)
        return max(1.0, peak)

    @staticmethod
    def _memory_budget_bytes(memory_budget_mb: float | None, memory_fraction: float, os_module) -> int:
        """Resolve the per-chunk memory budget in bytes."""
        if memory_budget_mb is not None:
            return int(memory_budget_mb * 1024 * 1024)

        try:
            pages = os_module.sysconf("SC_AVPHYS_PAGES")
            page_size = os_module.sysconf("SC_PAGE_SIZE")
            return max(1, int(pages * page_size * memory_fraction))
        except (AttributeError, OSError, ValueError):
            fallback_gb = 2
            logger.warning(
                "Could not determine available memory; using conservative {} GiB chunk budget",
                fallback_gb,
            )
            return int(fallback_gb * 1024**3 * memory_fraction)

    @staticmethod
    def _choose_chunk_count(
        n_times: int,
        n_channels: int,
        sample_bytes: int,
        peak_sample_multiplier: float,
        processing_memory_multiplier: float,
        memory_budget: int,
        min_chunks: int,
        max_chunks: int,
        math_module,
    ) -> int:
        """Choose the first chunk count whose estimated peak fits the budget."""
        for n_chunks in range(min_chunks, max_chunks + 1):
            chunk_samples = int(math_module.ceil(n_times / n_chunks))
            estimated = (
                n_channels
                * chunk_samples
                * sample_bytes
                * peak_sample_multiplier
                * processing_memory_multiplier
            )
            if estimated <= memory_budget:
                logger.info(
                    "Selected {} chunk(s): estimated peak {:.2f} GiB <= budget {:.2f} GiB",
                    n_chunks,
                    estimated / 1024**3,
                    memory_budget / 1024**3,
                )
                return n_chunks

        logger.warning(
            "Estimated chunk memory still exceeds budget at max_chunks={}; using max_chunks",
            max_chunks,
        )
        return max_chunks

    @staticmethod
    def _make_fixed_length_chunks(
        raw,
        n_chunks: int,
        output_dir: Path,
        output_stem: str,
        extension: str,
        mne_module,
        math_module,
    ) -> list[ChunkSpec]:
        """Create chunk windows using MNE fixed-length epochs."""
        n_times = int(raw.n_times)
        sfreq = float(raw.info["sfreq"])
        chunk_samples = max(1, int(math_module.ceil(n_times / n_chunks)))
        duration = chunk_samples / sfreq
        epochs = mne_module.make_fixed_length_epochs(
            raw,
            duration=duration,
            preload=False,
            reject_by_annotation=False,
            overlap=0.0,
            verbose=False,
        )
        starts = [int(event[0] - raw.first_samp) for event in epochs.events]
        starts = sorted({start for start in starts if 0 <= start < n_times})

        if not starts:
            starts = [0]
        while starts[-1] + chunk_samples < n_times:
            starts.append(starts[-1] + chunk_samples)

        total = len(starts)
        width = max(3, len(str(total)))
        chunks: list[ChunkSpec] = []
        for index, start in enumerate(starts, start=1):
            stop = min(start + chunk_samples, n_times)
            output_path = output_dir / f"{output_stem}_chunk_{index:0{width}d}_of_{total:0{width}d}{extension}"
            chunks.append(
                ChunkSpec(
                    index=index,
                    total=total,
                    start_sample=start,
                    stop_sample=stop,
                    sfreq=sfreq,
                    output_path=output_path,
                )
            )
        return chunks

    @staticmethod
    def _make_trigger_section_chunks(
        raw,
        triggers,
        output_dir: Path,
        output_stem: str,
        extension: str,
        padding_seconds: float,
        min_triggers: int,
        gap_seconds: float | None,
        max_sections: int,
    ) -> list[ChunkSpec]:
        """Create padded chunk windows around dense trigger sections."""
        import numpy as np

        n_times = int(raw.n_times)
        sfreq = float(raw.info["sfreq"])
        trigger_samples = np.asarray(triggers, dtype=np.int64).ravel()
        trigger_samples = trigger_samples[(trigger_samples >= 0) & (trigger_samples < n_times)]
        trigger_samples = np.unique(trigger_samples)

        if trigger_samples.size == 0:
            raise PipelineError("Cannot build trigger-section chunks from an empty trigger array")

        sections = Pipeline._split_trigger_sections(
            trigger_samples=trigger_samples,
            sfreq=sfreq,
            min_triggers=min_triggers,
            gap_seconds=gap_seconds,
            max_sections=max_sections,
        )

        padding_samples = int(round(padding_seconds * sfreq))
        windows: list[dict[str, int]] = []
        for index, section in enumerate(sections, start=1):
            first_trigger = int(section[0])
            last_trigger = int(section[-1])
            start = max(0, first_trigger - padding_samples)
            stop = min(n_times, last_trigger + padding_samples + 1)
            windows.append(
                {
                    "start": start,
                    "stop": stop,
                    "trigger_count": int(section.size),
                    "first_trigger": first_trigger,
                    "last_trigger": last_trigger,
                }
            )
            logger.info(
                "Trigger section {}: {} trigger(s), samples {}:{} -> padded chunk {}:{}",
                index,
                int(section.size),
                first_trigger,
                last_trigger,
                start,
                stop,
            )

        windows = Pipeline._merge_trigger_windows(windows)
        total = len(windows)
        width = max(3, len(str(total)))
        chunks: list[ChunkSpec] = []
        for index, window in enumerate(windows, start=1):
            output_path = output_dir / f"{output_stem}_chunk_{index:0{width}d}_of_{total:0{width}d}{extension}"
            chunks.append(
                ChunkSpec(
                    index=index,
                    total=total,
                    start_sample=window["start"],
                    stop_sample=window["stop"],
                    sfreq=sfreq,
                    output_path=output_path,
                )
            )
        return chunks

    @staticmethod
    def _split_trigger_sections(
        trigger_samples,
        sfreq: float,
        min_triggers: int,
        gap_seconds: float | None,
        max_sections: int,
    ) -> list:
        """Split sorted trigger samples into scan sections."""
        import numpy as np

        if trigger_samples.size <= 1:
            return [trigger_samples]

        gaps = np.diff(trigger_samples)
        if gap_seconds is None:
            positive_gaps = gaps[gaps > 0]
            typical_gap = float(np.median(positive_gaps)) if positive_gaps.size else 1.0
            threshold_samples = max(int(round(10.0 * sfreq)), int(round(5.0 * typical_gap)), 1)
        else:
            threshold_samples = max(int(round(gap_seconds * sfreq)), 1)

        split_points = np.flatnonzero(gaps > threshold_samples) + 1
        sections = [section for section in np.split(trigger_samples, split_points) if section.size > 0]
        logger.info(
            "Detected {} trigger-section candidate(s) using a {:.2f}s gap threshold",
            len(sections),
            threshold_samples / sfreq,
        )

        valid_sections = [section for section in sections if section.size >= min_triggers]
        skipped = len(sections) - len(valid_sections)
        if skipped > 0:
            logger.info(
                "Ignoring {} trigger-section candidate(s) with fewer than {} triggers",
                skipped,
                min_triggers,
            )

        if not valid_sections:
            logger.warning(
                "No trigger section reached {} triggers; using all {} trigger(s) as one section",
                min_triggers,
                trigger_samples.size,
            )
            valid_sections = [trigger_samples]

        # if len(valid_sections) > max_sections:
        #     logger.warning(
        #         "Detected {} trigger sections; keeping the {} largest section(s)",
        #         len(valid_sections),
        #         max_sections,
        #     )
        #     valid_sections = sorted(valid_sections, key=lambda section: section.size, reverse=True)[:max_sections]
        #     valid_sections = sorted(valid_sections, key=lambda section: int(section[0]))

        if max_sections is not None and len(valid_sections) > max_sections:
            logger.warning(
                "Detected {} trigger sections; keeping the {} largest section(s)",
                len(valid_sections),
                max_sections,
            )
            valid_sections = sorted(
                valid_sections,
                key=lambda section: section.size,
                reverse=True,
            )[:max_sections]
            valid_sections = sorted(
                valid_sections,
                key=lambda section: int(section[0]),
            )

        return valid_sections

    @staticmethod
    def _merge_trigger_windows(windows: list[dict[str, int]]) -> list[dict[str, int]]:
        """Merge padded trigger-section windows that overlap."""
        if not windows:
            return []

        merged = [dict(windows[0])]
        for window in windows[1:]:
            current = merged[-1]
            if window["start"] > current["stop"]:
                merged.append(dict(window))
                continue

            current["stop"] = max(current["stop"], window["stop"])
            current["trigger_count"] += window["trigger_count"]
            current["last_trigger"] = max(current["last_trigger"], window["last_trigger"])

        if len(merged) < len(windows):
            logger.info(
                "Merged {} overlapping padded trigger window(s) into {} output chunk(s)",
                len(windows),
                len(merged),
            )

        return merged

    @staticmethod
    def _write_chunk_manifest(
        output_dir: Path,
        source_path: str,
        chunks: list[ChunkSpec],
        results: list[PipelineResult],
        execution_time: float,
        memory_budget: int,
        chunking_mode: str = "fixed_length",
    ) -> Path:
        """Write a manifest describing all chunk outputs."""
        import json

        manifest_path = output_dir / "chunks_manifest.json"
        payload = {
            "source_path": source_path,
            "execution_time": execution_time,
            "memory_budget_bytes": memory_budget,
            "chunking_mode": chunking_mode,
            "chunks": [
                {
                    "index": chunk.index,
                    "total": chunk.total,
                    "start_sample": chunk.start_sample,
                    "stop_sample": chunk.stop_sample,
                    "duration_seconds": chunk.duration_seconds,
                    "output_path": str(chunk.output_path),
                    "success": result.success if i < len(results) else False,
                    "error": str(result.error) if i < len(results) and result.error is not None else None,
                }
                for i, (chunk, result) in enumerate(zip(chunks, results, strict=False))
            ],
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return manifest_path

    @staticmethod
    def _is_memory_failure(error: Exception | None) -> bool:
        """Best-effort detection of memory-related failures."""
        if error is None:
            return False
        if isinstance(error, MemoryError):
            return True
        text = str(error).lower()
        return "memory" in text or "cannot allocate" in text or "allocation" in text

    def __len__(self) -> int:
        """Get number of processors."""
        return len(self.processors)

    def __getitem__(self, index: int) -> Processor:
        """Get processor by index."""
        return self.processors[index]

    def __repr__(self) -> str:
        """String representation."""
        return f"Pipeline(name='{self.name}', n_processors={len(self.processors)})"


class PipelineBuilder:
    """
    Fluent builder for constructing pipelines.

    Example::

        pipeline = (PipelineBuilder()
            .add(Loader("data.edf"))
            .highpass(1.0)
            .upsample(10)
            .detect_triggers(r"\\btrigger\\b")
            .aas_correction()
            .export_edf("output.edf")
            .build())
    """

    def __init__(self, name: str | None = None):
        """
        Initialize builder.

        Args:
            name: Optional pipeline name
        """
        self._processors: list[Processor] = []
        self._name = name

    def add(self, processor: Processor) -> "PipelineBuilder":
        """
        Add custom processor.

        Args:
            processor: Processor to add

        Returns:
            Self for chaining
        """
        self._processors.append(processor)
        return self

    def add_if(self, condition: bool, processor: Processor) -> "PipelineBuilder":
        """
        Add processor conditionally.

        Args:
            condition: Whether to add processor
            processor: Processor to add

        Returns:
            Self for chaining
        """
        if condition:
            self._processors.append(processor)
        return self

    def build(self) -> Pipeline:
        """
        Build the pipeline.

        Returns:
            Constructed Pipeline instance
        """
        return Pipeline(self._processors, name=self._name)

    # Convenience methods can be added here for common processors
    # These will be populated as we implement specific processors
