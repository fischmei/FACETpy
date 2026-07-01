"""
Executable homes for the pipeline patterns described in quickstart.rst.

These functions intentionally mirror the documentation snippets without running
on import. Update this file when the quickstart guide adds or changes a pattern.
"""

from __future__ import annotations

from pathlib import Path

from facet import (
    AASCorrection,
    DownSample,
    EDFExporter,
    Loader,
    MetricsReport,
    Pipeline,
    QRSTriggerDetector,
    SNRCalculator,
    TriggerDetector,
    UpSample,
    create_standard_pipeline,
    load,
)


def basic_correction_pipeline(input_path: str, output_path: str) -> Pipeline:
    """Create the quickstart standard pipeline factory pattern."""
    return create_standard_pipeline(
        input_path=input_path,
        output_path=output_path,
        trigger_regex=r"\b1\b",
        evaluate=True,
    )


def custom_pipeline(input_path: str, output_path: str) -> Pipeline:
    """Build the explicit custom Pipeline([...]) pattern."""
    return Pipeline(
        [
            Loader(path=input_path, preload=True),
            TriggerDetector(regex=r"\b1\b"),
            UpSample(factor=10),
            AASCorrection(window_size=30, correlation_threshold=0.975),
            DownSample(factor=10),
            SNRCalculator(),
            MetricsReport(),
            EDFExporter(path=output_path, overwrite=True),
        ],
        name="Quickstart Custom Pipeline",
    )


def step_by_step_processing(input_path: str, output_path: str) -> None:
    """Execute the docs step-by-step ProcessingContext pattern."""
    context = Loader(path=input_path, preload=True).execute(None)
    context = TriggerDetector(regex=r"\b1\b").execute(context)
    context = UpSample(factor=10).execute(context)
    context = AASCorrection(window_size=30).execute(context)
    context.get_raw().save(output_path, overwrite=True)


def pipe_operator_processing(input_path: str):
    """Return a context processed with the pipe-operator pattern."""
    return (
        load(input_path, preload=True)
        | TriggerDetector(regex=r"\b1\b")
        | UpSample(factor=10)
        | AASCorrection(window_size=30)
    )


def bcg_correction_pipeline(input_path: str, output_path: str) -> Pipeline:
    """Build the BCG/QRS correction pattern from the quickstart guide."""
    return Pipeline(
        [
            Loader(path=input_path, preload=True),
            QRSTriggerDetector(),
            AASCorrection(window_size=20),
            EDFExporter(path=output_path, overwrite=True),
        ],
        name="Quickstart BCG Pattern",
    )


def batch_processing_pattern(input_files: list[str], output_dir: str) -> None:
    """Process many files with the reusable batch pattern."""
    correction = Pipeline(
        [
            TriggerDetector(regex=r"\b1\b"),
            UpSample(factor=10),
            AASCorrection(window_size=30),
            DownSample(factor=10),
        ],
        name="Quickstart Batch Correction",
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for input_file in input_files:
        context = Loader(path=input_file, preload=True).execute(None)
        result = correction.run(initial_context=context, channel_sequential=True)
        if result.success:
            output_path = output_root / f"{Path(input_file).stem}_corrected.edf"
            EDFExporter(path=str(output_path), overwrite=True).execute(result.context)
