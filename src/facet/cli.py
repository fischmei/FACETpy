"""Command line tools for FACETpy processing and BIDS conversion."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from facet import (
    AnalyzeDataReport,
    BIDSExporter,
    CheckDataReport,
    FFTAllenCalculator,
    FFTNiazyCalculator,
    LegacySNRCalculator,
    Loader,
    MedianArtifactCalculator,
    MetricsReport,
    ProcessorValidationError,
    RawPlotter,
    RMSCalculator,
    RMSResidualCalculator,
    SNRCalculator,
    TriggerDetector,
)

from ._cli_bids import (
    CHUNK_RE,
    _build_bids_export_plan,
    _deduplicate_runs,
    _derive_bids_entities,
    _sanitize_bids_label,
)
from ._cli_bids import _run_to_bids as _run_to_bids_impl
from ._cli_inputs import (
    _is_supported_eeg_path,
    _normalise_extension,
    _path_extension,
    _read_path_list,
    _resolve_inputs,
    _scan_input_dir,
    _source_output_dir,
)
from ._cli_parser import ANALYSIS_METRIC_DESCRIPTIONS, _add_input_arguments
from ._cli_parser import _build_parser as _build_parser_impl
from ._cli_pipeline import (
    ADD_ON_MODE_DESCRIPTIONS,
    CORRECTION_MATRIX_DESCRIPTIONS,
    CORRECTION_MODE_DESCRIPTIONS,
    DEFAULT_EGI_DROP_REGEX,
    PATTERN_DESCRIPTIONS,
    PROCESS_PATTERN_DESCRIPTIONS,
    ANCCorrection,
    PCACorrection,
    _build_bcg_pattern,
    _build_mode_processors,
    _build_processing_pipeline,
    _build_quickstart_pattern,
    _build_standard_pattern,
    _build_template_correction,
    _common_template_kwargs,
    _drop_channel_processors,
    _parse_pca_components,
    _selected_add_on_modes,
    _unique_modes,
)
from ._cli_reports import (
    _chunk_report,
    _chunked_results,
    _collect_artifact_template_reports,
    _json_safe,
    _processing_failure_record,
    _processor_report,
    _write_artifact_template_matrix_report,
    _write_batch_failures,
    _write_pipeline_description,
    _write_processing_error,
)

__all__ = [
    "ADD_ON_MODE_DESCRIPTIONS",
    "ANCCorrection",
    "CHUNK_RE",
    "CORRECTION_MATRIX_DESCRIPTIONS",
    "CORRECTION_MODE_DESCRIPTIONS",
    "DEFAULT_EGI_DROP_REGEX",
    "PATTERN_DESCRIPTIONS",
    "PCACorrection",
    "PROCESS_PATTERN_DESCRIPTIONS",
    "_add_input_arguments",
    "_build_bcg_pattern",
    "_build_bids_export_plan",
    "_build_mode_processors",
    "_build_parser",
    "_build_processing_pipeline",
    "_build_quickstart_pattern",
    "_build_standard_pattern",
    "_build_template_correction",
    "_chunk_report",
    "_chunked_results",
    "_collect_artifact_template_reports",
    "_common_template_kwargs",
    "_deduplicate_runs",
    "_derive_bids_entities",
    "_drop_channel_processors",
    "_is_supported_eeg_path",
    "_json_safe",
    "_normalise_extension",
    "_parse_pca_components",
    "_path_extension",
    "_processing_failure_record",
    "_processor_report",
    "_read_path_list",
    "_resolve_inputs",
    "_run_to_bids",
    "_sanitize_bids_label",
    "_scan_input_dir",
    "_selected_add_on_modes",
    "_source_output_dir",
    "_unique_modes",
    "_with_default_command",
    "_write_artifact_template_matrix_report",
    "_write_batch_failures",
    "_write_pipeline_description",
    "_write_processing_error",
    "bids_main",
    "main",
]


def _run_modes(args: argparse.Namespace) -> int:
    """Print available correction modes."""
    del args

    lines = [
        "Correction modes replace the baseline AAS template-subtraction step:",
        *[f"  {name}: {description}" for name, description in CORRECTION_MODE_DESCRIPTIONS.items()],
        "",
        "Add-on modes are layered around the selected correction mode:",
        *[f"  {name}: {description}" for name, description in ADD_ON_MODE_DESCRIPTIONS.items()],
        "",
        "Example: facetpy-run process --input data.mff --output-dir output --correction-mode farm --mode pca --mode anc",
    ]
    print("\n".join(lines))
    return 0


def _run_patterns(args: argparse.Namespace) -> int:
    """Print available pipeline patterns."""
    del args

    lines = [
        "Pipeline patterns describe whole workflow shapes from the quickstart docs:",
        *[f"  {name}: {description}" for name, description in PATTERN_DESCRIPTIONS.items()],
        "",
        "Use runnable process patterns with --pattern:",
        f"  {', '.join(PROCESS_PATTERN_DESCRIPTIONS)}",
        "",
        "Example: facetpy-run process --pattern standard --input data.edf --output-dir output",
    ]
    print("\n".join(lines))
    return 0


def _load_context_for_cli(input_path: str):
    """Load one recording for viewer and analysis commands."""
    return Loader(path=str(Path(input_path).expanduser().resolve()), preload=True).execute(None)


DEFAULT_ANALYSIS_METRICS = tuple(ANALYSIS_METRIC_DESCRIPTIONS)


def _analysis_metric_processor(metric_name: str):
    """Build the processor registered to a CLI metric name."""
    builders = {
        "snr": SNRCalculator,
        "legacy-snr": LegacySNRCalculator,
        "rms": RMSCalculator,
        "rms-residual": RMSResidualCalculator,
        "median-artifact": MedianArtifactCalculator,
        "fft-allen": FFTAllenCalculator,
        "fft-niazy": FFTNiazyCalculator,
        "report": MetricsReport,
    }
    try:
        return builders[metric_name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported analysis metric: {metric_name}") from exc


def _selected_analysis_metrics(args: argparse.Namespace) -> list[str]:
    """Resolve requested analysis metrics while preserving user order."""
    metric_names = args.metric if args.metric else DEFAULT_ANALYSIS_METRICS if args.metrics else ()
    selected: list[str] = []
    seen: set[str] = set()

    for metric_name in metric_names:
        if metric_name not in seen:
            selected.append(metric_name)
            seen.add(metric_name)

    return selected


def _print_analysis_metrics() -> None:
    """Print metric names and short descriptions for the analysis command."""
    lines = [
        "Available analysis metrics:",
        *[f"  {name}: {description}" for name, description in ANALYSIS_METRIC_DESCRIPTIONS.items()],
    ]
    print("\n".join(lines))


def _record_skipped_metric(context, metric_name: str, exc: ProcessorValidationError) -> None:
    """Store skipped metric names and reasons in exported metadata."""
    skipped = context.metadata.custom.setdefault("skipped_metrics", [])
    if metric_name not in skipped:
        skipped.append(metric_name)
    context.metadata.custom.setdefault("skipped_metric_reasons", {})[metric_name] = str(exc)


def _run_analysis_metrics(
    context,
    metric_names: Sequence[str],
    *,
    skip_inapplicable: bool,
):
    """Execute metric processors, optionally continuing past validation failures."""
    for metric_name in metric_names:
        processor = _analysis_metric_processor(metric_name)
        try:
            context = processor.execute(context)
        except ProcessorValidationError as exc:
            if not skip_inapplicable:
                raise
            _record_skipped_metric(context, metric_name, exc)
            logger.warning("Skipping analysis metric '{}': {}", metric_name, exc)

    return context


def _run_viewer(args: argparse.Namespace) -> int:
    """Open or save a FACETpy viewer for one EEG recording."""
    context = _load_context_for_cli(args.input)
    mne_kwargs = {}
    if args.n_channels is not None:
        mne_kwargs["n_channels"] = args.n_channels
    if args.scalings is not None:
        mne_kwargs["scalings"] = args.scalings

    RawPlotter(
        mode=args.viewer_mode,
        channel=args.channel,
        start=args.start,
        duration=args.duration,
        save_path=args.output,
        show=args.show,
        title=args.title,
        mne_kwargs=mne_kwargs,
    ).execute(context)
    return 0


def _run_analysis(args: argparse.Namespace) -> int:
    """Run FACETpy analysis/report processors for one EEG recording."""
    if args.list_metrics:
        _print_analysis_metrics()
        return 0

    if args.input is None:
        print("facetpy-run analysis: error: --input is required unless --list-metrics is used", file=sys.stderr)
        return 2

    context = _load_context_for_cli(args.input)

    if args.detect_events:
        context = TriggerDetector(regex=args.trigger_regex).execute(context)

    processors = [
        AnalyzeDataReport(),
        CheckDataReport(require_triggers=args.require_triggers, strict=args.strict),
    ]

    for processor in processors:
        context = processor.execute(context)

    context = _run_analysis_metrics(
        context,
        _selected_analysis_metrics(args),
        skip_inapplicable=args.skip_inapplicable_metrics,
    )

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_json_safe(context.metadata.custom), indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote analysis metadata to {}", output_path)

    return 0


def _run_process(args: argparse.Namespace) -> int:
    """Run FACETpy correction over one input, a list, or a directory."""
    input_paths = _resolve_inputs(
        inputs=args.input,
        input_list=args.input_list,
        input_dir=args.input_dir,
        extensions=args.extensions,
        recursive=args.recursive,
    )
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pipeline = _build_processing_pipeline(args)
    success = True
    failures: list[dict[str, str]] = []

    for input_path in input_paths:
        target_dir = _source_output_dir(input_path, output_root, len(input_paths), args.flat_output)
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Processing '{}' -> '{}'", input_path, target_dir)

        try:
            chunk_by_trigger_sections = not args.fixed_length_chunks
            if args.pattern == "bcg":
                # QRSTriggerDetector has no regex-based probe path, so BCG uses
                # fixed-length chunks unless the user later builds a custom flow.
                chunk_by_trigger_sections = False

            result = pipeline.run_chunked(
                input_path=str(input_path),
                output_dir=str(target_dir),
                output_extension=args.output_extension,
                min_chunks=args.min_chunks,
                max_chunks=args.max_chunks,
                memory_budget_mb=args.memory_budget_mb,
                memory_fraction=args.memory_fraction,
                overwrite=args.overwrite,
                channel_sequential=args.channel_sequential,
                on_error=args.on_error,
                keep_raw=False,
                chunk_by_trigger_sections=chunk_by_trigger_sections,
                trigger_section_padding_seconds=args.trigger_section_padding_seconds,
                trigger_section_min_triggers=args.trigger_section_min_triggers,
                trigger_section_gap_seconds=args.trigger_section_gap_seconds,
                trigger_section_max_sections=args.trigger_section_max_sections,
            )
        except Exception as exc:
            success = False
            record = _processing_failure_record(input_path, target_dir, exc)
            failures.append(record)
            marker_path = _write_processing_error(target_dir, record)
            logger.error("Processing failed for '{}': {}", input_path, exc)
            logger.warning("Recorded failure marker: {}", marker_path)
            if args.on_error == "raise":
                raise
            logger.warning("Continuing to next input because --on-error=continue")
            continue

        result.print_summary()
        matrix_report_path = _write_artifact_template_matrix_report(target_dir, input_path, result)
        description_path = _write_pipeline_description(
            target_dir, input_path, args, pipeline, result, matrix_report_path
        )
        logger.info("Wrote pipeline description: {}", description_path)
        logger.info("Wrote artifact template matrix report: {}", matrix_report_path)
        success = success and result.was_successful()

    failure_manifest = _write_batch_failures(output_root, failures)
    if failure_manifest is not None:
        logger.warning("Batch completed with {} failed input(s): {}", len(failures), failure_manifest)

    return 0 if success else 1


def _run_to_bids(args: argparse.Namespace) -> int:
    """Convert one or more corrected output files into a BIDS dataset."""
    return _run_to_bids_impl(
        args,
        loader_cls=Loader,
        trigger_detector_cls=TriggerDetector,
        bids_exporter_cls=BIDSExporter,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the FACETpy command line parser."""
    return _build_parser_impl(
        run_process=_run_process,
        run_to_bids=_run_to_bids,
        run_modes=_run_modes,
        run_patterns=_run_patterns,
        run_viewer=_run_viewer,
        run_analysis=_run_analysis,
    )


def _with_default_command(argv: Sequence[str]) -> list[str]:
    """Default to the processing command for backwards-compatible usage."""
    args = list(argv)
    commands = {"process", "to-bids", "modes", "patterns", "viewer", "view", "analysis"}
    if args and args[0] in {"-h", "--help"}:
        return args
    if not args or args[0] not in commands:
        return ["process", *args]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """FACETpy CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(_with_default_command(sys.argv[1:] if argv is None else argv))
    return int(args.func(args))


def bids_main(argv: Sequence[str] | None = None) -> int:
    """Convenience entry point for BIDS conversion only."""
    return main(["to-bids", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
