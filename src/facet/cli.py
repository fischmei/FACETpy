"""Command line tools for FACETpy processing and BIDS conversion."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from facet import (
    AnalyzeDataReport,
    BIDSExporter,
    CheckDataReport,
    CutAcquisitionWindow,
    DownSample,
    DropChannelsMatching,
    FFTAllenCalculator,
    FFTNiazyCalculator,
    HighPassFilter,
    LegacySNRCalculator,
    Loader,
    LowPassFilter,
    MedianArtifactCalculator,
    MetricsReport,
    PasteAcquisitionWindow,
    Pipeline,
    QRSTriggerDetector,
    RawPlotter,
    RMSCalculator,
    RMSResidualCalculator,
    SliceAligner,
    SNRCalculator,
    SubsampleAligner,
    TriggerDetector,
    UpSample,
)
from facet.correction import (
    AASCorrection,
    CorrespondingSliceCorrection,
    FARMCorrection,
    MoosmannCorrection,
    SliceTriggerCorrection,
    VolumeArtifactCorrection,
    VolumeTriggerCorrection,
)
from facet.io.loaders import SUPPORTED_EXTENSIONS

try:
    from facet.correction import ANCCorrection
except ImportError:  # pragma: no cover - depends on optional scientific stack
    ANCCorrection = None

try:
    from facet.correction import PCACorrection
except ImportError:  # pragma: no cover - depends on optional scientific stack
    PCACorrection = None

DEFAULT_EGI_DROP_REGEX = r"^E(?:[1-9]|[1-9]\d|1[01]\d|12[0-8])$"
CHUNK_RE = re.compile(r"^(?P<stem>.+)_chunk_(?P<index>\d+)_of_(?P<total>\d+)$")
# Saved CLI mode tables used by the on-disk process command.
CORRECTION_MODE_DESCRIPTIONS = {
    "aas": "Baseline Averaged Artifact Subtraction.",
    "farm": "FACET FARM-style template weighting for similar artifact epochs.",
    "volume-trigger": "FACET volume-trigger template weighting.",
    "slice-trigger": "FACET slice-trigger odd/even template weighting.",
    "corresponding-slice": "Average corresponding slice positions across volumes.",
    "moosmann": "Motion-informed template weighting from an SPM realignment-parameter file.",
}
ADD_ON_MODE_DESCRIPTIONS = {
    "volume-artifact": "Correct transition artifacts around slice-trigger volume gaps before template subtraction.",
    "pca": "Apply PCA residual cleanup after template subtraction.",
    "anc": "Apply adaptive noise cancellation after downsampling, using the accumulated noise estimate.",
}
CORRECTION_MATRIX_DESCRIPTIONS = {
    "aas": "AAS builds A with correlation-selected epochs from sliding windows.",
    "farm": "FARM builds A from the most correlated neighboring epochs above threshold.",
    "volume-trigger": "Volume-trigger correction builds A from fixed neighboring volume-trigger epochs.",
    "slice-trigger": "Slice-trigger correction builds A from alternating odd/even slice-trigger epochs.",
    "corresponding-slice": "Corresponding-slice correction builds A from the same slice position across volumes.",
    "moosmann": "Moosmann correction builds A from motion-informed realignment-parameter weights.",
}
PROCESS_PATTERN_DESCRIPTIONS = {
    "quickstart": "Memory-light trigger-section chunks: trigger detection, upsample, correction, downsample.",
    "standard": "Docs standard pipeline: cut, high-pass, align, correction, PCA, downsample, paste, low-pass, ANC.",
    "bcg": "Ballistocardiogram pattern: QRS trigger detection plus AAS correction.",
}
PATTERN_DESCRIPTIONS = {
    **PROCESS_PATTERN_DESCRIPTIONS,
    "custom": "Python pattern for manually assembling Pipeline([...]) with chosen processors.",
    "step-by-step": "Python pattern for executing processors one at a time against a ProcessingContext.",
    "pipe": "Python pattern for chaining processors with the ProcessingContext pipe operator.",
    "batch": "CLI/input pattern using --input-list or --input-dir to process many recordings.",
}


def _normalise_extension(extension: str) -> str:
    """Return a lowercase extension with a leading dot."""
    extension = extension.strip().lower()
    return extension if extension.startswith(".") else f".{extension}"


def _path_extension(path: Path) -> str:
    """Return the supported extension for regular EEG files and MFF folders."""
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] == [".fif", ".gz"]:
        return ".fif.gz"
    return path.suffix.lower()


def _is_supported_eeg_path(path: Path, extensions: set[str]) -> bool:
    """Return whether *path* looks like a supported input recording."""
    extension = _path_extension(path)
    if extension == ".fif.gz":
        return ".fif" in extensions or ".fif.gz" in extensions
    return extension in extensions


def _read_path_list(path: Path) -> list[Path]:
    """Read newline-delimited input paths, ignoring blanks and comments."""
    entries: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue

        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        entries.append(candidate)

    if not entries:
        logger.warning("Input list '{}' did not contain any usable paths", path)
    return entries


def _scan_input_dir(directory: Path, extensions: set[str], recursive: bool) -> list[Path]:
    """Find supported EEG files or MFF directories in *directory*."""
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    paths = [
        path
        for path in iterator
        if (path.is_file() or path.suffix.lower() == ".mff") and _is_supported_eeg_path(path, extensions)
    ]
    return sorted(paths, key=lambda item: str(item))


def _resolve_inputs(
    *,
    inputs: Sequence[str] | None,
    input_list: str | None,
    input_dir: str | None,
    extensions: Sequence[str],
    recursive: bool,
) -> list[Path]:
    """Collect input paths from explicit files, list files, and folders."""
    selected_extensions = {_normalise_extension(ext) for ext in extensions}
    resolved: list[Path] = []

    for value in inputs or ():
        resolved.append(Path(value).expanduser().resolve())

    if input_list is not None:
        resolved.extend(_read_path_list(Path(input_list).expanduser().resolve()))

    if input_dir is not None:
        directory = Path(input_dir).expanduser().resolve()
        if not directory.exists():
            raise FileNotFoundError(f"Input directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Input directory is not a directory: {directory}")
        resolved.extend(_scan_input_dir(directory, selected_extensions, recursive))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        path = path.resolve()
        if path in seen:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
        if not _is_supported_eeg_path(path, selected_extensions):
            raise ValueError(
                f"Unsupported input extension for '{path}'. "
                f"Allowed extensions: {', '.join(sorted(selected_extensions))}"
            )
        seen.add(path)
        unique.append(path)

    if not unique:
        raise ValueError("No input files found. Pass --input, --input-list, or --input-dir.")

    return unique


def _source_output_dir(input_path: Path, output_root: Path, total_inputs: int, flat_output: bool) -> Path:
    """Return the output folder for one source recording."""
    if total_inputs == 1 or flat_output:
        return output_root
    return output_root / _sanitize_bids_label(input_path.stem)


def _parse_pca_components(value: str) -> int | float | str:
    """Parse PCA component settings from the CLI."""
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"

    try:
        if re.fullmatch(r"[+-]?\d+", normalized):
            return int(normalized)
        return float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("PCA components must be an integer, a 0-1 fraction, or 'auto'.") from exc


def _unique_modes(modes: Sequence[str] | None) -> list[str]:
    """Return add-on modes in user order without duplicates."""
    selected: list[str] = []
    for mode in modes or ():
        if mode not in selected:
            selected.append(mode)
    return selected


def _selected_add_on_modes(args: argparse.Namespace) -> tuple[list[str], bool]:
    """Return add-on modes and whether they were selected by a pattern."""
    if args.mode is not None:
        return _unique_modes(args.mode), False
    if args.pattern == "standard":
        return ["pca", "anc"], True
    return [], False


def _common_template_kwargs(args: argparse.Namespace) -> dict:
    """Build shared options for AAS-style template subtraction processors."""
    return {
        "window_size": args.window_size,
        "plot_artifacts": args.plot_artifacts,
        "realign_after_averaging": args.realign_after_averaging,
        "search_window_factor": args.search_window_factor,
        "apply_epoch_alpha_scaling": args.apply_epoch_alpha_scaling,
    }


def _build_template_correction(args: argparse.Namespace):
    """Create the selected template-subtraction correction processor."""
    mode = args.correction_mode
    common = _common_template_kwargs(args)

    if mode == "aas":
        return AASCorrection(
            **common,
            correlation_threshold=args.aas_correlation_threshold,
            interpolate_volume_gaps=args.interpolate_volume_gaps,
        )
    if mode == "farm":
        return FARMCorrection(
            **common,
            correlation_threshold=args.farm_correlation_threshold,
            search_half_window=args.farm_search_half_window,
            search_half_window_factor=args.farm_search_half_window_factor,
            interpolate_volume_gaps=args.interpolate_volume_gaps,
        )
    if mode == "volume-trigger":
        return VolumeTriggerCorrection(**common)
    if mode == "slice-trigger":
        return SliceTriggerCorrection(**common)
    if mode == "corresponding-slice":
        return CorrespondingSliceCorrection(slices_per_volume=args.slices_per_volume, **common)
    if mode == "moosmann":
        if args.motion_rp_file is None:
            raise ValueError("--motion-rp-file is required when --correction-mode=moosmann")
        rp_file = Path(args.motion_rp_file).expanduser().resolve()
        if not rp_file.exists():
            raise FileNotFoundError(f"Motion realignment parameter file not found: {rp_file}")
        return MoosmannCorrection(
            rp_file=str(rp_file),
            motion_threshold=args.motion_threshold,
            motion_window_size=args.motion_window_size,
            **common,
        )

    raise ValueError(f"Unsupported correction mode: {mode}")


def _build_mode_processors(args: argparse.Namespace) -> tuple[list, list, list]:
    """Build pre-template, post-template, and post-downsample mode processors."""
    pre_template = []
    post_template = []
    post_downsample = []
    modes, from_pattern = _selected_add_on_modes(args)

    for mode in modes:
        if mode == "volume-artifact":
            pre_template.append(
                VolumeArtifactCorrection(
                    template_count=args.volume_template_count,
                    weighting_position=args.volume_weighting_position,
                    weighting_slope=args.volume_weighting_slope,
                )
            )
        elif mode == "pca":
            if PCACorrection is None:
                if from_pattern:
                    continue
                raise ImportError("PCACorrection is not available in this installation.")
            post_template.append(
                PCACorrection(
                    n_components=args.pca_components,
                    hp_freq=args.pca_hp_freq,
                )
            )
        elif mode == "anc":
            if ANCCorrection is None:
                if from_pattern:
                    continue
                raise ImportError("ANCCorrection is not available in this installation.")
            post_downsample.append(
                ANCCorrection(
                    filter_order=args.anc_filter_order,
                    hp_freq=args.anc_hp_freq,
                    use_c_extension=args.anc_c_extension,
                    mu_factor=args.anc_mu_factor,
                    max_gain=args.anc_max_gain,
                )
            )
        else:
            raise ValueError(f"Unsupported add-on mode: {mode}")

    return pre_template, post_template, post_downsample


def _drop_channel_processors(args: argparse.Namespace) -> list:
    """Return optional channel-dropping processors for every process pattern."""
    processors = []
    # EGI channel removal is opt-in. By default, trigger-section processing keeps
    # every channel in the cut so AAS correction runs over the full segment.
    if args.drop_egi_e_channels:
        processors.append(DropChannelsMatching(regex=args.drop_channel_regex))
    return processors


def _build_quickstart_pattern(args: argparse.Namespace) -> list:
    """Build the memory-light quickstart processing pattern."""
    pre_template, post_template, post_downsample = _build_mode_processors(args)
    return [
        TriggerDetector(regex=args.trigger_regex),
        UpSample(factor=args.upsample_factor),
        *pre_template,
        _build_template_correction(args),
        *post_template,
        DownSample(factor=args.upsample_factor),
        *post_downsample,
    ]


def _build_standard_pattern(args: argparse.Namespace) -> list:
    """Build the docs standard pattern without loader/exporter steps."""
    pre_template, post_template, post_downsample = _build_mode_processors(args)
    return [
        TriggerDetector(regex=args.trigger_regex),
        CutAcquisitionWindow(),
        HighPassFilter(freq=args.highpass_freq),
        UpSample(factor=args.upsample_factor),
        SliceAligner(ref_trigger_index=0),
        SubsampleAligner(ref_trigger_index=0),
        *pre_template,
        _build_template_correction(args),
        *post_template,
        DownSample(factor=args.upsample_factor),
        PasteAcquisitionWindow(),
        LowPassFilter(freq=args.lowpass_freq),
        *post_downsample,
    ]


def _build_bcg_pattern(args: argparse.Namespace) -> list:
    """Build the BCG/QRS pattern from the quickstart documentation."""
    return [
        QRSTriggerDetector(),
        AASCorrection(
            window_size=args.bcg_window_size,
            correlation_threshold=args.aas_correlation_threshold,
            plot_artifacts=args.plot_artifacts,
            realign_after_averaging=args.realign_after_averaging,
            search_window_factor=args.search_window_factor,
            apply_epoch_alpha_scaling=args.apply_epoch_alpha_scaling,
        ),
    ]


def _build_processing_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build the selected FACET correction pipeline used by the CLI."""
    processors = _drop_channel_processors(args)

    if args.pattern == "standard":
        processors.extend(_build_standard_pattern(args))
    elif args.pattern == "bcg":
        processors.extend(_build_bcg_pattern(args))
    else:
        processors.extend(_build_quickstart_pattern(args))

    return Pipeline(processors, name="FACETpy CLI Pipeline")


def _processing_failure_record(input_path: Path, target_dir: Path, exc: Exception) -> dict[str, str]:
    """Build a serializable failure record for one input file."""
    return {
        "input_path": str(input_path),
        "output_dir": str(target_dir),
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }


def _write_processing_error(target_dir: Path, record: dict[str, str]) -> Path:
    """Write a per-recording processing error marker."""
    marker_path = target_dir / "processing_error.json"
    marker_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return marker_path


def _write_batch_failures(output_root: Path, failures: list[dict[str, str]]) -> Path | None:
    """Write a batch-level failure manifest when any inputs failed."""
    if not failures:
        return None

    manifest_path = output_root / "processing_failures.json"
    manifest_path.write_text(json.dumps({"failures": failures}, indent=2), encoding="utf-8")
    return manifest_path


def _processor_report(processor, index: int) -> dict:
    """Return a JSON-friendly processor description for process reports."""
    return {
        "index": index,
        "name": processor.name,
        "type": processor.__class__.__name__,
        "description": getattr(processor, "description", ""),
        "parameters": _json_safe(getattr(processor, "_parameters", {})),
    }


def _chunk_report(chunk, result) -> dict:
    """Return a compact result description for one processed chunk."""
    return {
        "index": int(chunk.index),
        "total": int(chunk.total),
        "start_sample": int(chunk.start_sample),
        "stop_sample": int(chunk.stop_sample),
        "duration_seconds": float(chunk.duration_seconds),
        "output_path": str(chunk.output_path),
        "success": bool(result.success),
        "error": None if result.error is None else str(result.error),
    }


def _chunked_results(chunked_result) -> list:
    """Return chunk results when the result object is iterable."""
    try:
        return list(chunked_result)
    except TypeError:
        return []


def _collect_artifact_template_reports(chunked_result) -> list[dict]:
    """Collect AAS-style template matrix reports from successful chunk contexts."""
    chunks = list(getattr(chunked_result, "chunks", []))
    reports: list[dict] = []

    for chunk, result in zip(chunks, _chunked_results(chunked_result), strict=False):
        context = getattr(result, "context", None)
        if context is None:
            continue

        custom = getattr(context.metadata, "custom", {})
        for report in custom.get("artifact_template_matrices", []):
            report_payload = _json_safe(report)
            report_payload.setdefault(
                "chunk",
                {
                    "index": int(chunk.index),
                    "total": int(chunk.total),
                    "start_sample": int(chunk.start_sample),
                    "stop_sample": int(chunk.stop_sample),
                    "output_path": str(chunk.output_path),
                },
            )
            reports.append(report_payload)

    return reports


def _write_artifact_template_matrix_report(target_dir: Path, input_path: Path, chunked_result) -> Path:
    """Write the per-recording artifact template matrix report."""
    reports = _collect_artifact_template_reports(chunked_result)
    report_path = target_dir / "artifact_template_matrices.json"
    payload = {
        "source_path": str(input_path),
        "description": (
            "AAS-style corrections build artifact templates with N = A @ D. "
            "D is the epoch data matrix, A is the averaging matrix, and N is the estimated artifact template matrix."
        ),
        "report_count": len(reports),
        "reports": reports,
    }
    if not reports:
        payload["note"] = "No AAS-style artifact template matrix report was produced for this run."

    report_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return report_path


def _write_pipeline_description(
    target_dir: Path,
    input_path: Path,
    args: argparse.Namespace,
    pipeline: Pipeline,
    chunked_result,
    matrix_report_path: Path,
) -> Path:
    """Write a per-recording JSON description of the process pipeline."""
    chunks = list(getattr(chunked_result, "chunks", []))
    results = _chunked_results(chunked_result)
    add_on_modes, selected_by_pattern = _selected_add_on_modes(args)
    payload = {
        "source_path": str(input_path),
        "output_dir": str(target_dir),
        "pattern": args.pattern,
        "pattern_description": PROCESS_PATTERN_DESCRIPTIONS.get(args.pattern),
        "correction_mode": args.correction_mode,
        "correction_mode_description": CORRECTION_MODE_DESCRIPTIONS.get(args.correction_mode),
        "correction_matrix_description": CORRECTION_MATRIX_DESCRIPTIONS.get(args.correction_mode),
        "add_on_modes": add_on_modes,
        "add_on_modes_selected_by_pattern": selected_by_pattern,
        "add_on_mode_descriptions": {mode: ADD_ON_MODE_DESCRIPTIONS[mode] for mode in add_on_modes},
        "channel_sequential": bool(args.channel_sequential),
        "process_options": {
            key: _json_safe(value)
            for key, value in vars(args).items()
            if key not in {"func", "input", "input_list", "input_dir"}
        },
        "processors": [_processor_report(processor, index) for index, processor in enumerate(pipeline.processors, 1)],
        "result": {
            "success": bool(chunked_result.was_successful()),
            "execution_time_seconds": float(getattr(chunked_result, "execution_time", 0.0)),
            "chunks_manifest": (
                str(chunked_result.manifest_path)
                if getattr(chunked_result, "manifest_path", None) is not None
                else None
            ),
            "artifact_template_matrix_report": str(matrix_report_path),
            "chunks": [_chunk_report(chunk, result) for chunk, result in zip(chunks, results, strict=False)],
        },
    }

    description_path = target_dir / "pipeline_description.json"
    description_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return description_path


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


def _json_safe(value):
    """Convert nested metadata values into JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _load_context_for_cli(input_path: str):
    """Load one recording for viewer and analysis commands."""
    return Loader(path=str(Path(input_path).expanduser().resolve()), preload=True).execute(None)


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
    context = _load_context_for_cli(args.input)

    if args.detect_events:
        context = TriggerDetector(regex=args.trigger_regex).execute(context)

    processors = [
        AnalyzeDataReport(),
        CheckDataReport(require_triggers=args.require_triggers, strict=args.strict),
    ]
    if args.metrics:
        processors.extend(
            [
                SNRCalculator(),
                LegacySNRCalculator(),
                RMSCalculator(),
                RMSResidualCalculator(),
                MedianArtifactCalculator(),
                FFTAllenCalculator(),
                FFTNiazyCalculator(),
                MetricsReport(),
            ]
        )

    for processor in processors:
        context = processor.execute(context)

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
        description_path = _write_pipeline_description(target_dir, input_path, args, pipeline, result, matrix_report_path)
        logger.info("Wrote pipeline description: {}", description_path)
        logger.info("Wrote artifact template matrix report: {}", matrix_report_path)
        success = success and result.was_successful()

    failure_manifest = _write_batch_failures(output_root, failures)
    if failure_manifest is not None:
        logger.warning("Batch completed with {} failed input(s): {}", len(failures), failure_manifest)

    return 0 if success else 1


def _sanitize_bids_label(value: str, fallback: str = "recording") -> str:
    """Convert free-form text into a BIDS entity label."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", value)
    return cleaned or fallback


def _derive_bids_entities(path: Path, ordinal: int, subject_override: str | None) -> tuple[str, str | None]:
    """Derive a subject label and optional run label from a corrected output file."""
    stem = path.name
    for suffix in reversed(path.suffixes):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    match = CHUNK_RE.match(stem)
    if match is None:
        source_stem = stem
        run = str(ordinal)
    else:
        source_stem = match.group("stem")
        run = str(int(match.group("index")))

    subject = subject_override or _sanitize_bids_label(source_stem)
    return _sanitize_bids_label(subject, fallback=f"sub{ordinal}"), run


def _deduplicate_runs(items: list[tuple[Path, str, str | None]]) -> list[tuple[Path, str, str | None]]:
    """Ensure every subject/run pair is unique before BIDS export."""
    grouped: dict[tuple[str, str | None], list[int]] = defaultdict(list)
    for index, (_, subject, run) in enumerate(items):
        grouped[(subject, run)].append(index)

    for (_, run), indexes in grouped.items():
        if run is not None or len(indexes) == 1:
            continue
        for offset, item_index in enumerate(indexes, start=1):
            path, subject, _ = items[item_index]
            items[item_index] = (path, subject, str(offset))

    return items


def _build_bids_export_plan(args: argparse.Namespace) -> list[tuple[Path, str, str | None]]:
    """Resolve files and attach BIDS subject/run entities to each input."""
    input_paths = _resolve_inputs(
        inputs=args.input,
        input_list=args.input_list,
        input_dir=args.input_dir,
        extensions=args.extensions,
        recursive=args.recursive,
    )
    if args.subject is not None and len(input_paths) > 1:
        logger.warning("Using subject '{}' for all {} converted files", args.subject, len(input_paths))

    plan = [
        (path, *_derive_bids_entities(path, ordinal=index, subject_override=args.subject))
        for index, path in enumerate(input_paths, start=1)
    ]
    return _deduplicate_runs(plan)


# def _run_to_bids(args: argparse.Namespace) -> int:
#     """Convert one or more corrected output files into a BIDS dataset."""
#     output_root = Path(args.output_dir).expanduser().resolve()
#     output_root.mkdir(parents=True, exist_ok=True)

#     success = True
#     for input_path, subject, run in _build_bids_export_plan(args):
#         logger.info(
#             "Converting '{}' -> BIDS subject={}, task={}, run={}",
#             input_path,
#             subject,
#             args.task,
#             run,
#         )

#         try:
#             context = Loader(path=str(input_path), preload=True).execute(None)
#             if args.detect_events:
#                 context = TriggerDetector(regex=args.trigger_regex).execute(context)
#             context = BIDSExporter(
#                 root=str(output_root),
#                 subject=subject,
#                 task=args.task,
#                 session=args.session,
#                 run=run,
#                 event_id={"trigger": 1} if context.has_triggers() else None,
#                 overwrite=args.overwrite,
#             ).execute(context)
#             logger.info("BIDS export complete for '{}'", input_path)
#         except Exception as exc:
#             success = False
#             logger.error("BIDS conversion failed for '{}': {}", input_path, exc)
#             if args.on_error == "raise":
#                 raise

#     return 0 if success else 1

def _run_to_bids(args: argparse.Namespace) -> int:
    """Convert one or more corrected output files into a BIDS dataset."""
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    success = True
    for input_path, subject, run in _build_bids_export_plan(args):
        logger.info(
            "Converting '{}' -> BIDS subject={}, task={}, run={}",
            input_path,
            subject,
            args.task,
            run,
        )

        try:
            context = Loader(path=str(input_path), preload=True).execute(None)

            if args.detect_events:
                try:
                    context = TriggerDetector(regex=args.trigger_regex).execute(context)
                except Exception as exc:
                    logger.warning(
                        "No matching trigger events found for '{}'. Exporting without events. Reason: {}",
                        input_path,
                        exc,
                    )

            context = BIDSExporter(
                root=str(output_root),
                subject=subject,
                task=args.task,
                session=args.session,
                run=run,
                event_id={"trigger": 1} if context.has_triggers() else None,
                overwrite=args.overwrite,
            ).execute(context)

            logger.info("BIDS export complete for '{}'", input_path)

        except Exception as exc:
            success = False
            logger.error("BIDS conversion failed for '{}': {}", input_path, exc)
            if args.on_error == "raise":
                raise
            logger.warning("Continuing to next input because --on-error=continue")

    return 0 if success else 1


def _add_input_arguments(parser: argparse.ArgumentParser, *, default_extensions: Sequence[str]) -> None:
    """Add shared single/list/folder input arguments."""
    parser.add_argument("--input", action="append", help="Input EEG file or MFF folder. May be passed multiple times.")
    parser.add_argument("--input-list", help="Text file containing one input path per line.")
    parser.add_argument("--input-dir", help="Folder containing inputs to process.")
    parser.add_argument("--recursive", action="store_true", help="Search input folders recursively.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(default_extensions),
        help="Input extensions to include when scanning folders.",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the FACETpy command line parser."""
    parser = argparse.ArgumentParser(
        description="Run FACETpy correction and convert corrected outputs to BIDS.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser(
        "process",
        help="Run AAS correction over one file, a path list, or an input folder.",
    )
    _add_input_arguments(process, default_extensions=SUPPORTED_EXTENSIONS)
    process.add_argument(
        "--output-dir",
        "--output",
        required=True,
        dest="output_dir",
        help="Output folder for corrected segmented files.",
    )
    process.add_argument("--output-extension", default=".edf", help="Corrected output extension.")
    process.add_argument("--trigger-regex", default=r"\b1\b", help="Regex used to detect scanner triggers.")
    process.add_argument("--upsample-factor", type=int, default=10, help="Upsampling factor before template correction.")
    process.add_argument("--window-size", type=int, default=30, help="Window size for template correction.")
    process.add_argument(
        "--pattern",
        choices=tuple(PROCESS_PATTERN_DESCRIPTIONS),
        default="quickstart",
        help="Whole-pipeline pattern. Use the 'patterns' command to list details.",
    )
    process.add_argument(
        "--highpass-freq",
        type=float,
        default=1.0,
        help="High-pass frequency used by --pattern=standard.",
    )
    process.add_argument(
        "--lowpass-freq",
        type=float,
        default=70.0,
        help="Low-pass frequency used by --pattern=standard.",
    )
    process.add_argument(
        "--bcg-window-size",
        type=int,
        default=20,
        help="AAS window size used by --pattern=bcg.",
    )
    process.add_argument(
        "--correction-mode",
        choices=tuple(CORRECTION_MODE_DESCRIPTIONS),
        default="aas",
        help="Template-subtraction strategy. Use 'modes' to list details.",
    )
    process.add_argument(
        "--mode",
        action="append",
        choices=tuple(ADD_ON_MODE_DESCRIPTIONS),
        default=None,
        help="Optional add-on correction mode. Repeat to combine modes, for example --mode pca --mode anc.",
    )
    process.add_argument(
        "--aas-correlation-threshold",
        type=float,
        default=0.975,
        help="Correlation threshold for the baseline AAS template strategy.",
    )
    process.add_argument(
        "--farm-correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for --correction-mode=farm.",
    )
    process.add_argument(
        "--farm-search-half-window",
        type=int,
        default=None,
        help="Explicit FARM candidate search half-window in epochs.",
    )
    process.add_argument(
        "--farm-search-half-window-factor",
        type=float,
        default=3.0,
        help="FARM search half-window multiplier when no explicit half-window is set.",
    )
    process.add_argument(
        "--slices-per-volume",
        type=int,
        default=None,
        help="Slice count used by --correction-mode=corresponding-slice when it cannot be inferred.",
    )
    process.add_argument(
        "--motion-rp-file",
        default=None,
        help="SPM-style realignment parameter file required by --correction-mode=moosmann.",
    )
    process.add_argument(
        "--motion-threshold",
        type=float,
        default=5.0,
        help="Motion threshold for --correction-mode=moosmann.",
    )
    process.add_argument(
        "--motion-window-size",
        type=int,
        default=None,
        help="Motion weighting window size for --correction-mode=moosmann.",
    )
    process.add_argument(
        "--plot-artifacts",
        action="store_true",
        help="Plot one representative averaged artifact for template-correction modes.",
    )
    process.add_argument(
        "--realign-after-averaging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Realign triggers to averaged artifact templates after template calculation.",
    )
    process.add_argument(
        "--search-window-factor",
        type=float,
        default=3.0,
        help="Trigger realignment search-window factor for AAS-style modes.",
    )
    process.add_argument(
        "--interpolate-volume-gaps",
        action="store_true",
        help="Interpolate estimated noise in volume gaps for AAS/FARM modes.",
    )
    process.add_argument(
        "--apply-epoch-alpha-scaling",
        action="store_true",
        help="Scale each epoch template before subtraction with a least-squares alpha factor.",
    )
    process.add_argument(
        "--volume-template-count",
        type=int,
        default=5,
        help="Neighboring slice count for --mode=volume-artifact.",
    )
    process.add_argument(
        "--volume-weighting-position",
        type=float,
        default=0.8,
        help="Logistic midpoint inside one artifact epoch for --mode=volume-artifact.",
    )
    process.add_argument(
        "--volume-weighting-slope",
        type=float,
        default=20.0,
        help="Logistic slope for --mode=volume-artifact.",
    )
    process.add_argument(
        "--pca-components",
        type=_parse_pca_components,
        default=0.95,
        help="PCA components for --mode=pca: integer, 0-1 variance fraction, or 'auto'.",
    )
    process.add_argument(
        "--pca-hp-freq",
        type=float,
        default=1.0,
        help="High-pass cutoff before --mode=pca. Use 0 to disable.",
    )
    process.add_argument(
        "--anc-filter-order",
        type=int,
        default=None,
        help="Adaptive filter order for --mode=anc. Defaults to artifact length.",
    )
    process.add_argument(
        "--anc-hp-freq",
        type=float,
        default=None,
        help="High-pass cutoff for --mode=anc. Defaults to trigger-rate-derived value.",
    )
    process.add_argument(
        "--anc-c-extension",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the FastRANC C extension for --mode=anc when available.",
    )
    process.add_argument(
        "--anc-mu-factor",
        type=float,
        default=0.05,
        help="Learning-rate numerator for --mode=anc.",
    )
    process.add_argument(
        "--anc-max-gain",
        type=float,
        default=50.0,
        help="Maximum stable filtered-noise gain for --mode=anc.",
    )
    process.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    process.add_argument("--flat-output", action="store_true", help="Write all batch outputs directly into output-dir.")
    process.add_argument(
        "--channel-sequential",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the high-memory correction steps one channel at a time.",
    )
    process.add_argument(
        "--fixed-length-chunks",
        action="store_true",
        help="Use memory-estimated fixed-length chunks instead of trigger-section chunks.",
    )
    process.add_argument("--min-chunks", type=int, default=2, help="Minimum fixed-length chunk count.")
    process.add_argument("--max-chunks", type=int, default=128, help="Maximum fixed-length chunk count.")
    process.add_argument("--memory-budget-mb", type=float, default=None, help="Explicit per-chunk memory budget.")
    process.add_argument("--memory-fraction", type=float, default=0.5, help="Fraction of available memory to use.")
    process.add_argument(
        "--trigger-section-padding-seconds",
        type=float,
        default=10.0,
        help="Seconds kept before first trigger and after last trigger in each section.",
    )
    process.add_argument(
        "--trigger-section-min-triggers",
        type=int,
        default=16,
        help="Minimum trigger count for a trigger section.",
    )
    process.add_argument(
        "--trigger-section-gap-seconds",
        type=float,
        default=None,
        help="Explicit no-trigger gap used to split trigger sections.",
    )
    process.add_argument(
        "--trigger-section-max-sections",
        type=int,
        # default=2,
        default=None,
        help="Maximum trigger sections exported per input. Current: None",
    )
    process.add_argument(
        "--drop-egi-e-channels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop EGI E1-E128 channels on a copy before trigger detection and correction. Disabled by default.",
    )
    process.add_argument(
        "--drop-channel-regex",
        default=DEFAULT_EGI_DROP_REGEX,
        help="Regex used when --drop-egi-e-channels is enabled.",
    )
    process.add_argument("--on-error", choices=("raise", "continue"), default="raise", help="Batch error behavior.")
    process.set_defaults(func=_run_process)

    bids = subparsers.add_parser(
        "to-bids",
        help="Convert corrected output files into a BIDS dataset.",
    )
    _add_input_arguments(bids, default_extensions=(".edf", ".bdf", ".gdf", ".vhdr", ".set", ".fif"))
    bids.add_argument(
        "--output-dir",
        "--bids-dir",
        required=True,
        dest="output_dir",
        help="Destination BIDS root folder.",
    )
    bids.add_argument("--task", default="facetcorrected", help="BIDS task label.")
    bids.add_argument("--subject", default=None, help="BIDS subject label. Defaults to source filename.")
    bids.add_argument("--session", default=None, help="Optional BIDS session label.")
    bids.add_argument("--trigger-regex", default=r"\b1\b", help="Regex used to recover event markers.")
    # bids.add_argument(
    #     "--detect-events",
    #     action=argparse.BooleanOptionalAction,
    #     default=True,
    #     help="Detect trigger events before BIDS export.",
    # )
    bids.add_argument(
        "--detect-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally detect trigger events before BIDS export.",
    )
    bids.add_argument("--overwrite", action="store_true", help="Overwrite existing BIDS files.")
    bids.add_argument("--on-error", choices=("raise", "continue"), default="raise", help="Batch error behavior.")
    bids.set_defaults(func=_run_to_bids)

    modes = subparsers.add_parser(
        "modes",
        help="List correction and add-on modes available to the process command.",
    )
    modes.set_defaults(func=_run_modes)

    patterns = subparsers.add_parser(
        "patterns",
        help="List whole-pipeline patterns from the quickstart documentation.",
    )
    patterns.set_defaults(func=_run_patterns)

    viewer = subparsers.add_parser(
        "viewer",
        aliases=("view",),
        help="View or save a raw EEG plot using FACETpy RawPlotter.",
    )
    viewer.add_argument("--input", required=True, help="Input EEG file or MFF folder.")
    viewer.add_argument("--output", default=None, help="Optional plot image path.")
    viewer.add_argument("--viewer-mode", choices=("matplotlib", "mne"), default="matplotlib", help="Viewer backend.")
    viewer.add_argument("--channel", default=None, help="Channel name or index for matplotlib mode.")
    viewer.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    viewer.add_argument("--duration", type=float, default=10.0, help="Duration in seconds. Use 0 for full span.")
    viewer.add_argument("--show", action="store_true", help="Show the plot interactively.")
    viewer.add_argument("--title", default=None, help="Optional plot title.")
    viewer.add_argument("--n-channels", type=int, default=None, help="MNE viewer channel count.")
    viewer.add_argument("--scalings", default=None, help="MNE viewer scaling mode, e.g. 'auto'.")
    viewer.set_defaults(func=_run_viewer)

    analysis = subparsers.add_parser(
        "analysis",
        help="Run FACETpy recording checks and optional quality metrics.",
    )
    analysis.add_argument("--input", required=True, help="Input EEG file or MFF folder.")
    analysis.add_argument("--trigger-regex", default=r"\b1\b", help="Regex used when detecting events.")
    analysis.add_argument(
        "--detect-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Detect trigger events before running reports.",
    )
    analysis.add_argument(
        "--require-triggers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat missing triggers as a failed data check.",
    )
    analysis.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Raise an error when CheckDataReport finds problems.",
    )
    analysis.add_argument("--metrics", action="store_true", help="Run standard quality metric processors.")
    analysis.add_argument("--output-json", default=None, help="Optional path for structured analysis metadata.")
    analysis.set_defaults(func=_run_analysis)

    return parser


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
