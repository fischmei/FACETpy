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
    AASCorrection,
    BIDSExporter,
    DownSample,
    DropChannelsMatching,
    Loader,
    Pipeline,
    TriggerDetector,
    UpSample,
)
from facet.io.loaders import SUPPORTED_EXTENSIONS

DEFAULT_EGI_DROP_REGEX = r"^E(?:[1-9]|[1-9]\d|1[01]\d|12[0-8])$"
CHUNK_RE = re.compile(r"^(?P<stem>.+)_chunk_(?P<index>\d+)_of_(?P<total>\d+)$")


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


def _build_processing_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build the AAS processing pipeline used by the CLI."""
    processors = []
    if args.drop_egi_e_channels:
        processors.append(DropChannelsMatching(regex=args.drop_channel_regex))

    processors.extend(
        [
            TriggerDetector(regex=args.trigger_regex),
            UpSample(factor=args.upsample_factor),
            AASCorrection(window_size=args.window_size),
            DownSample(factor=args.upsample_factor),
        ]
    )
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
                chunk_by_trigger_sections=not args.fixed_length_chunks,
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
    process.add_argument("--upsample-factor", type=int, default=10, help="Upsampling factor before AAS correction.")
    process.add_argument("--window-size", type=int, default=30, help="Window size for AASCorrection.")
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
        default=True,
        help="Drop EGI E1-E128 channels on a copy before trigger detection and correction.",
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

    return parser


def _with_default_command(argv: Sequence[str]) -> list[str]:
    """Default to the processing command for backwards-compatible usage."""
    args = list(argv)
    if not args or args[0] not in {"process", "to-bids"}:
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
