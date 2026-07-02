"""BIDS conversion helpers for the FACETpy CLI."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from loguru import logger

from facet import BIDSExporter, Loader, TriggerDetector

from ._cli_inputs import _resolve_inputs
from ._cli_labels import _compact_label

CHUNK_RE = re.compile(r"^(?P<stem>.+)_chunk_(?P<index>\d+)_of_(?P<total>\d+)$")


def _sanitize_bids_label(value: str, fallback: str = "recording") -> str:
    """Convert free-form text into a BIDS entity label."""
    return _compact_label(value, fallback=fallback)


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


def _run_to_bids(
    args: argparse.Namespace,
    *,
    loader_cls=Loader,
    trigger_detector_cls=TriggerDetector,
    bids_exporter_cls=BIDSExporter,
) -> int:
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
            context = loader_cls(path=str(input_path), preload=True).execute(None)

            if args.detect_events:
                try:
                    context = trigger_detector_cls(regex=args.trigger_regex).execute(context)
                except Exception as exc:
                    logger.warning(
                        "No matching trigger events found for '{}'. Exporting without events. Reason: {}",
                        input_path,
                        exc,
                    )

            context = bids_exporter_cls(
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
