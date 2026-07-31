"""Convert the EEG recordings in a BIDS dataset to EDF+."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import mne
from loguru import logger

from facet.io import EDFExporter, Loader

BIDS_EEG_EXTENSIONS = frozenset({".bdf", ".edf", ".fif", ".set", ".vhdr"})
EEG_COMPANION_EXTENSIONS = frozenset({".eeg", ".fdt", ".vmrk"})


def _is_below_eeg_directory(path: Path, bids_root: Path) -> bool:
    """Return whether *path* is contained in an ``eeg`` directory."""
    return "eeg" in path.relative_to(bids_root).parts[:-1]


def discover_bids_eeg_recordings(bids_root: Path) -> list[Path]:
    """Recursively find supported BIDS EEG recordings below *bids_root*."""
    recordings: list[Path] = []

    for path in sorted(bids_root.rglob("*")):
        if not path.is_file() or not _is_below_eeg_directory(path, bids_root):
            continue
        if path.suffix.lower() in EEG_COMPANION_EXTENSIONS:
            # BrainVision/EEGLAB companion files are represented by their
            # .vhdr/.set recording header.
            continue
        if path.suffix.lower() not in BIDS_EEG_EXTENSIONS:
            logger.warning("Skipping unsupported BIDS EEG recording: {}", path)
            continue
        recordings.append(path)

    return recordings


def matching_events_path(recording: Path) -> Path:
    """Return the recording-level BIDS events sidecar for *recording*."""
    return matching_bids_metadata(recording)["events"]


def matching_bids_metadata(recording: Path) -> dict[str, Path]:
    """Build recording-level BIDS sidecar paths from the source filename stem."""
    recording_stem = recording.stem.removesuffix("_eeg")
    return {
        "events": recording.with_name(f"{recording_stem}_events.tsv"),
        "channels": recording.with_name(f"{recording_stem}_channels.tsv"),
        "eeg": recording.with_name(f"{recording_stem}_eeg.json"),
        "electrodes": recording.with_name(f"{recording_stem}_electrodes.tsv"),
        "coordsystem": recording.with_name(f"{recording_stem}_coordsystem.json"),
    }


def _bids_value(value: str | None) -> str | None:
    """Normalize an optional scalar from a BIDS TSV file."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() == "n/a":
        return None
    return stripped


def read_bids_event_annotations(events_path: Path) -> mne.Annotations:
    """Read BIDS onset, duration, and event labels as MNE annotations."""
    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []

    with events_path.open(encoding="utf-8-sig", newline="") as events_file:
        reader = csv.DictReader(events_file, delimiter="\t")
        fields = set(reader.fieldnames or ())
        if "onset" not in fields:
            raise ValueError("events.tsv is missing the required 'onset' column")
        if not fields.intersection({"trial_type", "value"}):
            raise ValueError("events.tsv needs a 'trial_type' or 'value' label column")

        for line_number, row in enumerate(reader, start=2):
            onset_text = _bids_value(row.get("onset"))
            if onset_text is None:
                logger.warning("Skipping event without an onset in {} at line {}", events_path, line_number)
                continue

            try:
                onset = float(onset_text)
                duration = float(_bids_value(row.get("duration")) or 0.0)
            except ValueError:
                logger.warning("Skipping event with invalid timing in {} at line {}", events_path, line_number)
                continue

            if not math.isfinite(onset) or not math.isfinite(duration) or duration < 0:
                logger.warning("Skipping event with invalid timing in {} at line {}", events_path, line_number)
                continue

            description = _bids_value(row.get("trial_type")) or _bids_value(row.get("value"))
            if description is None:
                logger.warning("Skipping event without a label in {} at line {}", events_path, line_number)
                continue

            onsets.append(onset)
            durations.append(duration)
            descriptions.append(description)

    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


BIDS_CHANNEL_TYPES = {
    "BIO": "bio",
    "DBS": "dbs",
    "ECG": "ecg",
    "ECOG": "ecog",
    "EDA": "eda",
    "EEG": "eeg",
    "EMG": "emg",
    "EOG": "eog",
    "EYEGAZE": "eyegaze",
    "MISC": "misc",
    "RESP": "resp",
    "SEEG": "seeg",
    "TRIG": "stim",
}


def apply_bids_channels_metadata(raw: mne.io.BaseRaw, channels_path: Path) -> None:
    """Apply channel types and bad-channel status from a BIDS channels.tsv."""
    channel_types: dict[str, str] = {}
    bad_channels = set(raw.info["bads"])

    with channels_path.open(encoding="utf-8-sig", newline="") as channels_file:
        reader = csv.DictReader(channels_file, delimiter="\t")
        if "name" not in set(reader.fieldnames or ()):
            raise ValueError("channels.tsv is missing the required 'name' column")

        for line_number, row in enumerate(reader, start=2):
            name = _bids_value(row.get("name"))
            if name is None or name not in raw.ch_names:
                logger.warning(
                    "Ignoring unknown channel in {} at line {}: {}",
                    channels_path,
                    line_number,
                    name or "<missing>",
                )
                continue

            bids_type = (_bids_value(row.get("type")) or "").upper()
            if bids_type:
                mne_type = BIDS_CHANNEL_TYPES.get(bids_type)
                if mne_type is None:
                    logger.warning("Ignoring unsupported BIDS channel type '{}' for '{}'", bids_type, name)
                else:
                    channel_types[name] = mne_type

            if (_bids_value(row.get("status")) or "").lower() == "bad":
                bad_channels.add(name)

    if channel_types:
        raw.set_channel_types(channel_types, on_unit_change="ignore", verbose=False)
    raw.info["bads"] = sorted(bad_channels)


def apply_bids_eeg_metadata(raw: mne.io.BaseRaw, eeg_json_path: Path) -> None:
    """Apply portable recording metadata from a BIDS EEG JSON sidecar."""
    with eeg_json_path.open(encoding="utf-8-sig") as eeg_file:
        metadata = json.load(eeg_file)

    power_line_frequency = metadata.get("PowerLineFrequency")
    if isinstance(power_line_frequency, (int, float)) and math.isfinite(power_line_frequency):
        raw.info["line_freq"] = float(power_line_frequency)


def convert_bids_dataset(
    bids_root: Path,
    output_dir: Path,
    *,
    loader_cls=Loader,
    exporter_cls=EDFExporter,
) -> int:
    """Convert all supported recordings, continuing after per-file failures."""
    if not bids_root.is_dir():
        logger.error("BIDS root is not a directory: {}", bids_root)
        return 1

    recordings = discover_bids_eeg_recordings(bids_root)
    if not recordings:
        logger.warning("No supported BIDS EEG recordings found under {}", bids_root)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    reserved_outputs: set[Path] = set()

    for recording in recordings:
        output_path = output_dir / f"{recording.stem}.edf"
        if output_path in reserved_outputs:
            skipped += 1
            logger.warning(
                "Skipping '{}' because its BIDS-derived output filename collides with another recording: {}",
                recording,
                output_path,
            )
            continue
        reserved_outputs.add(output_path)

        logger.info("Converting BIDS EEG recording '{}' -> '{}'", recording, output_path)
        try:
            context = loader_cls(path=str(recording), preload=True).execute(None)
            metadata_paths = matching_bids_metadata(recording)
            located_metadata = [path.name for path in metadata_paths.values() if path.is_file()]
            if located_metadata:
                logger.info("Located BIDS metadata for '{}': {}", recording, ", ".join(located_metadata))

            events_path = metadata_paths["events"]
            if events_path.is_file():
                annotations = read_bids_event_annotations(events_path)
                context.get_raw().set_annotations(annotations)
                logger.info("Imported {} event annotation(s) from '{}'", len(annotations), events_path)
            else:
                logger.warning(
                    "No matching events.tsv for '{}'; exporting without BIDS event annotations",
                    recording,
                )

            channels_path = metadata_paths["channels"]
            if channels_path.is_file():
                apply_bids_channels_metadata(context.get_raw(), channels_path)

            eeg_json_path = metadata_paths["eeg"]
            if eeg_json_path.is_file():
                apply_bids_eeg_metadata(context.get_raw(), eeg_json_path)

            exporter_cls(path=str(output_path), overwrite=True).execute(context)
            converted += 1
        except Exception as exc:
            skipped += 1
            logger.warning("Skipping incomplete or unreadable recording '{}': {}", recording, exc)

    logger.info("BIDS conversion finished: {} converted, {} skipped", converted, skipped)
    return 0 if converted > 0 and skipped == 0 else 1


def build_convert_bids_parser() -> argparse.ArgumentParser:
    """Build the dedicated BIDS-to-EDF+ command parser."""
    parser = argparse.ArgumentParser(
        prog="facetpy-convert-bids",
        description="Recursively convert all BIDS EEG recordings to FACETpy-compatible EDF+ files.",
    )
    parser.add_argument("--bids-root", required=True, help="BIDS dataset root directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for converted EDF+ files.")
    return parser


def convert_bids_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``facetpy-convert-bids``."""
    args = build_convert_bids_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return convert_bids_dataset(
        Path(args.bids_root).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
    )
