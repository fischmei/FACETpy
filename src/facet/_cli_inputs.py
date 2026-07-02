"""Input discovery and path resolution helpers for FACETpy CLI commands."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from ._cli_labels import _compact_label


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
        if (path.is_file() or path.suffix.lower() == ".mff")
        and _is_supported_eeg_path(path, extensions)
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


def _source_output_dir(
    input_path: Path,
    output_root: Path,
    total_inputs: int,
    flat_output: bool,
) -> Path:
    """Return the output folder for one source recording."""
    if total_inputs == 1 or flat_output:
        return output_root
    return output_root / _compact_label(input_path.stem)
