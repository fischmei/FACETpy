"""
Data Loaders Module

This module contains processors for loading EEG data from various formats.

Author: FACETpy Team
Date: 2025-01-12
"""

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import mne
from loguru import logger
from mne.io import BaseRaw
from mne_bids import BIDSPath, read_raw_bids

from facet.logging_config import suppress_stdout

from ..core import (
    ProcessingContext,
    ProcessingMetadata,
    Processor,
    ProcessorValidationError,
    register_processor,
)

_EXTENSION_READERS: dict[str, tuple[Callable[..., BaseRaw], str]] = {
    ".edf": (mne.io.read_raw_edf, "EDF"),
    ".bdf": (mne.io.read_raw_bdf, "BDF"),
    ".gdf": (mne.io.read_raw_gdf, "GDF"),
    ".vhdr": (mne.io.read_raw_brainvision, "BrainVision"),
    ".set": (mne.io.read_raw_eeglab, "EEGLAB"),
    ".fif": (mne.io.read_raw_fif, "FIF"),
    ".mff": (mne.io.read_raw_egi, "MFF"),
}

SUPPORTED_EXTENSIONS: list[str] = sorted(_EXTENSION_READERS.keys())


def _ensure_mff_runtime_dependencies() -> None:
    """Ensure optional dependencies required by MFF loading are usable.

    Some dependency combinations ship ``defusedxml`` without a module-level
    ``__version__`` attribute while older MNE code still expects it.
    """
    try:
        import defusedxml  # type: ignore
    except ImportError as exc:
        raise ProcessorValidationError(
            "Missing dependency for .mff loading: defusedxml. Install with `uv sync` or `pip install defusedxml`."
        ) from exc

    if getattr(defusedxml, "__version__", None):
        return

    try:
        defusedxml.__version__ = pkg_version("defusedxml")
    except PackageNotFoundError:
        defusedxml.__version__ = "unknown"


def _detect_format(path: Path) -> tuple[Callable[..., BaseRaw], str]:
    """Detect the EEG file format from the file extension.

    Parameters
    ----------
    path : Path
        Path to the EEG data file (or directory for MFF format).

    Returns
    -------
    tuple of (callable, str)
        The MNE reader function and a human-readable format name.

    Raises
    ------
    ProcessorValidationError
        If the extension is not recognized.
    """
    suffixes = path.suffixes
    if len(suffixes) >= 2 and suffixes[-2] == ".fif" and suffixes[-1] == ".gz":
        return _EXTENSION_READERS[".fif"]

    ext = path.suffix.lower()
    if ext in _EXTENSION_READERS:
        return _EXTENSION_READERS[ext]

    raise ProcessorValidationError(
        f"Unsupported file extension '{ext}' for '{path.name}'. "
        f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}. "
        f"For BIDS datasets, use BIDSLoader directly."
    )


def _coerce_sample_index(value: float | None, default: int, name: str) -> int:
    """Convert a potentially optional numeric input into a valid sample index."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer-like value, got boolean")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        if not float(value).is_integer():
            raise ValueError(f"{name} must be an integer number of samples")
        return int(value)
    raise ValueError(f"{name} must be an integer number of samples")


def _apply_sample_window(
    raw: BaseRaw,
    start_sample: int | None,
    stop_sample: int | None,
) -> tuple[BaseRaw, int, int]:
    """Restrict a Raw object to the requested sample window."""
    n_times = raw.n_times
    if n_times == 0:
        raise ValueError("Cannot apply a sample window to an empty recording")

    start = _coerce_sample_index(start_sample, 0, "start_sample")
    stop = _coerce_sample_index(stop_sample, n_times, "stop_sample")

    if start < 0:
        raise ValueError("start_sample must be non-negative")
    if start >= n_times:
        raise ValueError(f"start_sample ({start}) is out of bounds for data with {n_times} samples")
    if stop <= start:
        raise ValueError(f"stop_sample ({stop}) must be greater than start_sample ({start})")
    if stop > n_times:
        raise ValueError(f"stop_sample ({stop}) exceeds total samples ({n_times})")

    if start > 0 or stop < n_times:
        sfreq = raw.info["sfreq"]
        tmin = start / sfreq
        tmax = (stop - 1) / sfreq
        raw.crop(tmin=tmin, tmax=tmax, verbose=False)

    return raw, start, stop


def _configure_bad_channels(raw: mne.io.Raw, bad_channels: list[str]) -> None:
    """Mark specified channels as bad in the Raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        The Raw object to update in-place.
    bad_channels : list of str
        Channel names to mark as bad.
    """
    if not bad_channels:
        return

    existing = set(raw.ch_names)
    valid_bads = [ch for ch in bad_channels if ch in existing]
    missing = set(bad_channels) - existing

    if missing:
        logger.warning(
            "Skipping {} bad channel(s) not present in data: {}",
            len(missing),
            ", ".join(sorted(missing)),
        )
        logger.debug("Available channels: {}", ", ".join(raw.ch_names))

    raw.info["bads"] = valid_bads

    if valid_bads:
        logger.debug("Marked {} bad channel(s): {}", len(valid_bads), ", ".join(valid_bads))
    else:
        logger.info("No valid bad channels found to mark; leaving dataset unchanged.")


def _build_context_from_raw(
    raw: BaseRaw,
    bad_channels: list[str],
    start_sample: int | None,
    stop_sample: int | None,
    artifact_to_trigger_offset: float,
    upsampling_factor: int,
    preload_after_crop: bool = False,
    extra_custom: dict[str, Any] | None = None,
) -> ProcessingContext:
    """Post-read pipeline shared by all loaders.

    Applies bad-channel marking, sample windowing, metadata construction,
    and result logging.

    Parameters
    ----------
    raw : BaseRaw
        The MNE Raw object returned by the format-specific reader.
    bad_channels : list of str
        Channel names to mark as bad.
    start_sample : int or None
        First sample to keep (inclusive).
    stop_sample : int or None
        Last sample to keep (exclusive).
    artifact_to_trigger_offset : float
        Offset in seconds stored in metadata.
    upsampling_factor : int
        Upsampling factor stored in metadata.
    preload_after_crop : bool, optional
        If ``True``, load data into memory after applying the sample window.
        This is useful for large recordings: callers can open the file lazily,
        crop to a chunk, and only then materialize that chunk.
    extra_custom : dict, optional
        Additional entries for ``metadata.custom``.

    Returns
    -------
    ProcessingContext
        New context wrapping the (possibly cropped) Raw object.
    """
    _configure_bad_channels(raw, bad_channels)

    full_n_times = raw.n_times
    try:
        raw, start_idx, stop_idx = _apply_sample_window(raw, start_sample, stop_sample)
    except ValueError as exc:
        raise ProcessorValidationError(f"Invalid sample window: {exc}") from exc

    if start_idx != 0 or stop_idx != full_n_times:
        logger.info(
            "Applied sample window: start={}, stop={} (exclusive), kept {}/{} samples",
            start_idx,
            stop_idx,
            raw.n_times,
            full_n_times,
        )

    if preload_after_crop and not raw.preload:
        logger.debug("Preloading data after sample-window crop")
        raw.load_data(verbose=False)

    acq_start = start_idx if start_sample is not None else None
    acq_end = stop_idx if stop_sample is not None else None

    metadata = ProcessingMetadata(
        artifact_to_trigger_offset=artifact_to_trigger_offset,
        upsampling_factor=upsampling_factor,
        acq_start_sample=acq_start,
        acq_end_sample=acq_end,
    )
    if extra_custom:
        metadata.custom.update(extra_custom)

    logger.info(
        "Loaded {} channels, {} samples, {} Hz",
        len(raw.ch_names),
        raw.n_times,
        raw.info["sfreq"],
    )

    return ProcessingContext(raw=raw, metadata=metadata)


@register_processor
class Loader(Processor):
    """Load EEG data with automatic file-format detection.

    Inspects the file extension and selects the appropriate MNE reader.
    Supports EDF, BDF, GDF, BrainVision (.vhdr), EEGLAB (.set),
    FIF (.fif / .fif.gz), and EGI MFF (.mff). For BIDS datasets use
    :class:`BIDSLoader`. Note: MFF format uses a directory structure;
    pass the path to the .mff directory (e.g. recording.mff).

    Parameters
    ----------
    path : str
        Path to the EEG data file.
    bad_channels : list of str, optional
        Channel names to mark as bad (default: none).
    preload : bool, optional
        Whether to load data into memory immediately (default: True).
    artifact_to_trigger_offset : float, optional
        Offset of the artifact relative to the trigger, in seconds (default: 0.0).
    upsampling_factor : int, optional
        Upsampling factor stored in metadata for downstream processors (default: 1).
    start_sample : int, optional
        First sample index to keep, inclusive (default: first sample).
    stop_sample : int, optional
        Last sample index to keep, exclusive (default: last sample).
    """

    name: str = "auto_loader"
    description: str = "Load EEG data with automatic format detection"
    version: str = "1.0.0"

    requires_triggers: bool = False
    requires_raw: bool = False
    modifies_raw: bool = True
    parallel_safe: bool = False

    def __init__(
        self,
        path: str,
        bad_channels: list[str] | None = None,
        preload: bool = True,
        artifact_to_trigger_offset: float = 0.0,
        upsampling_factor: int = 1,
        start_sample: int | None = None,
        stop_sample: int | None = None,
    ) -> None:
        self.path = path
        self.bad_channels = bad_channels or []
        self.preload = preload
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.upsampling_factor = upsampling_factor
        self.start_sample = start_sample
        self.stop_sample = stop_sample
        super().__init__()

    def validate(self, context: ProcessingContext | None) -> None:
        resolved = Path(self.path)
        if not resolved.exists():
            raise ProcessorValidationError(f"File not found: {self.path}")
        if resolved.is_dir() and resolved.suffix.lower() != ".mff":
            raise ProcessorValidationError(
                f"Path is a directory: {self.path}. "
                "For BIDS datasets, use BIDSLoader. For MFF format, the path must "
                "have a .mff extension (e.g. recording.mff)."
            )
        _detect_format(resolved)

    def process(self, context: ProcessingContext | None) -> ProcessingContext:
        # --- EXTRACT ---
        resolved = Path(self.path)
        reader_fn, format_name = _detect_format(resolved)
        has_sample_window = self.start_sample is not None or self.stop_sample is not None

        # --- LOG ---
        logger.info("Loading {} file: {}", format_name, self.path)

        # --- COMPUTE ---
        if format_name == "MFF":
            _ensure_mff_runtime_dependencies()

        with suppress_stdout():
            raw = reader_fn(str(resolved), preload=self.preload and not has_sample_window, verbose=False)

        # --- RETURN ---
        return _build_context_from_raw(
            raw,
            self.bad_channels,
            self.start_sample,
            self.stop_sample,
            self.artifact_to_trigger_offset,
            self.upsampling_factor,
            preload_after_crop=self.preload and has_sample_window,
            extra_custom={"source_format": format_name},
        )


@register_processor
class BIDSLoader(Processor):
    """Load EEG data from a BIDS dataset.

    Creates a new ProcessingContext by reading EEG data identified by
    subject, task, and optional session from a BIDS-compliant directory.
    Optionally restricts the recording to a sample window and marks
    a list of channels as bad before returning.

    Parameters
    ----------
    root : str
        Path to the BIDS root directory.
    subject : str
        Subject identifier (without the ``sub-`` prefix).
    task : str
        Task name.
    session : str, optional
        Session identifier (without the ``ses-`` prefix).
    run : str, optional
        Run identifier (without the ``run-`` prefix).
    bad_channels : list of str, optional
        Channel names to mark as bad (default: none).
    preload : bool, optional
        Whether to load data into memory immediately (default: True).
    artifact_to_trigger_offset : float, optional
        Offset of the artifact relative to the trigger, in seconds (default: 0.0).
    upsampling_factor : int, optional
        Upsampling factor stored in metadata for downstream processors (default: 1).
    start_sample : int, optional
        First sample index to keep, inclusive (default: first sample).
    stop_sample : int, optional
        Last sample index to keep, exclusive (default: last sample).
    """

    name = "bids_loader"
    description = "Load EEG data from BIDS dataset"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = False
    modifies_raw = True
    parallel_safe = False

    def __init__(
        self,
        root: str,
        subject: str,
        task: str,
        session: str | None = None,
        run: str | None = None,
        bad_channels: list[str] | None = None,
        preload: bool = True,
        artifact_to_trigger_offset: float = 0.0,
        upsampling_factor: int = 1,
        start_sample: int | None = None,
        stop_sample: int | None = None,
    ) -> None:
        self.root = root
        self.subject = subject
        self.task = task
        self.session = session
        self.run = run
        self.bad_channels = bad_channels or []
        self.preload = preload
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.upsampling_factor = upsampling_factor
        self.start_sample = start_sample
        self.stop_sample = stop_sample
        super().__init__()

    def validate(self, context: ProcessingContext | None) -> None:
        if not Path(self.root).exists():
            raise ProcessorValidationError(f"BIDS root directory not found: {self.root}")

    def process(self, context: ProcessingContext | None) -> ProcessingContext:
        # --- LOG ---
        logger.info(
            "Loading BIDS data: subject={}, task={}, session={}, run={}",
            self.subject,
            self.task,
            self.session,
            self.run,
        )

        # --- COMPUTE ---
        bids_path = BIDSPath(
            subject=self.subject,
            session=self.session,
            task=self.task,
            run=self.run,
            root=self.root,
        )

        with suppress_stdout():
            raw = read_raw_bids(bids_path, verbose=False)

        has_sample_window = self.start_sample is not None or self.stop_sample is not None
        if self.preload and not has_sample_window:
            raw.load_data()

        # --- RETURN ---
        return _build_context_from_raw(
            raw,
            self.bad_channels,
            self.start_sample,
            self.stop_sample,
            self.artifact_to_trigger_offset,
            self.upsampling_factor,
            preload_after_crop=self.preload and has_sample_window,
            extra_custom={"bids_path": bids_path},
        )
