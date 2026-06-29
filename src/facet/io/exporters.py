"""
Data Exporters Module

This module contains processors for exporting EEG data to various formats.

Author: FACETpy Team
Date: 2025-01-12
"""

from pathlib import Path

import mne
import numpy as np
from loguru import logger
from mne_bids import BIDSPath, write_raw_bids

from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor


def _detect_export_extension(path: Path) -> str:
    """Detect the export extension from a target path."""
    suffixes = path.suffixes
    if len(suffixes) >= 2 and suffixes[-2] == ".fif" and suffixes[-1] == ".gz":
        return ".fif"
    return path.suffix.lower()


def _resolve_exporter_class(path: Path) -> type[Processor]:
    """Resolve the processor class responsible for a file extension."""
    ext = _detect_export_extension(path)
    if ext in _EXTENSION_EXPORTERS:
        return _EXTENSION_EXPORTERS[ext]
    raise ProcessorValidationError(
        f"Unsupported export extension '{path.suffix.lower()}' for '{path.name}'. "
        f"Supported extensions: {', '.join(SUPPORTED_EXPORT_EXTENSIONS)}."
    )


@register_processor
class EDFExporter(Processor):
    """Export EEG data to EDF format.

    Writes the current Raw object to an EDF file at the specified path.
    Parent directories are created automatically if they do not exist.
    The context is returned unchanged; only the file is written.

    Parameters
    ----------
    path : str
        Destination file path for the exported EDF.
    overwrite : bool, optional
        Whether to overwrite an existing file (default: True).
    """

    name = "edf_exporter"
    description = "Export EEG data to EDF file"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        path: str,
        overwrite: bool = True,
    ) -> None:
        self.path = path
        self.overwrite = overwrite
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()

        # --- LOG ---
        logger.info("Exporting to EDF: {}", self.path)

        # --- COMPUTE ---
        # EDF header subfield equipment_code (raw.info['device_info']['type'])
        # must not contain spaces per the EDF spec; replace them with underscores.
        device_info = raw.info.get("device_info")
        if device_info is not None:
            device_type = device_info.get("type") or ""
            if " " in device_type:
                with raw.info._unlock():
                    raw.info["device_info"]["type"] = device_type.replace(" ", "_")
                logger.debug(
                    "Sanitized device_info.type for EDF export: '{}' -> '{}'",
                    device_type,
                    raw.info["device_info"]["type"],
                )

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        raw.export(self.path, fmt="edf", overwrite=self.overwrite)

        logger.info("Export completed")

        # --- RETURN ---
        return context


@register_processor
class BDFExporter(Processor):
    """Export EEG data to BDF format.

    Parameters
    ----------
    path : str
        Destination file path for the exported BDF.
    overwrite : bool, optional
        Whether to overwrite an existing file (default: True).
    """

    name = "bdf_exporter"
    description = "Export EEG data to BDF file"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, path: str, overwrite: bool = True) -> None:
        self.path = path
        self.overwrite = overwrite
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        logger.info("Exporting to BDF: {}", self.path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        raw.export(self.path, fmt="bdf", overwrite=self.overwrite)
        logger.info("Export completed")
        return context


@register_processor
class BrainVisionExporter(Processor):
    """Export EEG data to BrainVision format.

    Parameters
    ----------
    path : str
        Destination BrainVision header path (``.vhdr``).
    overwrite : bool, optional
        Whether to overwrite an existing file set (default: True).
    """

    name = "brainvision_exporter"
    description = "Export EEG data to BrainVision file set"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, path: str, overwrite: bool = True) -> None:
        self.path = path
        self.overwrite = overwrite
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        logger.info("Exporting to BrainVision: {}", self.path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        raw.export(self.path, fmt="brainvision", overwrite=self.overwrite)
        logger.info("Export completed")
        return context


@register_processor
class EEGLABExporter(Processor):
    """Export EEG data to MATLAB EEGLAB format.

    Parameters
    ----------
    path : str
        Destination file path for the exported EEGLAB ``.set`` file.
    overwrite : bool, optional
        Whether to overwrite an existing file (default: True).
    """

    name = "eeglab_exporter"
    description = "Export EEG data to MATLAB EEGLAB file"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, path: str, overwrite: bool = True) -> None:
        self.path = path
        self.overwrite = overwrite
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        logger.info("Exporting to EEGLAB (.set): {}", self.path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        raw.export(self.path, fmt="eeglab", overwrite=self.overwrite)
        logger.info("Export completed")
        return context


@register_processor
class FIFExporter(Processor):
    """Export EEG data to FIF format.

    Parameters
    ----------
    path : str
        Destination file path for the exported FIF file (``.fif`` or ``.fif.gz``).
    overwrite : bool, optional
        Whether to overwrite an existing file (default: True).
    """

    name = "fif_exporter"
    description = "Export EEG data to FIF file"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, path: str, overwrite: bool = True) -> None:
        self.path = path
        self.overwrite = overwrite
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        logger.info("Exporting to FIF: {}", self.path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        raw.save(self.path, overwrite=self.overwrite, verbose=False)
        logger.info("Export completed")
        return context


@register_processor
class GDFExporter(Processor):
    """Route target for GDF exports.

    MNE currently does not provide GDF writing support.
    """

    name = "gdf_exporter"
    description = "Export EEG data to GDF file (unsupported in current runtime)"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, path: str, overwrite: bool = True) -> None:
        self.path = path
        self.overwrite = overwrite
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        raise ProcessorValidationError(
            "GDF export is not supported by MNE 1.10.2. "
            "Use EDF (.edf), BDF (.bdf), BrainVision (.vhdr), EEGLAB (.set), or FIF (.fif/.fif.gz)."
        )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        return context


@register_processor
class MFFExporter(Processor):
    """Route target for EGI MFF exports.

    MNE currently does not provide MFF writing support.
    """

    name = "mff_exporter"
    description = "Export EEG data to MFF directory (unsupported in current runtime)"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, path: str, overwrite: bool = True) -> None:
        self.path = path
        self.overwrite = overwrite
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        raise ProcessorValidationError(
            "MFF export is not supported by MNE 1.10.2. "
            "Use EDF (.edf), BDF (.bdf), BrainVision (.vhdr), EEGLAB (.set), or FIF (.fif/.fif.gz)."
        )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        return context


_EXTENSION_EXPORTERS: dict[str, type[Processor]] = {
    ".edf": EDFExporter,
    ".bdf": BDFExporter,
    ".gdf": GDFExporter,
    ".vhdr": BrainVisionExporter,
    ".set": EEGLABExporter,
    ".fif": FIFExporter,
    ".mff": MFFExporter,
}

SUPPORTED_EXPORT_EXTENSIONS: list[str] = sorted([*list(_EXTENSION_EXPORTERS.keys()), ".fif.gz"])


@register_processor
class Exporter(Processor):
    """Export EEG data with automatic file-format routing.

    Routes export requests to the file-type-specific exporter based on
    the destination extension.

    Parameters
    ----------
    path : str
        Destination path; extension determines exporter selection.
    overwrite : bool, optional
        Whether to overwrite existing outputs (default: True).
    """

    name = "auto_exporter"
    description = "Export EEG data with automatic format detection"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, path: str, overwrite: bool = True) -> None:
        self.path = path
        self.overwrite = overwrite
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        _resolve_exporter_class(Path(self.path))

    def process(self, context: ProcessingContext) -> ProcessingContext:
        destination = Path(self.path)
        exporter_class = _resolve_exporter_class(destination)
        logger.info("Routing export '{}' to {}", destination.suffix.lower(), exporter_class.__name__)
        exporter = exporter_class(path=self.path, overwrite=self.overwrite)
        return exporter.execute(context)


@register_processor
class BIDSExporter(Processor):
    """Export EEG data to BIDS format.

    Writes the current Raw object into a BIDS-compliant directory structure
    using MNE-BIDS. Stimulus channels are dropped before writing as per BIDS
    convention. If triggers are available in the context they are written as
    events. Parent directories are created automatically.

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
        Run identifier used when exporting multiple segments for the same
        subject/task.
    event_id : dict, optional
        Mapping of event description strings to integer event codes.
    overwrite : bool, optional
        Whether to overwrite existing BIDS files (default: True).
    """

    name = "bids_exporter"
    description = "Export EEG data to BIDS dataset"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        root: str,
        subject: str,
        task: str,
        session: str | None = None,
        run: str | None = None,
        event_id: dict | None = None,
        overwrite: bool = True,
    ) -> None:
        self.root = root
        self.subject = subject
        self.task = task
        self.session = session
        self.run = run
        self.event_id = event_id
        self.overwrite = overwrite
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()

        # --- LOG ---
        logger.info(
            "Exporting to BIDS: subject={}, task={}, run={}",
            self.subject,
            self.task,
            self.run,
        )

        # --- COMPUTE ---
        Path(self.root).mkdir(parents=True, exist_ok=True)

        bids_path = BIDSPath(
            subject=self.subject,
            session=self.session,
            task=self.task,
            run=self.run,
            root=self.root,
        )

        # Drop stim channels (BIDS convention)
        stim_channels = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)
        if len(stim_channels) > 0:
            raw.drop_channels([raw.ch_names[ch] for ch in stim_channels])

        events = None
        if context.has_triggers():
            triggers = context.get_triggers()
            events = np.array([[t, 0, 1] for t in triggers], dtype=np.int32)

        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            overwrite=self.overwrite,
            format="EDF",
            allow_preload=True,
            events=events,
            event_id=self.event_id,
            verbose=False,
        )

        logger.info("Export completed")

        # --- RETURN ---
        return context
