"""Fixed downstream cleaning, usable independently of the Optuna runner.

Only existing FACETpy processors perform correction. This module neither
constructs nor tunes the scanner averaging matrix.
"""

from dataclasses import asdict, dataclass

import mne
import numpy as np
from loguru import logger

from facet import AASCorrection, BandPassFilter, PCACorrection, QRSTriggerDetector, RawTransform, Resample
from facet.core import ProcessingContext


@dataclass(frozen=True)
class CleaningSettings:
    """Immutable settings shared by every trial, including screening.

    The 1–70 Hz passband follows FACETpy's complete-pipeline example. The
    200 Hz target retains that passband with room for anti-alias filtering.
    ANC is disabled: the metrics path does not retain a validated noise
    reference. ASR and automated ICA rejection are unavailable in FACETpy.
    """

    sfreq: float = 200.0
    l_freq: float = 1.0
    h_freq: float = 70.0
    pca_components: float = 0.95
    pca_hp_freq: float | None = None
    bcg_window: int = 10
    bcg_threshold: float = 0.975
    qrs_l_freq: float = 1.0
    qrs_h_freq: float = 40.0

    def manifest(self) -> dict:
        return {
            "version": 2,
            "parameters": asdict(self),
            "ecg_scanner_matrix": "Reuse the first good EEG channel's candidate scanner matrix; never score cardiac similarity",
            "qrs_detection_interval": "First MRI trigger through last complete artifact window; exclude pre-acquisition padding; restore event indices to original timeline",
            "order": [
                "pca_correction",
                "ecg_detection_bandpass",
                "qrs_detection",
                "heartbeat_aas",
                "resample",
                "bandpass",
                "exclude_marked_bad_channels",
                "average_reference",
            ],
            "anc": {"enabled": False, "reason": "No validated retained ANC noise reference"},
            "asr": {"enabled": False, "reason": "No FACETpy automated ASR processor"},
            "ica": {"enabled": False, "reason": "No FACETpy ICA rejection processor"},
            "bad_segments": "Preserve timeline; exclude existing BAD annotations during comparison",
            "bad_channels": "Honor existing marks; no new automatic detection or interpolation",
            "reference_convention": "Average over each recording's good EEG; compare on shared good EEG",
            "chunking": "Run on padded chunks; score cores only; not equivalent to whole-record filtering",
        }


DEFAULT_CLEANING = CleaningSettings()


def standardize_auxiliary_types(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Recognize explicitly named auxiliary channels without renaming EEG."""
    result = raw.copy()
    types = {}
    for name in result.ch_names:
        if name.upper() in {"ECG", "EKG"}:
            types[name] = "ecg"
        elif name.upper() in {"EOGL", "EOGU", "EOG"}:
            types[name] = "eog"
    if types:
        result.set_channel_types(types, verbose=False)
    return result


def good_eeg_names(raw: mne.io.BaseRaw) -> list[str]:
    return [
        raw.ch_names[p]
        for p in mne.pick_types(raw.info, eeg=True, exclude="bads")
        if raw.ch_names[p].upper() not in {"ECG", "EKG", "EOGL", "EOGU", "EOG"}
    ]


def validate_cleaning_input(raw: mne.io.BaseRaw, settings: CleaningSettings = DEFAULT_CLEANING) -> None:
    """Fail before trials when fixed preprocessing cannot run."""
    if len(good_eeg_names(raw)) < 2:
        raise ValueError("Fixed cleaning requires at least two good EEG channels for average reference.")
    ecg = mne.pick_types(raw.info, ecg=True, exclude="bads")
    named = [n for n in raw.ch_names if n.upper() in {"ECG", "EKG"} and n not in raw.info["bads"]]
    if not len(ecg) and not named:
        raise ValueError("Fixed BCG cleaning requires a good ECG/EKG channel; none is available.")
    if min(float(raw.info["sfreq"]), settings.sfreq) <= 2 * settings.h_freq:
        raise ValueError("Recording or cleaning sampling rate cannot support the fixed 1–70 Hz passband.")


def _pick_good_eeg(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    return raw.copy().pick(good_eeg_names(raw))


def average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    result = _pick_good_eeg(raw)
    return result.set_eeg_reference("average", projection=False, verbose=False)


def _detect_qrs_in_acquisition(context, detector, settings):
    """Use the acquisition ECG, excluding uncorrected pre-scanner padding.

    The output EEG timeline is unchanged. Map detected event indices back to
    that timeline explicitly before heartbeat subtraction.
    """
    raw = context.get_raw()
    triggers = context.get_triggers()
    start = max(0, int(triggers[0]))
    stop = min(raw.n_times, int(triggers[-1]) + context.get_artifact_length())
    cropped = raw.copy().crop(start / raw.info["sfreq"], (stop - 1) / raw.info["sfreq"])
    detection = BandPassFilter(settings.qrs_l_freq, settings.qrs_h_freq, picks="ecg").execute(context.with_raw(cropped))
    detected = detector.execute(detection)
    metadata = detected.metadata.copy()
    metadata.triggers = detected.get_triggers() + start
    metadata.custom["qrs_detection"] = {"start_sample": start, "stop_sample": stop, "count": len(metadata.triggers)}
    return detected.with_raw(raw).with_metadata(metadata)


def clean_after_flex(context: ProcessingContext, settings: CleaningSettings = DEFAULT_CLEANING) -> ProcessingContext:
    """Apply fixed FACETpy processors to a Flex-corrected multichannel context.

    MRI triggers must still be present for PCA. QRS detection subsequently
    replaces them for heartbeat AAS, which is distinct from scanner correction.
    """
    validate_cleaning_input(context.get_raw(), settings)
    context = context.with_raw(standardize_auxiliary_types(context.get_raw()))
    stages = [
        PCACorrection(n_components=settings.pca_components, hp_freq=settings.pca_hp_freq, track_estimated_noise=False),
        QRSTriggerDetector(),
        AASCorrection(
            window_size=settings.bcg_window,
            correlation_threshold=settings.bcg_threshold,
            realign_after_averaging=False,
            track_estimated_noise=False,
        ),
        Resample(settings.sfreq),
        BandPassFilter(settings.l_freq, settings.h_freq, picks="eeg"),
        RawTransform("exclude_marked_bad_channels", _pick_good_eeg),
        RawTransform("average_reference", average_reference),
    ]
    # Existing PCA logs per-channel errors and continues. Treat those as a
    # failed evaluation without changing its correction algorithm.
    errors = []
    sink = logger.add(lambda message: errors.append(str(message)), level="ERROR")
    try:
        for stage in stages:
            context = (
                _detect_qrs_in_acquisition(context, stage, settings)
                if isinstance(stage, QRSTriggerDetector)
                else stage.execute(context)
            )
            if errors:
                raise ValueError(f"Fixed cleaning failed in {stage.name}: {errors[-1]}")
            if isinstance(stage, QRSTriggerDetector) and len(context.get_triggers()) < 3:
                raise ValueError("BCG cleaning requires at least three detected QRS events.")
            if not np.all(np.isfinite(context.get_raw().get_data())):
                raise ValueError(f"Non-finite output from {stage.name}.")
    finally:
        logger.remove(sink)
    metadata = context.metadata.copy()
    metadata.custom["fixed_cleaning"] = settings.manifest()
    return context.with_metadata(metadata)
