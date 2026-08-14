"""Self-contained scientific HTML reports for ``facetpy-run process``.

The CLI deliberately releases each processed chunk's signal arrays to keep
large EEG-fMRI runs memory bounded.  This module therefore reloads a small,
stratified set of exact exported *core* windows and their matching source
windows after correction.  Static report images are embedded as data URIs so
the resulting HTML can be moved or archived without a companion asset
directory.  Deliberately avoiding animation keeps report generation and later
browser rendering bounded.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import platform
import re
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fractions import Fraction
from html import escape
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
from loguru import logger
from PIL import Image
from scipy import signal

from ._cli_reports import _chunked_results, _json_safe
from .io import Loader

MAX_ANALYSIS_SECONDS = 30.0
MAX_ANALYSIS_CHUNKS = 4
MAX_ANALYSIS_SFREQ_HZ = 500.0
MAX_ANALYSIS_VALUES_PER_PHASE = 4_000_000
MAX_COHERENCE_CHANNELS = 24
MAX_DISTRIBUTION_VALUES = 250_000
MAX_MATRIX_PREVIEW_BYTES = 180_000
MAX_MATRIX_PREVIEW_SIZE = (1_200, 900)
MIN_SPATIAL_CHANNELS = 8
NETWORK_EDGE_DENSITY = 0.15
MIN_COHERENCE_WINDOWS = 8
LOW_PRECISION_COHERENCE_WINDOWS = 20
COHERENCE_FMIN_HZ = 8.0
COHERENCE_FMAX_HZ = 13.0
MATRIX_ASSET_PLACEHOLDER = "<!-- FACETPY_STREAMED_FLEX_MATRIX_ASSETS -->"


@dataclass
class PairedRecordingData:
    """Bounded, sample-aligned source and corrected EEG windows."""

    before_segments: list[np.ndarray]
    after_segments: list[np.ndarray]
    channel_names: list[str]
    sfreq: float
    source_info: mne.Info
    source_duration_seconds: float
    analyzed_duration_seconds: float
    segment_windows: list[dict[str, float | int | str]]
    source_sfreq: float | None = None
    corrected_sfreq: float | None = None
    corrected_info: mne.Info | None = None
    source_bad_channels: list[str] = field(default_factory=list)
    corrected_bad_channels: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TemporalDiagnostics:
    """Frequency and amplitude summaries evaluated on paired windows."""

    frequencies: np.ndarray
    amplitude_before: np.ndarray
    amplitude_after: np.ndarray
    psd_before: np.ndarray
    psd_after: np.ndarray
    histogram_edges: np.ndarray
    histogram_before: np.ndarray
    histogram_after: np.ndarray
    peak_to_peak_before: np.ndarray
    peak_to_peak_after: np.ndarray
    clipped_before_fraction: float
    clipped_after_fraction: float
    frequency_resolution_hz: float
    welch_segment_seconds: float
    channel_names: list[str]


@dataclass
class SpatialGeometry:
    """EEG channels with defensible source-file or template sensor positions."""

    channel_indices: np.ndarray
    info: mne.Info
    coordinates_3d: np.ndarray
    origin: str
    coordinate_frame: str
    coverage_note: str


@dataclass
class CoherenceDiagnostics:
    """Band-averaged sensor coherence and its visualization graph."""

    before: np.ndarray
    after: np.ndarray
    channel_indices: np.ndarray
    channel_names: list[str]
    labels: np.ndarray
    thresholded_after: np.ndarray
    edge_threshold: float
    modularity: float
    frequencies: np.ndarray
    fmin: float
    fmax: float
    nperseg: int
    frame_count: int
    edge_density: float
    low_precision: bool


def _report_stem(input_path: Path) -> str:
    """Return a stable filename stem for one source recording."""
    name = input_path.name
    name = name[:-7] if name.lower().endswith(".fif.gz") else input_path.stem
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return sanitized or "recording"


def cleaning_report_path(
    target_dir: Path,
    input_path: Path,
    *,
    disambiguate: bool = False,
) -> Path:
    """Return the unique per-recording HTML report path."""
    stem = _report_stem(input_path)
    if disambiguate:
        source_key = str(input_path.expanduser().resolve()).encode("utf-8")
        stem = f"{stem}_{hashlib.sha256(source_key).hexdigest()[:8]}"
    return target_dir / f"{stem}_cleaning_report.html"


def _successful_chunk_pairs(chunked_result) -> list[tuple[Any, Any]]:
    """Return successful chunks whose corrected output exists."""
    chunks = list(getattr(chunked_result, "chunks", []))
    results = _chunked_results(chunked_result)
    pairs: list[tuple[Any, Any]] = []
    for chunk, result in zip(chunks, results, strict=False):
        if not bool(getattr(result, "success", False)):
            continue
        raw_output_path = getattr(chunk, "output_path", None)
        if raw_output_path is None:
            continue
        output_path = Path(raw_output_path)
        if output_path.exists():
            pairs.append((chunk, result))
    return pairs


def _stratified_pairs(pairs: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """Select evenly distributed chunks while bounding report I/O."""
    if len(pairs) <= MAX_ANALYSIS_CHUNKS:
        return pairs
    indices = np.linspace(0, len(pairs) - 1, MAX_ANALYSIS_CHUNKS)
    return [pairs[int(round(index))] for index in indices]


def _close_raw(raw) -> None:
    """Close a lazily loaded MNE object when its reader owns resources."""
    close = getattr(raw, "close", None)
    if callable(close):
        close()


def _eeg_channel_names(raw) -> list[str]:
    """Return non-bad EEG channel names in recording order."""
    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")
    return [raw.ch_names[int(index)] for index in picks]


def _analysis_windows(n_times: int, sfreq: float, quota_seconds: float) -> list[tuple[int, int]]:
    """Return up to three temporal strata whose total duration fits a quota."""
    available_seconds = n_times / sfreq
    selected_seconds = min(quota_seconds, available_seconds)
    if selected_seconds <= 0:
        return []

    if selected_seconds >= available_seconds:
        return [(0, n_times)]

    # Long cores are sampled near their beginning, middle, and end instead of
    # using only one centered excerpt.  Keep each stratum long enough for the
    # four-second Welch window whenever the quota permits it.
    window_count = min(3, max(1, int(selected_seconds // 4.0)))
    window_samples = max(1, int(math.floor(selected_seconds * sfreq / window_count)))
    window_samples = min(window_samples, n_times)
    starts = np.linspace(0, n_times - window_samples, window_count, dtype=int)
    return [(int(start), int(start + window_samples)) for start in np.unique(starts)]


def _analysis_quota_seconds(
    *,
    channel_count: int,
    pair_count: int,
    report_sfreq: float,
    source_sfreq: float,
    output_sfreq: float,
) -> float:
    """Bound both retained report arrays and native-rate reads per phase."""
    if channel_count <= 0 or pair_count <= 0:
        return 0.0

    highest_rate = max(report_sfreq, source_sfreq, output_sfreq)
    # Each selected core produces at most three windows.  Reserve one sample
    # per channel/window for integer endpoint rounding so the advertised cap
    # remains strict rather than approximate.
    rounding_guard = 3 * channel_count * pair_count
    available_values = max(1, MAX_ANALYSIS_VALUES_PER_PHASE - rounding_guard)
    return min(
        MAX_ANALYSIS_SECONDS / pair_count,
        available_values / (channel_count * highest_rate * pair_count),
    )


def _artifact_active_output_bounds(
    chunk_result: Any,
    chunk: Any,
    *,
    source_sfreq: float,
    output_sfreq: float,
    output_n_times: int,
) -> tuple[int, int] | None:
    """Return the trigger-spanned correction interval on an exported core."""
    context = getattr(chunk_result, "context", None)
    metadata = getattr(context, "metadata", None)
    raw_triggers = getattr(metadata, "triggers", None)
    if raw_triggers is None:
        return None

    triggers = np.asarray(raw_triggers, dtype=float).ravel()
    triggers = triggers[np.isfinite(triggers)]
    if triggers.size == 0:
        return None

    # Trigger metadata follows the processed sampling grid, while ChunkSpec
    # overlap is stored on the source grid.  Crop-to-core does not rewrite the
    # metadata, so shift the retained trigger positions by the exported-core
    # offset before selecting report windows.
    left_overlap = int(getattr(chunk, "left_overlap_samples", 0))
    core_offset = left_overlap * output_sfreq / source_sfreq
    core_triggers = triggers - core_offset

    artifact_length = getattr(metadata, "artifact_length", None)
    if artifact_length is None or int(artifact_length) <= 0:
        positive_intervals = np.diff(np.sort(core_triggers))
        positive_intervals = positive_intervals[positive_intervals > 0]
        artifact_length = (
            int(round(float(np.median(positive_intervals)))) if positive_intervals.size else int(round(output_sfreq))
        )
    artifact_length = max(1, int(artifact_length))
    artifact_offset = float(getattr(metadata, "artifact_to_trigger_offset", 0.0)) * output_sfreq

    relevant = core_triggers[
        (core_triggers + artifact_offset + artifact_length > 0) & (core_triggers + artifact_offset < output_n_times)
    ]
    if relevant.size == 0:
        return None

    start = max(0, int(math.floor(float(np.min(relevant) + artifact_offset))))
    stop = min(
        output_n_times,
        int(math.ceil(float(np.max(relevant) + artifact_offset + artifact_length))),
    )
    return (start, stop) if stop > start else None


def _resample_polyphase(values: np.ndarray, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    """Anti-alias and resample a report window without assuming periodic edges."""
    if np.isclose(source_sfreq, target_sfreq):
        return values
    ratio = Fraction(target_sfreq / source_sfreq).limit_denominator(100_000)
    return signal.resample_poly(
        values,
        ratio.numerator,
        ratio.denominator,
        axis=1,
        padtype="line",
    )


def _capture_paired_recording(
    input_path: Path,
    chunked_result,
) -> tuple[PairedRecordingData | None, list[str]]:
    """Reload bounded, exactly aligned before/after windows for reporting."""
    warnings: list[str] = []
    pairs = _stratified_pairs(_successful_chunk_pairs(chunked_result))
    if not pairs:
        return None, ["No successful corrected chunk file was available for signal diagnostics."]

    try:
        source_context = Loader(path=str(input_path), preload=False).execute(None)
        source_raw = source_context.get_raw()
    except Exception as exc:
        return None, [f"The source recording could not be reloaded for comparison: {exc}"]

    source_sfreq = float(source_raw.info["sfreq"])
    source_duration = float(source_raw.n_times / source_sfreq)
    channel_names: list[str] | None = None
    source_indices: list[int] = []
    source_info: mne.Info | None = None
    report_sfreq: float | None = None
    corrected_sfreq: float | None = None
    before_segments: list[np.ndarray] = []
    after_segments: list[np.ndarray] = []
    segment_windows: list[dict[str, float | int | str]] = []
    corrected_info: mne.Info | None = None
    corrected_bad_channels: list[str] = []
    used_trigger_active_windows = False
    used_core_fallback_windows = False

    try:
        source_eeg_names = _eeg_channel_names(source_raw)
        for chunk, chunk_result in pairs:
            output_path = Path(chunk.output_path)
            output_raw = None
            try:
                output_context = Loader(path=str(output_path), preload=False).execute(None)
                output_raw = output_context.get_raw()
                output_sfreq = float(output_raw.info["sfreq"])

                if report_sfreq is None:
                    corrected_sfreq = output_sfreq
                    report_sfreq = min(output_sfreq, MAX_ANALYSIS_SFREQ_HZ)
                    if report_sfreq < output_sfreq:
                        warnings.append(
                            f"Report analytics were anti-aliased to {report_sfreq:g} Hz from the "
                            f"{output_sfreq:g} Hz corrected recording to bound memory."
                        )
                elif corrected_sfreq is None or not np.isclose(corrected_sfreq, output_sfreq):
                    warnings.append(
                        f"Skipped chunk {chunk.index}: corrected sampling rate "
                        f"{output_sfreq:g} Hz did not match {corrected_sfreq:g} Hz."
                    )
                    continue

                output_names = set(_eeg_channel_names(output_raw))
                if channel_names is None:
                    channel_names = [name for name in source_eeg_names if name in output_names]
                    if not channel_names:
                        warnings.append("No common non-bad EEG channels were present before and after cleaning.")
                        break
                    source_indices = [source_raw.ch_names.index(name) for name in channel_names]
                    source_info = mne.pick_info(source_raw.info, source_indices, copy=True)
                elif not set(channel_names).issubset(output_names):
                    warnings.append(f"Skipped chunk {chunk.index}: its corrected EEG channel set changed.")
                    continue

                output_indices = [output_raw.ch_names.index(name) for name in channel_names]
                if corrected_info is None:
                    corrected_info = output_raw.info.copy()
                    corrected_bad_channels = list(output_raw.info.get("bads", []))
                quota_seconds = _analysis_quota_seconds(
                    channel_count=len(channel_names),
                    pair_count=len(pairs),
                    report_sfreq=report_sfreq,
                    source_sfreq=source_sfreq,
                    output_sfreq=output_sfreq,
                )
                core_start = int(getattr(chunk, "resolved_core_start_sample", chunk.start_sample))
                active_bounds = _artifact_active_output_bounds(
                    chunk_result,
                    chunk,
                    source_sfreq=source_sfreq,
                    output_sfreq=output_sfreq,
                    output_n_times=int(output_raw.n_times),
                )
                if active_bounds is None:
                    analysis_start = 0
                    analysis_stop = int(output_raw.n_times)
                    selection_mode = "core-stratified fallback"
                    used_core_fallback_windows = True
                else:
                    analysis_start, analysis_stop = active_bounds
                    selection_mode = "trigger-active span"
                    used_trigger_active_windows = True

                relative_windows = _analysis_windows(
                    analysis_stop - analysis_start,
                    output_sfreq,
                    quota_seconds,
                )
                for relative_output_start, relative_output_stop in relative_windows:
                    output_start = analysis_start + relative_output_start
                    output_stop = analysis_start + relative_output_stop
                    relative_start_seconds = output_start / output_sfreq
                    requested_seconds = (output_stop - output_start) / output_sfreq
                    source_start = core_start + int(round(relative_start_seconds * source_sfreq))
                    source_count = max(1, int(round(requested_seconds * source_sfreq)))
                    source_stop = min(int(source_raw.n_times), source_start + source_count)
                    if source_stop <= source_start:
                        continue

                    before = source_raw.get_data(
                        picks=source_indices,
                        start=source_start,
                        stop=source_stop,
                    )
                    if before.size == 0:
                        continue

                    # Polyphase resampling anti-aliases any report-rate change;
                    # line padding avoids treating an isolated window as periodic.
                    # Convert each phase before loading the other native-rate
                    # window so high-rate recordings cannot create two large
                    # float64 input arrays at the same time.
                    before = np.asarray(
                        _resample_polyphase(before, source_sfreq, report_sfreq),
                        dtype=np.float32,
                    )
                    after = output_raw.get_data(
                        picks=output_indices,
                        start=output_start,
                        stop=output_stop,
                    )
                    if after.size == 0:
                        continue
                    after = np.asarray(
                        _resample_polyphase(after, output_sfreq, report_sfreq),
                        dtype=np.float32,
                    )
                    paired_samples = min(before.shape[1], after.shape[1])
                    before = before[:, :paired_samples]
                    after = after[:, :paired_samples]

                    finite = np.isfinite(before).all() and np.isfinite(after).all()
                    if not finite:
                        warnings.append(
                            f"Non-finite values in chunk {chunk.index} were replaced with zero for report plots."
                        )
                        before = np.nan_to_num(before, copy=False)
                        after = np.nan_to_num(after, copy=False)

                    before_segments.append(before)
                    after_segments.append(after)
                    absolute_start_seconds = source_start / source_sfreq
                    segment_windows.append(
                        {
                            "chunk_index": int(chunk.index),
                            "start_seconds": float(absolute_start_seconds),
                            "stop_seconds": float(absolute_start_seconds + paired_samples / report_sfreq),
                            "samples": int(paired_samples),
                            "selection": selection_mode,
                        }
                    )
            except Exception as exc:
                warnings.append(f"Skipped chunk {chunk.index} while loading report data: {exc}")
            finally:
                if output_raw is not None:
                    _close_raw(output_raw)
    finally:
        _close_raw(source_raw)

    if not before_segments or channel_names is None or source_info is None or report_sfreq is None:
        warnings.append("No paired source/corrected window was long enough for signal diagnostics.")
        return None, warnings

    analyzed_duration = float(sum(segment.shape[1] for segment in after_segments) / report_sfreq)
    if len(pairs) < len(_successful_chunk_pairs(chunked_result)):
        warnings.append(f"Signal plots use {len(pairs)} stratified chunks to keep report generation memory bounded.")
    if used_core_fallback_windows:
        warnings.append(
            "Some signal windows used time-stratified exported cores because retained trigger metadata "
            "did not define an artifact-active interval; those windows are not artifact-aware."
        )
    if used_trigger_active_windows:
        warnings.append(
            "Where retained trigger metadata were available, signal windows were selected inside the "
            "trigger-spanned correction interval before temporal stratification."
        )

    return (
        PairedRecordingData(
            before_segments=before_segments,
            after_segments=after_segments,
            channel_names=channel_names,
            sfreq=report_sfreq,
            source_info=source_info,
            source_duration_seconds=source_duration,
            analyzed_duration_seconds=analyzed_duration,
            segment_windows=segment_windows,
            source_sfreq=source_sfreq,
            corrected_sfreq=corrected_sfreq,
            corrected_info=corrected_info,
            source_bad_channels=list(source_raw.info.get("bads", [])),
            corrected_bad_channels=corrected_bad_channels,
            warnings=warnings.copy(),
        ),
        warnings,
    )


def _analysis_nperseg(segments: list[np.ndarray], sfreq: float) -> int:
    """Choose a shared Welch/FFT window with useful EEG resolution."""
    shortest = min(segment.shape[1] for segment in segments)
    if shortest < 8:
        raise ValueError("At least eight paired samples are required for temporal spectra.")
    requested = min(int(round(4.0 * sfreq)), 4096)
    return min(shortest, max(8, requested))


def _amplitude_spectrum(
    segments: list[np.ndarray],
    sfreq: float,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean one-sided, Hann-corrected amplitude per channel.

    The result is in volts.  DC and the Nyquist bin are not doubled.
    Segment boundaries are never treated as continuous data.
    """
    window = signal.windows.hann(nperseg, sym=False)
    window_sum = float(np.sum(window))
    step = max(1, nperseg // 2)
    accumulated: np.ndarray | None = None
    frame_count = 0

    for segment in segments:
        starts = np.arange(0, segment.shape[1] - nperseg + 1, step, dtype=int)
        for start in starts:
            frame = signal.detrend(segment[:, start : start + nperseg], axis=1, type="constant")
            transformed = np.fft.rfft(frame * window, axis=1)
            amplitude = 2.0 * np.abs(transformed) / window_sum
            amplitude[:, 0] *= 0.5
            if nperseg % 2 == 0:
                amplitude[:, -1] *= 0.5
            if accumulated is None:
                accumulated = np.zeros_like(amplitude, dtype=float)
            accumulated += amplitude
            frame_count += 1

    if accumulated is None or frame_count == 0:
        raise ValueError("No complete analysis window was available for the amplitude spectrum.")
    frequencies = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)
    return frequencies, accumulated / frame_count


def _welch_psd(
    segments: list[np.ndarray],
    sfreq: float,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a frame-count-weighted Welch PSD without crossing chunks."""
    noverlap = nperseg // 2
    step = max(1, nperseg - noverlap)
    accumulated: np.ndarray | None = None
    total_frames = 0
    frequencies: np.ndarray | None = None

    for segment in segments:
        frequencies, psd = signal.welch(
            segment,
            fs=sfreq,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            axis=1,
        )
        frames = max(1, 1 + (segment.shape[1] - nperseg) // step)
        if accumulated is None:
            accumulated = np.zeros_like(psd, dtype=float)
        accumulated += psd * frames
        total_frames += frames

    if accumulated is None or frequencies is None:
        raise ValueError("No complete analysis window was available for Welch PSD.")
    return frequencies, accumulated / total_frames


def _sample_amplitudes(segments: list[np.ndarray]) -> np.ndarray:
    """Return deterministic pooled values without materializing every sample."""
    total_values = sum(segment.size for segment in segments)
    stride = max(1, math.ceil(total_values / MAX_DISTRIBUTION_VALUES))
    values = [segment.ravel()[::stride] for segment in segments]
    return np.concatenate(values).astype(float, copy=False) * 1e6


def _window_peak_to_peak(
    segments: list[np.ndarray],
    sfreq: float,
    window_seconds: float = 2.0,
) -> np.ndarray:
    """Return channel/window peak-to-peak amplitudes in microvolts."""
    window_samples = max(1, int(round(window_seconds * sfreq)))
    values: list[np.ndarray] = []
    for segment in segments:
        for start in range(0, segment.shape[1] - window_samples + 1, window_samples):
            frame = segment[:, start : start + window_samples]
            values.append(np.ptp(frame, axis=1) * 1e6)
    if not values:
        return np.concatenate([np.ptp(segment, axis=1) * 1e6 for segment in segments])
    return np.concatenate(values)


def _compute_temporal_diagnostics(data: PairedRecordingData) -> TemporalDiagnostics:
    """Compute all paired temporal-frequency and amplitude summaries."""
    all_segments = [*data.before_segments, *data.after_segments]
    nperseg = _analysis_nperseg(all_segments, data.sfreq)
    frequencies, amplitude_before = _amplitude_spectrum(data.before_segments, data.sfreq, nperseg)
    frequencies_after, amplitude_after = _amplitude_spectrum(data.after_segments, data.sfreq, nperseg)
    psd_frequencies, psd_before = _welch_psd(data.before_segments, data.sfreq, nperseg)
    psd_frequencies_after, psd_after = _welch_psd(data.after_segments, data.sfreq, nperseg)
    if not (
        np.array_equal(frequencies, frequencies_after)
        and np.array_equal(frequencies, psd_frequencies)
        and np.array_equal(frequencies, psd_frequencies_after)
    ):
        raise ValueError("Before and after spectral frequency grids did not match.")

    before_values = _sample_amplitudes(data.before_segments)
    after_values = _sample_amplitudes(data.after_segments)
    combined = np.concatenate([before_values, after_values])
    amplitude_limit = float(np.nanpercentile(np.abs(combined), 99.75))
    if not np.isfinite(amplitude_limit) or amplitude_limit <= 0:
        amplitude_limit = max(float(np.nanmax(np.abs(combined))), 1.0)
    histogram_edges = np.linspace(-amplitude_limit, amplitude_limit, 81)
    histogram_before, _ = np.histogram(before_values, bins=histogram_edges)
    histogram_after, _ = np.histogram(after_values, bins=histogram_edges)
    bin_widths = np.diff(histogram_edges)
    histogram_before = histogram_before / (len(before_values) * bin_widths)
    histogram_after = histogram_after / (len(after_values) * bin_widths)

    return TemporalDiagnostics(
        frequencies=frequencies,
        amplitude_before=amplitude_before,
        amplitude_after=amplitude_after,
        psd_before=psd_before,
        psd_after=psd_after,
        histogram_edges=histogram_edges,
        histogram_before=histogram_before,
        histogram_after=histogram_after,
        peak_to_peak_before=_window_peak_to_peak(data.before_segments, data.sfreq),
        peak_to_peak_after=_window_peak_to_peak(data.after_segments, data.sfreq),
        clipped_before_fraction=float(np.mean(np.abs(before_values) > amplitude_limit)),
        clipped_after_fraction=float(np.mean(np.abs(after_values) > amplitude_limit)),
        frequency_resolution_hz=float(data.sfreq / nperseg),
        welch_segment_seconds=float(nperseg / data.sfreq),
        channel_names=data.channel_names.copy(),
    )


def _integrated_band_power(frequencies: np.ndarray, psd: np.ndarray, mask: np.ndarray) -> float:
    """Integrate a PSD over a mask, returning NaN when the grid is too short."""
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.trapezoid(psd[mask], frequencies[mask]))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return a finite ratio or ``None`` when it cannot be interpreted."""
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return None
    value = float(numerator / denominator)
    return value if np.isfinite(value) else None


def _scanner_peak_mask(
    frequencies: np.ndarray,
    median_psd: np.ndarray,
    *,
    minimum_hz: float = 13.0,
    maximum_hz: float = 80.0,
    prominence_db: float = 6.0,
    half_width_hz: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect narrow scanner-like spectral peaks outside theta/alpha bands."""
    search = (frequencies >= minimum_hz) & (frequencies <= maximum_hz)
    search_indices = np.flatnonzero(search)
    mask = np.zeros_like(frequencies, dtype=bool)
    if len(search_indices) < 3:
        return mask, np.array([], dtype=float)

    safe_psd = np.maximum(median_psd[search], np.finfo(float).tiny)
    psd_db = 10.0 * np.log10(safe_psd)
    frequency_step = float(np.median(np.diff(frequencies)))
    minimum_distance = max(1, int(round(0.5 / frequency_step)))
    local_peaks, _ = signal.find_peaks(
        psd_db,
        prominence=prominence_db,
        height=float(np.max(psd_db) - 40.0),
        distance=minimum_distance,
    )
    peak_indices = search_indices[local_peaks]
    peak_frequencies = frequencies[peak_indices]
    for peak_frequency in peak_frequencies:
        mask |= np.abs(frequencies - peak_frequency) <= half_width_hz
    return mask, peak_frequencies


def _preservation_interpretation(value: float | None, band_name: str) -> str:
    if value is None:
        return f"{band_name} preservation could not be estimated from the retained windows."
    if 0.8 <= value <= 1.25:
        return f"{band_name} stayed within ±25% of the source (1.0 is unchanged)."
    if value < 0.8:
        return f"{band_name} decreased by {(1.0 - value) * 100:.1f}%; inspect for possible over-correction."
    return f"{band_name} increased by {(value - 1.0) * 100:.1f}%; residual or newly emphasized activity may remain."


def _metric_record(
    value: float | int | list[float] | None,
    *,
    unit: str,
    description: str,
    interpretation: str,
) -> dict[str, Any]:
    """Build one self-describing metric record for JSON and HTML."""
    return {
        "value": value,
        "unit": unit,
        "description": description,
        "interpretation": interpretation,
    }


def _collect_pipeline_metrics(chunked_result: Any) -> list[dict[str, Any]]:
    """Collect any evaluation-processor metrics already retained per chunk."""
    records: list[dict[str, Any]] = []
    chunks = list(getattr(chunked_result, "chunks", []))
    for chunk, result in zip(chunks, _chunked_results(chunked_result), strict=False):
        context = getattr(result, "context", None)
        custom = getattr(getattr(context, "metadata", None), "custom", {})
        metrics = custom.get("metrics", {}) if isinstance(custom, dict) else {}
        if metrics:
            records.append(
                {
                    "chunk_index": int(chunk.index),
                    "metrics": _json_safe(metrics),
                    "guidance": {name: _pipeline_metric_guidance(name) for name in metrics},
                }
            )
    return records


def _pipeline_metric_guidance(name: str) -> dict[str, str]:
    """Return concise reading guidance for metrics emitted by processors."""
    known = {
        "snr": (
            "Signal-to-noise ratio produced by SNRCalculator.",
            "Higher is better; above 10 is strong, 3–10 is moderate, and below 3 is weak.",
        ),
        "legacy_snr": (
            "Legacy signal-to-noise ratio relative to the original recording.",
            "Higher generally indicates better separation, but compare only runs using the same reference.",
        ),
        "rms_ratio": (
            "RMS improvement ratio produced by RMSCalculator.",
            "Above 1 means RMS decreased; very high values can also indicate over-correction.",
        ),
        "rms_residual": (
            "Corrected-to-reference RMS residual ratio.",
            "A value near 1 means reference-like RMS; distance from 1 indicates under- or over-correction.",
        ),
        "median_artifact": (
            "Median artifact amplitude retained by MedianArtifactCalculator.",
            "Lower magnitude generally means less residual artifact when acquisition units and intervals match.",
        ),
        "median_artifact_ratio": (
            "Median artifact amplitude ratio relative to its reference.",
            "A value near 1 matches the reference; interpret the direction using the paired amplitude value.",
        ),
        "fft_allen": (
            "Band-wise spectral power difference from the Allen-style reference.",
            "Values closer to zero indicate better spectral preservation; inspect individual bands for selective loss.",
        ),
        "fft_niazy": (
            "Uncorrected/corrected power ratio at slice and volume harmonics.",
            "Larger positive dB values indicate stronger harmonic suppression; negative values warrant inspection.",
        ),
    }
    description, interpretation = known.get(
        name,
        (
            "Metric emitted by a processor in this pipeline.",
            "Interpret using the emitting processor's documented definition and compare like-for-like runs.",
        ),
    )
    return {"description": description, "interpretation": interpretation}


def _quality_metrics_payload(
    *,
    input_path: Path,
    data: PairedRecordingData | None,
    diagnostics: TemporalDiagnostics | None,
    chunked_result: Any,
    graph_metadata: dict[str, Any] | None = None,
    coherence: CoherenceDiagnostics | None = None,
) -> dict[str, Any]:
    """Calculate bounded before/after metrics and attach reading guidance."""
    metrics: dict[str, dict[str, Any]] = {}
    if data is None or diagnostics is None:
        reason = "Paired source and corrected windows were unavailable."
        for name, unit, description in (
            ("scanner_peak_residual", "ratio", "Corrected/source power at detected scanner peaks."),
            ("scanner_peak_suppression_db", "dB", "Scanner-peak power reduction in decibels."),
            ("scanner_peak_count", "peaks", "Number of prominent source-spectrum peaks detected."),
            ("scanner_peak_frequencies_hz", "Hz", "Detected scanner-peak center frequencies."),
            ("delta_preservation", "ratio", "Corrected/source power in 0.8–4 Hz."),
            ("theta_preservation", "ratio", "Corrected/source power in 4–8 Hz."),
            ("alpha_preservation", "ratio", "Corrected/source power in 8–13 Hz."),
            ("nonpeak_beta_preservation", "ratio", "Corrected/source beta power away from scanner peaks."),
            ("nonpeak_eeg_log_deviation", "mean absolute log ratio", "Combined non-peak preservation error."),
            ("rms_improvement_ratio", "ratio", "Source RMS divided by corrected RMS."),
            ("removed_signal_rms_fraction", "ratio", "Removed-signal RMS divided by source RMS."),
            ("median_peak_to_peak_preservation", "ratio", "Corrected/source median peak-to-peak amplitude."),
            ("waveform_correlation", "median Pearson r", "Aligned source/corrected waveform similarity."),
            ("source_extreme_sample_fraction", "fraction", "Source samples outside the shared display range."),
            ("corrected_extreme_sample_fraction", "fraction", "Corrected samples outside the shared display range."),
            ("low_graph_mode_preservation", "ratio", "Corrected/source energy in low graph modes."),
            ("high_graph_mode_preservation", "ratio", "Corrected/source energy in high graph modes."),
            ("coherence_before_mean", "mean coherence", "Mean source off-diagonal alpha coherence."),
            ("coherence_after_mean", "mean coherence", "Mean corrected off-diagonal alpha coherence."),
            ("coherence_change", "after − before", "Change in mean off-diagonal alpha coherence."),
            ("coherence_modularity", "Q", "Exploratory corrected-network modularity."),
        ):
            metrics[name] = _metric_record(None, unit=unit, description=description, interpretation=reason)
        return {
            "source_path": str(input_path),
            "metrics": metrics,
            "pipeline_metrics_by_chunk": _collect_pipeline_metrics(chunked_result),
            "method": reason,
        }

    frequencies = diagnostics.frequencies
    before_psd = np.median(diagnostics.psd_before, axis=0)
    after_psd = np.median(diagnostics.psd_after, axis=0)
    scanner_mask, scanner_peaks = _scanner_peak_mask(
        frequencies,
        before_psd,
        maximum_hz=min(80.0, data.sfreq / 2.0),
    )
    scanner_residual = _safe_ratio(
        _integrated_band_power(frequencies, after_psd, scanner_mask),
        _integrated_band_power(frequencies, before_psd, scanner_mask),
    )
    if scanner_residual is None:
        scanner_interpretation = (
            "No prominent scanner peak passed the fixed detection rule; this metric is unavailable."
        )
        scanner_suppression_db = None
    else:
        scanner_suppression_db = float(-10.0 * np.log10(max(scanner_residual, np.finfo(float).tiny)))
        if scanner_residual <= 0.1:
            scanner_interpretation = "At least 90% of detected scanner-peak power was removed; lower is better."
        elif scanner_residual <= 0.5:
            scanner_interpretation = "Detected scanner-peak power was reduced by at least half; lower is better."
        elif scanner_residual <= 1.0:
            scanner_interpretation = "Detected scanner peaks were reduced, but substantial residual power remains."
        else:
            scanner_interpretation = "Detected scanner-peak power increased after cleaning; inspect the result."
    metrics["scanner_peak_residual"] = _metric_record(
        scanner_residual,
        unit="corrected/source ratio",
        description="Integrated corrected power divided by source power around detected scanner peaks.",
        interpretation=scanner_interpretation,
    )
    metrics["scanner_peak_suppression_db"] = _metric_record(
        scanner_suppression_db,
        unit="dB",
        description="Scanner-peak reduction expressed as −10 log10(residual ratio).",
        interpretation=(
            "Higher is better: 3 dB is about half the power and 10 dB is about one tenth."
            if scanner_suppression_db is not None
            else "Unavailable because no scanner peak passed detection."
        ),
    )
    metrics["scanner_peak_count"] = _metric_record(
        int(len(scanner_peaks)),
        unit="peaks",
        description="Prominent source-spectrum peaks detected between 13 Hz and the bounded upper frequency.",
        interpretation="A count of zero means the fixed detector found no peak; it does not prove the recording was artifact-free.",
    )
    metrics["scanner_peak_frequencies_hz"] = _metric_record(
        [float(value) for value in scanner_peaks],
        unit="Hz",
        description="Center frequencies used for scanner-peak residual calculation.",
        interpretation="Inspect these frequencies against the scanner slice/volume timing and spectral plots.",
    )

    preservation_values: dict[str, float | None] = {}
    for name, label, low, high in (
        ("delta_preservation", "Delta", 0.8, 4.0),
        ("theta_preservation", "Theta", 4.0, 8.0),
        ("alpha_preservation", "Alpha", 8.0, 13.0),
        ("nonpeak_beta_preservation", "Non-peak beta", 13.0, 30.0),
    ):
        mask = (frequencies >= low) & (frequencies < min(high, data.sfreq / 2.0))
        if name == "nonpeak_beta_preservation":
            mask &= ~scanner_mask
        value = _safe_ratio(
            _integrated_band_power(frequencies, after_psd, mask),
            _integrated_band_power(frequencies, before_psd, mask),
        )
        preservation_values[name] = value
        metrics[name] = _metric_record(
            value,
            unit="corrected/source ratio",
            description=f"Integrated {label.lower()} power after cleaning divided by source power ({low:g}–{high:g} Hz).",
            interpretation=_preservation_interpretation(value, label),
        )

    physiological = [
        preservation_values[name]
        for name in ("theta_preservation", "alpha_preservation", "nonpeak_beta_preservation")
        if preservation_values[name] is not None and preservation_values[name] > 0.0
    ]
    log_deviation = float(np.mean([abs(math.log(value)) for value in physiological])) if physiological else None
    metrics["nonpeak_eeg_log_deviation"] = _metric_record(
        log_deviation,
        unit="mean absolute log ratio",
        description="Symmetric deviation from ideal preservation across theta, alpha, and non-peak beta.",
        interpretation=(
            "Zero is ideal; values below 0.223 keep the average multiplicative change within roughly 25%."
            if log_deviation is not None
            else "Unavailable because the required preservation ratios were not finite and positive."
        ),
    )

    before_square_sum = after_square_sum = residual_square_sum = 0.0
    sample_count = 0
    correlations: list[float] = []
    for before, after in zip(data.before_segments, data.after_segments, strict=True):
        common = min(before.shape[1], after.shape[1])
        source = np.asarray(before[:, :common], dtype=float)
        cleaned = np.asarray(after[:, :common], dtype=float)
        before_square_sum += float(np.sum(source**2))
        after_square_sum += float(np.sum(cleaned**2))
        residual_square_sum += float(np.sum((source - cleaned) ** 2))
        sample_count += source.size
        for source_channel, clean_channel in zip(source, cleaned, strict=True):
            if np.std(source_channel) > 0.0 and np.std(clean_channel) > 0.0:
                correlation = float(np.corrcoef(source_channel, clean_channel)[0, 1])
                if np.isfinite(correlation):
                    correlations.append(correlation)
    source_rms = math.sqrt(before_square_sum / sample_count) if sample_count else float("nan")
    clean_rms = math.sqrt(after_square_sum / sample_count) if sample_count else float("nan")
    residual_rms = math.sqrt(residual_square_sum / sample_count) if sample_count else float("nan")
    rms_improvement = _safe_ratio(source_rms, clean_rms)
    residual_fraction = _safe_ratio(residual_rms, source_rms)
    waveform_correlation = float(np.median(correlations)) if correlations else None
    metrics["rms_improvement_ratio"] = _metric_record(
        rms_improvement,
        unit="source/corrected ratio",
        description="RMS amplitude before cleaning divided by RMS amplitude after cleaning.",
        interpretation=(
            "Above 1 means overall RMS fell; a very high value can reflect useful artifact removal or over-correction."
            if rms_improvement is not None
            else "Unavailable because RMS power was zero or non-finite."
        ),
    )
    metrics["removed_signal_rms_fraction"] = _metric_record(
        residual_fraction,
        unit="removed/source ratio",
        description="RMS of the before-minus-after waveform divided by source RMS.",
        interpretation=(
            "Lower means less waveform change; this is not independently better because artifact removal requires change."
            if residual_fraction is not None
            else "Unavailable because source RMS was zero or non-finite."
        ),
    )
    before_ptp = float(np.median(diagnostics.peak_to_peak_before))
    after_ptp = float(np.median(diagnostics.peak_to_peak_after))
    ptp_ratio = _safe_ratio(after_ptp, before_ptp)
    metrics["median_peak_to_peak_preservation"] = _metric_record(
        ptp_ratio,
        unit="corrected/source ratio",
        description="Median two-second peak-to-peak amplitude after cleaning divided by source amplitude.",
        interpretation=_preservation_interpretation(ptp_ratio, "Peak-to-peak amplitude"),
    )
    metrics["waveform_correlation"] = _metric_record(
        waveform_correlation,
        unit="median Pearson r",
        description="Median channel/window correlation between aligned source and corrected signals.",
        interpretation=(
            "Closer to 1 means the overall waveform shape was retained; interpret with artifact-suppression metrics."
            if waveform_correlation is not None
            else "Unavailable because paired channels were constant or non-finite."
        ),
    )
    metrics["source_extreme_sample_fraction"] = _metric_record(
        diagnostics.clipped_before_fraction,
        unit="fraction",
        description="Fraction of source samples outside the shared robust amplitude-display range.",
        interpretation=(
            "Closer to zero means fewer unusually large source samples; this is a distribution-tail flag, "
            "not an ADC clipping test."
        ),
    )
    metrics["corrected_extreme_sample_fraction"] = _metric_record(
        diagnostics.clipped_after_fraction,
        unit="fraction",
        description="Fraction of corrected samples outside the shared robust amplitude-display range.",
        interpretation=(
            "Compare with the source fraction: a decrease means fewer extreme samples, while an increase "
            "warrants inspection."
        ),
    )

    low_graph_ratio = graph_metadata.get("low_graph_mode_preservation") if graph_metadata else None
    high_graph_ratio = graph_metadata.get("high_graph_mode_preservation") if graph_metadata else None
    metrics["low_graph_mode_preservation"] = _metric_record(
        low_graph_ratio,
        unit="corrected/source ratio",
        description="Corrected/source energy ratio in the lowest third of sensor-graph modes.",
        interpretation=_preservation_interpretation(low_graph_ratio, "Low graph-mode energy"),
    )
    metrics["high_graph_mode_preservation"] = _metric_record(
        high_graph_ratio,
        unit="corrected/source ratio",
        description="Corrected/source energy ratio in the highest third of sensor-graph modes.",
        interpretation=_preservation_interpretation(high_graph_ratio, "High graph-mode energy"),
    )

    coherence_before = coherence_after = coherence_change = None
    if coherence is not None:
        off_diagonal = ~np.eye(len(coherence.before), dtype=bool)
        coherence_before = float(np.mean(coherence.before[off_diagonal]))
        coherence_after = float(np.mean(coherence.after[off_diagonal]))
        coherence_change = coherence_after - coherence_before
    metrics["coherence_before_mean"] = _metric_record(
        coherence_before,
        unit="mean coherence",
        description="Mean off-diagonal source sensor coherence in the configured alpha band.",
        interpretation=("This is the baseline for the after-cleaning value; coherence is not anatomical connectivity."),
    )
    metrics["coherence_after_mean"] = _metric_record(
        coherence_after,
        unit="mean coherence",
        description="Mean off-diagonal corrected sensor coherence in the configured alpha band.",
        interpretation=(
            "Compare with the source value; a large shift can indicate artifact removal or altered "
            "physiological coupling."
        ),
    )
    metrics["coherence_change"] = _metric_record(
        coherence_change,
        unit="after − before",
        description="Corrected minus source mean off-diagonal alpha coherence.",
        interpretation=(
            "Near zero means little network-wide change; positive or negative values require the coherence "
            "plot for context."
        ),
    )
    metrics["coherence_modularity"] = _metric_record(
        coherence.modularity if coherence is not None else None,
        unit="Q",
        description="Weighted modularity of the thresholded corrected alpha-coherence network.",
        interpretation=(
            "Higher values indicate stronger separation into sensor communities; this exploratory value "
            "has no universal good cutoff."
        ),
    )

    return {
        "source_path": str(input_path),
        "method": (
            "Metrics use the same bounded, aligned source/corrected windows as the HTML plots. "
            "Band ratios compare scanner-on source power with corrected power; 1.0 means unchanged."
        ),
        "analysis": {
            "sampling_frequency_hz": data.sfreq,
            "analyzed_duration_seconds": data.analyzed_duration_seconds,
            "frequency_resolution_hz": diagnostics.frequency_resolution_hz,
            "window_count": len(data.segment_windows),
        },
        "metrics": metrics,
        "pipeline_metrics_by_chunk": _collect_pipeline_metrics(chunked_result),
    }


def _write_quality_metrics(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one self-describing quality-metric report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(_json_safe(payload), stream, indent=2, ensure_ascii=False)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _log_quality_metrics(payload: dict[str, Any]) -> None:
    """Log every exported metric and its direct interpretation."""
    for name, record in payload.get("metrics", {}).items():
        value = record.get("value")
        unit = record.get("unit", "")
        logger.info(
            "Quality metric {} = {} {} — {}",
            name,
            _format_value(value),
            unit,
            record.get("interpretation", ""),
        )
    for record in payload.get("pipeline_metrics_by_chunk", []):
        chunk_index = record.get("chunk_index", "?")
        for name, value in record.get("metrics", {}).items():
            guidance = record.get("guidance", {}).get(name, {})
            logger.info(
                "Pipeline metric chunk={} {} = {} — {}",
                chunk_index,
                name,
                _format_value(_json_safe(value)),
                guidance.get("interpretation", "See the emitting processor's documentation."),
            )


def _positioned_indices(info: mne.Info) -> np.ndarray:
    """Return channels whose stored 3-D locations are finite and nonzero."""
    valid: list[int] = []
    for index, channel in enumerate(info["chs"]):
        position = np.asarray(channel["loc"][:3], dtype=float)
        if np.all(np.isfinite(position)) and float(np.linalg.norm(position)) > 1e-6:
            valid.append(index)
    return np.asarray(valid, dtype=int)


def _infer_template_montage(info: mne.Info) -> tuple[mne.Info, str] | None:
    """Apply a recognized template only when channel names support it."""
    names = list(info.ch_names)
    egi_names = [name for name in names if re.fullmatch(r"E(?:[1-9]|[1-9]\d|1[01]\d|12[0-8])", name)]
    candidates: list[tuple[str, str]] = []
    if len(egi_names) >= MIN_SPATIAL_CHANNELS:
        montage_name = "GSN-HydroCel-129" if "Cz" in names else "GSN-HydroCel-128"
        candidates.append((montage_name, f"MNE {montage_name} template"))
    candidates.append(("standard_1020", "MNE standard 10-20/10-10 template"))

    for montage_name, description in candidates:
        candidate = info.copy()
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            candidate.set_montage(
                montage,
                match_case=False,
                on_missing="ignore",
                verbose=False,
            )
        except Exception:
            continue
        if len(_positioned_indices(candidate)) >= MIN_SPATIAL_CHANNELS:
            return candidate, description
    return None


def _coordinate_frame_label(info: mne.Info) -> str:
    """Describe EEG coordinate-frame codes without relying on private MNE APIs."""
    frame_names = {
        0: "unknown",
        1: "device",
        2: "isotrak",
        4: "head",
        5: "MRI",
        1004: "CTF head",
    }
    frames = sorted({int(channel["coord_frame"]) for channel in info["chs"]})
    return ", ".join(frame_names.get(frame, f"MNE frame {frame}") for frame in frames)


def _spatial_coverage_note(coordinates: np.ndarray) -> str | None:
    """Validate human-head scale and broad two-dimensional sensor coverage."""
    coordinates = np.asarray(coordinates, dtype=float)
    if len(coordinates) < MIN_SPATIAL_CHANNELS:
        return None
    pairwise = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    diameter = float(np.max(pairwise))
    if not 0.04 <= diameter <= 0.5:
        return None

    centered_xy = coordinates[:, :2] - np.median(coordinates[:, :2], axis=0)
    if np.linalg.matrix_rank(centered_xy, tol=1e-8) < 2:
        return None
    noncentral = np.linalg.norm(centered_xy, axis=1) > max(diameter * 0.02, 1e-6)
    quadrants = {(int(x >= 0), int(y >= 0)) for x, y in centered_xy[noncentral]}
    if len(quadrants) < 3:
        return None

    density = "sparse" if len(coordinates) < 16 else "broad"
    return (
        f"{density} sensor coverage: {len(coordinates)} positioned channels, "
        f"{diameter * 100:.1f} cm maximum separation, {len(quadrants)}/4 projected quadrants"
    )


def _resolve_spatial_geometry(data: PairedRecordingData) -> SpatialGeometry | None:
    """Resolve source-file or recognized-template geometry with coverage checks."""
    source_info = data.source_info.copy()
    candidates: list[tuple[mne.Info, str]] = [
        (
            source_info,
            "source-file sensor coordinates (digitized-versus-template provenance is not encoded)",
        )
    ]
    inferred = _infer_template_montage(source_info)
    if inferred is not None:
        candidates.append(inferred)

    for info, origin in candidates:
        positioned = _positioned_indices(info)
        if len(positioned) < MIN_SPATIAL_CHANNELS:
            continue
        spatial_info = mne.pick_info(info, positioned.tolist(), copy=True)
        coordinates = np.asarray(
            [channel["loc"][:3] for channel in spatial_info["chs"]],
            dtype=float,
        )
        coverage_note = _spatial_coverage_note(coordinates)
        if coverage_note is None:
            continue
        return SpatialGeometry(
            channel_indices=positioned,
            info=spatial_info,
            coordinates_3d=coordinates,
            origin=origin,
            coordinate_frame=_coordinate_frame_label(spatial_info),
            coverage_note=coverage_note,
        )
    return None


def _physical_sensor_graph(coordinates: np.ndarray, neighbors: int = 4) -> np.ndarray:
    """Build a symmetric distance-weighted k-nearest sensor graph."""
    coordinates = np.asarray(coordinates, dtype=float)
    n_channels = len(coordinates)
    if n_channels < 2:
        raise ValueError("At least two positioned EEG channels are required for a sensor graph.")

    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbor_count = min(max(1, neighbors), n_channels - 1)
    nearest = np.partition(distances, neighbor_count - 1, axis=1)[:, :neighbor_count]
    finite_distances = nearest[np.isfinite(nearest) & (nearest > 0)]
    scale = float(np.median(finite_distances)) if finite_distances.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    adjacency = np.zeros((n_channels, n_channels), dtype=float)
    for row in range(n_channels):
        columns = np.argsort(distances[row])[:neighbor_count]
        weights = np.exp(-0.5 * (distances[row, columns] / scale) ** 2)
        adjacency[row, columns] = weights
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _laplacian_eigendecomposition(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered eigenpairs of the symmetric combinatorial Laplacian."""
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not np.allclose(adjacency, adjacency.T, atol=1e-10):
        raise ValueError("adjacency must be symmetric")

    laplacian = np.diag(np.sum(adjacency, axis=1)) - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues[np.abs(eigenvalues) < 1e-12] = 0.0
    return eigenvalues, eigenvectors


def _graph_spectral_energy(
    segments: list[np.ndarray],
    channel_indices: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Return empirical mean-square graph Fourier coefficients in volts²."""
    accumulated = np.zeros(eigenvectors.shape[1], dtype=float)
    total_samples = 0
    for segment in segments:
        values = np.asarray(segment[channel_indices], dtype=float)
        values -= np.mean(values, axis=1, keepdims=True)
        # Limit only the visualization calculation, preserving evenly spaced
        # samples over every selected chunk when recordings are very dense.
        stride = max(1, math.ceil(values.shape[1] / 20_000))
        coefficients = eigenvectors.T @ values[:, ::stride]
        accumulated += np.sum(np.abs(coefficients) ** 2, axis=1)
        total_samples += coefficients.shape[1]
    if total_samples == 0:
        raise ValueError("No samples were available for graph spectral energy.")
    return accumulated / total_samples


def _coherence_matrix(
    segments: list[np.ndarray],
    channel_indices: np.ndarray,
    sfreq: float,
    nperseg: int,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate band-mean magnitude-squared coherence with streaming frames."""
    frequencies = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)
    frequency_mask = (frequencies >= fmin) & (frequencies <= fmax)
    if not np.any(frequency_mask):
        raise ValueError(f"No Fourier bins fell inside the {fmin:g}-{fmax:g} Hz coherence band.")

    window = signal.windows.hann(nperseg, sym=False)
    step = max(1, nperseg // 2)
    cross_spectrum = np.zeros(
        (len(channel_indices), len(channel_indices), int(np.count_nonzero(frequency_mask))),
        dtype=np.complex128,
    )
    frame_count = 0
    for segment in segments:
        starts = np.arange(0, segment.shape[1] - nperseg + 1, step, dtype=int)
        for start in starts:
            frame = np.asarray(segment[channel_indices, start : start + nperseg], dtype=float)
            frame -= np.mean(frame, axis=1, keepdims=True)
            transformed = np.fft.rfft(frame * window, axis=1)[:, frequency_mask]
            cross_spectrum += np.einsum(
                "if,jf->ijf",
                transformed,
                transformed.conj(),
                optimize=True,
            )
            frame_count += 1
    if frame_count < 2:
        raise ValueError("At least two complete windows are required for coherence.")
    cross_spectrum /= frame_count

    auto_spectrum = np.real(np.diagonal(cross_spectrum, axis1=0, axis2=1)).T
    denominator = auto_spectrum[:, None, :] * auto_spectrum[None, :, :]
    coherence = np.divide(
        np.abs(cross_spectrum) ** 2,
        denominator,
        out=np.zeros_like(np.real(cross_spectrum)),
        # Coherence is scale invariant. An absolute epsilon is invalid for EEG
        # stored in volts because ordinary microvolt spectra fall below it.
        where=denominator > 0.0,
    )
    matrix = np.clip(np.mean(coherence, axis=2), 0.0, 1.0)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)
    return frequencies[frequency_mask], matrix


def _threshold_by_density(matrix: np.ndarray, density: float) -> tuple[np.ndarray, float]:
    """Keep the strongest fixed proportion of undirected edges."""
    if not (0 < density <= 1):
        raise ValueError("density must lie in (0, 1]")
    matrix = np.asarray(matrix, dtype=float)
    rows, columns = np.triu_indices_from(matrix, k=1)
    weights = matrix[rows, columns]
    finite_mask = np.isfinite(weights)
    if not np.any(finite_mask):
        return np.zeros_like(matrix), float("nan")

    finite_rows = rows[finite_mask]
    finite_columns = columns[finite_mask]
    finite_weights = weights[finite_mask]
    keep_count = max(1, int(math.ceil(density * len(finite_weights))))
    # Stable sorting makes tied weights deterministic without retaining every
    # tie and silently exceeding the stated proportional graph density.
    selected = np.argsort(-finite_weights, kind="stable")[:keep_count]
    threshold = float(np.min(finite_weights[selected]))
    thresholded = np.zeros_like(matrix)
    thresholded[finite_rows[selected], finite_columns[selected]] = finite_weights[selected]
    thresholded[finite_columns[selected], finite_rows[selected]] = finite_weights[selected]
    return thresholded, threshold


def _spectral_modularity_labels(adjacency: np.ndarray) -> np.ndarray:
    """Partition a weighted graph using recursive modularity eigenvectors.

    This is Newman's deterministic leading-eigenvector approximation.  It is
    intentionally labelled as spectral modularity in the report rather than
    as Leiden/Louvain or the spatial CCB method from Ji et al.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n_nodes = adjacency.shape[0]
    if n_nodes == 0:
        return np.array([], dtype=int)
    total_weight = float(np.sum(adjacency))
    if total_weight <= np.finfo(float).eps:
        return np.zeros(n_nodes, dtype=int)

    degrees = np.sum(adjacency, axis=1)
    modularity_matrix = adjacency - np.outer(degrees, degrees) / total_weight
    communities: list[np.ndarray] = [np.arange(n_nodes, dtype=int)]
    finalized: list[np.ndarray] = []

    while communities:
        members = communities.pop(0)
        if len(members) < 2:
            finalized.append(members)
            continue

        subgroup = modularity_matrix[np.ix_(members, members)].copy()
        subgroup -= np.diag(np.sum(subgroup, axis=1))
        eigenvalues, eigenvectors = np.linalg.eigh(subgroup)
        leading = eigenvectors[:, -1]
        signs = np.where(leading >= 0, 1.0, -1.0)
        positive = members[signs > 0]
        negative = members[signs < 0]
        gain = float(signs @ subgroup @ signs / (2.0 * total_weight))

        if eigenvalues[-1] <= 1e-12 or gain <= 1e-12 or len(positive) == 0 or len(negative) == 0:
            finalized.append(members)
            continue
        communities.extend([positive, negative])

    finalized.sort(key=lambda members: int(np.min(members)))
    labels = np.zeros(n_nodes, dtype=int)
    for label, members in enumerate(finalized):
        labels[members] = label
    return labels


def _weighted_modularity(adjacency: np.ndarray, labels: np.ndarray) -> float:
    """Calculate weighted undirected modularity for one partition."""
    adjacency = np.asarray(adjacency, dtype=float)
    total_weight = float(np.sum(adjacency))
    if total_weight <= np.finfo(float).eps:
        return 0.0
    degrees = np.sum(adjacency, axis=1)
    expected = np.outer(degrees, degrees) / total_weight
    same_community = labels[:, None] == labels[None, :]
    return float(np.sum((adjacency - expected)[same_community]) / total_weight)


def _compute_coherence_diagnostics(
    data: PairedRecordingData,
    geometry: SpatialGeometry | None,
) -> CoherenceDiagnostics:
    """Compute matched before/after coherence and clean-data communities."""
    n_channels = len(data.channel_names)
    if n_channels < 2:
        raise ValueError("At least two common EEG channels are required for coherence.")

    # Coherence coverage is independent of coordinate availability.  If every
    # selected channel is positioned, the node view uses the scalp layout;
    # otherwise it truthfully falls back to an abstract graph layout.
    channel_indices = np.linspace(
        0,
        n_channels - 1,
        min(MAX_COHERENCE_CHANNELS, n_channels),
        dtype=int,
    )

    shortest = min(segment.shape[1] for segment in [*data.before_segments, *data.after_segments])
    nperseg = min(max(8, int(round(2.0 * data.sfreq))), shortest, 2048)
    nyquist = data.sfreq / 2.0
    fmin = COHERENCE_FMIN_HZ
    fmax = min(COHERENCE_FMAX_HZ, nyquist)
    if fmax <= fmin:
        raise ValueError("Sampling rate is too low for the configured coherence band.")

    step = max(1, nperseg // 2)
    frame_count = sum(max(0, 1 + (segment.shape[1] - nperseg) // step) for segment in data.after_segments)
    if frame_count < MIN_COHERENCE_WINDOWS:
        raise ValueError(
            f"At least {MIN_COHERENCE_WINDOWS} complete Hann windows are required for coherence; "
            f"only {frame_count} were available."
        )

    frequencies, before = _coherence_matrix(
        data.before_segments,
        channel_indices,
        data.sfreq,
        nperseg,
        fmin,
        fmax,
    )
    _, after = _coherence_matrix(
        data.after_segments,
        channel_indices,
        data.sfreq,
        nperseg,
        fmin,
        fmax,
    )
    thresholded, threshold = _threshold_by_density(after, NETWORK_EDGE_DENSITY)
    labels = _spectral_modularity_labels(thresholded)
    names = [data.channel_names[int(index)] for index in channel_indices]
    possible_edges = len(channel_indices) * (len(channel_indices) - 1) / 2
    retained_edges = np.count_nonzero(np.triu(thresholded, k=1))
    edge_density = retained_edges / possible_edges if possible_edges else 0.0
    return CoherenceDiagnostics(
        before=before,
        after=after,
        channel_indices=channel_indices,
        channel_names=names,
        labels=labels,
        thresholded_after=thresholded,
        edge_threshold=threshold,
        modularity=_weighted_modularity(thresholded, labels),
        frequencies=frequencies,
        fmin=fmin,
        fmax=fmax,
        nperseg=nperseg,
        frame_count=frame_count,
        edge_density=float(edge_density),
        low_precision=frame_count < LOW_PRECISION_COHERENCE_WINDOWS,
    )


def _figure_png(fig: plt.Figure, *, dpi: int = 100) -> bytes:
    """Serialize and close a Matplotlib figure."""
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    return buffer.getvalue()


def _close_new_figures(existing: set[int]) -> None:
    """Close figures opened by a failed report section without touching prior figures."""
    for figure_number in set(plt.get_fignums()) - existing:
        plt.close(figure_number)


def _data_uri(payload: bytes, mime_type: str) -> str:
    """Encode binary report content as an inline data URI."""
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _channel_summary(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median and interquartile range across channels."""
    return (
        np.median(values, axis=0),
        np.percentile(values, 25, axis=0),
        np.percentile(values, 75, axis=0),
    )


def _positive_limits(*arrays: np.ndarray) -> tuple[float, float]:
    """Return shared full-range limits suitable for a logarithmic axis."""
    combined = np.concatenate([np.asarray(array).ravel() for array in arrays])
    finite = combined[np.isfinite(combined) & (combined > 0)]
    if finite.size == 0:
        return 1e-12, 1.0
    upper = float(np.max(finite))
    lower = float(np.min(finite))
    upper = max(upper, np.finfo(float).eps)
    lower = max(min(lower, upper / 10.0), upper * 1e-12, np.finfo(float).tiny)
    return lower, upper * 1.05


def _shared_db_limits(*arrays: np.ndarray) -> tuple[float, float]:
    """Return a robust lower bound and full-range upper bound for heatmaps."""
    combined = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return -120.0, 0.0
    upper = float(np.max(finite))
    lower = max(float(np.percentile(finite, 2.0)), upper - 120.0)
    if not lower < upper:
        lower = upper - 1.0
    return lower, upper


def _set_channel_ticks(axis: plt.Axes, channel_names: list[str]) -> None:
    """Label every channel on small arrays and a stable subset on large ones."""
    count = len(channel_names)
    indices = np.arange(count, dtype=int) if count <= 32 else np.unique(np.linspace(0, count - 1, 12, dtype=int))
    axis.set_yticks(indices, [channel_names[int(index)] for index in indices])


def _plot_phase_diagnostics(
    diagnostics: TemporalDiagnostics,
    *,
    phase: str,
) -> bytes:
    """Plot one phase with axes fixed from both before and after data."""
    if phase not in {"before", "after"}:
        raise ValueError("phase must be 'before' or 'after'")
    is_before = phase == "before"
    label = "Before cleaning" if is_before else "After cleaning"
    color = "#d97706" if is_before else "#0f766e"
    amplitude = diagnostics.amplitude_before if is_before else diagnostics.amplitude_after
    psd = diagnostics.psd_before if is_before else diagnostics.psd_after
    histogram = diagnostics.histogram_before if is_before else diagnostics.histogram_after
    peak_to_peak = diagnostics.peak_to_peak_before if is_before else diagnostics.peak_to_peak_after
    clipped = diagnostics.clipped_before_fraction if is_before else diagnostics.clipped_after_fraction

    frequencies = diagnostics.frequencies
    fmax = float(frequencies[-1])
    frequency_mask = (frequencies >= 0.0) & (frequencies <= fmax)
    amplitude_before_uv = diagnostics.amplitude_before * 1e6
    amplitude_after_uv = diagnostics.amplitude_after * 1e6
    psd_before_uv = diagnostics.psd_before * 1e12
    psd_after_uv = diagnostics.psd_after * 1e12
    amplitude_limits = _positive_limits(
        amplitude_before_uv[:, frequency_mask],
        amplitude_after_uv[:, frequency_mask],
    )
    psd_limits = _positive_limits(
        psd_before_uv[:, frequency_mask],
        psd_after_uv[:, frequency_mask],
    )
    ptp_upper = float(np.max(np.concatenate([diagnostics.peak_to_peak_before, diagnostics.peak_to_peak_after])))
    ptp_upper = max(ptp_upper, 1.0)

    amplitude_floor = max(float(np.max(np.concatenate([amplitude_before_uv, amplitude_after_uv]))) * 1e-12, 1e-15)
    psd_floor = max(float(np.max(np.concatenate([psd_before_uv, psd_after_uv]))) * 1e-12, 1e-24)
    amplitude_before_db = 20.0 * np.log10(np.maximum(amplitude_before_uv, amplitude_floor))
    amplitude_after_db = 20.0 * np.log10(np.maximum(amplitude_after_uv, amplitude_floor))
    psd_before_db = 10.0 * np.log10(np.maximum(psd_before_uv, psd_floor))
    psd_after_db = 10.0 * np.log10(np.maximum(psd_after_uv, psd_floor))
    amplitude_db_limits = _shared_db_limits(
        amplitude_before_db[:, frequency_mask],
        amplitude_after_db[:, frequency_mask],
    )
    psd_db_limits = _shared_db_limits(
        psd_before_db[:, frequency_mask],
        psd_after_db[:, frequency_mask],
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    fig.suptitle(f"{label}: temporal and amplitude diagnostics", fontsize=16, fontweight="bold")

    median, lower, upper = _channel_summary(amplitude * 1e6)
    axes[0, 0].fill_between(
        frequencies[frequency_mask], lower[frequency_mask], upper[frequency_mask], color=color, alpha=0.2
    )
    axes[0, 0].plot(frequencies[frequency_mask], median[frequency_mask], color=color, linewidth=1.5)
    axes[0, 0].set(
        title="Typical one-sided amplitude spectrum",
        xlabel="Temporal frequency (Hz)",
        ylabel="Amplitude (µV)",
        xlim=(0.0, fmax),
        ylim=amplitude_limits,
        yscale="log",
    )

    median, lower, upper = _channel_summary(psd * 1e12)
    positive_lower = np.maximum(lower[frequency_mask], np.finfo(float).tiny)
    positive_upper = np.maximum(upper[frequency_mask], np.finfo(float).tiny)
    axes[0, 1].fill_between(
        frequencies[frequency_mask],
        positive_lower,
        positive_upper,
        color=color,
        alpha=0.2,
    )
    axes[0, 1].plot(
        frequencies[frequency_mask],
        np.maximum(median[frequency_mask], np.finfo(float).tiny),
        color=color,
        linewidth=1.5,
    )
    axes[0, 1].set(
        title="Typical Welch power spectral density",
        xlabel="Temporal frequency (Hz)",
        ylabel="PSD (µV²/Hz)",
        xlim=(0.0, fmax),
        ylim=psd_limits,
        yscale="log",
    )

    amplitude_db = amplitude_before_db if is_before else amplitude_after_db
    amplitude_image = axes[0, 2].imshow(
        amplitude_db[:, frequency_mask],
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, fmax, len(diagnostics.channel_names) - 0.5, -0.5),
        cmap="viridis",
        vmin=amplitude_db_limits[0],
        vmax=amplitude_db_limits[1],
    )
    axes[0, 2].set(title="All-channel amplitude spectrum", xlabel="Temporal frequency (Hz)", ylabel="EEG channel")
    _set_channel_ticks(axes[0, 2], diagnostics.channel_names)
    amplitude_colorbar = fig.colorbar(amplitude_image, ax=axes[0, 2], pad=0.02)
    amplitude_colorbar.set_label("Amplitude (dB re 1 µV)")

    psd_db = psd_before_db if is_before else psd_after_db
    psd_image = axes[1, 0].imshow(
        psd_db[:, frequency_mask],
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, fmax, len(diagnostics.channel_names) - 0.5, -0.5),
        cmap="magma",
        vmin=psd_db_limits[0],
        vmax=psd_db_limits[1],
    )
    axes[1, 0].set(title="All-channel Welch PSD", xlabel="Temporal frequency (Hz)", ylabel="EEG channel")
    _set_channel_ticks(axes[1, 0], diagnostics.channel_names)
    psd_colorbar = fig.colorbar(psd_image, ax=axes[1, 0], pad=0.02)
    psd_colorbar.set_label("PSD (dB re 1 µV²/Hz)")

    centers = 0.5 * (diagnostics.histogram_edges[:-1] + diagnostics.histogram_edges[1:])
    axes[1, 1].stairs(histogram, diagnostics.histogram_edges, color=color, linewidth=1.7, fill=True, alpha=0.25)
    axes[1, 1].axvline(0.0, color="#475569", linewidth=0.8)
    axes[1, 1].set(
        title=f"Sample-amplitude histogram ({clipped * 100:.3f}% outside view)",
        xlabel="Amplitude (µV)",
        ylabel="Probability density",
        xlim=(float(centers[0]), float(centers[-1])),
    )

    axes[1, 2].boxplot(
        peak_to_peak,
        vert=False,
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": color, "alpha": 0.35, "edgecolor": color},
        medianprops={"color": "#0f172a", "linewidth": 2},
        whiskerprops={"color": color},
        capprops={"color": color},
    )
    axes[1, 2].set(
        title="Two-second channel/window peak-to-peak amplitude (fliers hidden)",
        xlabel="Peak-to-peak amplitude (µV)",
        yticks=[],
        xlim=(0.0, ptp_upper),
    )

    for axis in (axes[0, 0], axes[0, 1], axes[1, 1], axes[1, 2]):
        axis.grid(True, alpha=0.2)
    fig.text(
        0.5,
        0.01,
        "Line = channel median; shaded band = channel IQR. Heatmaps expose every selected channel. "
        "Before and after use identical axes, color scales, bins, and windows.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    return _figure_png(fig)


def _plot_temporal_comparison(diagnostics: TemporalDiagnostics) -> bytes:
    """Plot direct frequency comparisons and the PSD change in decibels."""
    frequencies = diagnostics.frequencies
    fmax = float(frequencies[-1])
    frequency_mask = (frequencies >= 0.0) & (frequencies <= fmax)
    before_amplitude = np.median(diagnostics.amplitude_before, axis=0) * 1e6
    after_amplitude = np.median(diagnostics.amplitude_after, axis=0) * 1e6
    before_psd = np.median(diagnostics.psd_before, axis=0) * 1e12
    after_psd = np.median(diagnostics.psd_after, axis=0) * 1e12
    epsilon = np.finfo(float).tiny
    paired_change = 10.0 * np.log10(
        np.maximum(diagnostics.psd_after * 1e12, epsilon) / np.maximum(diagnostics.psd_before * 1e12, epsilon)
    )
    change_db, change_lower, change_upper = _channel_summary(paired_change)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].plot(frequencies[frequency_mask], before_amplitude[frequency_mask], color="#d97706", label="Before")
    axes[0].plot(frequencies[frequency_mask], after_amplitude[frequency_mask], color="#0f766e", label="After")
    axes[0].set_yscale("log")
    axes[0].set(title="Amplitude spectrum", xlabel="Frequency (Hz)", ylabel="Median amplitude (µV)", xlim=(0, fmax))
    axes[0].legend()

    axes[1].plot(frequencies[frequency_mask], before_psd[frequency_mask], color="#d97706", label="Before")
    axes[1].plot(frequencies[frequency_mask], after_psd[frequency_mask], color="#0f766e", label="After")
    axes[1].set_yscale("log")
    axes[1].set(title="Welch PSD", xlabel="Frequency (Hz)", ylabel="Median PSD (µV²/Hz)", xlim=(0, fmax))
    axes[1].legend()

    axes[2].fill_between(
        frequencies[frequency_mask],
        change_lower[frequency_mask],
        change_upper[frequency_mask],
        color="#64748b",
        alpha=0.18,
    )
    axes[2].fill_between(
        frequencies[frequency_mask],
        0.0,
        change_db[frequency_mask],
        where=change_db[frequency_mask] <= 0,
        color="#2563eb",
        alpha=0.35,
    )
    axes[2].fill_between(
        frequencies[frequency_mask],
        0.0,
        change_db[frequency_mask],
        where=change_db[frequency_mask] > 0,
        color="#9333ea",
        alpha=0.3,
    )
    axes[2].plot(frequencies[frequency_mask], change_db[frequency_mask], color="#334155", linewidth=0.9)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set(title="After / before PSD", xlabel="Frequency (Hz)", ylabel="Change (dB)", xlim=(0, fmax))

    for axis in axes:
        axis.grid(True, alpha=0.2)
    fig.suptitle("Matched before/after frequency comparison", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _figure_png(fig)


def _project_xy(coordinates: np.ndarray) -> np.ndarray:
    """Normalize head-coordinate x/y values for node-link plotting."""
    xy = np.asarray(coordinates[:, :2], dtype=float)
    xy -= np.mean(xy, axis=0)
    radius = float(np.max(np.linalg.norm(xy, axis=1)))
    if radius <= np.finfo(float).eps:
        raise ValueError("Sensor coordinates collapse to a single 2-D location.")
    return xy / radius


def _plot_graph_spectrum(
    data: PairedRecordingData,
    geometry: SpatialGeometry,
) -> tuple[bytes, dict[str, float | int | str]]:
    """Plot a compact sensor graph, eigenspectrum, and energy comparison."""
    adjacency = _physical_sensor_graph(geometry.coordinates_3d)
    eigenvalues, eigenvectors = _laplacian_eigendecomposition(adjacency)
    energy_before = _graph_spectral_energy(data.before_segments, geometry.channel_indices, eigenvectors) * 1e12
    energy_after = _graph_spectral_energy(data.after_segments, geometry.channel_indices, eigenvectors) * 1e12
    xy = _project_xy(geometry.coordinates_3d)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for row, column in zip(*np.where(np.triu(adjacency, k=1) > 0), strict=True):
        axes[0].plot(
            xy[[row, column], 0],
            xy[[row, column], 1],
            color="#94a3b8",
            linewidth=0.5 + adjacency[row, column],
            alpha=0.55,
            zorder=1,
        )
    axes[0].scatter(xy[:, 0], xy[:, 1], s=25, color="#0f766e", edgecolor="white", linewidth=0.4, zorder=2)
    neighbor_count = min(4, len(geometry.channel_indices) - 1)
    axes[0].set(title=f"Sensor {neighbor_count}-nearest-neighbor graph", aspect="equal")
    axes[0].axis("off")

    axes[1].plot(np.arange(len(eigenvalues)), eigenvalues, color="#2563eb", marker="o", markersize=3)
    axes[1].set(title="Laplacian eigenspectrum", xlabel="Graph mode k", ylabel="Eigenvalue λₖ")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(
        eigenvalues,
        np.maximum(energy_before, np.finfo(float).tiny),
        color="#d97706",
        marker="o",
        markersize=3,
        label="Before",
    )
    axes[2].plot(
        eigenvalues,
        np.maximum(energy_after, np.finfo(float).tiny),
        color="#0f766e",
        marker="o",
        markersize=3,
        label="After",
    )
    axes[2].set_yscale("log")
    axes[2].set(
        title="Empirical graph spectral energy", xlabel="Graph frequency λₖ (not Hz)", ylabel="Mean mode energy (µV²)"
    )
    axes[2].grid(True, alpha=0.2)
    axes[2].legend()

    orthogonality_error = float(np.linalg.norm(eigenvectors.T @ eigenvectors - np.eye(len(eigenvalues))))
    split = max(1, len(eigenvalues) // 3)
    low_ratio = _safe_ratio(float(np.sum(energy_after[:split])), float(np.sum(energy_before[:split])))
    high_ratio = _safe_ratio(float(np.sum(energy_after[-split:])), float(np.sum(energy_before[-split:])))
    fig.suptitle("Compact electrode-graph comparison (L = UΛUᵀ)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _figure_png(fig), {
        "channels": len(geometry.channel_indices),
        "geometry_origin": geometry.origin,
        "coordinate_frame": geometry.coordinate_frame,
        "coverage_note": geometry.coverage_note,
        "neighbor_count": neighbor_count,
        "orthogonality_error": orthogonality_error,
        "zero_eigenvalues": int(np.count_nonzero(np.isclose(eigenvalues, 0.0, atol=1e-10))),
        "low_graph_mode_preservation": low_ratio,
        "high_graph_mode_preservation": high_ratio,
    }


def _coherence_layout(
    diagnostics: CoherenceDiagnostics,
    geometry: SpatialGeometry | None,
) -> tuple[np.ndarray, str]:
    """Return scalp positions when complete, otherwise an abstract layout."""
    if geometry is not None:
        position_by_index = {
            int(global_index): geometry.coordinates_3d[local_index]
            for local_index, global_index in enumerate(geometry.channel_indices)
        }
        if all(int(index) in position_by_index for index in diagnostics.channel_indices):
            coordinates = np.asarray([position_by_index[int(index)] for index in diagnostics.channel_indices])
            return _project_xy(coordinates), "scalp-position layout"

    adjacency = diagnostics.thresholded_after
    if len(adjacency) >= 3 and np.count_nonzero(adjacency) > 0:
        _, eigenvectors = _laplacian_eigendecomposition(adjacency)
        coordinates = eigenvectors[:, 1:3]
        if coordinates.shape[1] == 2 and np.max(np.linalg.norm(coordinates, axis=1)) > 0:
            coordinates -= np.mean(coordinates, axis=0)
            coordinates /= np.max(np.linalg.norm(coordinates, axis=1))
            return coordinates, "abstract graph-spectral layout"
    angles = np.linspace(0.0, 2.0 * np.pi, len(adjacency), endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)]), "abstract circular layout"


def _plot_coherence_network(
    diagnostics: CoherenceDiagnostics,
    geometry: SpatialGeometry | None,
) -> tuple[bytes, str]:
    """Plot before/after/change matrices plus the clean community network."""
    order = np.lexsort((np.arange(len(diagnostics.labels)), diagnostics.labels))
    before = diagnostics.before[np.ix_(order, order)].copy()
    after = diagnostics.after[np.ix_(order, order)].copy()
    difference = after - before
    np.fill_diagonal(before, np.nan)
    np.fill_diagonal(after, np.nan)
    np.fill_diagonal(difference, np.nan)
    ordered_names = [diagnostics.channel_names[int(index)] for index in order]

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    before_image = axes[0, 0].imshow(before, origin="lower", aspect="equal", vmin=0.0, vmax=1.0, cmap="viridis")
    after_image = axes[0, 1].imshow(after, origin="lower", aspect="equal", vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0, 0].set_title("Before: band-mean magnitude-squared coherence")
    axes[0, 1].set_title("After: band-mean magnitude-squared coherence")
    fig.colorbar(before_image, ax=axes[0, 0], fraction=0.046, label="Coherence (0–1)")
    fig.colorbar(after_image, ax=axes[0, 1], fraction=0.046, label="Coherence (0–1)")

    difference_limit = max(float(np.nanpercentile(np.abs(difference), 99)), 0.05)
    difference_image = axes[1, 0].imshow(
        difference,
        origin="lower",
        aspect="equal",
        vmin=-difference_limit,
        vmax=difference_limit,
        cmap="RdBu_r",
    )
    axes[1, 0].set_title("After − before coherence")
    fig.colorbar(difference_image, ax=axes[1, 0], fraction=0.046, label="Coherence change")

    tick_step = max(1, math.ceil(len(ordered_names) / 24))
    tick_indices = np.arange(0, len(ordered_names), tick_step)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(tick_indices, [ordered_names[index] for index in tick_indices], rotation=90, fontsize=7)
        axis.set_yticks(tick_indices, [ordered_names[index] for index in tick_indices], fontsize=7)
        axis.set_xlabel("Channel, ordered by clean-data community")
        axis.set_ylabel("Channel")

    positions, layout_name = _coherence_layout(diagnostics, geometry)
    network_axis = axes[1, 1]
    edge_rows, edge_columns = np.where(np.triu(diagnostics.thresholded_after, k=1) > 0)
    edge_values = diagnostics.thresholded_after[edge_rows, edge_columns]
    edge_min = float(np.min(edge_values)) if edge_values.size else 0.0
    edge_span = max(float(np.ptp(edge_values)) if edge_values.size else 0.0, np.finfo(float).eps)
    for row, column, weight in zip(edge_rows, edge_columns, edge_values, strict=True):
        normalized = (float(weight) - edge_min) / edge_span
        network_axis.plot(
            positions[[row, column], 0],
            positions[[row, column], 1],
            color="#64748b",
            linewidth=0.4 + 2.2 * normalized,
            alpha=0.18 + 0.55 * normalized,
            zorder=1,
        )

    strengths = np.sum(diagnostics.thresholded_after, axis=1)
    if np.ptp(strengths) > 0:
        node_sizes = 45.0 + 120.0 * (strengths - np.min(strengths)) / np.ptp(strengths)
    else:
        node_sizes = np.full(len(strengths), 70.0)
    network_axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=diagnostics.labels,
        s=node_sizes,
        cmap="tab20",
        edgecolor="white",
        linewidth=0.7,
        zorder=2,
    )
    if len(diagnostics.channel_names) <= 32:
        for (x, y), name in zip(positions, diagnostics.channel_names, strict=True):
            network_axis.text(x, y + 0.045, name, ha="center", va="bottom", fontsize=7)
    if layout_name == "scalp-position layout":
        network_axis.add_patch(plt.Circle((0, 0), 1.04, fill=False, color="#334155", linewidth=1.0))
        network_axis.plot([-0.08, 0.0, 0.08], [1.03, 1.15, 1.03], color="#334155", linewidth=1.0)
    precision_label = "low-precision exploratory; " if diagnostics.low_precision else ""
    network_axis.set_title(
        f"After-cleaning spectral-modularity network\n{layout_name}; {precision_label}Q≈{diagnostics.modularity:.2f}"
    )
    network_axis.set_aspect("equal")
    network_axis.axis("off")

    fig.suptitle(
        f"Sensor coherence communities ({diagnostics.fmin:g}–{diagnostics.fmax:g} Hz)",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        f"Matrix diagonal masked. Rank-based network keeps {diagnostics.edge_density * 100:.1f}% of "
        f"after-cleaning edges (top-N ranks; weakest retained weight {diagnostics.edge_threshold:.3f}; "
        "cutoff ties broken deterministically); communities use Newman's "
        "leading modularity eigenvectors and are not a statistical-significance result.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    return _figure_png(fig), layout_name


def _read_json(path: Path) -> dict[str, Any]:
    """Read one internally generated JSON report."""
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, dict) else {"value": payload}


def _format_value(value: Any) -> str:
    """Format a compact human-readable pipeline value."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.5g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in value) or "—"
    if isinstance(value, dict):
        return json.dumps(_json_safe(value), sort_keys=True)
    return str(value)


def _pipeline_timeline(payload: dict[str, Any]) -> str:
    """Render a simple, always-visible list of processing stages."""
    cards: list[str] = []
    for processor in payload.get("processors", []):
        cards.append(
            """
            <article class="step-card">
              <div class="step-index">{index}</div>
              <div>
                <h3>{name}</h3>
                <p class="eyebrow">{processor_type}</p>
                <p>{description}</p>
              </div>
            </article>
            """.format(
                index=escape(str(processor.get("index", "?"))),
                name=escape(str(processor.get("name", "unnamed processor"))),
                processor_type=escape(str(processor.get("type", "unknown"))),
                description=escape(str(processor.get("description", ""))),
            )
        )
    return "".join(cards) or '<p class="unavailable">No processor description was recorded.</p>'


def _flex_decisions_html(payload: dict[str, Any]) -> str:
    """Render every active Flex decision at the top of the report."""
    corrections = payload.get("flex_corrections", [])
    if not corrections:
        return '<p class="unavailable">No Flex decision manifest was recorded.</p>'

    cards: list[str] = []
    for correction in corrections:
        decisions = correction.get("decisions", {})
        decision_rows = "".join(
            f"<tr><th>{escape(str(name).replace('_', ' ').title())}</th>"
            f"<td><code>{escape(json.dumps(_json_safe(value), sort_keys=True))}</code></td></tr>"
            for name, value in decisions.items()
        )
        cards.append(
            """
            <article class="decision-card">
              <h3>{preset}</h3>
              <p><strong>Closest legacy resemblance:</strong> {resemblance}</p>
              <table>{rows}</table>
            </article>
            """.format(
                preset=escape(str(correction.get("preset", "custom_flex"))),
                resemblance=escape(str(correction.get("legacy_algorithm_resemblance", "custom Flex"))),
                rows=decision_rows,
            )
        )
    return "".join(cards)


def _quality_metrics_html(payload: dict[str, Any]) -> str:
    """Render metric values with one direct reading for each value."""
    rows: list[str] = []
    for name, record in payload.get("metrics", {}).items():
        rows.append(
            "<tr>"
            f"<th>{escape(str(name).replace('_', ' ').title())}</th>"
            f"<td>{escape(_format_value(record.get('value')))} {escape(str(record.get('unit', '')))}</td>"
            f"<td>{escape(str(record.get('interpretation', '')))}</td>"
            "</tr>"
        )
    if not rows:
        return '<p class="unavailable">Quality metrics were unavailable.</p>'
    return (
        "<table><thead><tr><th>Metric</th><th>Value</th><th>Direct interpretation</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _compact_matrix_display_payload(chunked_result: Any) -> dict[str, list[dict[str, Any]]]:
    """Collect only the small Flex fields needed to render the HTML index."""
    chunks = list(getattr(chunked_result, "chunks", []))
    summaries: list[dict[str, Any]] = []
    plot_records: list[dict[str, Any]] = []
    for chunk, result in zip(chunks, _chunked_results(chunked_result), strict=False):
        context = getattr(result, "context", None)
        custom = getattr(getattr(context, "metadata", None), "custom", {})
        if not isinstance(custom, dict):
            continue
        for report in custom.get("artifact_template_matrices", []):
            if not isinstance(report, dict):
                continue
            report_chunk = report.get("chunk", {})
            summaries.append(
                {
                    "processor_name": report.get("processor_name", "Flex correction"),
                    "chunk_index": (
                        report_chunk.get("index", chunk.index) if isinstance(report_chunk, dict) else chunk.index
                    ),
                    "num_triggers": report.get("num_triggers", "?"),
                    "channel_count": len(report.get("channels", [])),
                }
            )
        for record in custom.get("artifact_template_matrix_plots", []):
            if not isinstance(record, dict):
                continue
            plot_records.append(
                {
                    "path": record.get("path"),
                    "processor_name": record.get("processor_name", "Flex correction"),
                    "channel_name": record.get("channel_name", "unknown"),
                    "stage": record.get("stage", "?"),
                    "chunk_index": record.get("chunk_index", chunk.index),
                }
            )
    return {"summaries": summaries, "diagnostic_plots": plot_records}


def _matrix_diagnostics_html(matrix_display: dict[str, list[dict[str, Any]]]) -> str:
    """Render compact Flex summaries around a streamed image placeholder."""
    reports = matrix_display.get("summaries", [])
    summaries: list[str] = []
    for report_index, report in enumerate(reports, start=1):
        summaries.append(
            """
            <div class="matrix-summary">
              <strong>{processor}</strong>
              <span>chunk {chunk}</span>
              <span>{triggers} triggers</span>
              <span>{channels} channels</span>
              <span>N = A @ D</span>
            </div>
            """.format(
                processor=escape(str(report.get("processor_name", f"Flex report {report_index}"))),
                chunk=escape(str(report.get("chunk_index", "?"))),
                triggers=escape(str(report.get("num_triggers", "?"))),
                channels=escape(str(report.get("channel_count", "?"))),
            )
        )
    return "".join(summaries) + f'<div class="asset-grid">{MATRIX_ASSET_PLACEHOLDER}</div>'


def _trusted_matrix_plot_path(record: dict[str, Any], target_dir: Path) -> Path | None:
    """Resolve a matrix PNG only when it remains inside the output folder."""
    raw_path = record.get("path")
    if not raw_path:
        return None
    plot_path = Path(str(raw_path)).expanduser().resolve()
    target_root = target_dir.resolve()
    if not plot_path.is_relative_to(target_root):
        return None
    if plot_path.suffix.lower() != ".png" or not plot_path.is_file():
        return None
    return plot_path


def _bounded_matrix_preview(plot_path: Path) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """Return a readable JPEG preview with a strict per-figure byte cap."""
    with Image.open(plot_path) as opened:
        original_size = opened.size
        image = opened.convert("RGB")
    image.thumbnail(MAX_MATRIX_PREVIEW_SIZE, Image.Resampling.LANCZOS)

    quality = 88
    while True:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
        payload = buffer.getvalue()
        if len(payload) <= MAX_MATRIX_PREVIEW_BYTES:
            return payload, original_size, image.size
        if quality > 48:
            quality -= 10
            continue
        width, height = image.size
        if width <= 400 and height <= 300:
            # A 400×300 JPEG is comfortably below the cap in practice.  Keep
            # the guard explicit in case a future Pillow encoder differs.
            if len(payload) > MAX_MATRIX_PREVIEW_BYTES:
                raise ValueError("Could not compress matrix preview below the configured byte cap.")
            return payload, original_size, image.size
        image = image.resize(
            (max(1, int(width * 0.8)), max(1, int(height * 0.8))),
            Image.Resampling.LANCZOS,
        )
        quality = 78


def _write_base64_stream(source: io.BytesIO, destination: Any) -> None:
    """Stream one binary payload as concatenable base64 text chunks."""
    source.seek(0)
    chunk_size = 57 * 1_024  # divisible by three, so only the final block pads
    while block := source.read(chunk_size):
        destination.write(base64.b64encode(block).decode("ascii"))


def _write_matrix_assets(
    destination: Any,
    records: list[dict[str, Any]],
    target_dir: Path,
) -> None:
    """Stream every trusted Flex image without accumulating aggregate base64."""
    written = 0
    for record in records:
        plot_path = _trusted_matrix_plot_path(record, target_dir)
        if plot_path is None:
            continue
        try:
            payload, original_size, preview_size = _bounded_matrix_preview(plot_path)
        except (OSError, ValueError) as exc:
            logger.warning("Could not embed Flex matrix preview '{}': {}", plot_path, exc)
            continue

        processor = escape(str(record.get("processor_name", "Flex correction")))
        chunk = escape(str(record.get("chunk_index", "?")))
        stage = escape(str(record.get("stage", "?")))
        channel = escape(str(record.get("channel_name", "unknown")))
        destination.write('<figure class="asset-card"><img src="data:image/jpeg;base64,')
        _write_base64_stream(io.BytesIO(payload), destination)
        destination.write(
            '" alt="Flex template matrix diagnostic for '
            f'{processor}"><figcaption>Chunk {chunk}, stage {stage}: '
            f"<strong>{processor}</strong>, representative channel {channel}. "
            f"Bounded HTML preview {preview_size[0]}×{preview_size[1]} px from "
            f"the retained {original_size[0]}×{original_size[1]} px companion PNG."
            "</figcaption></figure>"
        )
        written += 1

    if written == 0:
        destination.write(
            '<p class="unavailable">No trusted rendered Flex matrix image was available. '
            "Structured calculations remain in the companion matrix JSON.</p>"
        )


def _warning_list(warnings: list[str]) -> str:
    """Render unique report warnings in stable order."""
    unique = list(dict.fromkeys(str(item) for item in warnings if item))
    if not unique:
        return '<p class="ok">All requested report sections were generated.</p>'
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in unique) + "</ul>"


def _summary_metrics(
    data: PairedRecordingData | None,
    diagnostics: TemporalDiagnostics | None,
    pipeline_payload: dict[str, Any],
) -> str:
    """Render the compact acquisition and outcome cards."""
    chunks = pipeline_payload.get("result", {}).get("chunks", [])
    runtime = float(pipeline_payload.get("result", {}).get("execution_time_seconds", 0.0))
    items: list[tuple[str, str, str]] = [
        ("Pipeline", str(pipeline_payload.get("pattern", "unknown")), "run pattern"),
        ("Correction", str(pipeline_payload.get("correction_mode", "unknown")), "Flex strategy"),
        ("Chunks", str(len(chunks)), "exported cores"),
        ("Runtime", f"{runtime:.2f} s", "processing time"),
    ]
    if data is not None:
        items.extend(
            [
                ("EEG channels", str(len(data.channel_names)), "matched before/after"),
                ("Analysis sampling", f"{data.sfreq:g} Hz", "paired report grid"),
                ("Recording", f"{data.source_duration_seconds:.1f} s", "source duration"),
                ("Analyzed", f"{data.analyzed_duration_seconds:.1f} s", "stratified paired windows"),
            ]
        )
    if diagnostics is not None:
        before_ptp = float(np.median(diagnostics.peak_to_peak_before))
        after_ptp = float(np.median(diagnostics.peak_to_peak_after))
        change = 100.0 * (after_ptp / before_ptp - 1.0) if before_ptp > 0 else float("nan")
        items.append(("Median peak-to-peak", f"{before_ptp:.2f} → {after_ptp:.2f} µV", f"{change:+.1f}%"))

    return "".join(
        f'<div class="metric-card"><span>{escape(title)}</span><strong>{escape(value)}</strong><small>{escape(note)}</small></div>'
        for title, value, note in items
    )


def _asset_figure(payload: bytes | None, mime_type: str, alt: str, caption: str) -> str:
    """Render an embedded figure or an explicit unavailable panel."""
    if payload is None:
        return f'<p class="unavailable">{escape(caption)}</p>'
    return (
        '<figure class="asset-card">'
        f'<img src="{_data_uri(payload, mime_type)}" alt="{escape(alt)}">'
        f"<figcaption>{escape(caption)}</figcaption>"
        "</figure>"
    )


def _runtime_version() -> str:
    """Return the installed FACETpy version without assuming installation."""
    try:
        return version("facetpy")
    except PackageNotFoundError:
        return "development checkout"


def _info_provenance_text(info: mne.Info, bad_channels: list[str], *, label: str) -> str:
    """Summarize the acquisition fields that affect spatial/spectral reading."""
    line_frequency = info.get("line_freq")
    line_frequency_text = f"{float(line_frequency):g} Hz" if line_frequency is not None else "not recorded"
    projections = [str(projection.get("desc", "unnamed projection")) for projection in info.get("projs", [])]
    projection_text = ", ".join(projections) if projections else "none recorded"
    bad_text = ", ".join(bad_channels) if bad_channels else "none recorded"
    return (
        f"{label} passband {float(info['highpass']):g}–{float(info['lowpass']):g} Hz; "
        f"line frequency {line_frequency_text}; custom-reference flag "
        f"{info.get('custom_ref_applied', 'not recorded')}; projection records {projection_text}; "
        f"bad channels {bad_text}. Explicit reference-electrode identity is not represented by the "
        "available MNE Info unless named in a projection."
    )


def _build_html(
    *,
    input_path: Path,
    pipeline_payload: dict[str, Any],
    matrix_display: dict[str, list[dict[str, Any]]],
    matrix_report_name: str,
    manifest_payload: dict[str, Any],
    quality_metrics_payload: dict[str, Any],
    data: PairedRecordingData | None,
    diagnostics: TemporalDiagnostics | None,
    comparison_png: bytes | None,
    graph_png: bytes | None,
    graph_metadata: dict[str, Any] | None,
    coherence_png: bytes | None,
    coherence: CoherenceDiagnostics | None,
    coherence_layout: str | None,
    warnings: list[str],
) -> str:
    """Assemble the complete, offline report document."""
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    source_display = str(input_path)
    matrix_html = _matrix_diagnostics_html(matrix_display)
    flex_decisions_html = _flex_decisions_html(pipeline_payload)
    quality_metrics_html = _quality_metrics_html(quality_metrics_payload)
    low_graph_ratio = graph_metadata.get("low_graph_mode_preservation") if graph_metadata else None
    high_graph_ratio = graph_metadata.get("high_graph_mode_preservation") if graph_metadata else None
    geometry_note = (
        f"The shared sensor-coordinate graph used {str(graph_metadata['geometry_origin'])}; "
        f"coordinate frame {str(graph_metadata['coordinate_frame'])}; "
        f"{str(graph_metadata['coverage_note'])}; "
        f"low-mode corrected/source energy={_format_value(low_graph_ratio)}, "
        f"high-mode corrected/source energy={_format_value(high_graph_ratio)}. "
        "Ratios below 1 mean less spatial-mode energy after cleaning; compare low and high modes rather than "
        "treating either direction as automatically better."
        if graph_metadata is not None
        else f"Graph spectral analysis requires at least {MIN_SPATIAL_CHANNELS} defensibly scaled, broadly distributed electrode positions and was unavailable."
    )
    coherence_band = (
        f"{coherence.fmin:g}–{coherence.fmax:g} Hz"
        if coherence is not None
        else f"{COHERENCE_FMIN_HZ:g}–{COHERENCE_FMAX_HZ:g} Hz"
    )
    coherence_note = (
        f"Magnitude-squared coherence averaged over {coherence.fmin:g}–{coherence.fmax:g} Hz using "
        f"{coherence.frame_count} overlapping {coherence.nperseg / data.sfreq:.2f} s Hann windows, "
        f"{len(coherence.channel_names)} channels, "
        f"and {len(np.unique(coherence.labels))} spectral-modularity communities (Q≈{coherence.modularity:.2f}). "
        + (
            f"Because only {coherence.frame_count} overlapping windows were available, community/Q estimates are low precision and can be unstable. "
            if coherence.low_precision
            else ""
        )
        + f"Mean off-diagonal coherence changed from {_format_value(float(np.mean(coherence.before[~np.eye(len(coherence.before), dtype=bool)])))} "
        f"to {_format_value(float(np.mean(coherence.after[~np.eye(len(coherence.after), dtype=bool)])))}. "
        f"Node view uses a {coherence_layout or 'graph layout'}. Lower or higher coherence is not automatically "
        "better; use this only to spot large network-wide shifts."
        if coherence is not None and data is not None
        else "Coherence analysis requires at least two common channels and multiple complete windows."
    )
    analysis_windows = (
        "".join(
            "<tr>"
            f"<td>{int(window['chunk_index'])}</td>"
            f"<td>{float(window['start_seconds']):.3f}</td>"
            f"<td>{float(window['stop_seconds']):.3f}</td>"
            f"<td>{int(window['samples'])}</td>"
            f"<td>{escape(str(window.get('selection', 'not recorded')))}</td>"
            "</tr>"
            for window in data.segment_windows
        )
        if data is not None
        else '<tr><td colspan="5">No paired signal windows available</td></tr>'
    )
    temporal_method = (
        f"Welch estimates use {diagnostics.welch_segment_seconds:.3f} s windows and "
        f"{diagnostics.frequency_resolution_hz:.4g} Hz bin spacing (Hann equivalent-noise bandwidth "
        f"≈ {1.5 * diagnostics.frequency_resolution_hz:.4g} Hz)."
        if diagnostics is not None
        else "Temporal spectral diagnostics were unavailable for this run."
    )
    recording_method = "Source acquisition metadata were unavailable."
    sampling_method = "No paired analysis grid was available."
    if data is not None:
        source_recording = _info_provenance_text(
            data.source_info,
            data.source_bad_channels,
            label="Source",
        )
        corrected_recording = (
            _info_provenance_text(
                data.corrected_info,
                data.corrected_bad_channels,
                label="Corrected output",
            )
            if data.corrected_info is not None
            else "Corrected-output Info was unavailable."
        )
        recording_method = f"{source_recording} {corrected_recording}"
        source_rate = data.source_sfreq if data.source_sfreq is not None else data.sfreq
        corrected_rate = data.corrected_sfreq if data.corrected_sfreq is not None else data.sfreq
        sampling_method = (
            f"Source {source_rate:g} Hz; corrected output {corrected_rate:g} Hz; report analysis "
            f"{data.sfreq:g} Hz. Retained arrays and total native-rate reads are each capped at "
            f"{MAX_ANALYSIS_VALUES_PER_PHASE:,} channel-sample values per phase."
        )
    software_versions = " · ".join(
        f"{name} {escape(version(package))}"
        for name, package in (
            ("NumPy", "numpy"),
            ("SciPy", "scipy"),
            ("MNE", "mne"),
            ("Pillow", "pillow"),
            ("Matplotlib", "matplotlib"),
        )
    )

    css = """
    :root { --ink:#172b3a; --muted:#5f7180; --line:#d8e0e5; --paper:#f3f6f8; --card:#fff; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--paper); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
    nav { display:flex; gap:1rem; overflow-x:auto; padding:.7rem max(1rem,calc((100vw - 1050px)/2)); background:#17324d; }
    nav a { color:white; text-decoration:none; white-space:nowrap; font-size:.84rem; }
    header { padding:2.2rem max(1rem,calc((100vw - 1050px)/2)); color:white; background:#24536b; }
    header h1 { margin:.2rem 0; font-size:2.25rem; }
    header p { color:#dcecf2; }
    main { max-width:1050px; margin:auto; padding:1rem; }
    section { margin:1rem 0; padding:1.3rem; background:var(--card); border:1px solid var(--line); }
    h2 { margin:.1rem 0 .5rem; font-size:1.5rem; } h3 { margin:.35rem 0; }
    .eyebrow { margin:0; color:var(--muted); font-size:.74rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    .lead,.muted { color:var(--muted); }
    .disclaimer { margin-top:1rem; padding:.8rem; border-left:4px solid #d6a017; background:#fff8df; color:#5d4900; }
    .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:.6rem; margin:1rem 0; }
    .metric-card { display:flex; flex-direction:column; padding:.75rem; background:#edf5f5; }
    .metric-card span,.metric-card small { color:var(--muted); } .metric-card strong { margin:.15rem 0; font-size:1.15rem; }
    .phase-flow { display:flex; align-items:center; gap:.6rem; margin:1rem 0; }
    .phase { flex:1; padding:.7rem; text-align:center; font-weight:700; background:#edf2f5; }
    .asset-card { margin:1rem 0; border:1px solid var(--line); background:white; }
    .asset-card img { display:block; width:100%; height:auto; } .asset-card figcaption { padding:.7rem; color:var(--muted); background:#f8fafb; }
    .asset-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:.7rem; }
    .matrix-summary { display:flex; flex-wrap:wrap; gap:.7rem; padding:.55rem; margin:.4rem 0; background:#f2effa; }
    .matrix-summary span { color:var(--muted); }
    .decision-card { margin:.8rem 0; padding:.8rem; border-left:4px solid #6d55a4; background:#faf9fd; }
    .steps { display:grid; gap:.5rem; } .step-card { display:grid; grid-template-columns:2.5rem 1fr; gap:.6rem; padding:.7rem; border:1px solid var(--line); }
    .step-index { display:grid; place-items:center; width:2rem; height:2rem; border-radius:50%; color:white; background:#24536b; font-weight:700; }
    code { white-space:normal; overflow-wrap:anywhere; }
    table { width:100%; border-collapse:collapse; } th,td { padding:.45rem .55rem; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; } th { width:27%; }
    .unavailable { padding:.8rem; border:1px dashed #94a3b8; background:#f8fafb; color:var(--muted); }
    .ok { color:#0f766e; font-weight:700; }
    footer { max-width:1050px; margin:auto; padding:0 1rem 2rem; color:var(--muted); }
    @media(max-width:700px){ section{padding:.8rem}.phase-flow{display:block}.phase{margin:.4rem 0}.arrow{display:none}.asset-grid{grid-template-columns:1fr}table{font-size:.88rem} }
    @media print { nav{display:none} body{background:white} section{break-inside:avoid} }
    """

    metric_records = quality_metrics_payload.get("metrics", {})
    temporal_note = " ".join(
        str(metric_records.get(name, {}).get("interpretation", ""))
        for name in ("scanner_peak_residual", "theta_preservation", "alpha_preservation")
    )
    chunking_mode = manifest_payload.get("chunking_mode", "not recorded")
    memory_budget = manifest_payload.get("memory_budget_bytes", "not recorded")

    # Add user-managed citation links here later if the report should include
    # a references section. Citations are intentionally omitted by default.
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FACETpy cleaning report — {escape(input_path.name)}</title><style>{css}</style></head>
<body>
<nav aria-label="Report sections"><a href="#overview">Overview</a><a href="#decisions">Flex decisions</a><a href="#metrics">Metrics</a><a href="#during">Templates</a><a href="#temporal">Spectra</a><a href="#graph">Spatial</a><a href="#coherence">Coherence</a><a href="#pipeline">Pipeline</a><a href="#methods">Methods</a></nav>
<header><p class="eyebrow">FACETpy · EEG-fMRI artifact correction quality control</p><h1>Before, during, and after cleaning</h1><p>{escape(source_display)}</p><div class="disclaimer"><strong>Analytic/QC aid—not a diagnosis.</strong> Use these summaries with the raw EEG and acquisition context.</div></header>
<main>
<section id="overview"><p class="eyebrow">Run identity</p><h2>Recording overview</h2><div class="phase-flow"><div class="phase">Before<br><small>source EEG</small></div><div class="arrow">→</div><div class="phase">During<br><small>Flex D → A → N</small></div><div class="arrow">→</div><div class="phase">After<br><small>corrected EEG</small></div></div><div class="metrics">{_summary_metrics(data, diagnostics, pipeline_payload)}</div><h3>Notices</h3>{_warning_list(warnings)}</section>

<section id="decisions"><p class="eyebrow">Correction configuration</p><h2>Flex decisions and legacy resemblance</h2><p class="lead">These are the complete active matrix decisions. They are shown directly rather than hidden inside processor parameters.</p>{flex_decisions_html}</section>

<section id="metrics"><p class="eyebrow">Outcome summary</p><h2>Quality metrics</h2><p class="lead">Preservation ratios use 1.0 for unchanged power. Scanner residual uses 0 as the ideal. No single value proves a good correction; read suppression and preservation together.</p>{quality_metrics_html}</section>

<section id="during"><p class="eyebrow">Template construction</p><h2>Flex artifact templates</h2><p class="lead">D contains measured artifact epochs, A contains the selected weights, and N = A @ D is subtracted from each target epoch.</p>{matrix_html}<p class="muted">Complete numeric matrices remain in {escape(matrix_report_name)}.</p></section>

<section id="temporal"><p class="eyebrow">Temporal-frequency comparison</p><h2>Source versus corrected spectra</h2><p>{escape(temporal_note)}</p>{_asset_figure(comparison_png, "image/png", "Matched before and after temporal-frequency comparison", "Orange is source, green is corrected, and the change panel is corrected/source in dB. Negative values mean less power after cleaning; that is useful at artifact peaks but can indicate attenuation in physiological bands.")}</section>

<section id="graph"><p class="eyebrow">Spatial-frequency comparison</p><h2>Compact sensor-graph spectrum</h2><p class="lead">Graph frequency describes variation across neighboring sensors, not time in Hz.</p>{_asset_figure(graph_png, "image/png", "Sensor graph, Laplacian eigenspectrum, and graph energy", geometry_note)}</section>

<section id="coherence"><p class="eyebrow">Exploratory sensor coupling</p><h2>{coherence_band} coherence</h2><p class="lead">Coherence is statistical coupling in sensor space, not anatomical or causal connectivity.</p>{_asset_figure(coherence_png, "image/png", "Before and after coherence matrices and network", coherence_note)}</section>

<section id="pipeline"><p class="eyebrow">Reproducibility</p><h2>Processing stages</h2><p class="lead">The stage order is shown without the former hidden parameter blocks. Full CLI options and processor parameters remain in the companion pipeline JSON.</p><div class="steps">{_pipeline_timeline(pipeline_payload)}</div></section>

<section id="methods"><p class="eyebrow">Methods and limits</p><h2>What was analyzed</h2><table><tr><th>Acquisition metadata</th><td>{escape(recording_method)}</td></tr><tr><th>Bounded sampling</th><td>{escape(sampling_method)}</td></tr><tr><th>Temporal method</th><td>{escape(temporal_method)}</td></tr><tr><th>Chunking</th><td>{escape(str(chunking_mode))}; memory budget {escape(str(memory_budget))} bytes; only non-overlapping exported cores are compared.</td></tr><tr><th>Spatial graph</th><td>One distance-weighted four-neighbor sensor graph and its combinatorial Laplacian; graph frequency is not Hz.</td></tr><tr><th>Coherence</th><td>Magnitude-squared coherence in {coherence_band}, capped at {MAX_COHERENCE_CHANNELS} channels and interpreted only as an exploratory change summary.</td></tr><tr><th>Software</th><td>{software_versions}</td></tr></table><h3>Analyzed windows</h3><table><thead><tr><th>Chunk</th><th>Start (s)</th><th>Stop (s)</th><th>Samples</th><th>Selection</th></tr></thead><tbody>{analysis_windows}</tbody></table></section>
</main>
<footer>Generated {escape(generated_at)} · FACETpy {escape(_runtime_version())} · Python {escape(platform.python_version())} · self-contained static HTML</footer>
</body></html>"""


def write_cleaning_report(
    *,
    target_dir: Path,
    input_path: Path,
    chunked_result,
    pipeline_description_path: Path,
    matrix_report_path: Path,
    quality_metrics_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Create one self-contained scientific report for a completed run.

    Individual signal sections are failure tolerant: insufficient channel
    geometry or short data produces an explicit notice, while provenance and
    any successfully computed sections remain available.  Failure to write the
    HTML itself is intentionally not swallowed because the process output
    contract requires this file.
    """
    output_path = output_path or cleaning_report_path(target_dir, input_path)
    quality_metrics_path = quality_metrics_path or target_dir / "quality_metrics.json"
    pipeline_payload = _read_json(pipeline_description_path)
    matrix_display = _compact_matrix_display_payload(chunked_result)
    report_warnings: list[str] = []
    manifest_payload: dict[str, Any] = {"note": "No chunk manifest was available for this run."}
    manifest_value = getattr(chunked_result, "manifest_path", None)
    if manifest_value is None:
        manifest_value = pipeline_payload.get("result", {}).get("chunks_manifest")
    if manifest_value is not None:
        manifest_path = Path(str(manifest_value)).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = target_dir / manifest_path
        manifest_path = manifest_path.resolve()
        target_root = target_dir.resolve()
        if manifest_path.is_relative_to(target_root) and manifest_path.is_file():
            try:
                manifest_payload = _read_json(manifest_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                report_warnings.append(f"The chunk manifest could not be embedded: {exc}")
        else:
            report_warnings.append("The chunk manifest path was missing or outside the recording output folder.")

    logger.info("Building self-contained EEG cleaning report: {}", output_path)
    data, capture_warnings = _capture_paired_recording(input_path, chunked_result)
    report_warnings.extend(capture_warnings)
    diagnostics: TemporalDiagnostics | None = None
    comparison_png = None
    geometry: SpatialGeometry | None = None
    graph_png = None
    graph_metadata: dict[str, Any] | None = None
    coherence_png = None
    coherence: CoherenceDiagnostics | None = None
    coherence_layout: str | None = None

    if data is not None:
        existing_figures = set(plt.get_fignums())
        try:
            diagnostics = _compute_temporal_diagnostics(data)
            comparison_png = _plot_temporal_comparison(diagnostics)
        except Exception as exc:
            report_warnings.append(f"Temporal/amplitude diagnostics were unavailable: {exc}")
            logger.warning("Could not build temporal report diagnostics: {}", exc)
        finally:
            _close_new_figures(existing_figures)

        geometry = _resolve_spatial_geometry(data)
        if geometry is None:
            report_warnings.append(
                f"Fewer than {MIN_SPATIAL_CHANNELS} defensibly scaled, broadly distributed EEG sensor "
                "positions were available; the spatial graph section was skipped."
            )
        else:
            report_warnings.append(
                f"Spatial geometry: {geometry.coverage_note}; coordinate frame {geometry.coordinate_frame}; "
                f"origin {geometry.origin}."
            )
            if len(geometry.channel_indices) < len(data.channel_names):
                report_warnings.append(
                    f"Spatial plots use {len(geometry.channel_indices)} of {len(data.channel_names)} channels with valid positions."
                )
            existing_figures = set(plt.get_fignums())
            try:
                graph_png, graph_metadata = _plot_graph_spectrum(data, geometry)
            except Exception as exc:
                report_warnings.append(f"Graph Laplacian diagnostics were unavailable: {exc}")
                logger.warning("Could not build graph spectral diagnostics: {}", exc)
            finally:
                _close_new_figures(existing_figures)

        existing_figures = set(plt.get_fignums())
        try:
            coherence = _compute_coherence_diagnostics(data, geometry)
            if len(coherence.channel_names) < len(data.channel_names):
                report_warnings.append(
                    f"Coherence uses {len(coherence.channel_names)} of {len(data.channel_names)} channels for bounded runtime and legibility."
                )
            if coherence.low_precision:
                report_warnings.append(
                    f"Coherence communities use only {coherence.frame_count} overlapping windows; "
                    "the network partition and Q estimate are marked low precision and may be unstable."
                )
            coherence_png, coherence_layout = _plot_coherence_network(coherence, geometry)
        except Exception as exc:
            report_warnings.append(f"Coherence/community diagnostics were unavailable: {exc}")
            logger.warning("Could not build coherence diagnostics: {}", exc)
        finally:
            _close_new_figures(existing_figures)

    quality_metrics_payload = _quality_metrics_payload(
        input_path=input_path,
        data=data,
        diagnostics=diagnostics,
        chunked_result=chunked_result,
        graph_metadata=graph_metadata,
        coherence=coherence,
    )
    _write_quality_metrics(quality_metrics_path, quality_metrics_payload)
    _log_quality_metrics(quality_metrics_payload)
    pipeline_payload["quality_metrics"] = quality_metrics_payload
    pipeline_payload.setdefault("result", {})["quality_metrics_report"] = str(quality_metrics_path)
    pipeline_description_path.write_text(
        json.dumps(_json_safe(pipeline_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    html_shell = _build_html(
        input_path=input_path,
        pipeline_payload=pipeline_payload,
        matrix_display=matrix_display,
        matrix_report_name=matrix_report_path.name,
        manifest_payload=manifest_payload,
        quality_metrics_payload=quality_metrics_payload,
        data=data,
        diagnostics=diagnostics,
        comparison_png=comparison_png,
        graph_png=graph_png,
        graph_metadata=graph_metadata,
        coherence_png=coherence_png,
        coherence=coherence,
        coherence_layout=coherence_layout,
        warnings=report_warnings,
    )
    if html_shell.count(MATRIX_ASSET_PLACEHOLDER) != 1:
        raise RuntimeError("The HTML shell must contain exactly one Flex matrix asset placeholder.")
    html_prefix, html_suffix = html_shell.split(MATRIX_ASSET_PLACEHOLDER)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(html_prefix)
            _write_matrix_assets(
                stream,
                matrix_display.get("diagnostic_plots", []),
                target_dir,
            )
            stream.write(html_suffix)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    logger.info("Wrote self-contained EEG cleaning report: {}", output_path)
    return output_path
