"""Diagnose Flex extraction, subtraction, and evaluation alignment on one EEG.

The diagnostic runs the selected matrix recipe with conservative lifecycle
settings, retains Flex's exact subtracted artifact estimate, and compares:

* original EEG ``D``;
* estimated artifact/template signal;
* corrected EEG ``D - estimate``;
* the per-trigger subtraction mask;
* the contiguous acquisition and reference intervals used by the evaluators.

It writes machine-readable summaries and representative waveform plots.  The
reported recommendations are evidence-based checks, not automatic fixes.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from loguru import logger
from run_matrix_optimization import _iter_result_items, _result_context, release_process_memory
from scipy.signal import coherence, welch

from facet import DownSample, Flex, MatrixDecisions, Pipeline, Processor, TriggerDetector, UpSample
from facet.core import ProcessingContext

FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "high_frequency": (30.0, 80.0),
    "broadband": (0.5, 80.0),
}


def load_matrix_recipe(path: Path | None) -> MatrixDecisions:
    """Load a raw decision manifest or optimizer ``selected_recipe.json``."""
    if path is None:
        return MatrixDecisions()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Recipe JSON must contain an object.")
    manifest = payload.get("recipe", payload)
    if not isinstance(manifest, Mapping):
        raise ValueError("The JSON 'recipe' field must contain an object.")
    return MatrixDecisions.from_dict(manifest)


def interval_masks(
    *,
    n_samples: int,
    sfreq: float,
    triggers: np.ndarray,
    artifact_length: int,
    artifact_offset_seconds: float,
    reference_buffer_seconds: float,
) -> dict[str, np.ndarray]:
    """Reproduce Flex subtraction and evaluator interval definitions."""
    template = np.zeros(n_samples, dtype=bool)
    offset = int(round(artifact_offset_seconds * sfreq))
    for trigger in triggers:
        start = max(0, int(trigger) + offset)
        stop = min(n_samples, int(trigger) + offset + artifact_length)
        if stop > start:
            template[start:stop] = True

    acquisition = np.zeros(n_samples, dtype=bool)
    acq_start = max(0, int(triggers[0]) - int(artifact_length * 0.5))
    acq_stop = min(n_samples, int(triggers[-1]) + int(artifact_length * 1.5))
    acquisition[acq_start:acq_stop] = True

    reference = np.zeros(n_samples, dtype=bool)
    buffer_samples = int(round(reference_buffer_seconds * sfreq))
    pre_stop = max(0, int(triggers[0]) - buffer_samples)
    post_start = min(n_samples, int(triggers[-1]) + artifact_length + buffer_samples)
    reference[:pre_stop] = True
    reference[post_start:] = True
    return {"template": template, "acquisition": acquisition, "reference": reference}


def _rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(data)))) if data.size else float("nan")


def _peak_to_peak(data: np.ndarray) -> float:
    return float(np.ptp(data)) if data.size else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if np.isfinite(denominator) and denominator > 0.0 else float("nan")


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size != left.size or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else float("nan")


def channel_diagnostics(
    original: np.ndarray,
    corrected: np.ndarray,
    estimate: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Calculate telling energy, cancellation, and identity checks."""
    template_mask = masks["template"]
    acquisition_mask = masks["acquisition"]
    reference_mask = masks["reference"]
    original_template = original[template_mask]
    corrected_template = corrected[template_mask]
    estimate_template = estimate[template_mask]
    original_acquisition = original[acquisition_mask]
    corrected_acquisition = corrected[acquisition_mask]
    original_reference = original[reference_mask]
    identity_error = corrected - (original - estimate)

    original_template_rms = _rms(original_template)
    corrected_template_rms = _rms(corrected_template)
    estimate_template_rms = _rms(estimate_template)
    reference_rms = _rms(original_reference)
    return {
        "original_template_rms": original_template_rms,
        "estimate_template_rms": estimate_template_rms,
        "corrected_template_rms": corrected_template_rms,
        "original_acquisition_rms": _rms(original_acquisition),
        "corrected_acquisition_rms": _rms(corrected_acquisition),
        "original_reference_rms": reference_rms,
        "original_template_p2p": _peak_to_peak(original_template),
        "estimate_template_p2p": _peak_to_peak(estimate_template),
        "corrected_template_p2p": _peak_to_peak(corrected_template),
        "estimate_to_original_template_rms": _safe_ratio(estimate_template_rms, original_template_rms),
        "corrected_to_original_template_rms": _safe_ratio(corrected_template_rms, original_template_rms),
        "corrected_acquisition_to_reference_rms": _safe_ratio(_rms(corrected_acquisition), reference_rms),
        "original_acquisition_to_reference_rms": _safe_ratio(_rms(original_acquisition), reference_rms),
        "template_original_estimate_correlation": _safe_correlation(original_template, estimate_template),
        "subtraction_identity_error_rms": _rms(identity_error),
        "subtraction_identity_relative_error": _safe_ratio(_rms(identity_error), _rms(original)),
    }


def spectral_diagnostics(
    original: np.ndarray,
    roundtrip: np.ndarray,
    corrected: np.ndarray,
    estimate: np.ndarray,
    masks: Mapping[str, np.ndarray],
    sfreq: float,
) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray]]:
    """Measure preservation by frequency without using selection scores as evidence."""
    acquisition = masks["acquisition"]
    reference = masks["reference"]
    signals = {
        "original": original[acquisition],
        "roundtrip": roundtrip[acquisition],
        "corrected": corrected[acquisition],
        "estimate": estimate[acquisition],
        "reference": original[reference],
    }
    minimum_length = min((len(value) for value in signals.values()), default=0)
    if minimum_length < 8:
        return [], {}
    nperseg = min(4096, minimum_length)
    spectra: dict[str, np.ndarray] = {}
    frequencies = None
    for name, signal in signals.items():
        frequencies, spectra[name] = welch(
            signal,
            fs=sfreq,
            nperseg=nperseg,
            detrend="constant",
            scaling="density",
        )
    coherence_frequencies, coherence_values = coherence(
        roundtrip[acquisition],
        estimate[acquisition],
        fs=sfreq,
        nperseg=nperseg,
        detrend="constant",
    )
    rows: list[dict[str, float | str]] = []
    assert frequencies is not None
    nyquist = sfreq / 2.0
    for band, (low, configured_high) in FREQUENCY_BANDS.items():
        high = min(configured_high, nyquist)
        frequency_mask = (frequencies >= low) & (frequencies < high)
        coherence_mask = (coherence_frequencies >= low) & (coherence_frequencies < high)
        if np.count_nonzero(frequency_mask) < 2:
            continue
        powers = {
            name: float(np.trapezoid(values[frequency_mask], frequencies[frequency_mask]))
            for name, values in spectra.items()
        }
        rows.append(
            {
                "band": band,
                "low_hz": low,
                "high_hz": high,
                **{f"{name}_power": power for name, power in powers.items()},
                "corrected_to_reference_power": _safe_ratio(powers["corrected"], powers["reference"]),
                "corrected_to_roundtrip_power": _safe_ratio(powers["corrected"], powers["roundtrip"]),
                "estimate_to_roundtrip_power": _safe_ratio(powers["estimate"], powers["roundtrip"]),
                "roundtrip_to_original_power": _safe_ratio(powers["roundtrip"], powers["original"]),
                "original_to_reference_power": _safe_ratio(powers["original"], powers["reference"]),
                "original_estimate_coherence": (
                    float(np.mean(coherence_values[coherence_mask])) if np.any(coherence_mask) else float("nan")
                ),
            }
        )
    return rows, {"frequencies": frequencies, **spectra}


def matrix_self_weight_diagnostics(context: ProcessingContext) -> list[dict[str, Any]]:
    """Extract diagonal A weights from dense or sparse serialized reports."""
    reports = context.metadata.custom.get("artifact_template_matrices", [])
    rows: list[dict[str, Any]] = []
    for report_index, report in enumerate(reports):
        for channel in report.get("channels", []):
            payload = channel.get("averaging_matrix_A", {})
            shape = payload.get("shape", [0, 0])
            diagonal_length = min(shape) if len(shape) == 2 else 0
            weights: list[float] = []
            if payload.get("storage") == "dense":
                matrix = np.asarray(payload.get("matrix", []), dtype=float)
                weights = np.diag(matrix).astype(float).tolist() if matrix.ndim == 2 else []
            elif payload.get("storage") == "sparse_rows":
                for sparse_row in payload.get("rows", []):
                    row_index = int(sparse_row["row"])
                    columns = sparse_row.get("columns", [])
                    values = sparse_row.get("weights", [])
                    self_weight = 0.0
                    if row_index in columns:
                        self_weight = float(values[columns.index(row_index)])
                    weights.append(self_weight)
            rows.append(
                {
                    "report_index": report_index,
                    "channel": channel.get("channel_name"),
                    "storage": payload.get("storage"),
                    "rows_audited": len(weights),
                    "rows_total": diagonal_length,
                    "audit_truncated": len(weights) < diagonal_length,
                    "self_weight_nonzero_fraction": (
                        float(np.mean(np.asarray(weights) != 0.0)) if weights else float("nan")
                    ),
                    "self_weight_mean": float(np.mean(weights)) if weights else float("nan"),
                    "self_weight_max": float(np.max(weights)) if weights else float("nan"),
                }
            )
    return rows


def infer_findings(summary: Mapping[str, Any]) -> list[dict[str, str]]:
    """Translate diagnostic measurements into actionable next checks."""
    findings: list[dict[str, str]] = []
    identity = float(summary["subtraction_identity_relative_error_median"])
    estimate_ratio = float(summary["estimate_to_original_template_rms_median"])
    corrected_ratio = float(summary["corrected_to_original_template_rms_median"])
    mask_fraction = float(summary["template_fraction_of_acquisition_interval_median"])
    overlap = float(summary["estimate_energy_outside_template_fraction_median"])

    if identity > 1e-6:
        findings.append(
            {
                "severity": "critical",
                "finding": "Corrected data does not equal original minus the retained Flex estimate.",
                "next_fix": "Inspect noise resampling and trigger realignment before changing A.",
            }
        )
    if estimate_ratio > 0.9 and corrected_ratio < 0.2:
        findings.append(
            {
                "severity": "critical",
                "finding": "The template reproduces most target-epoch energy and leaves less than 20% RMS.",
                "next_fix": "Inspect D/N overlays and target self-weight; constrain A or template scaling only if masks are correct.",
            }
        )
    if mask_fraction < 0.8:
        findings.append(
            {
                "severity": "warning",
                "finding": "The evaluator's contiguous acquisition interval contains substantial samples Flex never subtracts.",
                "next_fix": "Evaluate per-trigger artifact masks separately from between-trigger acquisition data.",
            }
        )
    if overlap > 1e-6:
        findings.append(
            {
                "severity": "warning",
                "finding": "The retained artifact estimate has energy outside the calculated subtraction mask.",
                "next_fix": "Check trigger realignment, gap interpolation, and resampling mask expansion.",
            }
        )
    if not findings:
        findings.append(
            {
                "severity": "info",
                "finding": "No gross subtraction identity or mask mismatch was detected.",
                "next_fix": "Use the channel table and overlays to inspect recipe-specific signal removal.",
            }
        )
    return findings


class AlignmentDiagnostic(Processor):
    """Attach exact alignment diagnostics to a corrected chunk context."""

    name = "flex_alignment_diagnostic"
    description = "Compare original, estimated, corrected, and evaluation intervals"
    version = "1.0.0"
    requires_raw = True
    requires_triggers = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, reference_buffer_seconds: float = 0.1, upsample_factor: int = 10) -> None:
        self.reference_buffer_seconds = reference_buffer_seconds
        self.upsample_factor = upsample_factor
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw()
        original_raw = context.get_raw_original()
        estimate = context.get_estimated_noise()
        if original_raw is None or estimate is None:
            raise RuntimeError("Diagnostic requires original raw and Flex track_estimated_noise=True.")

        triggers = np.asarray(context.get_triggers(), dtype=int)
        artifact_length = context.get_artifact_length()
        if artifact_length is None or len(triggers) == 0:
            raise RuntimeError("Diagnostic requires triggers and artifact length.")
        masks = interval_masks(
            n_samples=raw.n_times,
            sfreq=float(raw.info["sfreq"]),
            triggers=triggers,
            artifact_length=int(artifact_length),
            artifact_offset_seconds=float(context.metadata.artifact_to_trigger_offset),
            reference_buffer_seconds=self.reference_buffer_seconds,
        )
        picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude="bads")
        original = original_raw.get_data(picks=picks)
        corrected = raw.get_data(picks=picks)
        estimate = np.asarray(estimate)[picks]

        # Compare against the signal Flex actually receives after the same
        # resampling round trip, rather than against the pre-resampling input.
        roundtrip_raw = original_raw.copy()
        original_sfreq = float(roundtrip_raw.info["sfreq"])
        roundtrip_raw.resample(
            original_sfreq * self.upsample_factor,
            window="boxcar",
            n_jobs=1,
            verbose=False,
        )
        roundtrip_raw.resample(original_sfreq, window="boxcar", n_jobs=1, verbose=False)
        roundtrip = roundtrip_raw.get_data(picks=picks)
        common_samples = min(original.shape[1], corrected.shape[1], estimate.shape[1], roundtrip.shape[1])
        original = original[:, :common_samples]
        corrected = corrected[:, :common_samples]
        estimate = estimate[:, :common_samples]
        roundtrip = roundtrip[:, :common_samples]
        masks = {name: mask[:common_samples] for name, mask in masks.items()}
        rows = []
        spectral_rows = []
        channel_spectra = []
        for local_index, channel_index in enumerate(picks):
            rows.append(
                {
                    "channel": raw.ch_names[channel_index],
                    **channel_diagnostics(roundtrip[local_index], corrected[local_index], estimate[local_index], masks),
                    "resampling_roundtrip_relative_error": _safe_ratio(
                        _rms(roundtrip[local_index] - original[local_index]),
                        _rms(original[local_index]),
                    ),
                }
            )
            band_rows, spectra = spectral_diagnostics(
                original[local_index],
                roundtrip[local_index],
                corrected[local_index],
                estimate[local_index],
                masks,
                float(raw.info["sfreq"]),
            )
            spectral_rows.extend({"channel": raw.ch_names[channel_index], **band_row} for band_row in band_rows)
            if spectra:
                channel_spectra.append(spectra)

        template_count = int(np.count_nonzero(masks["template"]))
        acquisition_count = int(np.count_nonzero(masks["acquisition"]))
        outside = ~masks["template"]
        total_estimate_energy = float(np.sum(np.square(estimate)))
        outside_estimate_energy = float(np.sum(np.square(estimate[:, outside])))
        plot_trigger = int(triggers[len(triggers) // 2])
        plot_start = max(0, plot_trigger - int(artifact_length))
        plot_stop = min(raw.n_times, plot_trigger + (2 * int(artifact_length)))
        report = {
            "sampling_rate_hz": float(raw.info["sfreq"]),
            "n_samples": int(raw.n_times),
            "n_triggers": int(len(triggers)),
            "artifact_length_samples": int(artifact_length),
            "artifact_length_seconds": float(artifact_length / raw.info["sfreq"]),
            "artifact_offset_seconds": float(context.metadata.artifact_to_trigger_offset),
            "median_trigger_spacing_samples": float(np.median(np.diff(triggers))) if len(triggers) > 1 else None,
            "template_mask_samples": template_count,
            "acquisition_interval_samples": acquisition_count,
            "reference_samples": int(np.count_nonzero(masks["reference"])),
            "template_fraction_of_recording": template_count / raw.n_times,
            "acquisition_fraction_of_recording": acquisition_count / raw.n_times,
            "template_fraction_of_acquisition_interval": (
                template_count / acquisition_count if acquisition_count else float("nan")
            ),
            "estimate_energy_outside_template_fraction": (
                outside_estimate_energy / total_estimate_energy if total_estimate_energy > 0.0 else 0.0
            ),
            "channels": rows,
            "spectral_rows": spectral_rows,
            "matrix_self_weights": matrix_self_weight_diagnostics(context),
            "mean_spectra": (
                {
                    name: np.mean([item[name] for item in channel_spectra], axis=0).tolist()
                    for name in channel_spectra[0]
                }
                if channel_spectra
                else {}
            ),
            "plot": {
                "channel": raw.ch_names[picks[0]] if len(picks) else None,
                "original": roundtrip[0, plot_start:plot_stop].tolist() if len(picks) else [],
                "corrected": corrected[0, plot_start:plot_stop].tolist() if len(picks) else [],
                "estimate": estimate[0, plot_start:plot_stop].tolist() if len(picks) else [],
                "trigger_within_plot": plot_trigger - plot_start,
            },
        }
        metadata = context.metadata.copy()
        metadata.custom["flex_alignment_diagnostic"] = report
        return context.with_metadata(metadata, copy_estimated_noise=False)


def summarize_reports(reports: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine chunk reports using medians robust to channel outliers."""
    channel_rows = []
    for chunk_index, report in enumerate(reports):
        for row in report["channels"]:
            channel_rows.append({"chunk": chunk_index, **row})
    channels = pd.DataFrame(channel_rows)
    summary: dict[str, Any] = {
        "n_chunks": len(reports),
        "n_channel_rows": len(channels),
    }
    scalar_report_keys = (
        "sampling_rate_hz",
        "n_samples",
        "n_triggers",
        "artifact_length_samples",
        "artifact_length_seconds",
        "artifact_offset_seconds",
        "median_trigger_spacing_samples",
        "template_fraction_of_recording",
        "acquisition_fraction_of_recording",
        "template_fraction_of_acquisition_interval",
        "estimate_energy_outside_template_fraction",
    )
    for key in scalar_report_keys:
        values = [report[key] for report in reports if report.get(key) is not None]
        summary[key + "_median"] = float(np.median(values)) if values else None
    for key in channels.select_dtypes(include=[np.number]).columns:
        if key != "chunk":
            summary[key + "_median"] = float(channels[key].median())
    summary["findings"] = infer_findings(summary)
    return channels, summary


def summarize_spectral_reports(
    reports: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Flatten spectral and A audits and summarize physiological bands."""
    spectral_rows = []
    self_weight_rows = []
    for chunk_index, report in enumerate(reports):
        spectral_rows.extend({"chunk": chunk_index, **row} for row in report["spectral_rows"])
        self_weight_rows.extend({"chunk": chunk_index, **row} for row in report["matrix_self_weights"])
    spectral = pd.DataFrame(spectral_rows)
    self_weights = pd.DataFrame(self_weight_rows)
    band_summary: dict[str, Any] = {}
    if not spectral.empty:
        numeric_columns = [
            column
            for column in spectral.select_dtypes(include=[np.number]).columns
            if column not in {"chunk", "low_hz", "high_hz"}
        ]
        for band, group in spectral.groupby("band", sort=False):
            band_summary[str(band)] = {column + "_median": float(group[column].median()) for column in numeric_columns}
    return spectral, self_weights, band_summary


def spectral_findings(bands: Mapping[str, Mapping[str, float]]) -> list[dict[str, str]]:
    """Flag patterns that distinguish signal loss from artifact suppression."""
    findings = []
    for band in ("delta", "theta", "alpha", "beta"):
        metrics = bands.get(band)
        if not metrics:
            continue
        preservation = metrics.get("corrected_to_reference_power_median", float("nan"))
        resampling = metrics.get("roundtrip_to_original_power_median", float("nan"))
        if np.isfinite(preservation) and preservation < 0.25:
            findings.append(
                {
                    "severity": "critical",
                    "finding": f"Corrected {band} power is below 25% of scanner-off reference power.",
                    "next_fix": "Do not optimize on time-domain SNR alone; inspect whether this band is present in N and add a band-preservation constraint.",
                }
            )
        if np.isfinite(resampling) and not 0.8 <= resampling <= 1.25:
            findings.append(
                {
                    "severity": "warning",
                    "finding": f"The resampling round trip materially changes {band} power.",
                    "next_fix": "Test a smaller upsampling factor or evaluate correction before downsampling.",
                }
            )
    return findings


def plot_mean_spectra(report: Mapping[str, Any], output_path: Path) -> None:
    """Plot channel-mean PSDs for the four diagnostic signal stages."""
    spectra = report.get("mean_spectra", {})
    if not spectra:
        return
    frequencies = np.asarray(spectra["frequencies"], dtype=float)
    keep = (frequencies >= 0.5) & (frequencies <= 80.0)
    figure, axis = plt.subplots(figsize=(13, 7))
    for name, label in (
        ("original", "original acquisition"),
        ("roundtrip", "uncorrected after resampling"),
        ("estimate", "Flex estimate N"),
        ("corrected", "corrected D-N"),
        ("reference", "scanner-off reference"),
    ):
        values = np.asarray(spectra[name], dtype=float)
        axis.semilogy(frequencies[keep], values[keep], label=label)
    for boundary in (4.0, 8.0, 13.0, 30.0):
        axis.axvline(boundary, color="grey", linewidth=0.7, alpha=0.5)
    axis.set(xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)", title="Frequency-resolved signal preservation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_representative(report: Mapping[str, Any], output_path: Path) -> None:
    """Plot original, estimate, and corrected data around a middle trigger."""
    payload = report["plot"]
    if not payload["original"]:
        return
    sfreq = float(report["sampling_rate_hz"])
    trigger = int(payload["trigger_within_plot"])
    times = (np.arange(len(payload["original"])) - trigger) / sfreq
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(times, payload["original"], label="original D", alpha=0.8)
    axis.plot(times, payload["estimate"], label="Flex estimate", alpha=0.8)
    axis.plot(times, payload["corrected"], label="corrected D-estimate", alpha=0.8)
    offset = float(report["artifact_offset_seconds"])
    axis.axvspan(offset, offset + report["artifact_length_seconds"], color="orange", alpha=0.12)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1, label="trigger")
    axis.set(title=f"Alignment diagnostic: {payload['channel']}", xlabel="Seconds from trigger", ylabel="V")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--recipe-json", type=Path)
    parser.add_argument("--trigger-regex", default=r"^R128$")
    parser.add_argument("--upsample-factor", type=int, default=10)
    parser.add_argument("--chunk-padding-seconds", type=float, default=5.0)
    parser.add_argument("--chunk-min-triggers", type=int, default=20)
    parser.add_argument("--chunk-gap-seconds", type=float)
    parser.add_argument("--reference-buffer-seconds", type=float, default=0.1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input_path.is_file():
        raise ValueError(f"Input does not exist: {args.input_path}")
    if args.recipe_json is not None and not args.recipe_json.is_file():
        raise ValueError(f"Recipe does not exist: {args.recipe_json}")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    decisions = load_matrix_recipe(args.recipe_json)
    pipeline = Pipeline(
        [
            TriggerDetector(regex=args.trigger_regex),
            UpSample(factor=args.upsample_factor),
            Flex(
                matrix_decisions=decisions,
                plot_artifacts=False,
                realign_after_averaging=True,
                interpolate_volume_gaps=False,
                apply_epoch_alpha_scaling=False,
                track_estimated_noise=True,
            ),
            DownSample(factor=args.upsample_factor),
            AlignmentDiagnostic(
                reference_buffer_seconds=args.reference_buffer_seconds,
                upsample_factor=args.upsample_factor,
            ),
        ],
        name="Flex extraction/subtraction/evaluation alignment diagnostic",
    )
    with tempfile.TemporaryDirectory(prefix="facetpy_alignment_") as temporary_directory:
        result = pipeline.run_chunked(
            input_path=str(args.input_path),
            output_dir=temporary_directory,
            output_extension=".edf",
            overwrite=True,
            channel_sequential=True,
            on_error="continue",
            keep_raw=False,
            chunk_by_trigger_sections=True,
            trigger_section_padding_seconds=args.chunk_padding_seconds,
            trigger_section_min_triggers=args.chunk_min_triggers,
            trigger_section_gap_seconds=args.chunk_gap_seconds,
        )
        reports = []
        errors = []
        for item in _iter_result_items(result):
            if not bool(getattr(item, "success", True)):
                errors.append(str(getattr(item, "error", "unknown failure")))
                continue
            context = _result_context(item)
            if context is not None:
                report = context.metadata.custom.get("flex_alignment_diagnostic")
                if report is not None:
                    reports.append(report)
        del result
    release_process_memory()
    if not reports:
        raise RuntimeError("No diagnostic chunks succeeded: " + " | ".join(errors))

    channels, summary = summarize_reports(reports)
    spectral, self_weights, band_summary = summarize_spectral_reports(reports)
    summary["frequency_bands"] = band_summary
    summary["findings"].extend(spectral_findings(band_summary))
    if not self_weights.empty:
        summary["matrix_self_weight"] = {
            "nonzero_fraction_median": float(self_weights["self_weight_nonzero_fraction"].median()),
            "mean_median": float(self_weights["self_weight_mean"].median()),
            "maximum": float(self_weights["self_weight_max"].max()),
            "any_truncated_audit": bool(self_weights["audit_truncated"].any()),
        }
    summary["input_path"] = str(args.input_path.resolve())
    summary["recipe"] = decisions.to_dict()
    summary["chunk_errors"] = errors
    channels.to_csv(args.output_directory / "alignment_channel_diagnostics.csv", index=False)
    spectral.to_csv(args.output_directory / "frequency_band_diagnostics.csv", index=False)
    self_weights.to_csv(args.output_directory / "matrix_self_weight_audit.csv", index=False)
    (args.output_directory / "alignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot_representative(reports[0], args.output_directory / "alignment_waveform_overlay.png")
    plot_mean_spectra(reports[0], args.output_directory / "frequency_resolved_psd.png")

    print("\nKey measurements")
    for key in (
        "artifact_length_seconds_median",
        "template_fraction_of_acquisition_interval_median",
        "estimate_to_original_template_rms_median",
        "corrected_to_original_template_rms_median",
        "template_original_estimate_correlation_median",
        "subtraction_identity_relative_error_median",
        "resampling_roundtrip_relative_error_median",
        "estimate_energy_outside_template_fraction_median",
    ):
        print(f"  {key}: {summary.get(key)}")
    print("\nFindings")
    for finding in summary["findings"]:
        print(f"  [{finding['severity'].upper()}] {finding['finding']}")
        print(f"    Next: {finding['next_fix']}")
    if band_summary:
        print("\nMedian band preservation (corrected acquisition / scanner-off reference)")
        for band in FREQUENCY_BANDS:
            if band in band_summary:
                value = band_summary[band].get("corrected_to_reference_power_median")
                print(f"  {band}: {value}")
    if "matrix_self_weight" in summary:
        print("\nA-matrix target self-weight audit")
        for key, value in summary["matrix_self_weight"].items():
            print(f"  {key}: {value}")
    logger.info("Diagnostic written to {}", args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
