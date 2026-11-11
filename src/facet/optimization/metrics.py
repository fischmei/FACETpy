"""Original-defined spectral masks and independently cleaned EEG agreement.

Preservation is a spectral proxy, never proof of preserved neural activity.
Independent references are comparison recordings, not artifact-free truth.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS = {"theta": (4, 8), "alpha": (8, 13), "nonpeak_beta": (13, 30)}
METRIC_VERSION = 4
MIN_CLEAN_BAND_COVERAGE = 0.25
MIN_PRESERVATION_COVERAGE = 0.50


def original_spectrum(data, sfreq, triggers, settings):
    """Cache per-channel PSD and a fixed contamination mask before correction."""
    from facet.optimization.search import detect_scanner_peak_mask

    intervals = np.diff(np.asarray(triggers, dtype=float))
    if not len(intervals) or not np.all(np.isfinite(intervals)) or np.any(intervals <= 0):
        raise ValueError("Spectral evaluation requires at least two strictly increasing finite triggers.")
    period = float(np.median(intervals)) / sfreq
    fundamental = 1 / period
    # Resolve the trigger comb in Hz, independently of acquisition sample rate.
    # Sixteen periods provide bins much narrower than the harmonic spacing.
    segment_samples = min(len(data), int(np.ceil(sfreq * max(16.0, 16 * period))))
    if segment_samples < 2:
        raise ValueError("Spectral evaluation requires at least two signal samples.")
    frequencies, psd = welch(data, fs=sfreq, nperseg=segment_samples)
    frequency_step = frequencies[1] - frequencies[0]
    # Require prominent peaks in the ORIGINAL spectrum: trigger timing alone
    # must not mark every possible harmonic as contaminated. Below the scanner
    # search range, accept only peaks close to a trigger harmonic.
    _, peaks = detect_scanner_peak_mask(
        frequencies,
        psd,
        minimum_hz=min(fundamental, settings.scanner_peak_minimum_hz),
        maximum_hz=min(settings.scanner_peak_maximum_hz, sfreq / 2),
        prominence_db=settings.scanner_peak_prominence_db,
        half_width_hz=settings.scanner_peak_half_width_hz,
    )
    harmonic_number = np.rint(peaks / fundamental)
    near_harmonic = (harmonic_number >= 1) & (np.abs(peaks - harmonic_number * fundamental) <= 2 * frequency_step)
    peaks = peaks[(peaks >= settings.scanner_peak_minimum_hz) | near_harmonic]
    # Cap the requested half-width at a quarter of the harmonic spacing, but
    # retain two Welch bins for Hann leakage. Unresolved measurements become
    # unavailable rather than narrowing masks below resolution.
    width = max(min(settings.scanner_peak_half_width_hz, fundamental / 4), 2 * frequency_step)
    mask = np.zeros_like(frequencies, dtype=bool)
    for peak in peaks:
        mask |= np.abs(frequencies - peak) <= width
    return frequencies, psd, mask


def power(psd, mask, frequencies):
    # Integrating masked points with trapezoid bridges excluded frequency
    # intervals. Integrate Welch density as bin mass instead.
    return float(np.sum(psd[mask]) * (frequencies[1] - frequencies[0]))


def clean_band_diagnostics(frequencies, original_psd, scanner_mask):
    """Describe original-defined preservation support before any correction."""
    diagnostics = {}
    for name, (low, high) in BANDS.items():
        band = (frequencies >= low) & (frequencies < high)
        clean = band & ~scanner_mask
        coverage = float(clean.sum() / max(band.sum(), 1))
        denominator = power(original_psd, clean, frequencies)
        diagnostics[name] = {
            "mask": clean,
            "coverage": coverage,
            "bins": int(clean.sum()),
            "power": denominator,
            "valid": (
                coverage >= MIN_CLEAN_BAND_COVERAGE
                and clean.sum() >= 2
                and np.isfinite(denominator)
                and denominator > np.finfo(float).tiny
            ),
        }
        reasons = []
        if coverage < MIN_CLEAN_BAND_COVERAGE:
            reasons.append("clean frequency coverage below 25%")
        if clean.sum() < 2:
            reasons.append("fewer than two clean bins")
        if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
            reasons.append("non-finite or non-positive original power")
        diagnostics[name]["invalid_reason"] = "; ".join(reasons)
    return diagnostics


def validate_clean_band_coverage(frequencies, original_psd, scanner_mask, *, context):
    """Classify original-only support; isolated failures are unavailable.

    Dataset/fidelity coverage is enforced after all measurements are classified.
    ``context`` is retained for compatibility with existing callers.
    """
    return clean_band_diagnostics(frequencies, original_psd, scanner_mask)


def spectral_metrics(corrected, sfreq, frequencies, original_psd, scanner_mask, band_valid=None):
    if not np.all(np.isfinite(corrected)):
        raise ValueError("Candidate produced non-finite samples on the fixed evaluation set.")
    segment_samples = int(round(sfreq / (frequencies[1] - frequencies[0])))
    _, psd = welch(corrected, fs=sfreq, nperseg=segment_samples)
    tiny = np.finfo(float).tiny
    base = power(original_psd, scanner_mask, frequencies)
    residual = power(psd, scanner_mask, frequencies)
    result = {
        "scanner_artifact_residual_power": residual,
        "scanner_peak_residual": residual / base if base > tiny else float("nan"),
        "scanner_suppression_db": 10 * np.log10(base / max(residual, tiny)) if base > tiny else float("nan"),
        "scanner_peak_count": int(scanner_mask.sum()),
        "signal_rms": float(np.sqrt(np.mean(corrected**2))),
        "signal_peak_to_peak": float(np.ptp(corrected)),
        "finite_sample_fraction": float(np.mean(np.isfinite(corrected))),
    }
    deviations = []
    sufficient = True
    diagnostics_by_band = clean_band_diagnostics(frequencies, original_psd, scanner_mask)
    if band_valid is not None and not np.array_equal(
        band_valid, [item["valid"] for item in diagnostics_by_band.values()]
    ):
        raise ValueError("Cached original preservation validity is inconsistent; rebuild preparation.")
    for name, diagnostics in diagnostics_by_band.items():
        clean, denominator, valid = diagnostics["mask"], diagnostics["power"], diagnostics["valid"]
        result[f"{name}_clean_coverage"] = diagnostics["coverage"]
        ratio = power(psd, clean, frequencies) / denominator if valid else float("nan")
        result[f"{name}_preservation"] = ratio
        result[f"{name}_power_change_db"] = 10 * np.log10(max(ratio, tiny)) if valid else float("nan")
        result[f"{name}_log_deviation"] = abs(np.log(max(ratio, tiny))) if valid else float("nan")
        result[f"{name}_valid_measurements"] = int(valid)
        result[f"{name}_invalid_measurements"] = int(not valid)
        for metric in ("preservation", "power_change_db", "log_deviation"):
            result[f"{name}_{metric}_weight"] = float(valid)
        if valid and not np.isfinite(ratio):
            raise ValueError(f"Candidate produced non-finite {name} power on the fixed evaluation set.")
        if valid:
            deviations.append(abs(np.log(max(ratio, tiny))))
        sufficient &= valid
    result["clean_band_coverage_sufficient"] = sufficient
    result["nonpeak_eeg_log_deviation"] = float(np.mean(deviations)) if sufficient else float("nan")
    return result


def reference_metrics(corrected, reference, *, sfreq=None):
    if (
        corrected.shape != reference.shape
        or corrected.size < 2
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(corrected))
    ):
        raise ValueError("Reference usable interval or finite sample validation failed.")
    residual = corrected - reference
    mse = float(np.mean(residual**2))
    reference_power = float(np.mean(reference**2))
    tiny = np.finfo(float).tiny
    result = {
        "reference_snr_db": 10 * np.log10(max(reference_power, tiny) / max(mse, tiny)),
        "reference_rms_residual": np.sqrt(mse),
        "reference_median_residual": float(np.median(np.abs(residual))),
        "reference_normalized_mse": mse / max(reference_power, tiny),
        "reference_correlation": float(np.corrcoef(corrected, reference)[0, 1])
        if np.std(reference) > 0 and np.std(corrected) > 0
        else float("nan"),
    }
    if sfreq is not None:
        frequencies, x = welch(corrected, fs=sfreq, nperseg=min(len(corrected), int(4 * sfreq)))
        _, y = welch(reference, fs=sfreq, nperseg=min(len(reference), int(4 * sfreq)))
        # Fixed common comparison band; downstream filtering is never fitted
        # to improve agreement with a particular reference or candidate.
        band = (frequencies >= 1) & (frequencies <= 40)
        result["reference_spectral_log_deviation"] = float(
            np.mean(np.abs(np.log(np.maximum(x[band], tiny) / np.maximum(y[band], tiny))))
        )
        result["reference_spectral_similarity"] = float(
            np.dot(x[band], y[band]) / max(np.linalg.norm(x[band]) * np.linalg.norm(y[band]), tiny)
        )
        for name, (low, high) in {**BANDS, "stimulus_12hz": (11.5, 12.5), "stimulus_24hz": (23.5, 24.5)}.items():
            mask = (frequencies >= low) & (frequencies < high)
            ratio = max(power(x, mask, frequencies), tiny) / max(power(y, mask, frequencies), tiny)
            result[f"reference_{name}_bandpower_log_deviation"] = float(abs(np.log(ratio)))
    return result


def objectives(row, mode):
    if mode == "non-reference":
        return row["scanner_peak_residual"], row["nonpeak_eeg_log_deviation"]
    if mode == "reference":
        return row.get("reference_normalized_mse", float("nan")), row.get(
            "reference_spectral_log_deviation", float("nan")
        )
    # Both objectives are dimensionless, without adding incomparable scales.
    return row["scanner_peak_residual"], row.get("reference_normalized_mse", float("nan"))


def trigger_locked_metrics(data, triggers, epoch_length):
    """Phase-locked mean and residual RMS on complete, nonoverlapping epochs.

    These are diagnostics, since periodic neural activity can also be locked
    to acquisition. They are not claimed to identify scanner artifact alone.
    """
    starts = np.asarray(triggers, dtype=int)
    starts = starts[(starts >= 0) & (starts + epoch_length <= len(data))]
    if len(starts) < 3 or epoch_length < 2 or np.any(np.diff(starts) < epoch_length):
        return {"trigger_locked_rms": float("nan"), "non_trigger_locked_rms": float("nan")}
    epochs = np.stack([data[start : start + epoch_length] for start in starts])
    locked = np.mean(epochs, axis=0)
    return {
        "trigger_locked_rms": float(np.sqrt(np.mean(locked**2))),
        "non_trigger_locked_rms": float(np.sqrt(np.mean((epochs - locked) ** 2))),
    }
