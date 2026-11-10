"""A mandatory, bounded single-recording gate before reference optimization."""

import json
import resource
import sys
import time
import traceback
from dataclasses import replace

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import welch
from threadpoolctl import threadpool_limits

from facet import Loader, Resample
from facet.core import ProcessingContext
from facet.correction.flex.decisions import DirectionalQuota, MatrixDecisions
from facet.optimization.cache import digest
from facet.optimization.cleaning import DEFAULT_CLEANING, average_reference
from facet.optimization.prepared import atomic_json, evaluate_prepared, prepare_dataset


def comparison_plots(dataset, flex_raw, preview, directory, *, chunk_start):
    """Export four-stage traces/PSDs for an explicitly mapped valid interval."""
    directory.mkdir(parents=True, exist_ok=True)
    sfreq = preview["sfreq"]
    start = float(preview["raw_start"])
    duration = min(30.0, preview["candidate"].shape[1] / sfreq)
    source = Loader(str(dataset.path), preload=False).execute(None).get_raw()
    try:
        rate = float(source.info["sfreq"])
        raw = source.copy().crop(start, start + duration - 1 / rate).load_data()
    finally:
        source.close()
    flex = flex_raw.copy().crop(start - chunk_start, start - chunk_start + duration - 1 / rate)
    stages = {}
    for name, recording in (("Raw", raw), ("Flex only", flex)):
        recording = average_reference(recording.copy().pick(preview["channels"]))
        recording = Resample(sfreq).execute(ProcessingContext(recording)).get_raw()
        stages[name] = recording.get_data()
    count = int(round(duration * sfreq))
    stages["Full FACETpy"] = preview["candidate"][:, :count]
    stages["NATVIEW reference"] = preview["reference"][:, :count]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for name, data in stages.items():
        n = min(data.shape[1], int(5 * sfreq))
        axes[0].plot(start + np.arange(n) / sfreq, data[0, :n] * 1e6, label=name, alpha=0.8)
        if name != "Raw":
            axes[1].plot(start + np.arange(n) / sfreq, data[0, :n] * 1e6, label=name, alpha=0.8)
        frequencies, psd = welch(data, fs=sfreq, nperseg=min(data.shape[1], int(4 * sfreq)), axis=-1)
        axes[2].semilogy(frequencies, np.mean(psd, axis=0), label=name)
    axes[0].set(xlabel="Raw acquisition time (s)", ylabel="µV", title=preview["channels"][0])
    axes[1].set(xlabel="Raw acquisition time (s)", ylabel="µV", title="Corrected signals at a readable scale")
    axes[2].set(xlabel="Frequency (Hz)", ylabel="Mean PSD (V²/Hz)", xlim=(1, 80))
    for frequency in (12, 24):
        axes[2].axvline(frequency, color="gray", linestyle=":", linewidth=0.7)
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(directory / "four_stage_comparison.png", dpi=150)
    plt.close(fig)
    np.savez_compressed(
        directory / "comparison_preview.npz",
        sfreq=sfreq,
        raw_start=start,
        channels=preview["channels"],
        **{k.replace(" ", "_"): v for k, v in stages.items()},
    )


def validate_single_recording(dataset, settings, cache, directory):
    """Record a useful failure report as well as successful validation evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        report = _validate_single_recording(dataset, settings, cache, directory)
    except Exception as exc:
        atomic_json(
            directory / "validation_status.json",
            {
                "passed": False,
                "dataset_id": dataset.dataset_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "fixed_preprocessing": DEFAULT_CLEANING.manifest(),
                "execution_seconds": time.perf_counter() - started,
            },
        )
        raise
    atomic_json(directory / "validation_status.json", {"passed": True, "identity": report["identity"]})
    return report


def _validate_single_recording(dataset, settings, cache, directory):
    """Check one full fixed pipeline and a different Flex matrix before TPE.

    This gate tests execution/finite metrics, not reference-score thresholds;
    its reference values are never used to choose a recipe or change settings.
    """
    prepared = prepare_dataset(dataset, settings, cache.database.parent / "prepared")
    identity = digest((prepared.name, DEFAULT_CLEANING.manifest(), "validation_v1"))
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "validation.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("identity") == identity and previous.get("passed"):
            return previous
    started = time.perf_counter()
    baseline = MatrixDecisions()
    variant = replace(baseline, quota=DirectionalQuota.symmetric(20))
    fixed_before = DEFAULT_CLEANING.manifest()
    with threadpool_limits(limits=1):
        first = evaluate_prepared(baseline, dataset, settings, prepared, cache, "full", diagnostics_directory=directory)
        baseline_screening = evaluate_prepared(baseline, dataset, settings, prepared, cache, "screening")
        second = evaluate_prepared(variant, dataset, settings, prepared, cache, "screening")
    required = [
        "reference_normalized_mse",
        "reference_rms_residual",
        "reference_median_residual",
        "reference_correlation",
        "reference_spectral_similarity",
        "reference_snr_db",
    ]
    for result in (first, second):
        if not result["success"] or not all(np.isfinite(result.get(key, np.nan)) for key in required):
            raise ValueError("Single-recording validation failed: non-finite final reference metrics.")
    if fixed_before != DEFAULT_CLEANING.manifest():
        raise ValueError("Downstream settings changed during validation.")
    report = {
        "identity": identity,
        "passed": True,
        "dataset_id": dataset.dataset_id,
        "baseline_matrix": baseline.to_dict(),
        "variant_matrix": variant.to_dict(),
        "baseline_metrics": first,
        "variant_screening_metrics": second,
        "baseline_screening_metrics": baseline_screening,
        "flex_checkpoint_changed": any(
            not np.isclose(baseline_screening[key], second[key], rtol=1e-8, atol=0)
            for key in ("scanner_peak_residual", "nonpeak_eeg_log_deviation", "signal_rms")
        ),
        "fixed_preprocessing": fixed_before,
        "alignment": json.loads((prepared / "prepared.json").read_text())["reference_plan"],
        "execution_seconds": time.perf_counter() - started,
        "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        "interpretation": "Reference is a published comparison output, not neural ground truth. "
        "Execution gate only; no reference threshold selects or tunes the recipe.",
    }
    atomic_json(marker, report)
    pd.DataFrame(
        [
            {"configuration": "baseline_full", **first},
            {"configuration": "baseline_screening", **baseline_screening},
            {"configuration": "variant_screening", **second},
        ]
    ).to_csv(directory / "validation_metrics.csv", index=False)
    return report
