"""Inspect SET provenance using selected MAT variables only, never EEG signals.

Run ``python -m facet.optimization.eeglab_diagnostic RAW.set REFERENCE.set``.
Output goes to stdout; no EEG or metadata files are created or modified.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from facet.optimization.eeglab_alignment import _events, published_sample_intervals

METADATA_FIELDS = ["srate", "pnts", "trials", "nbchan", "event", "urevent", "etc", "history", "setname"]


def inspect_provenance(raw_path, reference_path):
    """Return scalar summaries and selected event metadata, excluding signal data."""
    original = loadmat(raw_path, simplify_cells=True, variable_names=METADATA_FIELDS)
    cleaned = loadmat(reference_path, simplify_cells=True, variable_names=METADATA_FIELDS)
    for name, metadata in (("raw", original), ("reference", cleaned)):
        if "srate" not in metadata or "pnts" not in metadata:
            raise ValueError(
                f"{name}: metadata-only reading requires a flat SET with srate/pnts; "
                "nested EEG structs are not loaded because they may contain signals."
            )
        if metadata.get("trials", 1) != 1:
            raise ValueError(f"{name}: epoched SET provenance is unsupported.")
        if not np.isfinite(metadata["srate"]) or metadata["srate"] <= 0 or metadata["pnts"] <= 0:
            raise ValueError(f"{name}: sampling rate and sample count must be positive and finite.")
    rate = float(cleaned["srate"])
    mask = np.asarray(cleaned.get("etc", {}).get("clean_sample_mask", []))
    binary = mask.ndim == 1 and bool(np.all(np.isin(mask, [0, 1]))) and mask.size > 0
    report = {
        "raw": {
            "sfreq": float(original["srate"]),
            "samples": int(original["pnts"]),
            "duration_seconds": int(original["pnts"]) / float(original["srate"]),
        },
        "reference": {
            "sfreq": rate,
            "samples": int(cleaned["pnts"]),
            "duration_seconds": int(cleaned["pnts"]) / rate,
            "sample_count_source": "pnts header; signal array not inspected",
        },
        "mask": {"length": int(mask.size), "binary": binary, "retained": int(mask.sum()) if binary else None},
        "boundaries": [],
        "event_mappings": [],
        "segments": [],
        "reference_history": str(cleaned.get("history", "")),
        "reference_setname": str(cleaned.get("setname", "")),
        "signals_loaded": False,
        "etc_fields": sorted(cleaned.get("etc", {})),
    }
    if binary:
        extra = int(mask.sum()) - int(cleaned["pnts"])
        report["mask"].update(additional_missing_samples=extra, additional_missing_seconds=extra / rate)
        report["mask"]["interpretation"] = (
            "Retained count matches final EEG; event timestamps must still verify the map."
            if extra == 0
            else "Mask retains fewer than final EEG; further deletion alone cannot explain the difference."
            if extra < 0
            else "Possible later rejection or stale/intermediate mask; counts alone cannot identify the stage or locations."
        )
        report["mask"]["duration_at_reference_rate_seconds"] = len(mask) / rate
    events, urevents = _events(cleaned.get("event", [])), _events(cleaned.get("urevent", []))
    markers = {int(e["bvmknum"]): e for e in _events(original.get("event", [])) if np.size(e.get("bvmknum", [])) == 1}
    for index, event in enumerate(events):
        if str(event.get("type", "")).strip().lower() in {"boundary", "-99"}:
            duration = np.asarray(event.get("duration", []))
            report["boundaries"].append(
                {
                    "event_index": index + 1,
                    "latency": float(event["latency"]),
                    "duration": float(duration.item()) if duration.size == 1 and np.isfinite(duration.item()) else None,
                }
            )
            continue
        pointer = np.asarray(event.get("urevent", []))
        if (
            pointer.size != 1
            or not np.isfinite(pointer.item())
            or not float(pointer.item()).is_integer()
            or not 1 <= pointer.item() <= len(urevents)
        ):
            report["event_mappings"].append({"event_index": index + 1, "error": "missing/invalid urevent pointer"})
            continue
        source = urevents[int(pointer.item()) - 1]
        mapping = {
            "event_index": index + 1,
            "type": str(event["type"]),
            "final_latency": float(event["latency"]),
            "urevent": int(pointer.item()),
            "source_latency": float(source["latency"]),
        }
        key = event.get("bvmknum", [])
        marker = markers.get(int(key)) if np.size(key) == 1 else None
        if marker is not None:
            mapping.update(
                bvmknum=int(key),
                raw_latency=float(marker["latency"]),
                raw_type=str(marker["type"]),
                raw_clock_error_seconds=abs(
                    (float(source["latency"]) - 1) / rate - (float(marker["latency"]) - 1) / float(original["srate"])
                ),
            )
        report["event_mappings"].append(mapping)
    cuts = sorted(
        {
            0,
            int(cleaned["pnts"]),
            *[
                int(round(row["latency"] - 0.5))
                for row in report["boundaries"]
                if 0.5 <= row["latency"] <= cleaned["pnts"] + 0.5
            ],
        }
    )
    for left, right in zip(cuts, cuts[1:], strict=False):
        links = [
            row
            for row in report["event_mappings"]
            if "error" not in row and left + 1 <= row["final_latency"] - 1 < right - 1
        ]
        origins = [row["source_latency"] - row["final_latency"] + left for row in links]
        report["segments"].append(
            {
                "reference_start": left,
                "samples": right - left,
                "urevent_links": len(links),
                "origin_min": min(origins) if origins else None,
                "origin_max": max(origins) if origins else None,
                "original_markers": sum("bvmknum" in row for row in links),
            }
        )
    objects = [
        SimpleNamespace(n_times=int(meta["pnts"]), info={"sfreq": float(meta["srate"])}) for meta in (original, cleaned)
    ]
    try:
        result = published_sample_intervals(raw_path, reference_path, *objects)
        if result is None:
            raise ValueError("No supported published sample mask; an explicit verified timeline is required.")
        intervals, evidence = result
        report.update(reconstruction="verified", intervals=intervals, evidence=evidence)
    except (ValueError, TypeError) as exc:
        report.update(reconstruction="blocked", diagnostic=str(exc))
    return report


def main(argv=None):
    """Print metadata diagnostics without running any processing pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument(
        "--max-events", type=int, default=12, help="Maximum event examples printed; 0 prints every mapping."
    )
    parser.add_argument("--json", action="store_true", help="Print the full diagnostic report as JSON to stdout.")
    args = parser.parse_args(argv)
    if args.max_events < 0:
        parser.error("--max-events must be nonnegative")
    try:
        report = inspect_provenance(args.raw, args.reference)
    except (ValueError, OSError, NotImplementedError) as exc:
        print(f"Metadata-only diagnostic failed: {exc}. No EEG signals were loaded.")
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for name in ("raw", "reference", "mask"):
            print(f"{name}: {json.dumps(report[name])}")
        print("Finite boundary durations are not assumed to be cumulative across rejection stages.")
        for row in report["boundaries"]:
            print("boundary:", json.dumps(row))
        for row in report["segments"]:
            print("segment:", json.dumps(row))
        mappings = report["event_mappings"]
        checked = [row["raw_clock_error_seconds"] for row in mappings if "raw_clock_error_seconds" in row]
        print(
            f"original marker clock checks: {len(checked)}; maximum error seconds: {max(checked) if checked else 'unavailable'}"
        )
        chosen = (
            mappings
            if not args.max_events
            else [
                mappings[index]
                for index in np.unique(
                    np.linspace(0, len(mappings) - 1, min(args.max_events, len(mappings)), dtype=int)
                )
            ]
        )
        print(f"event/urevent mappings: {len(mappings)} total; {len(chosen)} shown")
        for row in chosen:
            print("event:", json.dumps(row))
        print("reconstruction:", report["reconstruction"])
        print(json.dumps(report.get("evidence", report.get("diagnostic")), indent=2))
        print("setname:", report["reference_setname"])
        print("available etc fields:", ", ".join(report["etc_fields"]))
        print("history:", report["reference_history"])
        print("EEG signals were not loaded, exported, or modified.")
    return 0 if report["reconstruction"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
