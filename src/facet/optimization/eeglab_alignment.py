"""Read published EEGLAB sample provenance; never estimate a signal lag."""

import numpy as np
from scipy.io import loadmat

from facet.optimization.eeglab_boundaries import boundary_sample_mask


def published_sample_intervals(raw_path, reference_path, raw, reference):
    """Recover final intervals, allowing a verifiable intermediate sample mask.

    Flat EEGLAB SET metadata only. Other layouts require an alignment sidecar.
    Cross-check original event marker IDs and urevent timestamps independently
    of EEG values. This reads provenance, not MATLAB preprocessing algorithms.
    """
    if raw_path is None or raw_path.suffix.lower() != ".set" or reference_path.suffix.lower() != ".set":
        return None
    fields = ["etc", "event", "urevent", "history", "setname", "srate", "pnts"]
    cleaned = loadmat(reference_path, simplify_cells=True, variable_names=fields)
    mask = cleaned.get("etc", {}).get("clean_sample_mask")
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim != 1 or not np.all(np.isin(mask, [0, 1])):
        raise ValueError("Published EEGLAB clean_sample_mask must be a one-dimensional binary mask.")
    mask = mask.astype(bool)
    retained = int(mask.sum())
    if retained < reference.n_times:
        raise ValueError(
            f"Published mask retains {retained} samples, fewer than final EEG {reference.n_times}; "
            "sample deletion alone cannot explain the provenance."
        )
    rate = float(reference.info["sfreq"])
    if ("srate" in cleaned and cleaned["srate"] != rate) or (
        "pnts" in cleaned and cleaned["pnts"] != reference.n_times
    ):
        raise ValueError("EEGLAB header and loaded reference sampling rate/sample count disagree.")
    if abs(len(mask) / rate - raw.n_times / raw.info["sfreq"]) > 1 / rate:
        raise ValueError("Published sample mask does not cover the raw recording duration.")
    original = loadmat(raw_path, simplify_cells=True, variable_names=["event", "srate", "pnts"])
    extra = retained - reference.n_times
    try:
        events, links, errors = _event_evidence(original, cleaned, raw, rate)
        final_mask, reconstruction = (
            (mask, {}) if not extra else boundary_sample_mask(mask, events, links, reference.n_times)
        )
        mapped_errors, boundary_events = _validate_mapping(final_mask, links, rate)
    except ValueError as exc:
        if not extra:
            raise
        raise ValueError(
            f"Published mask retains {retained} samples; final EEG has {reference.n_times}: "
            f"{extra} additional samples ({extra / rate:g} seconds) need rejection provenance. "
            f"Cannot verify final timeline: {exc} Supply a verified rejection log or explicit alignment sidecar."
        ) from exc
    if int(final_mask.sum()) != reference.n_times:
        raise ValueError("Final provenance intervals do not equal the final EEG sample count.")
    intervals = _intervals(final_mask, rate)
    return intervals, {
        "method": "Published EEG.etc.clean_sample_mask and original bvmknum/urevent timestamps"
        if not extra
        else "Final EEGLAB boundary segments and urevent origins, independently checked against raw bvmknum timestamps and intermediate mask",
        "matched_events": len(errors),
        "maximum_event_error_seconds": max(errors),
        "maximum_mapped_event_error_seconds": max(mapped_errors),
        "fractional_events_adjacent_to_rejection": boundary_events,
        "published_mask_samples": len(mask),
        "published_mask_retained_samples": retained,
        "final_retained_samples": int(final_mask.sum()),
        "additional_deleted_samples": extra,
        "additional_deleted_seconds": extra / rate,
        "additional_deleted_source_intervals": _deleted_spans(mask & ~final_mask, rate),
        "additional_deleted_intermediate_intervals": _deleted_spans(~final_mask[mask], rate),
        "reference_history": str(cleaned.get("history", "")),
        "reference_setname": str(cleaned.get("setname", "")),
        **reconstruction,
    }


def _events(value):
    return [value] if isinstance(value, dict) else list(value)


def _event_evidence(original, cleaned, raw, rate):
    """Verify the urevent clock against a separate file's original markers."""
    if ("srate" in original and original["srate"] != raw.info["sfreq"]) or (
        "pnts" in original and original["pnts"] != raw.n_times
    ):
        raise ValueError("EEGLAB header and loaded raw sampling rate/sample count disagree.")
    markers = {}
    for event in _events(original.get("event", [])):
        if str(event.get("type", "")).lower() == "boundary" or np.size(event.get("bvmknum", [])) != 1:
            continue
        key = int(event["bvmknum"])
        if key in markers and (markers[key]["latency"] != event["latency"] or markers[key]["type"] != event["type"]):
            raise ValueError("Original marker IDs are ambiguous; independent event matching is impossible.")
        markers[key] = event
    events, urevents = _events(cleaned.get("event", [])), _events(cleaned.get("urevent", []))
    links, errors, timestamps = [], [], set()
    for event in events:
        if str(event.get("type", "")).strip().lower() in {"boundary", "-99"} or np.size(event.get("urevent", [])) != 1:
            continue
        pointer = float(event["urevent"])
        if not np.isfinite(pointer) or pointer != round(pointer) or not 1 <= pointer <= len(urevents):
            raise ValueError("Retained event has an invalid urevent pointer.")
        source_event = urevents[int(pointer) - 1]
        if str(source_event["type"]) != str(event["type"]):
            raise ValueError("Retained event and urevent identities disagree.")
        source_position, position = float(source_event["latency"]) - 1, float(event["latency"]) - 1
        if not np.all(np.isfinite([source_position, position])):
            raise ValueError("Retained event has a non-finite timestamp.")
        links.append((source_position, position))
        marker = markers.get(int(event["bvmknum"])) if np.size(event.get("bvmknum", [])) == 1 else None
        if marker is not None:
            if str(marker["type"]) != str(event["type"]):
                raise ValueError("EEGLAB original/reference marker identity differs.")
            errors.append(abs(source_position / rate - (float(marker["latency"]) - 1) / raw.info["sfreq"]))
            timestamps.add(float(marker["latency"]))
    if len(timestamps) < 3 or max(errors) > 1 / raw.info["sfreq"]:
        raise ValueError(
            "Published sample mask lacks consistent original-event timestamp evidence; urevent sampling clock must match raw time after downsampling."
        )
    return events, links, errors


def _validate_mapping(mask, links, rate):
    removed_prefix = np.r_[0, np.cumsum(~mask)]
    mapped_errors, boundary_events = [], 0
    for source_position, position in links:
        index = int(np.floor(source_position))
        if index < 0 or index >= len(mask) or not np.any(mask[max(0, index - 1) : min(len(mask), index + 2)]):
            raise ValueError("A retained EEGLAB event maps outside the published retained-sample mask.")
        # EEGLAB events have fractional latencies and may straddle a rejection
        # boundary. Check their mapping within one reference sample; the EEG
        # sample map itself remains exactly the published binary mask.
        boundary_events += int(not mask[index])
        expected = source_position - removed_prefix[index]
        mapped_errors.append(abs(expected - position) / rate)
    if max(mapped_errors) > 1 / rate + 1e-9:
        raise ValueError("Reference event latencies disagree with the published retained-sample mask.")
    return mapped_errors, boundary_events


def _intervals(mask, rate):
    edges = np.diff(np.r_[False, mask.astype(bool), False].astype(int))
    intervals, reference_start = [], 0
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        intervals.append(
            {"raw_start": start / rate, "reference_start": reference_start / rate, "duration": (stop - start) / rate}
        )
        reference_start += stop - start
    return intervals


def _deleted_spans(mask, rate):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [
        {"start_sample": int(start), "stop_sample": int(stop), "seconds": (stop - start) / rate}
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    ]
