"""Read published EEGLAB sample provenance; never estimate a signal lag."""

import numpy as np
from scipy.io import loadmat


def published_sample_intervals(raw_path, reference_path, raw, reference):
    """Recover retained intervals from an explicit clean_sample_mask.

    Flat EEGLAB SET metadata only. Other layouts require an alignment sidecar.
    Cross-check original event marker IDs and urevent timestamps independently
    of EEG values. This reads provenance, not MATLAB preprocessing algorithms.
    """
    if raw_path is None or raw_path.suffix.lower() != ".set" or reference_path.suffix.lower() != ".set":
        return None
    fields = ["etc", "event", "urevent", "history", "setname"]
    cleaned = loadmat(reference_path, simplify_cells=True, variable_names=fields)
    mask = cleaned.get("etc", {}).get("clean_sample_mask")
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim != 1 or not np.all(np.isin(mask, [0, 1])) or int(mask.sum()) != reference.n_times:
        raise ValueError("Published EEGLAB clean_sample_mask does not match the reference sample count.")
    rate = float(reference.info["sfreq"])
    if abs(len(mask) / rate - raw.n_times / raw.info["sfreq"]) > 1 / rate:
        raise ValueError("Published sample mask does not cover the raw recording duration.")
    original = loadmat(raw_path, simplify_cells=True, variable_names=["event"])
    markers = {
        int(event["bvmknum"]): event for event in original.get("event", []) if np.size(event.get("bvmknum", [])) == 1
    }
    errors, mapped_errors, boundary_events = [], [], 0
    removed_prefix = np.r_[0, np.cumsum(~mask.astype(bool))]
    for event in cleaned.get("event", []):
        if np.size(event.get("bvmknum", [])) != 1 or np.size(event.get("urevent", [])) != 1:
            continue
        marker = markers.get(int(event["bvmknum"]))
        if marker is None:
            continue
        source_event = cleaned["urevent"][int(event["urevent"]) - 1]
        if str(marker["type"]) != str(event["type"]):
            raise ValueError("EEGLAB original/reference marker identity differs.")
        errors.append(
            abs((float(source_event["latency"]) - 1) / rate - (float(marker["latency"]) - 1) / raw.info["sfreq"])
        )
        source_position = float(source_event["latency"]) - 1
        index = int(np.floor(source_position))
        if index < 0 or index >= len(mask) or not np.any(mask[max(0, index - 1) : min(len(mask), index + 2)]):
            raise ValueError("A retained EEGLAB event maps outside the published retained-sample mask.")
        # EEGLAB events have fractional latencies and may straddle a rejection
        # boundary. Check their mapping within one reference sample; the EEG
        # sample map itself remains exactly the published binary mask.
        boundary_events += int(not mask[index])
        expected = source_position - removed_prefix[index]
        mapped_errors.append(abs(expected - (float(event["latency"]) - 1)) / rate)
    if len(errors) < 3 or max(errors) > 1 / raw.info["sfreq"]:
        raise ValueError("Published sample mask lacks consistent original-event timestamp evidence.")
    if max(mapped_errors) > 1 / rate + 1e-9:
        raise ValueError("Reference event latencies disagree with the published retained-sample mask.")
    edges = np.diff(np.r_[False, mask.astype(bool), False].astype(int))
    intervals, reference_start = [], 0
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        intervals.append(
            {"raw_start": start / rate, "reference_start": reference_start / rate, "duration": (stop - start) / rate}
        )
        reference_start += stop - start
    return intervals, {
        "method": "Published EEG.etc.clean_sample_mask and original bvmknum/urevent timestamps",
        "matched_events": len(errors),
        "maximum_event_error_seconds": max(errors),
        "maximum_mapped_event_error_seconds": max(mapped_errors),
        "fractional_events_adjacent_to_rejection": boundary_events,
        "reference_history": str(cleaned.get("history", "")),
        "reference_setname": str(cleaned.get("setname", "")),
    }
