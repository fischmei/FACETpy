"""Recover a final retained-sample mask from EEGLAB boundary/event provenance.

Boundary cuts use EEGLAB's half-sample convention (latency - 0.5).
Urevent clocks must be independently checked against the original recording
by the caller. No EEG values, guessed lag or length-only truncation are used.
"""

import numpy as np


def boundary_sample_mask(mask, events, links, final_samples):
    """Reconstruct only when each final continuous segment has a unique origin."""
    boundaries = {}
    for event in events:
        if str(event.get("type", "")).strip().lower() not in {"boundary", "-99"}:
            continue
        latency = float(event["latency"])
        duration = np.asarray(event.get("duration", []))
        # An original New Segment marker at the first sample carries no loss.
        if duration.size != 1 or not np.isfinite(float(duration.item())):
            if latency <= 1 or latency >= final_samples + 0.5:
                continue
            raise ValueError(
                "Final boundary has unknown duration; a verified rejection log or explicit alignment sidecar is required."
            )
        cut, deleted = _integer(latency - 0.5, "boundary cut"), _integer(float(duration.item()), "boundary duration")
        if cut < 0 or cut > final_samples or deleted < 0:
            raise ValueError("Final boundary lies outside the EEG or has a negative deletion duration.")
        boundaries[cut] = boundaries.get(cut, 0) + deleted
    if not boundaries:
        raise ValueError("No finite final boundary rejection metadata; additional sample deletion cannot be located.")
    cuts = sorted({0, final_samples, *boundaries})
    lengths = np.diff(cuts)
    starts = [None] * len(lengths)
    rejected_prefix = np.r_[0, np.cumsum(~mask)]

    def valid(start, length):
        return start >= 0 and start + length <= len(mask) and rejected_prefix[start + length] == rejected_prefix[start]

    # A retained event and its urevent locate a segment without inspecting EEG.
    # Ignore fractional events immediately adjacent to a rejection boundary
    # when choosing anchors; all events are subsequently checked by the caller.
    anchored = 0
    for index, (left, right) in enumerate(zip(cuts, cuts[1:], strict=False)):
        offsets = [source - (position - left) for source, position in links if left + 1 <= position < right - 1]
        if not offsets:
            continue
        origin = _integer(offsets[0], "urevent segment origin")
        if any(abs(value - origin) > 1e-6 for value in offsets):
            raise ValueError(
                f"Conflicting urevent origins in final segment [{left}, {right}); missing boundary or incompatible event clock."
            )
        if not valid(origin, right - left):
            raise ValueError(f"Final segment [{left}, {right}) is not contained in the published retained-sample mask.")
        starts[index] = origin
        anchored += 1

    # A boundary duration can undercount an original-clock gap after earlier
    # rejection. Never choose a clock interpretation for an unanchored span:
    # require both neighboring origins and exact sample conservation. If their
    # entire source span equals retained lengths plus recorded gap durations,
    # no unrecorded gap remains and each intermediate origin is determined.
    missing_indices = np.asarray([index for index, start in enumerate(starts) if start is None], dtype=int)
    for run in np.split(missing_indices, np.flatnonzero(np.diff(missing_indices) > 1) + 1):
        if not len(run):
            continue
        first, last = int(run[0]), int(run[-1])
        previous_end = 0 if first == 0 else starts[first - 1] + lengths[first - 1]
        following = len(mask) if last == len(starts) - 1 else starts[last + 1]
        gaps = [boundaries.get(cuts[index]) for index in range(first, last + 2)]
        if any(gap is None for gap in gaps) or following - previous_end != sum(lengths[first : last + 1]) + sum(gaps):
            continue
        origin = previous_end
        resolved = []
        for index in range(first, last + 1):
            origin += boundaries[cuts[index]]
            if not valid(origin, lengths[index]):
                break
            resolved.append(int(origin))
            origin += lengths[index]
        if len(resolved) == len(run):
            starts[first : last + 1] = resolved
    if any(start is None for start in starts):
        missing = [(cuts[index], cuts[index + 1]) for index, start in enumerate(starts) if start is None]
        raise ValueError(
            f"Unanchored/ambiguous final segments {missing}: retained-event urevent links or sample-exact rejection provenance are missing."
        )
    final_mask = np.zeros(len(mask), dtype=bool)
    previous_end = 0
    evidence = []
    for index, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        if start < previous_end:
            raise ValueError("Reconstructed source intervals overlap or reverse time.")
        gap = start - previous_end
        intermediate_gap = int(mask[previous_end:start].sum())
        cut = cuts[index]
        if cut in boundaries and boundaries[cut] not in {gap, intermediate_gap}:
            raise ValueError(
                f"Boundary duration at final sample {cut} contradicts the recovered original/intermediate sample gap."
            )
        final_mask[start : start + length] = True
        evidence.append(
            {
                "reference_start_sample": cut,
                "raw_resampled_start_sample": int(start),
                "samples": int(length),
                "original_gap_samples": int(gap),
                "intermediate_gap_samples": intermediate_gap,
                "boundary_duration_samples": boundaries.get(cut),
            }
        )
        previous_end = start + length
    if final_samples in boundaries:
        gap = len(mask) - previous_end
        if boundaries[final_samples] not in {gap, int(mask[previous_end:].sum())}:
            raise ValueError("Trailing boundary duration contradicts the recovered source endpoint.")
    if int(final_mask.sum()) != final_samples or np.any(final_mask & ~mask):
        raise ValueError("Reconstructed intervals do not exactly account for final EEG samples.")
    return final_mask, {
        "boundary_segments": evidence,
        "urevent_anchored_segments": anchored,
        "duration_resolved_segments": len(starts) - anchored,
    }


def _integer(value, label):
    if not np.isfinite(value) or abs(value - round(value)) > 1e-6:
        raise ValueError(
            f"Non-integer {label}; its sampling-rate/stage provenance is insufficient for exact sample mapping."
        )
    return int(round(value))
