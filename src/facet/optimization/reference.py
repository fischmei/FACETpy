"""Explicit reference provenance, timeline mapping, and bounded comparisons."""

import json
from math import gcd, lcm
from pathlib import Path

import numpy as np

from facet import Resample
from facet.core import ProcessingContext
from facet.optimization.cleaning import average_reference, good_eeg_names
from facet.optimization.eeglab_alignment import published_sample_intervals
from facet.optimization.metrics import reference_metrics


def alignment_path(reference_path: Path) -> Path:
    return Path(str(reference_path) + ".alignment.json")


def reference_plan(raw, reference, reference_path, *, assume_aligned=False, raw_path=None):
    """Validate a timeline using timestamps or an explicit interval sidecar.

    Sidecar intervals are in seconds from each file's first stored sample.
    They describe acquisition correspondence, never a score-fitted shift.
    """
    source_names, reference_names = good_eeg_names(raw), good_eeg_names(reference)
    shared = [name for name in source_names if name in reference_names]
    if len(shared) < 2:
        raise ValueError("Reference comparison requires at least two shared good EEG channels.")
    for recording in (raw, reference):
        for name in shared:
            if recording.info["chs"][recording.ch_names.index(name)]["unit"] != 107:
                raise ValueError(f"Reference comparison requires volt units for {name}.")
    source_duration = raw.n_times / float(raw.info["sfreq"])
    reference_duration = reference.n_times / float(reference.info["sfreq"])
    sidecar = alignment_path(reference_path)
    published = None if sidecar.exists() else published_sample_intervals(raw_path, reference_path, raw, reference)
    if sidecar.exists():
        payload = json.loads(sidecar.read_text())
        intervals = payload["intervals"]
        if not payload.get("provenance"):
            raise ValueError("Alignment sidecar requires provenance explaining the interval correspondence.")
        evidence = payload["provenance"]
    elif published is not None:
        intervals, evidence = published
    else:
        boundaries = any("boundary" in str(d).lower() for d in reference.annotations.description)
        tolerance = 1 / min(raw.info["sfreq"], reference.info["sfreq"])
        if boundaries or abs(source_duration - reference_duration) > tolerance:
            raise ValueError(
                f"Reference has removed samples or a different duration; provide {sidecar} "
                "with explicit raw_start/reference_start/duration intervals and provenance."
            )
        dates = raw.info["meas_date"], reference.info["meas_date"]
        if None in dates:
            if not assume_aligned:
                raise ValueError(
                    "Reference timestamps missing; verify alignment and use --assume-reference-aligned "
                    "or supply an alignment sidecar."
                )
            evidence = "User assertion: equal file starts and no removed samples"
        else:
            source_start = dates[0].timestamp() + raw.first_samp / raw.info["sfreq"]
            reference_start = dates[1].timestamp() + reference.first_samp / reference.info["sfreq"]
            if abs(source_start - reference_start) > 1e-6:
                raise ValueError("Reference start timestamp differs; supply an explicit alignment sidecar.")
            evidence = "Matching absolute start timestamps and recording durations"
        intervals = [{"raw_start": 0, "reference_start": 0, "duration": min(source_duration, reference_duration)}]
    if not intervals:
        raise ValueError("Reference alignment contains no valid intervals.")
    last_raw, last_ref = 0.0, 0.0
    for item in intervals:
        start, ref_start, duration = (float(item[k]) for k in ("raw_start", "reference_start", "duration"))
        if not np.all(np.isfinite([start, ref_start, duration])) or duration <= 0:
            raise ValueError("Alignment intervals require finite starts and positive duration.")
        if start < last_raw - 1e-9 or ref_start < last_ref - 1e-9:
            raise ValueError("Alignment intervals must be ordered, disjoint and nonnegative in both recordings.")
        if start + duration > source_duration + 1e-6 or ref_start + duration > reference_duration + 1e-6:
            raise ValueError("Alignment interval exceeds recording bounds.")
        last_raw, last_ref = start + duration, ref_start + duration
    return {
        "channels": shared,
        "missing_reference_channels": sorted(set(source_names) - set(reference_names)),
        "reference_only_channels": sorted(set(reference_names) - set(source_names)),
        "intervals": intervals,
        "provenance": evidence,
        "coverage": sum(float(item["duration"]) for item in intervals) / source_duration,
        "reference_sfreq": float(reference.info["sfreq"]),
        "reference_highpass": float(reference.info["highpass"]),
        "reference_lowpass": float(reference.info["lowpass"]),
        "harmonization": "Explicit resampling to least common integer rate (max 2000 Hz) for sample-exact "
        "metadata alignment; average reference on shared good EEG; volts",
    }


def valid_samples(raw):
    """Keep annotated bad samples on the timeline, but exclude them from scores."""
    valid = np.ones(raw.n_times, dtype=bool)
    sfreq = float(raw.info["sfreq"])
    for onset, duration, description in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description, strict=True
    ):
        if str(description).lower().startswith("bad"):
            first = max(0, int(np.floor((onset - raw.first_time) * sfreq)))
            stop = min(raw.n_times, int(np.ceil((onset - raw.first_time + duration) * sfreq)))
            valid[first:stop] = False
    return valid


def _sample(time, sfreq):
    sample = time * sfreq
    if abs(sample - round(sample)) > 1e-5:
        raise ValueError(
            f"Alignment time {time:g}s is not on the {sfreq:g} Hz sample grid; "
            "provide sample-exact interval boundaries. No fractional shift is fitted."
        )
    return int(round(sample))


def compare_chunk(cleaned, reference, plan, *, raw_start, core_start, core_stop, channel_names=None):
    """Score only mapped, valid core intervals; never concatenate gaps for PSD."""
    rates = float(cleaned.info["sfreq"]), float(reference.info["sfreq"])
    if any(rate != round(rate) for rate in rates):
        raise ValueError("Reference comparison currently requires integer sampling frequencies.")
    sfreq = float(lcm(*(int(rate) for rate in rates)))
    if sfreq > 2000:
        raise ValueError("Sample-exact common comparison rate exceeds 2000 Hz; provide a compatible reference.")
    names = plan["channels"]
    candidate = average_reference(cleaned.copy().pick(names))
    if candidate.info["sfreq"] != sfreq:
        candidate = Resample(sfreq).execute(ProcessingContext(candidate)).get_raw()
    candidate_data = candidate.get_data()
    candidate_valid = valid_samples(candidate)
    rows, weights, previews = [], [], []
    for interval in plan["intervals"]:
        begin = max(core_start, float(interval["raw_start"]))
        end = min(core_stop, float(interval["raw_start"]) + float(interval["duration"]))
        if end <= begin:
            continue
        ref_rate = float(reference.info["sfreq"])
        if sfreq != round(sfreq) or ref_rate != round(ref_rate):
            raise ValueError("Reference comparison currently requires integer sampling frequencies.")
        grid = gcd(int(sfreq), int(ref_rate))
        # Explicit deterministic intersection of sample grids, independent of
        # the signal values. Report usable duration rather than hiding rounding.
        begin = np.ceil((begin - raw_start) * grid - 1e-8) / grid + raw_start
        end = np.floor((end - raw_start) * grid + 1e-8) / grid + raw_start
        if end <= begin:
            continue
        a, b = _sample(begin - raw_start, sfreq), _sample(end - raw_start, sfreq)
        ref_begin = float(interval["reference_start"]) + begin - float(interval["raw_start"])
        ref_end = ref_begin + end - begin
        ra, rb = _sample(ref_begin, ref_rate), _sample(ref_end, ref_rate)
        # Read only this interval, including its BAD annotations. No unexplained
        # crop or fitted scaling, lag, polarity or channel permutation is used.
        ref = reference.copy().crop(ra / ref_rate, (rb - 1) / ref_rate).load_data().pick(names)
        ref = Resample(sfreq).execute(ProcessingContext(ref)).get_raw() if ref_rate != sfreq else ref
        ref = average_reference(ref)
        x, y = candidate_data[:, a:b], ref.get_data()
        if x.shape != y.shape:
            raise ValueError(f"Mapped reference sample counts differ: {x.shape} versus {y.shape}.")
        valid = candidate_valid[a:b] & valid_samples(ref)
        # Split gaps before Welch; joining rejected segments invents spectral edges.
        edges = np.diff(np.r_[False, valid, False].astype(int))
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            if right - left < 2 * sfreq:
                continue
            for channel, name in enumerate(names):
                if channel_names is not None and name not in channel_names:
                    continue
                metrics = reference_metrics(x[channel, left:right], y[channel, left:right], sfreq=sfreq)
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise ValueError(f"Reference metrics are non-finite for {name}; check shared-reference variance.")
                rows.append(metrics)
                weights.append(right - left)
            if not previews:
                previews = {
                    "candidate": x[:, left:right],
                    "reference": y[:, left:right],
                    "raw_start": begin + left / sfreq,
                    "channels": names,
                    "sfreq": sfreq,
                }
    if not rows:
        raise ValueError("No shared valid interval of at least two seconds in this chunk.")
    metrics = {key: float(np.average([row[key] for row in rows], weights=weights)) for key in rows[0]}
    count_channels = len(names if channel_names is None else set(names) & set(channel_names))
    metrics.update(
        reference_valid_seconds=sum(weights) / max(count_channels, 1) / sfreq,
        reference_channel_coverage=len(names) / len(good_eeg_names(cleaned)),
        reference_timeline_coverage=plan["coverage"],
        reference_metric_valid=True,
    )
    return metrics, previews
