"""Metadata-only single-stage and multistage EEGLAB alignment regressions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from facet.optimization.eeglab_alignment import published_sample_intervals
from facet.optimization.eeglab_diagnostic import inspect_provenance, main


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def metadata_fixture(tmp_path, mask, final, *, native_rate=5000, rate=250, markerless=(), durations=None):
    """Encode retained intervals and urevent origins, without any EEG arrays."""
    durations = durations or {}
    original_events, urevents, events, expected = [], [], [], []
    reference_start, previous_end, marker_id = 0, 0, 10
    for segment, (start, stop) in enumerate(_runs(final)):
        if start > previous_end:
            events.append(
                {
                    "type": "boundary",
                    "latency": reference_start + 0.5,
                    "duration": durations.get(int(previous_end), int(start - previous_end)),
                }
            )
        length = stop - start
        if segment not in markerless:
            for offset in sorted({5, int(length) - 5}):
                if not 1 <= offset < length - 1:
                    continue
                # Fractional event latencies survive downsampling exactly.
                offset += 0.25
                source = start + offset
                original_events.append(
                    {"type": "R128", "latency": source * native_rate / rate + 1, "bvmknum": marker_id}
                )
                urevents.append({"type": "R128", "latency": source + 1, "bvmknum": marker_id})
                events.append(
                    {
                        "type": "R128",
                        "latency": reference_start + offset + 1,
                        "urevent": len(urevents),
                        "bvmknum": marker_id,
                    }
                )
                marker_id += 1
        expected.append(
            {"raw_start": start / rate, "reference_start": reference_start / rate, "duration": length / rate}
        )
        reference_start += length
        previous_end = stop
    if previous_end < len(mask):
        events.append({"type": "boundary", "latency": reference_start + 0.5, "duration": len(mask) - previous_end})
    raw_path, ref_path = tmp_path / "raw.set", tmp_path / "reference.set"
    raw_meta = {
        "srate": native_rate,
        "pnts": int(len(mask) * native_rate / rate),
        "trials": 1,
        "event": original_events,
    }
    ref_meta = {
        "srate": rate,
        "pnts": int(final.sum()),
        "trials": 1,
        "etc": {"clean_sample_mask": mask},
        "event": events,
        "urevent": urevents,
        "history": "synthetic documented rejections",
    }
    savemat(raw_path, raw_meta)
    savemat(ref_path, ref_meta)
    raw = SimpleNamespace(n_times=raw_meta["pnts"], info={"sfreq": native_rate})
    ref = SimpleNamespace(n_times=ref_meta["pnts"], info={"sfreq": rate})
    return raw_path, ref_path, raw, ref, ref_meta, expected


def subject_two_fixture(tmp_path):
    mask = np.ones(161730, dtype=bool)
    for start, stop in ((500, 1500), (20000, 22000), (90000, 92629)):
        mask[start:stop] = False
    final = mask.copy()
    for start, stop in ((0, 200), (22000, 22800), (70000, 72000), (140000, 141116)):
        final[start:stop] = False
    assert mask.sum() == 156101 and final.sum() == 151985
    # The merged boundary records a later-stage loss of 800 retained samples,
    # not the 2800-sample original-clock gap including the earlier rejection.
    return metadata_fixture(tmp_path, mask, final, durations={20000: 800})


def test_single_stage_reported_subject_one_counts_are_unchanged(tmp_path):
    mask = np.ones(155600, dtype=bool)
    mask[1000:4000] = False
    mask[80000:82220] = False
    fixture = metadata_fixture(tmp_path, mask, mask)
    intervals, evidence = published_sample_intervals(*fixture[:4])
    assert mask.sum() == 150380
    assert intervals == fixture[-1]
    assert evidence["additional_deleted_samples"] == 0
    assert evidence["maximum_event_error_seconds"] < 1e-10


def test_multistage_reported_subject_two_counts_without_guessing_positions(tmp_path):
    fixture = subject_two_fixture(tmp_path)
    intervals, evidence = published_sample_intervals(*fixture[:4])
    assert intervals == fixture[-1]
    assert evidence["final_retained_samples"] == 151985
    assert sum(item["duration"] for item in intervals) * 250 == pytest.approx(151985)
    assert evidence["additional_deleted_samples"] == 4116
    assert evidence["additional_deleted_seconds"] == pytest.approx(16.464)
    assert sum(item["seconds"] for item in evidence["additional_deleted_intermediate_intervals"]) == pytest.approx(
        16.464
    )
    assert evidence["maximum_event_error_seconds"] < 1e-10
    assert evidence["maximum_mapped_event_error_seconds"] < 1e-10
    assert evidence["urevent_anchored_segments"] == len(intervals)


def test_short_unanchored_segment_is_resolved_uniquely_from_boundaries(tmp_path):
    mask = np.ones(1000, dtype=bool)
    mask[300:350] = False
    final = mask.copy()
    final[500:520] = False
    fixture = metadata_fixture(tmp_path, mask, final, markerless=(1,))
    intervals, evidence = published_sample_intervals(*fixture[:4])
    assert intervals == fixture[-1]
    assert evidence["duration_resolved_segments"] == 1


def test_unanchored_span_requires_exact_sample_conservation(tmp_path):
    mask = np.ones(1000, dtype=bool)
    mask[200:220] = False
    mask[300:330] = False
    final = mask.copy()
    final[400:420] = False
    fixture = metadata_fixture(tmp_path, mask, final, markerless=(1, 2))
    intervals, evidence = published_sample_intervals(*fixture[:4])
    assert intervals == fixture[-1]
    assert evidence["duration_resolved_segments"] == 2
    meta = fixture[4]
    next(e for e in meta["event"] if e["type"] == "boundary")["duration"] -= 1
    savemat(fixture[1], meta)
    with pytest.raises(ValueError, match="Unanchored/ambiguous"):
        published_sample_intervals(*fixture[:4])


@pytest.mark.parametrize(
    "damage",
    [
        "no_boundaries",
        "missing_boundary",
        "bad_duration",
        "unknown_duration",
        "bad_pointer",
        "wrong_clock",
        "wrong_identity",
        "non_integer_cut",
        "missing_urevents",
    ],
)
def test_inconsistent_or_missing_provenance_blocks_multistage_alignment(tmp_path, damage):
    fixture = subject_two_fixture(tmp_path)
    raw_path, ref_path, raw, reference, meta, _ = fixture
    boundaries = [e for e in meta["event"] if e["type"] == "boundary"]
    retained = [e for e in meta["event"] if e["type"] != "boundary"]
    if damage == "no_boundaries":
        meta["event"] = retained
    elif damage == "missing_boundary":
        meta["event"].remove(boundaries[2])
    elif damage == "bad_duration":
        boundaries[2]["duration"] += 1
    elif damage == "unknown_duration":
        boundaries[2]["duration"] = np.nan
    elif damage == "bad_pointer":
        retained[0]["urevent"] = 9999
    elif damage == "wrong_clock":
        meta["urevent"][0]["latency"] += 20
    elif damage == "wrong_identity":
        meta["urevent"][0]["type"] = "different"
    elif damage == "non_integer_cut":
        boundaries[2]["latency"] += 0.1
    elif damage == "missing_urevents":
        meta.pop("urevent")
    savemat(ref_path, meta)
    with pytest.raises(ValueError, match=r"4116 additional samples \(16.464 seconds\).*Cannot verify final timeline"):
        published_sample_intervals(raw_path, ref_path, raw, reference)


def test_unanchored_segment_without_duration_does_not_get_interpolated(tmp_path):
    mask = np.ones(1000, dtype=bool)
    mask[300:350] = False
    final = mask.copy()
    final[500:520] = False
    fixture = metadata_fixture(tmp_path, mask, final, markerless=(1,))
    meta = fixture[4]
    next(e for e in meta["event"] if e["type"] == "boundary")["duration"] = np.nan
    savemat(fixture[1], meta)
    with pytest.raises(ValueError, match="unknown duration"):
        published_sample_intervals(*fixture[:4])


def test_independent_raw_timestamps_are_required(tmp_path):
    fixture = subject_two_fixture(tmp_path)
    original = loadmat(fixture[0], simplify_cells=True)
    original["event"][0]["latency"] += 100
    savemat(fixture[0], {key: value for key, value in original.items() if not key.startswith("__")})
    with pytest.raises(ValueError, match="original-event timestamp evidence"):
        published_sample_intervals(*fixture[:4])


def test_fewer_mask_samples_than_final_eeg_is_not_padding_or_truncation(tmp_path):
    mask = np.ones(1000, dtype=bool)
    final = mask.copy()
    fixture = metadata_fixture(tmp_path, mask, final)
    meta = fixture[4]
    meta["etc"]["clean_sample_mask"][100:110] = False
    savemat(fixture[1], meta)
    with pytest.raises(ValueError, match="fewer than final EEG"):
        published_sample_intervals(*fixture[:4])


def test_metadata_cli_never_requests_signal_variables_or_writes_files(tmp_path, monkeypatch, capsys):
    fixture = subject_two_fixture(tmp_path)
    from facet.optimization import eeglab_alignment, eeglab_diagnostic

    reads = []

    def selected_read(path, *args, **kwargs):
        fields = kwargs.get("variable_names")
        assert fields and "data" not in fields and "EEG" not in fields and "icaact" not in fields
        reads.append(fields)
        return loadmat(path, *args, **kwargs)

    monkeypatch.setattr(eeglab_alignment, "loadmat", selected_read)
    monkeypatch.setattr(eeglab_diagnostic, "loadmat", selected_read)
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    assert main([str(fixture[0]), str(fixture[1]), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mask"]["additional_missing_samples"] == 4116
    assert report["mask"]["additional_missing_seconds"] == 16.464
    assert report["boundaries"] and report["event_mappings"] and report["segments"]
    assert report["reconstruction"] == "verified" and not report["signals_loaded"]
    assert before == {path: path.read_bytes() for path in tmp_path.iterdir()}
    assert len(reads) >= 2


def test_metadata_cli_prints_blocking_evidence_without_exporting_signals(tmp_path, capsys):
    fixture = subject_two_fixture(tmp_path)
    meta = fixture[4]
    meta["event"] = [e for e in meta["event"] if e["type"] != "boundary"]
    savemat(fixture[1], meta)
    assert main([str(fixture[0]), str(fixture[1])]) == 2
    text = capsys.readouterr().out
    assert "4116" in text and "16.464" in text and "No finite final boundary" in text
    assert "not loaded, exported, or modified" in text


def test_diagnostic_reports_instead_of_accepting_missing_mask(tmp_path):
    fixture = subject_two_fixture(tmp_path)
    meta = fixture[4]
    meta["etc"] = {}
    savemat(fixture[1], meta)
    report = inspect_provenance(fixture[0], fixture[1])
    assert report["reconstruction"] == "blocked"
    assert "No supported published sample mask" in report["diagnostic"]
