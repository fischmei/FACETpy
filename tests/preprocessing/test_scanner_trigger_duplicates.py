"""Exact-sample scanner deduplication and continuous-acquisition safeguards."""

import mne
import numpy as np
import pytest
from loguru import logger

from facet.core import ProcessingContext, ProcessorValidationError
from facet.optimization.search import preflight_datasets
from facet.preprocessing import TriggerDetector


def annotation_recording(samples, *, first_samp=0, boundaries=(0.0,)):
    sfreq = 100.0
    raw = mne.io.RawArray(
        np.zeros((1, int(max(samples)) + 300)),
        mne.create_info(["Cz"], sfreq, "eeg"),
        first_samp=first_samp,
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            np.r_[np.asarray(samples) / sfreq, boundaries],
            np.zeros(len(samples) + len(boundaries)),
            ["R128"] * len(samples) + ["boundary"] * len(boundaries),
        )
    )
    return raw


def test_550_annotations_become_288_triggers_before_metadata_and_preflight(tmp_path, monkeypatch):
    distinct = 100 + np.arange(288) * 210  # 2.1 s at 100 Hz
    raw = annotation_recording(np.r_[distinct, distinct[:262]])
    assert len(TriggerDetector("^R128$")._find_events(raw)) == 550
    annotations_before = raw.annotations.copy()
    data_before = raw.get_data()
    original = TriggerDetector._compute_artifact_metadata
    seen = []

    def inspect(self, triggers):
        seen.append(triggers.copy())
        return original(self, triggers)

    monkeypatch.setattr(TriggerDetector, "_compute_artifact_metadata", inspect)
    messages = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        result = TriggerDetector("^R128$").execute(ProcessingContext(raw))
    finally:
        logger.remove(sink)
    np.testing.assert_array_equal(seen[0], distinct)
    np.testing.assert_array_equal(result.get_triggers(), distinct)
    assert result.get_artifact_length() == 210
    assert not result.metadata.volume_gaps
    assert result.metadata.slices_per_volume is None
    assert any("original=550, duplicates_removed=262, final=288" in message for message in messages)
    assert result.get_raw() is raw
    np.testing.assert_array_equal(raw.get_data(), data_before)
    np.testing.assert_array_equal(raw.annotations.onset, annotations_before.onset)
    np.testing.assert_array_equal(raw.annotations.description, annotations_before.description)

    path = tmp_path / "sub-16_ses-02_task-rest_eeg.fif"
    raw.save(path, fmt="double", verbose=False)
    dataset = preflight_datasets({"session": path}, trigger_regex="^R128$", motion_directory=None, rotation_scale=0.0)[
        "session"
    ]
    np.testing.assert_array_equal(dataset.trigger_samples, distinct)
    assert dataset.slices_per_volume is None


def test_recording_without_duplicates_is_unaffected():
    distinct = 100 + np.arange(288) * 210
    reference = TriggerDetector("^R128$").execute(ProcessingContext(annotation_recording(distinct)))
    duplicates = TriggerDetector("^R128$").execute(
        ProcessingContext(annotation_recording(np.r_[distinct, distinct[:262]]))
    )
    np.testing.assert_array_equal(reference.get_triggers(), duplicates.get_triggers())
    assert reference.metadata.to_dict() == duplicates.metadata.to_dict()


def test_preflight_still_rejects_only_one_distinct_scanner_trigger(tmp_path):
    raw = annotation_recording([100, 100, 100])
    path = tmp_path / "single_eeg.fif"
    raw.save(path, verbose=False)
    with pytest.raises(ValueError, match="Invalid scanner triggers"):
        preflight_datasets({"single": path}, trigger_regex="^R128$", motion_directory=None, rotation_scale=0.0)


@pytest.mark.parametrize("first_samp", [0, 1000])
def test_distinct_adjacent_sample_triggers_are_retained(first_samp):
    samples = np.array([10, 11, 20, 21, 30, 31])
    raw = annotation_recording(np.r_[samples, samples[::2]], first_samp=first_samp)
    result = TriggerDetector("^R128$").execute(ProcessingContext(raw))
    np.testing.assert_array_equal(result.get_triggers(), samples)
    assert np.min(np.diff(result.get_triggers())) == 1


def test_real_slice_and_volume_gaps_are_preserved():
    samples = np.array([10, 20, 30, 60, 70, 80, 110, 120, 130])
    expected = TriggerDetector("^R128$").execute(ProcessingContext(annotation_recording(samples)))
    result = TriggerDetector("^R128$").execute(ProcessingContext(annotation_recording(np.r_[samples, samples])))
    np.testing.assert_array_equal(result.get_triggers(), samples)
    assert result.get_artifact_length() == 10
    assert result.metadata.volume_gaps
    assert result.metadata.slices_per_volume == expected.metadata.slices_per_volume


def test_duplicate_events_from_multiple_stim_channels():
    samples = np.array([20, 120, 220])
    data = np.zeros((2, 300))
    for sample in samples:
        data[:, sample : sample + 2] = 1
    raw = mne.io.RawArray(data, mne.create_info(["STI001", "STI002"], 100, "stim"), first_samp=1000, verbose=False)
    detector = TriggerDetector("^1$")
    assert len(detector._find_events(raw)) == 6
    np.testing.assert_array_equal(detector.execute(ProcessingContext(raw)).get_triggers(), samples)


@pytest.mark.parametrize("description", ["boundary", "BAD boundary", "EDGE boundary", "BOUNDARY"])
@pytest.mark.parametrize("first_samp", [0, 1000])
def test_internal_boundary_fails_before_timing_inference(description, first_samp, monkeypatch):
    raw = annotation_recording([100, 310, 520, 730], first_samp=first_samp, boundaries=())
    raw.set_annotations(mne.Annotations([1.0, 3.1, 5.2, 7.3, 4.0], np.zeros(5), ["R128"] * 4 + [description]))
    detector = TriggerDetector("^R128$")
    monkeypatch.setattr(
        detector, "_compute_artifact_metadata", lambda *_: pytest.fail("Inferred timing across a boundary")
    )
    with pytest.raises(ProcessorValidationError, match="Internal boundary.*sample 400.*Split the recording"):
        detector.execute(ProcessingContext(raw))


def test_end_boundary_does_not_create_a_second_acquisition():
    raw = annotation_recording([100, 310, 520])
    raw.set_annotations(raw.annotations + mne.Annotations([raw.n_times / 100], [0.0], ["boundary"]))
    result = TriggerDetector("^R128$").execute(ProcessingContext(raw))
    np.testing.assert_array_equal(result.get_triggers(), [100, 310, 520])


@pytest.mark.parametrize("samples", [[10, 20, 20, 15], [-1, 10, 10], [10, 10, 10000], [10, 10, np.nan]])
def test_invalid_sequences_are_not_repaired_by_sorting_or_clipping(samples, monkeypatch):
    raw = annotation_recording([10, 20])
    detector = TriggerDetector("^R128$")
    monkeypatch.setattr(detector, "_find_events", lambda _: [[sample, 0, 1] for sample in samples])
    with pytest.raises(ProcessorValidationError, match="finite, in bounds, and strictly increasing"):
        detector.execute(ProcessingContext(raw))
