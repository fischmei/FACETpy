"""Tests for the universal BIDS-to-EDF+ command."""

from __future__ import annotations

from types import SimpleNamespace

import mne
import numpy as np
import pytest

from facet import _cli_convert_bids as converter

pytestmark = pytest.mark.unit


def _recording(root, name="sub-01_task-rest_eeg.set"):
    path = root / "sub-01" / "eeg" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_discovery_finds_supported_bids_eeg_recordings_only(tmp_path):
    set_path = _recording(tmp_path)
    vhdr_path = tmp_path / "sub-02" / "ses-01" / "eeg" / "sub-02_ses-01_task-test_run-1_eeg.vhdr"
    vhdr_path.parent.mkdir(parents=True)
    vhdr_path.touch()
    vhdr_path.with_suffix(".eeg").touch()
    vhdr_path.with_suffix(".vmrk").touch()
    (vhdr_path.parent / "sub-02_ses-01_task-test_run-1_eeg.fif").touch()
    (tmp_path / "sub-03_task-rest_eeg.edf").touch()
    invalid_layout = tmp_path / "misc" / "eeg" / "sub-04_task-rest_eeg.bdf"
    invalid_layout.parent.mkdir(parents=True)
    invalid_layout.touch()

    assert converter.discover_bids_eeg_recordings(tmp_path) == [
        invalid_layout,
        set_path,
        vhdr_path.parent / "sub-02_ses-01_task-test_run-1_eeg.fif",
        vhdr_path,
    ]


def test_discovery_accepts_session_filename_without_enforcing_layout(tmp_path):
    recording = (
        tmp_path
        / "imported"
        / "eeg"
        / "sub-01_ses-01_task-checker_eeg.set"
    )
    recording.parent.mkdir(parents=True)
    recording.touch()

    assert converter.discover_bids_eeg_recordings(tmp_path) == [recording]


def test_matching_metadata_uses_recording_stem(tmp_path):
    recording = tmp_path / "eeg" / "sub-01_ses-01_task-checker_eeg.set"

    metadata = converter.matching_bids_metadata(recording)

    assert metadata["events"].name == "sub-01_ses-01_task-checker_events.tsv"
    assert metadata["channels"].name == "sub-01_ses-01_task-checker_channels.tsv"
    assert metadata["eeg"].name == "sub-01_ses-01_task-checker_eeg.json"


def test_events_tsv_preserves_onsets_durations_and_labels(tmp_path):
    events_path = tmp_path / "sub-01_task-rest_events.tsv"
    events_path.write_text(
        "onset\tduration\ttrial_type\tvalue\n"
        "0.25\t0.1\tstimulus\t1\n"
        "1.5\tn/a\tn/a\tresponse\n",
        encoding="utf-8",
    )

    annotations = converter.read_bids_event_annotations(events_path)

    assert annotations.onset.tolist() == [0.25, 1.5]
    assert annotations.duration.tolist() == [0.1, 0.0]
    assert annotations.description.tolist() == ["stimulus", "response"]


def test_conversion_imports_events_and_uses_bids_filename(tmp_path):
    recording = _recording(tmp_path / "bids")
    recording.with_name("sub-01_task-rest_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0.5\t0.25\tbutton press\n",
        encoding="utf-8",
    )
    raw = mne.io.RawArray(np.zeros((1, 200)), mne.create_info(["Cz"], 100.0, "eeg"), verbose=False)
    context = SimpleNamespace(get_raw=lambda: raw)
    loaded_context = context
    exports = []

    class FakeLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def execute(self, context):
            return loaded_context

    class FakeExporter:
        def __init__(self, **kwargs):
            exports.append(kwargs)

        def execute(self, context):
            return context

    status = converter.convert_bids_dataset(
        tmp_path / "bids",
        tmp_path / "output",
        loader_cls=FakeLoader,
        exporter_cls=FakeExporter,
    )

    assert status == 0
    assert raw.annotations.onset.tolist() == [0.5]
    assert raw.annotations.duration.tolist() == [0.25]
    assert raw.annotations.description.tolist() == ["button press"]
    assert exports == [{"path": str(tmp_path / "output" / "sub-01_task-rest_eeg.edf"), "overwrite": True}]


def test_conversion_applies_channels_and_eeg_sidecars(tmp_path):
    recording = _recording(tmp_path / "bids", name="recording.set")
    recording.with_name("recording_channels.tsv").write_text(
        "name\ttype\tstatus\nCz\tEEG\tgood\nECG\tECG\tbad\n",
        encoding="utf-8",
    )
    recording.with_name("recording_eeg.json").write_text(
        '{"PowerLineFrequency": 50}',
        encoding="utf-8",
    )
    raw = mne.io.RawArray(
        np.zeros((2, 20)),
        mne.create_info(["Cz", "ECG"], 10.0, ["misc", "misc"]),
        verbose=False,
    )
    loaded_context = SimpleNamespace(get_raw=lambda: raw)

    class FakeLoader:
        def __init__(self, **kwargs):
            pass

        def execute(self, context):
            return loaded_context

    class FakeExporter:
        def __init__(self, **kwargs):
            pass

        def execute(self, context):
            return context

    status = converter.convert_bids_dataset(
        tmp_path / "bids",
        tmp_path / "output",
        loader_cls=FakeLoader,
        exporter_cls=FakeExporter,
    )

    assert status == 0
    assert raw.get_channel_types() == ["eeg", "ecg"]
    assert raw.info["bads"] == ["ECG"]
    assert raw.info["line_freq"] == 50.0


def test_edf_plus_round_trip_preserves_bids_annotations(tmp_path):
    recording = _recording(tmp_path / "bids", name="sub-01_task-rest_eeg.edf")
    source_raw = mne.io.RawArray(
        np.zeros((1, 200)),
        mne.create_info(["Cz"], 100.0, "eeg"),
        verbose=False,
    )
    source_raw.export(recording, fmt="edf", overwrite=True, verbose=False)
    recording.with_name("sub-01_task-rest_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0.5\t0.25\tbutton press\n",
        encoding="utf-8",
    )

    status = converter.convert_bids_dataset(tmp_path / "bids", tmp_path / "output")
    converted = mne.io.read_raw_edf(
        tmp_path / "output" / "sub-01_task-rest_eeg.edf",
        preload=False,
        verbose=False,
    )

    assert status == 0
    assert converted.annotations.onset.tolist() == [0.5]
    assert converted.annotations.duration.tolist() == [0.25]
    assert converted.annotations.description.tolist() == ["button press"]


def test_conversion_continues_after_unreadable_recording(tmp_path):
    first = _recording(tmp_path / "bids")
    second = tmp_path / "bids" / "sub-02" / "eeg" / "sub-02_task-rest_eeg.edf"
    second.parent.mkdir(parents=True)
    second.touch()
    exported = []

    class FakeLoader:
        def __init__(self, path, **kwargs):
            self.path = path

        def execute(self, context):
            if self.path == str(first):
                raise OSError("missing EEGLAB payload")
            raw = mne.io.RawArray(np.zeros((1, 10)), mne.create_info(["Cz"], 10.0, "eeg"), verbose=False)
            return SimpleNamespace(get_raw=lambda: raw)

    class FakeExporter:
        def __init__(self, path, **kwargs):
            self.path = path

        def execute(self, context):
            exported.append(self.path)

    status = converter.convert_bids_dataset(
        tmp_path / "bids",
        tmp_path / "output",
        loader_cls=FakeLoader,
        exporter_cls=FakeExporter,
    )

    assert status == 1
    assert exported == [str(tmp_path / "output" / "sub-02_task-rest_eeg.edf")]


def test_dedicated_parser_accepts_only_required_paths():
    parser = converter.build_convert_bids_parser()
    args = parser.parse_args(["--bids-root", "dataset", "--output-dir", "converted"])

    assert vars(args) == {"bids_root": "dataset", "output_dir": "converted"}
    with pytest.raises(SystemExit):
        parser.parse_args(["--bids-root", "dataset", "--output-dir", "converted", "--overwrite"])
