"""Tests for chunked pipeline execution."""

import json
import math

import mne
import numpy as np
import pytest

from facet.core import Pipeline, ProcessingContext, Processor
from facet.io.loaders import _EXTENSION_READERS
from facet.preprocessing import DropChannelsMatching, TriggerDetector

pytestmark = pytest.mark.unit


class _TouchExporter(Processor):
    """Tiny exporter used by chunking tests to avoid format-specific I/O."""

    name = "touch_exporter"
    requires_raw = False
    modifies_raw = False

    def __init__(self, path: str):
        self.path = path
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        from pathlib import Path

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.path).write_text("ok", encoding="utf-8")
        return context


@pytest.fixture
def chunk_raw_factory():
    """Return a small Raw factory used by chunking tests."""

    def _factory(n_times: int = 120, sfreq: float = 10.0) -> mne.io.RawArray:
        rng = np.random.RandomState(7)
        info = mne.create_info(["EEG001", "EEG002"], sfreq=sfreq, ch_types="eeg")
        data = rng.standard_normal((2, n_times)) * 1e-6
        return mne.io.RawArray(data, info, verbose=False)

    return _factory


def test_run_chunked_writes_numbered_outputs(monkeypatch, tmp_path, chunk_raw_factory):
    """Chunked runs should crop lazily and write one numbered output per chunk."""
    read_preload_values = []

    def fake_read_raw_edf(path, *args, **kwargs):
        read_preload_values.append(kwargs.get("preload"))
        return chunk_raw_factory()

    monkeypatch.setitem(_EXTENSION_READERS, ".edf", (fake_read_raw_edf, "EDF"))

    input_path = tmp_path / "recording.edf"
    input_path.write_text("placeholder", encoding="utf-8")
    output_dir = tmp_path / "chunks"

    result = Pipeline([], name="Chunk test").run_chunked(
        input_path=str(input_path),
        output_dir=str(output_dir),
        output_extension=".edf",
        min_chunks=2,
        max_chunks=2,
        memory_budget_mb=512,
    )

    assert result.was_successful()
    assert len(result) == 2
    assert read_preload_values == [False, False, False]
    assert [path.name for path in result.output_paths] == [
        "recording_chunk_001_of_002.edf",
        "recording_chunk_002_of_002.edf",
    ]
    assert all(path.exists() for path in result.output_paths)

    manifest = json.loads((output_dir / "chunks_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_path"] == str(input_path)
    assert len(manifest["chunks"]) == 2
    assert manifest["chunks"][0]["start_sample"] == 0
    assert manifest["chunks"][0]["stop_sample"] == 60
    assert manifest["chunks"][1]["start_sample"] == 60
    assert manifest["chunks"][1]["stop_sample"] == 120


def test_run_chunked_uses_trigger_section_windows(monkeypatch, tmp_path):
    """AAS-style chunking should keep only trigger blocks with padding."""
    read_preload_values = []

    def fake_read_raw_edf(path, *args, **kwargs):
        read_preload_values.append(kwargs.get("preload"))
        sfreq = 10.0
        n_times = 1000
        ch_names = ["E1", "E128", "E129", "TREV", "ECG"]
        ch_types = ["eeg", "eeg", "eeg", "stim", "ecg"]
        data = np.zeros((len(ch_names), n_times))
        data[:3] = np.random.RandomState(11).standard_normal((3, n_times)) * 1e-6
        trigger_samples = np.r_[np.arange(200, 300, 5), np.arange(700, 800, 5)]
        data[3, trigger_samples] = 1
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        return mne.io.RawArray(data, info, verbose=False)

    monkeypatch.setitem(_EXTENSION_READERS, ".edf", (fake_read_raw_edf, "EDF"))

    input_path = tmp_path / "recording.edf"
    input_path.write_text("placeholder", encoding="utf-8")
    output_dir = tmp_path / "chunks"
    e1_to_e128 = r"^E(?:[1-9]|[1-9]\d|1[01]\d|12[0-8])$"

    result = Pipeline(
        [
            DropChannelsMatching(regex=e1_to_e128),
            TriggerDetector(regex=r"\b1\b"),
        ],
        name="Trigger section test",
    ).run_chunked(
        input_path=str(input_path),
        output_dir=str(output_dir),
        output_extension=".edf",
        exporter_factory=lambda path: _TouchExporter(path),
        trigger_section_padding_seconds=10.0,
        trigger_section_min_triggers=16,
    )

    assert result.was_successful()
    assert len(result) == 2
    assert read_preload_values == [False, False, False]
    assert [(chunk.start_sample, chunk.stop_sample) for chunk in result.chunks] == [
        (100, 396),
        (600, 896),
    ]
    assert [path.name for path in result.output_paths] == [
        "recording_chunk_001_of_002.edf",
        "recording_chunk_002_of_002.edf",
    ]
    assert all(path.exists() for path in result.output_paths)

    manifest = json.loads((output_dir / "chunks_manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunking_mode"] == "trigger_sections"
    assert [chunk["start_sample"] for chunk in manifest["chunks"]] == [100, 600]


def test_chunk_count_increases_until_estimate_fits():
    """The chunk-count selector should try 2 chunks, then 3, and so on."""
    n_chunks = Pipeline._choose_chunk_count(
        n_times=120,
        n_channels=2,
        sample_bytes=8,
        peak_sample_multiplier=10,
        processing_memory_multiplier=4,
        memory_budget=30_000,
        min_chunks=2,
        max_chunks=8,
        math_module=math,
    )

    assert n_chunks == 3
