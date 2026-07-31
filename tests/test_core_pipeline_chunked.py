"""Tests for chunked pipeline execution."""

import json
import math

import mne
import numpy as np
import pytest

from facet.core import Pipeline, PipelineError, ProcessingContext, Processor
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


class _FailOnceWithMemoryError(Processor):
    """Processor that simulates one underestimated peak-memory failure."""

    name = "fail_once_with_memory_error"
    requires_raw = False
    modifies_raw = False

    def __init__(self):
        self.failed = False
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        if not self.failed:
            self.failed = True
            raise MemoryError("simulated allocation failure")
        return context


class _CaptureLengthExporter(Processor):
    """Record exported core lengths without writing a real EEG file."""

    name = "capture_length_exporter"
    requires_raw = True
    modifies_raw = False

    def __init__(self, path: str, lengths: list[int]):
        self.path = path
        self.lengths = lengths
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        self.lengths.append(context.get_raw().n_times)
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


def _scanner_run_raw(n_triggers: int = 96) -> mne.io.RawArray:
    """Return one synthetic MRI run with regularly spaced scanner triggers."""
    sfreq = 100.0
    trigger_samples = 100 + np.arange(n_triggers) * 10
    n_times = int(trigger_samples[-1] + 101)
    data = np.zeros((2, n_times))
    data[1, trigger_samples] = 1
    info = mne.create_info(["EEG001", "R128"], sfreq, ["eeg", "stim"])
    return mne.io.RawArray(data, info, verbose=False)


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


def test_trigger_sections_are_split_at_safe_memory_boundaries(tmp_path, chunk_raw_factory):
    """Oversized trigger sections should remain contiguous and keep context."""
    raw = chunk_raw_factory(n_times=500, sfreq=10.0)
    triggers = np.arange(50, 250, 5)

    chunks = Pipeline._make_trigger_section_chunks(
        raw=raw,
        triggers=triggers,
        output_dir=tmp_path,
        output_stem="recording",
        extension=".edf",
        padding_seconds=1.0,
        min_triggers=8,
        gap_seconds=5.0,
        max_sections=None,
        max_chunk_samples=120,
        min_chunk_triggers=8,
    )

    assert len(chunks) == 2
    assert chunks[0].start_sample == 40
    assert chunks[-1].stop_sample == 256
    assert chunks[0].stop_sample > chunks[1].start_sample
    assert chunks[0].resolved_core_stop_sample == chunks[1].resolved_core_start_sample
    assert chunks[0].resolved_core_stop_sample in triggers
    assert all(chunk.stop_sample - chunk.start_sample <= 120 for chunk in chunks)

    trigger_counts = [
        int(np.count_nonzero((triggers >= chunk.start_sample) & (triggers < chunk.stop_sample))) for chunk in chunks
    ]
    assert trigger_counts == [22, 20]
    assert all(count >= 8 for count in trigger_counts)


def test_run_chunked_applies_budget_to_trigger_sections(monkeypatch, tmp_path):
    """The public chunk runner should enforce its budget in trigger mode."""
    read_preload_values = []

    def fake_read_raw_edf(path, *args, **kwargs):
        read_preload_values.append(kwargs.get("preload"))
        sfreq = 10.0
        n_times = 500
        data = np.zeros((2, n_times))
        triggers = np.arange(50, 250, 5)
        data[1, triggers] = 1
        info = mne.create_info(
            ["EEG001", "TREV"],
            sfreq=sfreq,
            ch_types=["eeg", "stim"],
        )
        return mne.io.RawArray(data, info, verbose=False)

    monkeypatch.setitem(_EXTENSION_READERS, ".edf", (fake_read_raw_edf, "EDF"))
    input_path = tmp_path / "recording.edf"
    input_path.write_text("placeholder", encoding="utf-8")

    # Two float64 channels and the default 4x processing multiplier consume
    # 64 estimated bytes per input sample. This budget therefore permits at
    # most 120 samples in one processing window.
    memory_budget_mb = (120 * 2 * 8 * 4) / 1024**2
    result = Pipeline(
        [TriggerDetector(regex=r"\b1\b")],
        name="Budgeted trigger section test",
    ).run_chunked(
        input_path=str(input_path),
        output_dir=str(tmp_path / "chunks"),
        output_extension=".edf",
        exporter_factory=lambda path: _TouchExporter(path),
        memory_budget_mb=memory_budget_mb,
        trigger_section_padding_seconds=1.0,
        trigger_section_min_triggers=8,
        trigger_section_gap_seconds=5.0,
    )

    assert result.was_successful()
    assert len(result) == 2
    assert all(chunk.stop_sample - chunk.start_sample <= 120 for chunk in result.chunks)
    assert read_preload_values == [False, False, False]


def test_trigger_section_memory_failure_retries_channel_sequentially(
    monkeypatch,
    tmp_path,
):
    """A real allocation failure should prefer channel-sequential fallback."""

    def fake_read_raw_edf(path, *args, **kwargs):
        sfreq = 10.0
        n_times = 500
        data = np.zeros((2, n_times))
        data[1, np.arange(50, 250, 5)] = 1
        info = mne.create_info(
            ["EEG001", "TREV"],
            sfreq=sfreq,
            ch_types=["eeg", "stim"],
        )
        return mne.io.RawArray(data, info, verbose=False)

    monkeypatch.setitem(_EXTENSION_READERS, ".edf", (fake_read_raw_edf, "EDF"))
    input_path = tmp_path / "recording.edf"
    input_path.write_text("placeholder", encoding="utf-8")

    # The first estimate keeps the section whole. The simulated MemoryError
    # retries the same scientifically valid window channel-sequentially.
    memory_budget_mb = (300 * 2 * 8 * 4) / 1024**2
    result = Pipeline(
        [
            TriggerDetector(regex=r"\b1\b"),
            _FailOnceWithMemoryError(),
        ],
        name="Retry trigger section test",
    ).run_chunked(
        input_path=str(input_path),
        output_dir=str(tmp_path / "chunks"),
        output_extension=".edf",
        exporter_factory=lambda path: _TouchExporter(path),
        memory_budget_mb=memory_budget_mb,
        trigger_section_padding_seconds=1.0,
        trigger_section_min_triggers=8,
        trigger_section_gap_seconds=5.0,
    )

    assert result.was_successful()
    assert len(result) == 1


def test_trigger_section_split_rejects_budget_without_required_context(
    tmp_path,
    chunk_raw_factory,
):
    """The RAM limit must not create correction chunks with too few triggers."""
    raw = chunk_raw_factory(n_times=500, sfreq=10.0)
    triggers = np.arange(50, 250, 5)

    with pytest.raises(PipelineError, match="at least 8 triggers must remain together"):
        Pipeline._make_trigger_section_chunks(
            raw=raw,
            triggers=triggers,
            output_dir=tmp_path,
            output_stem="recording",
            extension=".edf",
            padding_seconds=1.0,
            min_triggers=8,
            gap_seconds=5.0,
            max_sections=None,
            max_chunk_samples=30,
            min_chunk_triggers=8,
        )


def test_adaptive_enlargement_uses_minimum_valid_31_trigger_chunks(
    monkeypatch,
    tmp_path,
):
    """An automatic estimate below 31 triggers should enlarge safely."""

    def fake_read_raw_edf(path, *args, **kwargs):
        return _scanner_run_raw()

    monkeypatch.setitem(_EXTENSION_READERS, ".edf", (fake_read_raw_edf, "EDF"))
    monkeypatch.setattr(Pipeline, "_available_memory_bytes", staticmethod(lambda os_module: 100_000))
    input_path = tmp_path / "natview.edf"
    input_path.touch()

    result = Pipeline([TriggerDetector(regex=r"\b1\b")]).run_chunked(
        input_path=str(input_path),
        output_dir=str(tmp_path / "chunks"),
        exporter_factory=lambda path: _TouchExporter(path),
        memory_fraction=0.1,
        trigger_section_padding_seconds=0.1,
        trigger_section_min_triggers=31,
        channel_sequential=False,
    )

    assert result.was_successful()
    assert 2 <= len(result.chunks) < 10
    assert all(
        np.count_nonzero(
            (100 + np.arange(96) * 10 >= chunk.start_sample)
            & (100 + np.arange(96) * 10 < chunk.stop_sample)
        )
        >= 31
        for chunk in result.chunks
    )


def test_trigger_overlap_exports_exact_non_overlapping_core_samples(
    monkeypatch,
    tmp_path,
):
    """Overlap is processing-only; exported core ranges have no gaps or duplicates."""

    def fake_read_raw_edf(path, *args, **kwargs):
        return _scanner_run_raw(n_triggers=40)

    monkeypatch.setitem(_EXTENSION_READERS, ".edf", (fake_read_raw_edf, "EDF"))
    input_path = tmp_path / "run.edf"
    input_path.touch()
    exported_lengths: list[int] = []

    result = Pipeline([TriggerDetector(regex=r"\b1\b")]).run_chunked(
        input_path=str(input_path),
        output_dir=str(tmp_path / "chunks"),
        exporter_factory=lambda path: _CaptureLengthExporter(path, exported_lengths),
        memory_budget_mb=(140 * 2 * 8 * 4) / 1024**2,
        trigger_section_padding_seconds=0.1,
        trigger_section_min_triggers=8,
    )

    cores = [
        (chunk.resolved_core_start_sample, chunk.resolved_core_stop_sample)
        for chunk in result.chunks
    ]
    assert all(left[1] == right[0] for left, right in zip(cores, cores[1:], strict=False))
    assert sum(stop - start for start, stop in cores) == cores[-1][1] - cores[0][0]
    assert exported_lengths == [stop - start for start, stop in cores]
    assert any(chunk.left_overlap_samples or chunk.right_overlap_samples for chunk in result.chunks)


def test_minimum_trigger_chunk_reports_genuine_ram_failure(monkeypatch, tmp_path):
    """A minimum valid context that cannot fit real RAM should fail clearly."""

    def fake_read_raw_edf(path, *args, **kwargs):
        return _scanner_run_raw()

    monkeypatch.setitem(_EXTENSION_READERS, ".edf", (fake_read_raw_edf, "EDF"))
    monkeypatch.setattr(Pipeline, "_available_memory_bytes", staticmethod(lambda os_module: 1_000))
    input_path = tmp_path / "natview.edf"
    input_path.touch()

    with pytest.raises(PipelineError, match="minimum scientifically valid Flex chunk cannot fit"):
        Pipeline([TriggerDetector(regex=r"\b1\b")]).run_chunked(
            input_path=str(input_path),
            output_dir=str(tmp_path / "chunks"),
            exporter_factory=lambda path: _TouchExporter(path),
            memory_fraction=0.5,
            trigger_section_padding_seconds=0.1,
            trigger_section_min_triggers=31,
        )


def test_disable_chunking_processes_checked_full_run(monkeypatch, tmp_path, chunk_raw_factory):
    """The controlled full-run mode should emit one complete core."""

    def fake_read_raw_edf(path, *args, **kwargs):
        return chunk_raw_factory(n_times=120)

    monkeypatch.setitem(_EXTENSION_READERS, ".edf", (fake_read_raw_edf, "EDF"))
    monkeypatch.setattr(Pipeline, "_available_memory_bytes", staticmethod(lambda os_module: 10_000_000))
    input_path = tmp_path / "recording.edf"
    input_path.touch()
    exported_lengths: list[int] = []

    result = Pipeline([]).run_chunked(
        input_path=str(input_path),
        output_dir=str(tmp_path / "chunks"),
        exporter_factory=lambda path: _CaptureLengthExporter(path, exported_lengths),
        disable_chunking=True,
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].resolved_core_start_sample == 0
    assert result.chunks[0].resolved_core_stop_sample == 120
    assert exported_lengths == [120]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["chunking_mode"] == "full_run"


def test_disable_chunking_refuses_unsafe_full_run(monkeypatch, tmp_path, chunk_raw_factory):
    """Full-run mode must check memory unless force_full_run is explicit."""

    def fake_read_raw_edf(path, *args, **kwargs):
        return chunk_raw_factory(n_times=120)

    monkeypatch.setitem(_EXTENSION_READERS, ".edf", (fake_read_raw_edf, "EDF"))
    monkeypatch.setattr(Pipeline, "_available_memory_bytes", staticmethod(lambda os_module: 100))
    input_path = tmp_path / "recording.edf"
    input_path.touch()

    with pytest.raises(PipelineError, match="Full-run processing is not memory-safe"):
        Pipeline([]).run_chunked(
            input_path=str(input_path),
            output_dir=str(tmp_path / "chunks"),
            exporter_factory=lambda path: _TouchExporter(path),
            disable_chunking=True,
        )


def test_minimum_trigger_context_does_not_mutate_flex_parameters():
    """Chunk planning may inspect Flex, but must not alter its semantics."""
    from facet.correction.flex import FlexCorrection

    flex = FlexCorrection(window_size=30, min_accepted=5, threshold=0.9)
    before = (flex.window_size, flex.min_accepted, flex.threshold)

    required = Pipeline([flex])._minimum_trigger_context(16)

    assert required == 31
    assert (flex.window_size, flex.min_accepted, flex.threshold) == before
