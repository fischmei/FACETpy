"""Tests for FACETpy command line helpers."""

import argparse
import json

import pytest

from facet import cli
from facet.core import Pipeline

pytestmark = pytest.mark.unit


class _DummyChunkedResult:
    """Small stand-in for a successful ChunkedPipelineResult."""

    def __init__(self):
        self.printed = False

    def print_summary(self) -> None:
        self.printed = True

    def was_successful(self) -> bool:
        return True


def test_resolve_inputs_reads_lists_and_scans_mff_dirs(tmp_path):
    """Inputs can come from a text file and from an input directory."""
    listed = tmp_path / "listed.mff"
    listed.mkdir()
    scanned = tmp_path / "inputs" / "scanned.mff"
    scanned.mkdir(parents=True)
    nested = tmp_path / "inputs" / "nested" / "nested.edf"
    nested.parent.mkdir()
    nested.write_text("placeholder", encoding="utf-8")

    input_list = tmp_path / "inputs.txt"
    input_list.write_text(f"# comment\n{listed}\n", encoding="utf-8")

    result = cli._resolve_inputs(
        inputs=None,
        input_list=str(input_list),
        input_dir=str(tmp_path / "inputs"),
        extensions=[".mff", ".edf"],
        recursive=True,
    )

    assert result == [listed.resolve(), nested.resolve(), scanned.resolve()]


def test_process_command_batches_into_recording_folders(monkeypatch, tmp_path):
    """Batch processing should call run_chunked once per input."""
    first = tmp_path / "sub-01.mff"
    second = tmp_path / "sub-02.mff"
    first.mkdir()
    second.mkdir()
    input_list = tmp_path / "inputs.txt"
    input_list.write_text(f"{first}\n{second}\n", encoding="utf-8")

    calls = []

    def fake_run_chunked(self, **kwargs):
        calls.append(kwargs)
        return _DummyChunkedResult()

    monkeypatch.setattr(Pipeline, "run_chunked", fake_run_chunked)

    status = cli.main(
        [
            "process",
            "--input-list",
            str(input_list),
            "--output-dir",
            str(tmp_path / "corrected"),
            "--overwrite",
        ]
    )

    assert status == 0
    assert [call["input_path"] for call in calls] == [str(first.resolve()), str(second.resolve())]
    assert [call["output_dir"] for call in calls] == [
        str((tmp_path / "corrected" / "sub01").resolve()),
        str((tmp_path / "corrected" / "sub02").resolve()),
    ]
    assert all(call["chunk_by_trigger_sections"] for call in calls)


def test_process_command_continues_after_failed_input(monkeypatch, tmp_path):
    """--on-error=continue should skip failed recordings and process the next."""
    first = tmp_path / "sub-01.mff"
    second = tmp_path / "sub-02.mff"
    first.mkdir()
    second.mkdir()
    input_list = tmp_path / "inputs.txt"
    input_list.write_text(f"{first}\n{second}\n", encoding="utf-8")

    calls = []

    def fake_run_chunked(self, **kwargs):
        calls.append(kwargs)
        if kwargs["input_path"] == str(first.resolve()):
            raise RuntimeError("Cannot build trigger-section chunks: no triggers were detected")
        return _DummyChunkedResult()

    monkeypatch.setattr(Pipeline, "run_chunked", fake_run_chunked)

    output_dir = tmp_path / "corrected"
    status = cli.main(
        [
            "process",
            "--input-list",
            str(input_list),
            "--output-dir",
            str(output_dir),
            "--on-error",
            "continue",
        ]
    )

    assert status == 1
    assert [call["input_path"] for call in calls] == [str(first.resolve()), str(second.resolve())]

    per_recording = json.loads((output_dir / "sub01" / "processing_error.json").read_text(encoding="utf-8"))
    assert per_recording["input_path"] == str(first.resolve())
    assert "no triggers" in per_recording["error"]

    batch = json.loads((output_dir / "processing_failures.json").read_text(encoding="utf-8"))
    assert len(batch["failures"]) == 1
    assert batch["failures"][0]["input_path"] == str(first.resolve())


def test_bids_plan_derives_subject_and_run_from_chunk_names(tmp_path):
    """Chunked outputs should map to one subject with numbered BIDS runs."""
    chunk_1 = tmp_path / "my_recording_chunk_001_of_002.edf"
    chunk_2 = tmp_path / "my_recording_chunk_002_of_002.edf"
    chunk_1.write_text("placeholder", encoding="utf-8")
    chunk_2.write_text("placeholder", encoding="utf-8")

    args = argparse.Namespace(
        input=[str(chunk_1), str(chunk_2)],
        input_list=None,
        input_dir=None,
        extensions=[".edf"],
        recursive=False,
        subject=None,
    )

    plan = cli._build_bids_export_plan(args)

    assert plan == [
        (chunk_1.resolve(), "myrecording", "1"),
        (chunk_2.resolve(), "myrecording", "2"),
    ]
