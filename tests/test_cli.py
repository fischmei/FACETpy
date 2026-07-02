"""Tests for FACETpy command line helpers."""

import argparse
import json
from types import SimpleNamespace

import pytest

from facet import cli
from facet.core import Pipeline, ProcessorValidationError

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
        calls.append(
            {
                "kwargs": kwargs,
                "processor_names": [processor.name for processor in self.processors],
            }
        )
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
    assert [call["kwargs"]["input_path"] for call in calls] == [str(first.resolve()), str(second.resolve())]
    assert [call["kwargs"]["output_dir"] for call in calls] == [
        str((tmp_path / "corrected" / "sub01").resolve()),
        str((tmp_path / "corrected" / "sub02").resolve()),
    ]
    assert all(call["kwargs"]["chunk_by_trigger_sections"] for call in calls)
    assert all("drop_channels_matching" not in call["processor_names"] for call in calls)


def test_process_command_can_opt_in_to_egi_channel_drop(monkeypatch, tmp_path):
    """EGI channel dropping should remain available when explicitly requested."""
    source = tmp_path / "sub-01.mff"
    source.mkdir()

    processor_names = []

    def fake_run_chunked(self, **kwargs):
        processor_names.append([processor.name for processor in self.processors])
        return _DummyChunkedResult()

    monkeypatch.setattr(Pipeline, "run_chunked", fake_run_chunked)

    status = cli.main(
        [
            "process",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "corrected"),
            "--drop-egi-e-channels",
        ]
    )

    assert status == 0
    assert processor_names == [
        [
            "drop_channels_matching",
            "trigger_detector",
            "upsample",
            "aas_correction",
            "downsample",
        ]
    ]


def test_process_command_builds_selected_correction_modes(monkeypatch, tmp_path):
    """Correction and add-on modes should be ordered like the full examples."""
    source = tmp_path / "sub-01.mff"
    source.mkdir()

    captured_processors = []

    def fake_run_chunked(self, **kwargs):
        captured_processors.append(self.processors)
        return _DummyChunkedResult()

    monkeypatch.setattr(Pipeline, "run_chunked", fake_run_chunked)

    status = cli.main(
        [
            "process",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "corrected"),
            "--correction-mode",
            "farm",
            "--farm-correlation-threshold",
            "0.91",
            "--mode",
            "volume-artifact",
            "--mode",
            "pca",
            "--pca-components",
            "auto",
            "--pca-hp-freq",
            "0.5",
            "--mode",
            "anc",
            "--no-anc-c-extension",
        ]
    )

    assert status == 0
    processors = captured_processors[0]
    assert [processor.name for processor in processors] == [
        "trigger_detector",
        "upsample",
        "volume_artifact_correction",
        "farm_correction",
        "pca_correction",
        "downsample",
        "anc_correction",
    ]
    assert processors[3].correlation_threshold == 0.91
    assert processors[4].n_components == "auto"
    assert processors[4].hp_freq == 0.5
    assert processors[6].use_c_extension is False


def test_process_command_writes_pipeline_and_matrix_reports(monkeypatch, tmp_path):
    """Every processed input should receive pipeline and template-matrix JSON."""
    source = tmp_path / "sub-01.mff"
    source.mkdir()

    chunk_output = tmp_path / "corrected" / "sub-01_chunk_001_of_001.edf"
    matrix_report = {
        "processor_name": "aas_correction",
        "matrix_equation": {"equation": "N = A @ D"},
        "channels": [
            {
                "channel_name": "Cz",
                "averaging_matrix_A": {
                    "shape": [2, 2],
                    "storage": "dense",
                    "matrix": [[0.5, 0.5], [0.5, 0.5]],
                },
            }
        ],
    }

    class FakePipelineResult:
        success = True
        error = None
        context = SimpleNamespace(metadata=SimpleNamespace(custom={"artifact_template_matrices": [matrix_report]}))

    class FakeChunkedResult:
        chunks = [
            SimpleNamespace(
                index=1,
                total=1,
                start_sample=0,
                stop_sample=100,
                duration_seconds=1.0,
                output_path=chunk_output,
            )
        ]
        source_path = str(source)
        output_dir = str(tmp_path / "corrected")
        execution_time = 1.25
        manifest_path = tmp_path / "corrected" / "chunks_manifest.json"

        def __iter__(self):
            return iter([FakePipelineResult()])

        def print_summary(self) -> None:
            return None

        def was_successful(self) -> bool:
            return True

    monkeypatch.setattr(Pipeline, "run_chunked", lambda self, **kwargs: FakeChunkedResult())

    output_dir = tmp_path / "corrected"
    status = cli.main(
        [
            "process",
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    pipeline_payload = json.loads((output_dir / "pipeline_description.json").read_text(encoding="utf-8"))
    matrix_payload = json.loads((output_dir / "artifact_template_matrices.json").read_text(encoding="utf-8"))

    assert pipeline_payload["correction_mode"] == "aas"
    assert pipeline_payload["result"]["artifact_template_matrix_report"].endswith("artifact_template_matrices.json")
    assert matrix_payload["description"].startswith("AAS-style corrections build artifact templates")
    assert matrix_payload["reports"][0]["processor_name"] == "aas_correction"
    assert matrix_payload["reports"][0]["chunk"]["index"] == 1


def test_process_command_builds_standard_pattern(monkeypatch, tmp_path):
    """The standard pattern should mirror the docs full correction flow."""
    source = tmp_path / "sub-01.mff"
    source.mkdir()

    captured_processors = []

    def fake_run_chunked(self, **kwargs):
        captured_processors.append(self.processors)
        return _DummyChunkedResult()

    monkeypatch.setattr(Pipeline, "run_chunked", fake_run_chunked)

    status = cli.main(
        [
            "process",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "corrected"),
            "--pattern",
            "standard",
            "--no-anc-c-extension",
        ]
    )

    assert status == 0
    assert [processor.name for processor in captured_processors[0]] == [
        "trigger_detector",
        "cut_acquisition_window",
        "highpass_filter",
        "upsample",
        "slice_aligner",
        "subsample_aligner",
        "aas_correction",
        "pca_correction",
        "downsample",
        "paste_acquisition_window",
        "lowpass_filter",
        "anc_correction",
    ]


def test_process_command_builds_bcg_pattern_with_fixed_chunks(monkeypatch, tmp_path):
    """BCG uses QRS triggers and fixed chunks because trigger probing is regex-based."""
    source = tmp_path / "sub-01.mff"
    source.mkdir()

    calls = []

    def fake_run_chunked(self, **kwargs):
        calls.append(
            {
                "kwargs": kwargs,
                "processor_names": [processor.name for processor in self.processors],
            }
        )
        return _DummyChunkedResult()

    monkeypatch.setattr(Pipeline, "run_chunked", fake_run_chunked)

    status = cli.main(
        [
            "process",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "corrected"),
            "--pattern",
            "bcg",
        ]
    )

    assert status == 0
    assert calls[0]["kwargs"]["chunk_by_trigger_sections"] is False
    assert calls[0]["processor_names"] == ["qrs_trigger_detector", "aas_correction"]


def test_process_command_moosmann_requires_motion_file(tmp_path):
    """Motion-informed mode should fail early without realignment parameters."""
    source = tmp_path / "sub-01.mff"
    source.mkdir()

    with pytest.raises(ValueError, match="--motion-rp-file"):
        cli.main(
            [
                "process",
                "--input",
                str(source),
                "--output-dir",
                str(tmp_path / "corrected"),
                "--correction-mode",
                "moosmann",
            ]
        )


def test_modes_command_lists_cli_modes(capsys):
    """The discovery command should explain correction and add-on modes."""
    status = cli.main(["modes"])

    captured = capsys.readouterr()
    assert status == 0
    assert "Correction modes replace the baseline AAS" in captured.out
    assert "farm:" in captured.out
    assert "pca:" in captured.out
    assert "anc:" in captured.out


def test_patterns_command_lists_cli_patterns(capsys):
    """The pattern discovery command should keep modes and patterns separate."""
    status = cli.main(["patterns"])

    captured = capsys.readouterr()
    assert status == 0
    assert "Pipeline patterns" in captured.out
    assert "quickstart:" in captured.out
    assert "standard:" in captured.out
    assert "batch:" in captured.out


def test_default_command_preserves_root_help():
    """Top-level help should expose discovery commands instead of process help."""
    assert cli._with_default_command(["--help"]) == ["--help"]
    assert cli._with_default_command(["-h"]) == ["-h"]
    assert cli._with_default_command(["--input", "recording.edf"]) == ["process", "--input", "recording.edf"]


def test_viewer_command_uses_raw_plotter(monkeypatch, tmp_path):
    """Viewer CLI should delegate to FACETpy RawPlotter."""
    context = object()
    loader_calls = []
    plotter_calls = []

    class DummyLoader:
        def __init__(self, **kwargs):
            loader_calls.append(kwargs)

        def execute(self, context_arg):
            assert context_arg is None
            return context

    class DummyPlotter:
        def __init__(self, **kwargs):
            plotter_calls.append(kwargs)

        def execute(self, context_arg):
            assert context_arg is context
            return context_arg

    monkeypatch.setattr(cli, "Loader", DummyLoader)
    monkeypatch.setattr(cli, "RawPlotter", DummyPlotter)

    output_path = tmp_path / "plot.png"
    status = cli.main(
        [
            "viewer",
            "--input",
            str(tmp_path / "recording.edf"),
            "--output",
            str(output_path),
            "--channel",
            "Cz",
            "--show",
            "--n-channels",
            "30",
            "--scalings",
            "auto",
        ]
    )

    assert status == 0
    assert loader_calls[0]["preload"] is True
    assert plotter_calls[0]["channel"] == "Cz"
    assert plotter_calls[0]["save_path"] == str(output_path)
    assert plotter_calls[0]["show"] is True
    assert plotter_calls[0]["mne_kwargs"] == {"n_channels": 30, "scalings": "auto"}


def test_analysis_command_writes_report_json(monkeypatch, tmp_path):
    """Analysis CLI should delegate to report processors and write metadata."""
    context = SimpleNamespace(metadata=SimpleNamespace(custom={}))

    class DummyLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def execute(self, context_arg):
            assert context_arg is None
            return context

    class DummyProcessor:
        def __init__(self, name, **kwargs):
            self.name = name
            self.kwargs = kwargs

        def execute(self, context_arg):
            context_arg.metadata.custom[self.name] = {"kwargs": self.kwargs}
            return context_arg

    monkeypatch.setattr(cli, "Loader", DummyLoader)
    monkeypatch.setattr(cli, "AnalyzeDataReport", lambda: DummyProcessor("analyze_data_report"))
    monkeypatch.setattr(
        cli,
        "CheckDataReport",
        lambda **kwargs: DummyProcessor("check_data_report", **kwargs),
    )

    report_path = tmp_path / "analysis.json"
    status = cli.main(
        [
            "analysis",
            "--input",
            str(tmp_path / "recording.edf"),
            "--require-triggers",
            "--strict",
            "--output-json",
            str(report_path),
        ]
    )

    assert status == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "analyze_data_report" in payload
    assert payload["check_data_report"]["kwargs"] == {
        "require_triggers": True,
        "strict": True,
    }


def test_analysis_command_lists_metrics_without_input(capsys):
    """Metric listing should not require loading a recording."""
    status = cli.main(["analysis", "--list-metrics"])

    output = capsys.readouterr().out
    assert status == 0
    assert "Available analysis metrics:" in output
    assert "legacy-snr:" in output
    assert "fft-niazy:" in output


def test_analysis_command_selects_metrics_and_skips_inapplicable(monkeypatch, tmp_path):
    """Selected metrics should continue when one metric cannot apply."""
    context = SimpleNamespace(metadata=SimpleNamespace(custom={}))
    calls = []

    class DummyLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def execute(self, context_arg):
            assert context_arg is None
            return context

    class NoOpProcessor:
        def execute(self, context_arg):
            return context_arg

    class DummyMetric:
        def __init__(self, name, failure=None):
            self.name = name
            self.failure = failure

        def execute(self, context_arg):
            calls.append(self.name)
            if self.failure is not None:
                raise ProcessorValidationError(self.failure)
            context_arg.metadata.custom.setdefault("metrics", {})[self.name] = True
            return context_arg

    monkeypatch.setattr(cli, "Loader", DummyLoader)
    monkeypatch.setattr(cli, "AnalyzeDataReport", NoOpProcessor)
    monkeypatch.setattr(cli, "CheckDataReport", lambda **kwargs: NoOpProcessor())
    monkeypatch.setattr(cli, "RMSCalculator", lambda: DummyMetric("rms"))
    monkeypatch.setattr(
        cli,
        "LegacySNRCalculator",
        lambda: DummyMetric("legacy-snr", "Original raw data not available."),
    )
    monkeypatch.setattr(cli, "MetricsReport", lambda: DummyMetric("report"))

    report_path = tmp_path / "analysis.json"
    status = cli.main(
        [
            "analysis",
            "--input",
            str(tmp_path / "recording.edf"),
            "--metric",
            "rms",
            "--metric",
            "legacy-snr",
            "--metric",
            "report",
            "--output-json",
            str(report_path),
        ]
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert status == 0
    assert calls == ["rms", "legacy-snr", "report"]
    assert payload["metrics"] == {"rms": True, "report": True}
    assert payload["skipped_metrics"] == ["legacy-snr"]
    assert payload["skipped_metric_reasons"]["legacy-snr"] == "Original raw data not available."


def test_analysis_command_can_fail_on_inapplicable_metric(monkeypatch, tmp_path):
    """Strict metric mode should preserve the original validation failure."""
    context = SimpleNamespace(metadata=SimpleNamespace(custom={}))

    class DummyLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def execute(self, context_arg):
            assert context_arg is None
            return context

    class NoOpProcessor:
        def execute(self, context_arg):
            return context_arg

    class InvalidMetric:
        def execute(self, context_arg):
            raise ProcessorValidationError("Original raw data not available.")

    monkeypatch.setattr(cli, "Loader", DummyLoader)
    monkeypatch.setattr(cli, "AnalyzeDataReport", NoOpProcessor)
    monkeypatch.setattr(cli, "CheckDataReport", lambda **kwargs: NoOpProcessor())
    monkeypatch.setattr(cli, "LegacySNRCalculator", InvalidMetric)

    with pytest.raises(ProcessorValidationError, match="Original raw data not available"):
        cli.main(
            [
                "analysis",
                "--input",
                str(tmp_path / "recording.edf"),
                "--metric",
                "legacy-snr",
                "--no-skip-inapplicable-metrics",
            ]
        )


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
