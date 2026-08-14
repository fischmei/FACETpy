"""Tests for the self-contained FACETpy cleaning report."""

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pytest
from PIL import Image

import facet._cli_html_report as html_report
from facet._cli_html_report import (
    MAX_MATRIX_PREVIEW_BYTES,
    MAX_MATRIX_PREVIEW_SIZE,
    PairedRecordingData,
    _analysis_quota_seconds,
    _artifact_active_output_bounds,
    _bounded_matrix_preview,
    _coherence_matrix,
    _compute_coherence_diagnostics,
    _compute_temporal_diagnostics,
    _laplacian_eigendecomposition,
    _physical_sensor_graph,
    _resolve_spatial_geometry,
    _spectral_modularity_labels,
    _threshold_by_density,
    _write_base64_stream,
    _write_matrix_assets,
    cleaning_report_path,
    write_cleaning_report,
)

pytestmark = pytest.mark.unit


def _paired_sine_data(*, duration: float = 8.0) -> PairedRecordingData:
    """Return a small, positioned before/after recording with a 10 Hz peak."""
    sfreq = 100.0
    channel_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
    info = mne.create_info(channel_names, sfreq, "eeg")
    info.set_montage("standard_1020")
    times = np.arange(int(duration * sfreq)) / sfreq
    rng = np.random.default_rng(17)
    before = np.stack(
        [
            10e-6 * np.sin(2.0 * np.pi * 10.0 * times + index / 6.0) + 1e-6 * rng.normal(size=len(times))
            for index in range(len(channel_names))
        ]
    )
    after = 0.6 * before
    return PairedRecordingData(
        before_segments=[before.astype(np.float32)],
        after_segments=[after.astype(np.float32)],
        channel_names=channel_names,
        sfreq=sfreq,
        source_info=info,
        source_duration_seconds=duration,
        analyzed_duration_seconds=duration,
        segment_windows=[
            {
                "chunk_index": 1,
                "start_seconds": 0.0,
                "stop_seconds": duration,
                "samples": before.shape[1],
            }
        ],
    )


def test_temporal_diagnostics_find_known_peak_on_shared_grid():
    """Amplitude and Welch spectra should recover a known 10 Hz component."""
    diagnostics = _compute_temporal_diagnostics(_paired_sine_data())
    median_amplitude = np.median(diagnostics.amplitude_before, axis=0)
    median_psd = np.median(diagnostics.psd_before, axis=0)

    assert diagnostics.frequencies[np.argmax(median_amplitude)] == pytest.approx(10.0, abs=0.26)
    assert diagnostics.frequencies[np.argmax(median_psd)] == pytest.approx(10.0, abs=0.26)
    assert diagnostics.amplitude_before.shape == diagnostics.amplitude_after.shape
    assert diagnostics.psd_before.shape == diagnostics.psd_after.shape
    assert np.all(diagnostics.psd_before >= 0)
    assert np.allclose(diagnostics.histogram_edges, np.sort(diagnostics.histogram_edges))


def test_analysis_quota_bounds_native_rate_reads_and_retained_values():
    """High native sampling rates must not bypass the report-memory cap."""
    channel_count = 256
    pair_count = 2
    source_sfreq = 5_000.0
    output_sfreq = 10_000.0
    report_sfreq = 1_000.0

    quota_seconds = _analysis_quota_seconds(
        channel_count=channel_count,
        pair_count=pair_count,
        report_sfreq=report_sfreq,
        source_sfreq=source_sfreq,
        output_sfreq=output_sfreq,
    )

    for sfreq in (source_sfreq, output_sfreq, report_sfreq):
        values = quota_seconds * channel_count * sfreq * pair_count
        assert values <= html_report.MAX_ANALYSIS_VALUES_PER_PHASE

    assert _analysis_quota_seconds(
        channel_count=8,
        pair_count=2,
        report_sfreq=100.0,
        source_sfreq=100.0,
        output_sfreq=100.0,
    ) == pytest.approx(html_report.MAX_ANALYSIS_SECONDS / 2)


def test_trigger_active_bounds_are_shifted_from_overlap_to_exported_core():
    """Retained triggers should focus report windows on corrected core time."""
    metadata = SimpleNamespace(
        triggers=np.array([100, 200]),
        artifact_length=50,
        artifact_to_trigger_offset=0.0,
    )
    result = SimpleNamespace(context=SimpleNamespace(metadata=metadata))
    chunk = SimpleNamespace(left_overlap_samples=50)

    assert _artifact_active_output_bounds(
        result,
        chunk,
        source_sfreq=100.0,
        output_sfreq=100.0,
        output_n_times=400,
    ) == (50, 200)


def test_laplacian_basis_is_ordered_orthonormal_and_energy_preserving():
    """The graph basis should be a valid orthonormal eigendecomposition."""
    data = _paired_sine_data()
    geometry = _resolve_spatial_geometry(data)
    assert geometry is not None
    adjacency = _physical_sensor_graph(geometry.coordinates_3d)
    eigenvalues, eigenvectors = _laplacian_eigendecomposition(adjacency)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency

    assert np.all(np.diff(eigenvalues) >= -1e-12)
    np.testing.assert_allclose(eigenvectors.T @ eigenvectors, np.eye(len(eigenvalues)), atol=1e-12)
    np.testing.assert_allclose(laplacian, eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T, atol=1e-12)

    sample = data.before_segments[0][:, :100]
    coefficients = eigenvectors.T @ sample
    assert np.sum(coefficients**2) == pytest.approx(np.sum(sample**2), rel=1e-10)


def test_coherence_is_symmetric_bounded_and_community_labels_deterministic():
    """Coherence and spectral community outputs should obey graph invariants."""
    data = _paired_sine_data()
    indices = np.arange(len(data.channel_names))
    _, coherence = _coherence_matrix(
        data.after_segments,
        indices,
        data.sfreq,
        nperseg=200,
        fmin=8.0,
        fmax=12.0,
    )
    adjacency = coherence.copy()
    np.fill_diagonal(adjacency, 0.0)

    np.testing.assert_allclose(coherence, coherence.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(coherence), 1.0)
    assert np.all((coherence >= 0.0) & (coherence <= 1.0))
    np.testing.assert_array_equal(
        _spectral_modularity_labels(adjacency),
        _spectral_modularity_labels(adjacency),
    )


def test_coherence_is_invariant_to_volt_scale():
    """Scaling the same signal to EEG volts must not erase coherence."""
    rng = np.random.default_rng(23)
    values = rng.normal(size=(4, 2_000))
    indices = np.arange(values.shape[0])

    _, unit_scale = _coherence_matrix([values], indices, 100.0, 200, 8.0, 13.0)
    _, microvolt_scale = _coherence_matrix([values * 1e-6], indices, 100.0, 200, 8.0, 13.0)

    np.testing.assert_allclose(microvolt_scale, unit_scale, rtol=1e-12, atol=1e-12)
    assert np.any(microvolt_scale[np.triu_indices_from(microvolt_scale, k=1)] > 0.0)


def test_short_coherence_estimate_is_marked_low_precision():
    """Community summaries near the minimum window count need a warning flag."""
    diagnostics = _compute_coherence_diagnostics(_paired_sine_data(duration=10.0), geometry=None)

    assert diagnostics.frame_count == 9
    assert diagnostics.low_precision


def test_proportional_threshold_keeps_exact_edge_count_with_ties():
    """Tied coherence values must not inflate the documented graph density."""
    matrix = np.ones((6, 6), dtype=float)
    np.fill_diagonal(matrix, 1.0)

    thresholded, threshold = _threshold_by_density(matrix, 0.15)

    # Six nodes have 15 possible edges; ceil(15% * 15) keeps exactly three.
    assert np.count_nonzero(np.triu(thresholded, k=1)) == 3
    assert threshold == pytest.approx(1.0)
    np.testing.assert_allclose(thresholded, thresholded.T)


def test_missing_geometry_is_not_fabricated():
    """Generic channel labels should not be assigned arbitrary scalp positions."""
    sfreq = 100.0
    names = ["EEG001", "EEG002", "EEG003", "EEG004"]
    info = mne.create_info(names, sfreq, "eeg")
    values = np.zeros((len(names), 500), dtype=np.float32)
    data = PairedRecordingData(
        [values],
        [values],
        names,
        sfreq,
        info,
        5.0,
        5.0,
        [{"chunk_index": 1, "start_seconds": 0.0, "stop_seconds": 5.0, "samples": 500}],
    )

    assert _resolve_spatial_geometry(data) is None


def test_report_paths_are_unique_for_flat_batch(tmp_path):
    """Source-derived HTML names must not collide in a shared output folder."""
    first = cleaning_report_path(tmp_path, tmp_path / "sub-01.edf")
    second = cleaning_report_path(tmp_path, tmp_path / "sub-02.edf")

    assert first != second
    assert first.name == "sub-01_cleaning_report.html"
    assert second.name == "sub-02_cleaning_report.html"


def test_streamed_base64_round_trips_across_chunk_boundary():
    """Independent base64 blocks must concatenate into the original payload."""
    payload = bytes(range(256)) * 500
    encoded = io.StringIO()

    _write_base64_stream(io.BytesIO(payload), encoded)

    assert base64.b64decode(encoded.getvalue()) == payload


def test_matrix_previews_are_bounded_and_streamed_in_order(tmp_path):
    """Every trusted matrix plot should get one ordered bounded HTML preview."""
    records = []
    for stage, color in enumerate(((180, 20, 20), (20, 20, 180)), start=1):
        path = tmp_path / f"stage_{stage}.png"
        Image.new("RGB", (2_000, 1_400), color).save(path)
        records.append(
            {
                "path": str(path),
                "processor_name": f"flex_stage_{stage}",
                "channel_name": "Cz",
                "chunk_index": 1,
                "stage": stage,
            }
        )

    preview, _, preview_size = _bounded_matrix_preview(Path(records[0]["path"]))
    destination = io.StringIO()
    _write_matrix_assets(destination, records, tmp_path)
    html = destination.getvalue()

    assert len(preview) <= MAX_MATRIX_PREVIEW_BYTES
    assert preview_size[0] <= MAX_MATRIX_PREVIEW_SIZE[0]
    assert preview_size[1] <= MAX_MATRIX_PREVIEW_SIZE[1]
    assert html.count("data:image/jpeg;base64,") == 2
    assert html.index("flex_stage_1") < html.index("flex_stage_2")


def test_full_report_embeds_ordered_assets_and_pipeline(tmp_path):
    """A real paired FIF run should produce one offline before/during/after report."""
    data = _paired_sine_data(duration=10.0)
    source_path = tmp_path / "source_raw.fif"
    output_path = tmp_path / "source_chunk_001_of_001_raw.fif"
    source_raw = mne.io.RawArray(data.before_segments[0], data.source_info, verbose=False)
    clean_raw = mne.io.RawArray(data.after_segments[0], data.source_info, verbose=False)
    source_raw.save(source_path, overwrite=True, verbose=False)
    clean_raw.save(output_path, overwrite=True, verbose=False)

    # A tiny valid PNG stands in for the already-rendered Flex matrix figure;
    # report code verifies that embedded paths remain inside the output folder.
    matrix_png = tmp_path / "source_chunk_001_of_001.aas_matrices.png"
    matrix_png.write_bytes(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    )
    pipeline_path = tmp_path / "pipeline_description.json"
    matrix_path = tmp_path / "artifact_template_matrices.json"
    manifest_path = tmp_path / "chunks_manifest.json"
    pipeline_path.write_text(
        json.dumps(
            {
                "source_path": str(source_path),
                "pattern": "quickstart",
                "correction_mode": "flex",
                "flex_corrections": [
                    {
                        "processor_index": 1,
                        "processor_name": "flex_correction",
                        "preset": "flex_default",
                        "legacy_algorithm_resemblance": "default Flex",
                        "decisions": {
                            "quota": {"window_size": 10, "past": 0, "future": 10, "global_mode": False},
                            "sampling": {"mode": "consecutive", "stride": 1, "start_offset": 1},
                            "motion": {
                                "same_motion_segment": False,
                                "motion_stable_only": False,
                                "max_motion_distance": None,
                            },
                            "target_policy": "exclude_target",
                            "scoring": {"mode": "signed_pearson", "threshold": 0.975},
                            "template_size": {"mode": "minimum_k", "k": 5},
                            "weighting": {"kernel": "equal"},
                        },
                    }
                ],
                "processors": [
                    {
                        "index": 1,
                        "name": "flex_correction",
                        "type": "Flex",
                        "description": "Template subtraction",
                        "parameters": {"threshold": 0.975},
                    }
                ],
                "result": {
                    "execution_time_seconds": 1.25,
                    "chunks": [{"index": 1, "output_path": str(output_path), "success": True}],
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "chunking_mode": "fixed_length",
                "memory_budget_bytes": 123456,
                "chunks": [{"index": 1, "core_start_sample": 0, "core_stop_sample": int(clean_raw.n_times)}],
            }
        ),
        encoding="utf-8",
    )
    matrix_payload = {
        "artifact_template_matrices": [
            {
                "processor_name": "flex_correction",
                "num_triggers": 10,
                "channels": [{"channel_name": "Fp1"}],
                "chunk": {"index": 1},
            }
        ],
        "artifact_template_matrix_plots": [
            {
                "path": str(matrix_png),
                "processor_name": "flex_correction",
                "channel_name": "Fp1",
                "chunk_index": 1,
                "stage": 1,
            }
        ],
    }
    matrix_path.write_text(json.dumps(matrix_payload), encoding="utf-8")

    chunk = SimpleNamespace(
        index=1,
        total=1,
        start_sample=0,
        stop_sample=clean_raw.n_times,
        resolved_core_start_sample=0,
        resolved_core_stop_sample=clean_raw.n_times,
        output_path=output_path,
    )
    result = SimpleNamespace(
        success=True,
        context=SimpleNamespace(
            metadata=SimpleNamespace(
                custom=matrix_payload,
                triggers=None,
            )
        ),
    )

    # SimpleNamespace special methods are class-level, so use a tiny concrete
    # result container for the iterable interface expected by the CLI helper.
    class ChunkedResult:
        chunks = [chunk]

        def __iter__(self):
            return iter([result])

    ChunkedResult.manifest_path = manifest_path

    report_path = write_cleaning_report(
        target_dir=tmp_path,
        input_path=source_path,
        chunked_result=ChunkedResult(),
        pipeline_description_path=pipeline_path,
        matrix_report_path=matrix_path,
    )
    html = report_path.read_text(encoding="utf-8")

    assert html.index('id="decisions"') < html.index('id="during"') < html.index('id="temporal"')
    assert "data:image/png;base64," in html
    assert "data:image/jpeg;base64," in html
    assert "data:image/gif;base64," not in html
    assert "flex_correction" in html
    assert "threshold" in html
    assert "Complete numeric matrices remain in artifact_template_matrices.json" in html
    assert "Technical references" not in html
    assert "<details" not in html
    assert "123456" in html
    assert "8–13 Hz" in html
    assert "9 overlapping 2.00 s Hann windows" in html
    assert "Analytic/QC aid—not a diagnosis" in html
    metrics = json.loads((tmp_path / "quality_metrics.json").read_text(encoding="utf-8"))
    expected_metrics = {
        "scanner_peak_residual",
        "scanner_peak_suppression_db",
        "delta_preservation",
        "theta_preservation",
        "alpha_preservation",
        "nonpeak_beta_preservation",
        "nonpeak_eeg_log_deviation",
        "rms_improvement_ratio",
        "removed_signal_rms_fraction",
        "median_peak_to_peak_preservation",
        "waveform_correlation",
        "source_extreme_sample_fraction",
        "corrected_extreme_sample_fraction",
        "low_graph_mode_preservation",
        "high_graph_mode_preservation",
        "coherence_before_mean",
        "coherence_after_mean",
        "coherence_change",
        "coherence_modularity",
    }
    assert expected_metrics <= set(metrics["metrics"])
    assert all(record["description"] for record in metrics["metrics"].values())
    assert all(record["interpretation"] for record in metrics["metrics"].values())


def test_atomic_report_write_preserves_existing_file_on_stream_failure(monkeypatch, tmp_path):
    """A failed matrix stream must not expose a partial replacement report."""
    pipeline_path = tmp_path / "pipeline_description.json"
    matrix_path = tmp_path / "artifact_template_matrices.json"
    output_path = tmp_path / "source_cleaning_report.html"
    pipeline_path.write_text("{}", encoding="utf-8")
    # The HTML writer intentionally does not deserialize the potentially large
    # matrix companion; compact display records come from retained metadata.
    matrix_path.write_text("this is intentionally not parsed", encoding="utf-8")
    output_path.write_text("previous complete report", encoding="utf-8")

    class EmptyResult:
        chunks = []
        manifest_path = None

        def __iter__(self):
            return iter(())

    def fail_stream(*args, **kwargs):
        raise RuntimeError("stream failed")

    monkeypatch.setattr(html_report, "_write_matrix_assets", fail_stream)

    with pytest.raises(RuntimeError, match="stream failed"):
        write_cleaning_report(
            target_dir=tmp_path,
            input_path=tmp_path / "source.edf",
            chunked_result=EmptyResult(),
            pipeline_description_path=pipeline_path,
            matrix_report_path=matrix_path,
            output_path=output_path,
        )

    assert output_path.read_text(encoding="utf-8") == "previous complete report"
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))
