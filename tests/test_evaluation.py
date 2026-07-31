"""
Tests for evaluation processors.
"""

import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessorValidationError
from facet.evaluation import (
    FFTAllenCalculator,
    FFTNiazyCalculator,
    LegacySNRCalculator,
    MedianArtifactCalculator,
    MetricsReport,
    ReferenceIntervalSelector,
    RMSCalculator,
    RMSResidualCalculator,
    SignalIntervalSelector,
    SNRCalculator,
)


def _assert_nested_numeric_equal(actual, expected) -> None:
    """Compare nested metric structures while tolerating floating roundoff."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_numeric_equal(actual[key], expected[key])
        return

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-13,
        atol=1e-15,
        equal_nan=True,
    )


@pytest.mark.unit
class TestReferenceIntervalSelector:
    """Tests for ReferenceIntervalSelector processor."""

    def test_reference_interval_selector_stores_interval(self, sample_context, monkeypatch):
        """Selected interval should be written to metadata.custom."""
        selector = ReferenceIntervalSelector(channel=0, min_duration=0.2)
        monkeypatch.setattr(
            selector,
            "_show_selector",
            lambda **kwargs: (0.5, 1.5),
        )

        result = selector.execute(sample_context)
        interval = result.metadata.custom.get("reference_interval", {})

        assert interval.get("tmin") == pytest.approx(0.5)
        assert interval.get("tmax") == pytest.approx(1.5)
        assert interval.get("source") == "reference_interval_selector"

    def test_reference_interval_selector_cancel_keeps_context(self, sample_context, monkeypatch):
        """Cancelling selection should keep metadata unchanged."""
        selector = ReferenceIntervalSelector()
        monkeypatch.setattr(selector, "_show_selector", lambda **kwargs: None)

        result = selector.execute(sample_context)
        assert "reference_interval" not in result.metadata.custom

    def test_reference_interval_selector_uses_explicit_bounds(self, sample_context, monkeypatch):
        """Explicit tmin/tmax should skip GUI and store configured interval."""
        selector = ReferenceIntervalSelector(channel=0, tmin=0.25, tmax=1.25)

        def _must_not_be_called(**kwargs):
            raise AssertionError("_show_selector should not be called when explicit bounds are provided")

        monkeypatch.setattr(selector, "_show_selector", _must_not_be_called)

        result = selector.execute(sample_context)
        interval = result.metadata.custom.get("reference_interval", {})

        assert interval.get("tmin") == pytest.approx(0.25)
        assert interval.get("tmax") == pytest.approx(1.25)

    def test_reference_interval_selector_rejects_invalid_bounds(self, sample_context):
        """Invalid explicit interval should raise validation error."""
        selector = ReferenceIntervalSelector(tmin=2.0, tmax=1.0)

        with pytest.raises(ProcessorValidationError):
            selector.execute(sample_context)


@pytest.mark.unit
class TestSignalIntervalSelector:
    """Tests for SignalIntervalSelector processor."""

    def test_signal_interval_selector_stores_interval(self, sample_context, monkeypatch):
        """Selected interval should be written to metadata.custom."""
        selector = SignalIntervalSelector(channel=0, min_duration=0.2)
        monkeypatch.setattr(
            selector,
            "_show_selector",
            lambda **kwargs: (1.0, 2.0),
        )

        result = selector.execute(sample_context)
        interval = result.metadata.custom.get("evaluation_interval", {})

        assert interval.get("tmin") == pytest.approx(1.0)
        assert interval.get("tmax") == pytest.approx(2.0)
        assert interval.get("source") == "signal_interval_selector"

    def test_signal_interval_selector_cancel_keeps_context(self, sample_context, monkeypatch):
        """Cancelling selection should keep metadata unchanged."""
        selector = SignalIntervalSelector()
        monkeypatch.setattr(selector, "_show_selector", lambda **kwargs: None)

        result = selector.execute(sample_context)
        assert "evaluation_interval" not in result.metadata.custom

    def test_signal_interval_selector_uses_explicit_bounds(self, sample_context, monkeypatch):
        """Explicit tmin/tmax should skip GUI and store configured interval."""
        selector = SignalIntervalSelector(channel=0, tmin=1.0, tmax=2.5)

        def _must_not_be_called(**kwargs):
            raise AssertionError("_show_selector should not be called when explicit bounds are provided")

        monkeypatch.setattr(selector, "_show_selector", _must_not_be_called)

        result = selector.execute(sample_context)
        interval = result.metadata.custom.get("evaluation_interval", {})

        assert interval.get("tmin") == pytest.approx(1.0)
        assert interval.get("tmax") == pytest.approx(2.5)

    def test_signal_interval_selector_rejects_invalid_bounds(self, sample_context):
        """Invalid explicit interval should raise validation error."""
        selector = SignalIntervalSelector(tmin=3.0, tmax=2.0)

        with pytest.raises(ProcessorValidationError):
            selector.execute(sample_context)

    def test_signal_interval_selector_requires_triggers(self, sample_raw):
        """Signal interval selection requires triggers."""
        selector = SignalIntervalSelector()
        context = ProcessingContext(raw=sample_raw)

        with pytest.raises(ProcessorValidationError):
            selector.execute(context)


@pytest.mark.unit
class TestSNRCalculator:
    """Tests for SNRCalculator processor."""

    def test_snr_calculation(self, sample_context):
        """Test SNR calculation."""
        snr_calc = SNRCalculator()
        result = snr_calc.execute(sample_context)

        # Check SNR was calculated
        metrics = result.metadata.custom.get("metrics", {})
        assert "snr" in metrics
        assert isinstance(metrics["snr"], float)
        assert metrics["snr"] > 0

    def test_snr_requires_triggers(self, sample_raw):
        """Test that SNR calculator requires triggers."""
        context = ProcessingContext(raw=sample_raw)

        snr_calc = SNRCalculator()

        with pytest.raises(ProcessorValidationError):
            snr_calc.execute(context)

    def test_snr_requires_original_raw(self, sample_context):
        """Test that SNR requires original raw data."""
        # Create context without original
        context = ProcessingContext(raw=sample_context.get_raw())

        snr_calc = SNRCalculator()

        with pytest.raises(ProcessorValidationError):
            snr_calc.execute(context)

    def test_snr_per_channel(self, sample_context):
        """Test per-channel SNR calculation."""
        snr_calc = SNRCalculator()
        result = snr_calc.execute(sample_context)

        metrics = result.metadata.custom.get("metrics", {})
        assert "snr_per_channel" in metrics
        assert len(metrics["snr_per_channel"]) > 0

    def test_snr_uses_selected_reference_interval(self, sample_context):
        """User-selected reference interval should override automatic extraction."""
        metadata = sample_context.metadata.copy()
        metadata.custom["reference_interval"] = {"tmin": 0.0, "tmax": 1.0}
        context = sample_context.with_metadata(metadata)

        snr_calc = SNRCalculator()
        result = snr_calc.execute(context)

        metrics = result.metadata.custom.get("metrics", {})
        assert "snr" in metrics

    def test_snr_uses_selected_evaluation_interval(self, sample_context):
        """User-selected evaluation interval should override automatic extraction."""
        metadata = sample_context.metadata.copy()
        metadata.custom["evaluation_interval"] = {"tmin": 1.0, "tmax": 2.0}
        context = sample_context.with_metadata(metadata)

        snr_calc = SNRCalculator()
        raw = context.get_raw()
        acquisition_data = snr_calc.get_acquisition_data(
            raw=raw,
            triggers=context.get_triggers(),
            artifact_length=context.get_artifact_length(),
            context=context,
        )

        eeg_picks = snr_calc.get_eeg_channels(raw)
        expected = raw.get_data(picks=eeg_picks, tmin=1.0, tmax=2.0)

        assert acquisition_data.shape == expected.shape
        np.testing.assert_allclose(acquisition_data, expected)


@pytest.mark.unit
class TestRMSCalculator:
    """Tests for RMSCalculator processor."""

    def test_rms_calculation(self, sample_context):
        """Test RMS calculation."""
        rms_calc = RMSCalculator()
        result = rms_calc.execute(sample_context)

        # Check RMS ratio was calculated
        metrics = result.metadata.custom.get("metrics", {})
        assert "rms_ratio" in metrics
        assert isinstance(metrics["rms_ratio"], float)
        assert metrics["rms_ratio"] > 0

    def test_rms_requires_triggers(self, sample_raw):
        """Test that RMS calculator requires triggers."""
        context = ProcessingContext(raw=sample_raw)

        rms_calc = RMSCalculator()

        with pytest.raises(ProcessorValidationError):
            rms_calc.execute(context)

    def test_rms_requires_original_raw(self, sample_context):
        """Test that RMS requires original raw data."""
        context = ProcessingContext(raw=sample_context.get_raw())

        rms_calc = RMSCalculator()

        with pytest.raises(ProcessorValidationError):
            rms_calc.execute(context)

    def test_rms_per_channel(self, sample_context):
        """Test per-channel RMS calculation."""
        rms_calc = RMSCalculator()
        result = rms_calc.execute(sample_context)

        metrics = result.metadata.custom.get("metrics", {})
        assert "rms_ratio_per_channel" in metrics


@pytest.mark.unit
class TestRMSResidualCalculator:
    """Tests for RMSResidualCalculator processor."""

    def test_rms_residual_calculation(self, sample_context):
        """Test RMS residual calculation."""
        rms_calc = RMSResidualCalculator()
        result = rms_calc.execute(sample_context)

        metrics = result.metadata.custom.get("metrics", {})
        assert "rms_residual" in metrics
        assert isinstance(metrics["rms_residual"], float)
        assert metrics["rms_residual"] > 0

    def test_rms_residual_requires_triggers(self, sample_raw):
        """Test that RMS residual calculator requires triggers."""
        context = ProcessingContext(raw=sample_raw)
        rms_calc = RMSResidualCalculator()
        with pytest.raises(ProcessorValidationError):
            rms_calc.execute(context)

    def test_rms_residual_per_channel(self, sample_context):
        """Test per-channel RMS residual calculation."""
        rms_calc = RMSResidualCalculator()
        result = rms_calc.execute(sample_context)

        metrics = result.metadata.custom.get("metrics", {})
        assert "rms_residual_per_channel" in metrics
        assert len(metrics["rms_residual_per_channel"]) > 0


@pytest.mark.unit
class TestLegacySNRCalculator:
    """Tests for LegacySNRCalculator processor."""

    def test_legacy_snr_calculation(self, sample_context):
        """Test legacy SNR calculation."""
        snr_calc = LegacySNRCalculator()
        result = snr_calc.execute(sample_context)

        metrics = result.metadata.custom.get("metrics", {})
        assert "legacy_snr" in metrics
        assert isinstance(metrics["legacy_snr"], float)
        assert metrics["legacy_snr"] > 0

    def test_legacy_snr_requires_original_raw(self, sample_context):
        """Test that legacy SNR requires original raw data."""
        context = ProcessingContext(raw=sample_context.get_raw())
        snr_calc = LegacySNRCalculator()
        with pytest.raises(ProcessorValidationError):
            snr_calc.execute(context)


@pytest.mark.unit
class TestMedianArtifactCalculator:
    """Tests for MedianArtifactCalculator processor."""

    def test_median_artifact_calculation(self, sample_context):
        """Test median artifact calculation."""
        median_calc = MedianArtifactCalculator()
        result = median_calc.execute(sample_context)

        # Check median artifact was calculated
        metrics = result.metadata.custom.get("metrics", {})
        assert "median_artifact" in metrics
        assert isinstance(metrics["median_artifact"], float)
        assert metrics["median_artifact"] >= 0

        # Check for reference ratio (might be missing if sample data is too short for reference extraction)
        # But the sample_context usually provides enough data
        if "median_artifact_reference" in metrics:
            assert "median_artifact_ratio" in metrics

    def test_median_requires_triggers(self, sample_raw):
        """Test that median calculator requires triggers."""
        context = ProcessingContext(raw=sample_raw)

        median_calc = MedianArtifactCalculator()

        with pytest.raises(ProcessorValidationError):
            median_calc.execute(context)

    def test_median_requires_artifact_length(self, sample_context):
        """Test that median calculator requires artifact length."""
        metadata = sample_context.metadata.copy()
        metadata.artifact_length = None
        context = sample_context.with_metadata(metadata)

        median_calc = MedianArtifactCalculator()

        with pytest.raises(ProcessorValidationError):
            median_calc.execute(context)


@pytest.mark.unit
class TestFFTAllenCalculator:
    """Tests for FFTAllenCalculator processor."""

    def test_fft_allen_calculation(self, sample_context):
        """Test FFT Allen calculation."""
        fft_calc = FFTAllenCalculator()
        result = fft_calc.execute(sample_context)

        metrics = result.metadata.custom.get("metrics", {})
        assert "fft_allen" in metrics
        assert isinstance(metrics["fft_allen"], dict)

        # Check bands
        bands = ["Delta", "Theta", "Alpha", "Beta"]
        for band in bands:
            if band in metrics["fft_allen"]:
                assert isinstance(metrics["fft_allen"][band], float)
                assert metrics["fft_allen"][band] >= 0

    def test_fft_allen_requires_triggers(self, sample_raw):
        context = ProcessingContext(raw=sample_raw)
        fft_calc = FFTAllenCalculator()
        with pytest.raises(ProcessorValidationError):
            fft_calc.execute(context)


@pytest.mark.unit
class TestFFTNiazyCalculator:
    """Tests for FFTNiazyCalculator processor."""

    def test_fft_niazy_calculation(self, sample_context):
        """Test FFT Niazy calculation."""
        fft_calc = FFTNiazyCalculator()
        result = fft_calc.execute(sample_context)

        metrics = result.metadata.custom.get("metrics", {})
        assert "fft_niazy" in metrics
        assert "slice" in metrics["fft_niazy"]
        # volume might be missing if slices_per_volume is not set

    def test_fft_niazy_with_volume(self, sample_context):
        """Test FFT Niazy with volume information."""
        metadata = sample_context.metadata.copy()
        metadata.slices_per_volume = 10  # fake value
        context = sample_context.with_metadata(metadata)

        fft_calc = FFTNiazyCalculator()
        result = fft_calc.execute(context)

        metrics = result.metadata.custom.get("metrics", {})
        assert "volume" in metrics["fft_niazy"]


@pytest.mark.unit
def test_evaluation_processors_accept_verbose_mode(sample_context):
    """All main evaluation processors should run with verbose diagnostics enabled."""
    processors = [
        (SNRCalculator(verbose=True), "snr"),
        (RMSCalculator(verbose=True), "rms_ratio"),
        (RMSResidualCalculator(verbose=True), "rms_residual"),
        (MedianArtifactCalculator(verbose=True), "median_artifact"),
        (LegacySNRCalculator(verbose=True), "legacy_snr"),
        (FFTAllenCalculator(verbose=True), "fft_allen"),
        (FFTNiazyCalculator(verbose=True), "fft_niazy"),
    ]

    context = sample_context
    for proc, metric_key in processors:
        context = proc.execute(context)
        metrics = context.metadata.custom.get("metrics", {})
        assert metric_key in metrics


@pytest.mark.unit
@pytest.mark.parametrize(
    ("processor_type", "metric_key"),
    [
        (SNRCalculator, "snr"),
        (RMSCalculator, "rms_ratio"),
        (RMSResidualCalculator, "rms_residual"),
        (MedianArtifactCalculator, "median_artifact"),
        (LegacySNRCalculator, "legacy_snr"),
        (FFTAllenCalculator, "fft_allen"),
        (FFTNiazyCalculator, "fft_niazy"),
    ],
)
def test_channel_blocking_preserves_metric_values(
    sample_context,
    processor_type,
    metric_key,
):
    """One-channel blocks must produce the same metrics as one full block."""
    full_result = processor_type(channel_block_size=None).execute(sample_context)
    blocked_result = processor_type(channel_block_size=1).execute(sample_context)

    full_metric = full_result.metadata.custom["metrics"][metric_key]
    blocked_metric = blocked_result.metadata.custom["metrics"][metric_key]
    _assert_nested_numeric_equal(blocked_metric, full_metric)


@pytest.mark.unit
def test_metric_metadata_update_does_not_copy_estimated_noise(
    sample_context_with_noise,
):
    """Read-only metric processors share the unchanged noise allocation."""
    original_noise = sample_context_with_noise.get_estimated_noise()

    result = SNRCalculator(channel_block_size=1).execute(sample_context_with_noise)

    assert result.get_estimated_noise() is original_noise


@pytest.mark.unit
def test_metric_channel_block_size_must_be_positive():
    """Invalid block sizes should fail before a potentially expensive run."""
    with pytest.raises(ValueError, match="channel_block_size"):
        SNRCalculator(channel_block_size=0)


@pytest.mark.unit
class TestMetricsReport:
    """Tests for MetricsReport processor."""

    def test_metrics_report(self, sample_context, capsys):
        """Test metrics report generation."""
        # Add some metrics
        metadata = sample_context.metadata.copy()
        metadata.custom["metrics"] = {
            "snr": 15.5,
            "rms_ratio": 2.3,
            "median_artifact": 1.5e-5,
            "rms_residual": 1.1,
            "fft_allen": {"Alpha": 5.0},
            "fft_niazy": {"slice": {"h1": -10.0}},
        }
        context = sample_context.with_metadata(metadata)

        # Generate report
        report = MetricsReport()
        result = report.execute(context)

        # Check it doesn't fail
        assert result is not None

    def test_metrics_report_storage(self, sample_context):
        """Test metrics report storage."""
        metadata = sample_context.metadata.copy()
        metadata.custom["metrics"] = {"snr": 15.5, "fft_allen": {"Alpha": 5.0}}
        context = sample_context.with_metadata(metadata)

        results = {}
        report = MetricsReport(store=results)
        report.execute(context)

        assert len(results) == 1
        key = list(results.keys())[0]
        assert results[key]["snr"] == 15.5
        assert results[key]["fft_allen_Alpha"] == 5.0


@pytest.mark.integration
class TestEvaluationPipeline:
    """Integration tests for evaluation pipeline."""

    def test_full_evaluation_pipeline(self, sample_context):
        """Test full evaluation workflow."""
        from facet.core import Pipeline

        pipeline = Pipeline(
            [
                SNRCalculator(),
                RMSCalculator(),
                RMSResidualCalculator(),
                MedianArtifactCalculator(),
                LegacySNRCalculator(),
                FFTAllenCalculator(),
                FFTNiazyCalculator(),
                MetricsReport(),
            ]
        )

        result = pipeline.run(initial_context=sample_context)

        assert result.success is True

        # Check all metrics were calculated
        metrics = result.context.metadata.custom.get("metrics", {})
        assert "snr" in metrics
        assert "rms_ratio" in metrics
        assert "rms_residual" in metrics
        assert "median_artifact" in metrics
        assert "legacy_snr" in metrics
        assert "fft_allen" in metrics
        assert "fft_niazy" in metrics

    def test_evaluation_after_correction(self, sample_edf_file):
        """Test evaluation after correction."""
        from facet.core import Pipeline
        from facet.correction import AASCorrection
        from facet.io import Loader
        from facet.preprocessing import DownSample, TriggerDetector, UpSample

        pipeline = Pipeline(
            [
                Loader(path=str(sample_edf_file), preload=True),
                TriggerDetector(regex=r"\b1\b"),
                UpSample(factor=2),
                AASCorrection(window_size=5),
                DownSample(factor=2),
                SNRCalculator(),
                RMSCalculator(),
                RMSResidualCalculator(),
                MedianArtifactCalculator(),
                FFTAllenCalculator(),
                MetricsReport(),
            ]
        )

        result = pipeline.run()

        assert result.success is True

        # All metrics should be present
        metrics = result.context.metadata.custom.get("metrics", {})
        assert "snr" in metrics
