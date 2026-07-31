"""
Tests for Pipeline class.
"""

import numpy as np
import pytest

from facet.core import Pipeline
from tests.conftest import create_mock_processor


@pytest.mark.unit
class TestPipeline:
    """Tests for Pipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        proc1 = create_mock_processor("proc1")
        proc2 = create_mock_processor("proc2")

        pipeline = Pipeline([proc1, proc2], name="Test Pipeline")

        assert pipeline.name == "Test Pipeline"
        assert len(pipeline.processors) == 2

    def test_pipeline_execution(self, sample_context):
        """Test basic pipeline execution."""
        proc1 = create_mock_processor("proc1")
        proc2 = create_mock_processor("proc2")
        proc3 = create_mock_processor("proc3")

        pipeline = Pipeline([proc1, proc2, proc3])
        result = pipeline.run(initial_context=sample_context)

        # Check success
        assert result.success is True
        assert result.error is None
        assert result.context is not None

        # Check all processors were called
        assert proc1.call_count == 1
        assert proc2.call_count == 1
        assert proc3.call_count == 1

    def test_pipeline_with_loader(self, sample_edf_file):
        """Test pipeline with loader creating initial context."""
        from facet.io import Loader

        loader = Loader(path=str(sample_edf_file), preload=True)
        proc1 = create_mock_processor("proc1")

        pipeline = Pipeline([loader, proc1])
        result = pipeline.run()

        # Should create context from loader
        assert result.success is True
        assert result.context is not None
        assert proc1.call_count == 1

    def test_pipeline_error_handling(self, sample_context):
        """Test pipeline handles processor errors."""
        proc1 = create_mock_processor("proc1")

        # Processor that raises error
        def error_fn(ctx):
            raise ValueError("Test error")

        proc2 = create_mock_processor("error_proc", process_fn=error_fn)
        proc3 = create_mock_processor("proc3")

        pipeline = Pipeline([proc1, proc2, proc3])
        result = pipeline.run(initial_context=sample_context)

        # Check failure
        assert result.success is False
        assert result.error is not None
        assert result.failed_processor == "error_proc"

        # Only proc1 should have been called
        assert proc1.call_count == 1
        assert proc3.call_count == 0

    def test_pipeline_timing(self, sample_context):
        """Test pipeline records execution time."""
        proc1 = create_mock_processor("proc1")

        pipeline = Pipeline([proc1])
        result = pipeline.run(initial_context=sample_context)

        assert result.execution_time > 0
        assert isinstance(result.execution_time, float)

    def test_pipeline_history_tracking(self, sample_context):
        """Test pipeline tracks processing history."""
        proc1 = create_mock_processor("proc1")
        proc2 = create_mock_processor("proc2")

        pipeline = Pipeline([proc1, proc2])
        result = pipeline.run(initial_context=sample_context)

        history = result.context.get_history()
        assert len(history) >= 2

        # Check processors are recorded
        processor_names = [h.name for h in history]
        assert "proc1" in processor_names
        assert "proc2" in processor_names

    def test_empty_pipeline(self, sample_context):
        """Test empty pipeline."""
        pipeline = Pipeline([])
        result = pipeline.run(initial_context=sample_context)

        # Should succeed with unchanged context
        assert result.success is True
        assert result.context.get_raw() is sample_context.get_raw()

    def test_pipeline_repr(self):
        """Test pipeline string representation."""
        proc1 = create_mock_processor("proc1")
        pipeline = Pipeline([proc1], name="My Pipeline")

        repr_str = repr(pipeline)
        assert "My Pipeline" in repr_str
        assert "n_processors=1" in repr_str


@pytest.mark.unit
class TestPipelineResult:
    """Tests for PipelineResult class."""

    def test_result_attributes(self, sample_context):
        """Test PipelineResult attributes."""
        pipeline = Pipeline([create_mock_processor("proc1")])
        result = pipeline.run(initial_context=sample_context)

        # Check all attributes exist
        assert hasattr(result, "success")
        assert hasattr(result, "context")
        assert hasattr(result, "error")
        assert hasattr(result, "execution_time")
        assert hasattr(result, "failed_processor")

    def test_successful_result(self, sample_context):
        """Test attributes of successful result."""
        pipeline = Pipeline([create_mock_processor("proc1")])
        result = pipeline.run(initial_context=sample_context)

        assert result.success is True
        assert result.context is not None
        assert result.error is None
        assert result.failed_processor is None
        assert result.execution_time > 0

    def test_release_raw_discards_signal_data_but_retains_summaries(self, sample_context):
        """Released batch results retain metrics/history without large arrays."""
        sample_context.set_estimated_noise(np.ones_like(sample_context.get_raw()._data))
        sample_context.cache_set("derived", np.ones_like(sample_context.get_raw()._data))
        sample_context.metadata.custom["metrics"] = {"snr": 12.5}
        sample_context.add_history_entry(processor_name="prepared")

        result = Pipeline([]).run(initial_context=sample_context)
        metrics = result.metrics.copy()
        history = result.get_history()

        result.release_raw()

        assert not result.context.has_raw()
        assert result.context.get_raw_original() is None
        assert not result.context.has_estimated_noise()
        assert not result.context.cache_has("derived")
        assert result.metrics == metrics
        assert result.get_history() == history

    def test_failed_result(self, sample_context):
        """Test attributes of failed result."""

        def error_fn(ctx):
            raise ValueError("Test error")

        proc_error = create_mock_processor("error_proc", process_fn=error_fn)
        pipeline = Pipeline([proc_error])
        result = pipeline.run(initial_context=sample_context)

        assert result.success is False
        assert result.error is not None
        assert "Test error" in str(result.error)
        assert result.failed_processor == "error_proc"


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for pipeline with real processors."""

    def test_simple_correction_pipeline(self, sample_edf_file):
        """Test a simple correction pipeline."""
        import tempfile

        from facet.correction import AASCorrection
        from facet.io import EDFExporter, Loader
        from facet.preprocessing import DownSample, TriggerDetector, UpSample

        # Create output path
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            output_path = tmp.name

        pipeline = Pipeline(
            [
                Loader(path=str(sample_edf_file), preload=True),
                TriggerDetector(regex=r"\b1\b"),
                UpSample(factor=2),  # Small factor for speed
                AASCorrection(window_size=5),  # Small window for speed
                DownSample(factor=2),
                EDFExporter(path=output_path, overwrite=True),
            ]
        )

        result = pipeline.run()

        # Check success
        assert result.success is True
        assert result.context is not None

        # Check file was created
        import os

        assert os.path.exists(output_path)

        # Cleanup
        os.unlink(output_path)

    @pytest.mark.slow
    def test_full_correction_pipeline(self, sample_edf_file):
        """Test a full correction pipeline with multiple steps."""
        from facet.correction import AASCorrection
        from facet.evaluation import RMSCalculator, SNRCalculator
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
            ]
        )

        result = pipeline.run()

        assert result.success is True

        # Check metrics were calculated
        metrics = result.context.metadata.custom.get("metrics", {})
        assert "snr" in metrics
        assert "rms_ratio" in metrics


@pytest.mark.unit
class TestPipelineCallableNormalisation:
    """Pipeline should accept plain callables alongside Processor instances."""

    def test_lambda_in_constructor(self, sample_context):
        """A lambda is silently wrapped and executed."""
        called = []

        pipeline = Pipeline([lambda ctx: called.append(1) or ctx])
        result = pipeline.run(initial_context=sample_context)

        assert result.success is True
        assert called == [1]

    def test_named_function_in_constructor(self, sample_context):
        """A def function is wrapped and its __name__ is used as step label."""

        def my_step(ctx):
            return ctx

        pipeline = Pipeline([my_step])
        result = pipeline.run(initial_context=sample_context)

        assert result.success is True
        names = [h.name for h in result.context.get_history()]
        assert "my_step" in names

    def test_mixed_processor_and_callable(self, sample_context):
        """Processor and callable can be mixed freely in one list."""
        proc = create_mock_processor("proc1")
        pipeline = Pipeline([proc, lambda ctx: ctx])
        result = pipeline.run(initial_context=sample_context)

        assert result.success is True
        assert proc.call_count == 1

    def test_invalid_item_raises_type_error(self):
        """A non-callable, non-Processor item raises TypeError."""
        with pytest.raises(TypeError):
            Pipeline([42])

    def test_add_callable(self, sample_context):
        """Pipeline.add() accepts a callable."""
        called = []
        pipeline = Pipeline([])
        pipeline.add(lambda ctx: called.append(True) or ctx)

        result = pipeline.run(initial_context=sample_context)
        assert result.success is True
        assert called == [True]

    def test_insert_callable(self, sample_context):
        """Pipeline.insert() accepts a callable."""
        order = []
        proc = create_mock_processor("first")

        def second(ctx):
            order.append("second")
            return ctx

        pipeline = Pipeline([proc])
        pipeline.insert(0, lambda ctx: order.append("inserted") or ctx)

        result = pipeline.run(initial_context=sample_context)
        assert result.success is True
        assert order[0] == "inserted"

    def test_extend_callables(self, sample_context):
        """Pipeline.extend() accepts a list of callables."""
        called = []
        pipeline = Pipeline([])
        pipeline.extend(
            [
                lambda ctx: called.append(1) or ctx,
                lambda ctx: called.append(2) or ctx,
            ]
        )

        result = pipeline.run(initial_context=sample_context)
        assert result.success is True
        assert called == [1, 2]


@pytest.mark.unit
class TestPipelineResultMetrics:
    """Tests for PipelineResult.metrics and .metrics_df properties."""

    def _make_result_with_metrics(self, sample_context):
        """Run a pipeline that populates metrics via SNRCalculator."""
        from facet.correction import AASCorrection
        from facet.evaluation import SNRCalculator
        from facet.preprocessing import DownSample, UpSample

        pipeline = Pipeline(
            [
                UpSample(factor=2),
                AASCorrection(window_size=3),
                DownSample(factor=2),
                SNRCalculator(),
            ]
        )
        return pipeline.run(initial_context=sample_context)

    def test_metrics_returns_dict(self, sample_context):
        """result.metrics is a dict."""
        result = self._make_result_with_metrics(sample_context)
        assert isinstance(result.metrics, dict)

    def test_metrics_contains_snr(self, sample_context):
        """result.metrics contains 'snr' after SNRCalculator."""
        result = self._make_result_with_metrics(sample_context)
        assert "snr" in result.metrics

    def test_metrics_empty_on_failure(self, sample_context):
        """result.metrics is empty dict when pipeline fails."""

        def explode(ctx):
            raise RuntimeError("boom")

        result = Pipeline([explode]).run(initial_context=sample_context)
        assert result.metrics == {}

    def test_metrics_df_is_series(self, sample_context):
        """result.metrics_df is a pandas Series (or None if pandas absent)."""
        pytest.importorskip("pandas")
        result = self._make_result_with_metrics(sample_context)
        import pandas as pd

        assert isinstance(result.metrics_df, pd.Series)

    def test_metrics_df_contains_snr(self, sample_context):
        """result.metrics_df includes the snr entry."""
        pytest.importorskip("pandas")
        result = self._make_result_with_metrics(sample_context)
        assert "snr" in result.metrics_df.index


@pytest.mark.unit
class TestPipelineMap:
    """Tests for Pipeline.map() batch helper."""

    def test_map_with_contexts(self, sample_context):
        """map() accepts a list of ProcessingContext objects."""
        pipeline = Pipeline([create_mock_processor("p")])
        results = pipeline.map([sample_context, sample_context])

        assert len(results) == 2
        assert all(r.success for r in results)

    def test_map_with_loader_factory(self, sample_edf_file):
        """map() uses loader_factory to load each file path."""
        from facet.io import Loader
        from facet.preprocessing import TriggerDetector

        pipeline = Pipeline([TriggerDetector(regex=r"\b1\b")])
        results = pipeline.map(
            [str(sample_edf_file)],
            loader_factory=lambda p: Loader(path=p, preload=True),
        )

        assert len(results) == 1
        assert results[0].success is True

    def test_map_on_error_continue(self, sample_context):
        """on_error='continue' skips failed inputs instead of raising."""

        def explode(ctx):
            raise RuntimeError("fail")

        pipeline = Pipeline([explode])
        results = pipeline.map(
            [sample_context, sample_context],
            on_error="continue",
        )

        assert len(results) == 2
        assert all(not r.success for r in results)

    def test_map_on_error_raise(self, sample_context):
        """on_error='raise' re-raises the first failure."""

        def explode(ctx):
            raise RuntimeError("fail")

        pipeline = Pipeline([explode])
        with pytest.raises(RuntimeError, match="fail"):
            pipeline.map([sample_context], on_error="raise")
