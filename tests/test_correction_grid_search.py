"""Tests for correction grid-search setup."""

import numpy as np
import pytest

from facet.correction import CORRELATION_THRESHOLD_GRID, CorrectionGridSearch, FARMCorrection


@pytest.mark.unit
class TestCorrectionGridSearch:
    """Tests for lazy AAS/FARM grid-search construction."""

    def test_default_threshold_grid_uses_one_percent_steps(self, temp_dir):
        """Correlation thresholds should cover 0.01 through 1.00."""
        search = CorrectionGridSearch(
            output_csv=temp_dir / "grid.csv",
            rel_window_positions=[0.0],
            search_window_factors=[1.0],
            window_sizes=[5],
            models=["aas"],
        )

        assert CORRELATION_THRESHOLD_GRID[0] == pytest.approx(0.01)
        assert CORRELATION_THRESHOLD_GRID[-1] == pytest.approx(1.0)
        assert len(CORRELATION_THRESHOLD_GRID) == 100
        assert search.n_combinations == 100

    def test_parameter_grid_uses_same_parameters_for_aas_and_farm(self, temp_dir):
        """AAS and FARM should receive the same grid rows and editable columns."""
        search = CorrectionGridSearch(
            output_csv=temp_dir / "grid.csv",
            correlation_thresholds=[0.1],
            rel_window_positions=[-1.0, 0.0, 1.0],
            search_window_factors=[2.0],
            window_sizes=[5],
            models=["aas", "farm"],
        )

        params = search.iter_parameter_grid()
        aas_rows = [row for row in params if row["model"] == "aas"]
        farm_rows = [row for row in params if row["model"] == "farm"]

        assert len(aas_rows) == 3
        assert len(farm_rows) == 3
        assert {row["rel_window_position"] for row in aas_rows} == {-1.0, 0.0, 1.0}
        assert {row["rel_window_position"] for row in farm_rows} == {-1.0, 0.0, 1.0}
        assert set(aas_rows[0]) == set(farm_rows[0])

    def test_build_pipeline_uses_requested_correction_order(self, temp_dir):
        """Each generated pipeline should start trigger, upsample, model, downsample."""
        search = CorrectionGridSearch(
            output_csv=temp_dir / "grid.csv",
            correlation_thresholds=[0.2],
            rel_window_positions=[0.5],
            search_window_factors=[3.0],
            window_sizes=[7],
            models=["aas"],
        )

        [params] = search.iter_parameter_grid()
        pipeline = search.build_pipeline(params)

        assert [processor.name for processor in pipeline.processors[:4]] == [
            "trigger_detector",
            "upsample",
            "aas_correction",
            "downsample",
        ]
        assert pipeline.processors[2].window_size == 7
        assert pipeline.processors[2].rel_window_position == pytest.approx(0.5)
        assert pipeline.processors[2].search_window_factor == pytest.approx(3.0)

    def test_farm_receives_same_shared_parameters_as_aas(self, temp_dir):
        """FARM should be built from the same shared grid parameter names."""
        search = CorrectionGridSearch(
            output_csv=temp_dir / "grid.csv",
            correlation_thresholds=[0.2],
            rel_window_positions=[0.25],
            search_window_factors=[4.0],
            window_sizes=[8],
            models=["farm"],
        )

        [params] = search.iter_parameter_grid()
        pipeline = search.build_pipeline(params)
        model = pipeline.processors[2]

        assert isinstance(model, FARMCorrection)
        assert model.window_size == 8
        assert model.correlation_threshold == pytest.approx(0.2)
        assert model.rel_window_position == pytest.approx(0.25)
        assert model.search_window_factor == pytest.approx(4.0)

    def test_auto_score_is_finite_for_scalar_metrics(self, temp_dir):
        """Default scoring should combine FACETpy metrics into a finite scalar."""
        search = CorrectionGridSearch(output_csv=temp_dir / "grid.csv")

        score = search.score_metrics(
            {
                "snr": 12.0,
                "legacy_snr": 8.0,
                "rms_ratio": 2.5,
                "rms_residual": 1.05,
                "median_artifact_ratio": 0.95,
                "fft_allen_Delta": 3.0,
            }
        )

        assert np.isfinite(score)
