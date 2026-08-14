"""Tests for correction grid-search setup."""

import numpy as np
import pytest

from facet.correction import CorrectionGridSearch, Flex


@pytest.mark.unit
class TestCorrectionGridSearch:
    """Tests for Flex-only grid-search construction."""

    def test_default_threshold_grid_is_a_bounded_starter_set(self, temp_dir):
        """Default thresholds should remain a moderate, reproducible set."""
        search = CorrectionGridSearch(
            output_csv=temp_dir / "grid.csv",
            window_sizes=[5],
            min_accepted_values=[3],
            N_distributions=["equal"],
            realign_after_averaging_values=[False],
            search_window_factors=[1.0],
            interpolate_volume_gaps_values=[False],
            apply_epoch_alpha_scaling_values=[False],
        )

        assert search.thresholds == pytest.approx((0.50, 0.75, 0.90, 0.95))
        assert search.n_combinations == 4

    def test_parameter_grid_combines_only_flex_parameters(self, temp_dir):
        """The grid should contain complete, valid Flex constructor rows."""
        search = CorrectionGridSearch(
            output_csv=temp_dir / "grid.csv",
            thresholds=[0.1],
            window_sizes=[5],
            min_accepted_values=[2, 4],
            N_distributions=["equal", "normal"],
            realign_after_averaging_values=[False],
            search_window_factors=[2.0],
            interpolate_volume_gaps_values=[False],
            apply_epoch_alpha_scaling_values=[False],
        )

        params = search.iter_parameter_grid()

        assert len(params) == 4
        assert {row["min_accepted"] for row in params} == {2, 4}
        assert {row["N_distribution"] for row in params} == {"equal", "normal"}
        assert all("model" not in row for row in params)

    def test_build_pipeline_uses_requested_correction_order(self, temp_dir):
        """Each generated pipeline should start trigger, upsample, model, downsample."""
        search = CorrectionGridSearch(
            output_csv=temp_dir / "grid.csv",
            thresholds=[0.2],
            min_accepted_values=[3],
            N_distributions=["equal"],
            realign_after_averaging_values=[False],
            search_window_factors=[3.0],
            window_sizes=[7],
            interpolate_volume_gaps_values=[False],
            apply_epoch_alpha_scaling_values=[False],
        )

        [params] = search.iter_parameter_grid()
        pipeline = search.build_pipeline(params)

        assert [processor.name for processor in pipeline.processors[:4]] == [
            "trigger_detector",
            "upsample",
            "flex_correction",
            "downsample",
        ]
        assert pipeline.processors[2].window_size == 7
        assert pipeline.processors[2].threshold == pytest.approx(0.2)
        assert pipeline.processors[2].search_window_factor == pytest.approx(3.0)

    def test_pipeline_correction_is_flex(self, temp_dir):
        """Grid search must not instantiate one of the archived algorithms."""
        search = CorrectionGridSearch(
            output_csv=temp_dir / "grid.csv",
            thresholds=[0.2],
            min_accepted_values=[4],
            N_distributions=["normal"],
            realign_after_averaging_values=[True],
            search_window_factors=[4.0],
            window_sizes=[8],
            interpolate_volume_gaps_values=[True],
            apply_epoch_alpha_scaling_values=[True],
        )

        [params] = search.iter_parameter_grid()
        pipeline = search.build_pipeline(params)
        model = pipeline.processors[2]

        assert type(model) is Flex
        assert model.window_size == 8
        assert model.threshold == pytest.approx(0.2)
        assert model.min_accepted == 4
        assert model.N_distribution == "normal"
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
