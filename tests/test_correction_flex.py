"""Tests for flexible correlation-based artifact correction."""

from __future__ import annotations

import inspect

import mne
import numpy as np
import pytest

import facet
from facet.core import (
    Pipeline,
    ProcessingContext,
    ProcessingMetadata,
    Processor,
    ProcessorValidationError,
    get_processor,
)
from facet.correction import AASCorrection, Flex, FlexCorrection


def _make_flex_context(
    *,
    n_triggers: int = 5,
    n_channels: int = 2,
    artifact_length: int = 12,
) -> ProcessingContext:
    """Create deterministic, trigger-locked artifacts for integration tests."""
    sfreq = 100.0
    trigger_spacing = artifact_length + 6
    triggers = 5 + np.arange(n_triggers, dtype=int) * trigger_spacing
    n_samples = int((triggers[-1] if n_triggers else 5) + artifact_length + 5)

    phase = np.linspace(0.0, 2.0 * np.pi, artifact_length, endpoint=False)
    templates = [
        np.sin(phase) + 0.25 * np.cos(2.0 * phase),
        np.cos(phase) - 0.20 * np.sin(3.0 * phase),
    ]
    data = np.zeros((n_channels, n_samples), dtype=float)

    for channel_idx in range(n_channels):
        template = templates[channel_idx % len(templates)]
        for epoch_idx, trigger in enumerate(triggers):
            # Positive amplitude drift keeps the epochs perfectly correlated
            # while ensuring that each target receives a non-trivial template.
            amplitude = (channel_idx + 1) * (1.0 + 0.08 * epoch_idx) * 1e-5
            data[channel_idx, trigger : trigger + artifact_length] = amplitude * template

    ch_names = [f"EEG{channel_idx + 1:03d}" for channel_idx in range(n_channels)]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
    raw = mne.io.RawArray(data, info, verbose=False)

    metadata = ProcessingMetadata()
    metadata.triggers = triggers
    metadata.artifact_length = artifact_length
    metadata.upsampling_factor = 1
    metadata.artifact_to_trigger_offset = 0.0
    metadata.custom["sentinel"] = {"preserved": True}
    return ProcessingContext(raw=raw, raw_original=raw.copy(), metadata=metadata)


def _averaging_matrix(processor: Flex, epochs: np.ndarray) -> np.ndarray:
    """Call the shared Flex matrix hook with public configuration."""
    return processor._calc_averaging_matrix(
        epochs,
        window_size=processor.window_size,
        rel_window_offset=processor.rel_window_position,
        correlation_threshold=processor.correlation_threshold,
    )


@pytest.mark.unit
class TestFlexPublicAPI:
    """Test construction, exports, registration, and parameter provenance."""

    def test_public_exports_alias_and_registry(self):
        """Flex should be discoverable through every supported public entry point."""
        assert facet.Flex is Flex
        assert facet.FlexCorrection is Flex
        assert FlexCorrection is Flex
        assert get_processor("flex_correction") is Flex
        assert Flex.__bases__ == (Processor,)
        assert AASCorrection.__bases__ == (Flex,)

    def test_requested_constructor_parameters_and_defaults(self):
        """The four flexible controls should lead the constructor API."""
        parameters = list(inspect.signature(Flex).parameters.values())

        assert [parameter.name for parameter in parameters[:4]] == [
            "window_size",
            "threshold",
            "min_accepted",
            "N_distribution",
        ]
        assert [parameter.default for parameter in parameters[:4]] == [10, 0.975, 5, "equal"]

    def test_parameters_are_canonical_and_constructor_compatible(self):
        """History parameters should recreate the exact processor configuration."""
        processor = Flex(
            window_size=7,
            threshold=0.82,
            min_accepted=3,
            N_distribution=" NORMAL ",
            plot_artifacts=False,
            realign_after_averaging=False,
            search_window_factor=2.5,
            interpolate_volume_gaps=True,
            apply_epoch_alpha_scaling=True,
        )

        expected = {
            "window_size": 7,
            "threshold": 0.82,
            "min_accepted": 3,
            "N_distribution": "normal",
            "plot_artifacts": False,
            "realign_after_averaging": False,
            "search_window_factor": 2.5,
            "interpolate_volume_gaps": True,
            "apply_epoch_alpha_scaling": True,
            "track_estimated_noise": True,
        }
        assert processor.correlation_threshold == processor.threshold == 0.82
        assert processor.rel_window_position == 0.0
        assert processor._get_parameters() == expected
        assert processor._parameters == expected
        assert Flex(**expected)._get_parameters() == expected

    def test_execution_flags_preserve_full_context_semantics(self):
        """Template corrections should stream internally in one full context."""
        default_processor = Flex()
        channel_processor = Flex(realign_after_averaging=False)

        assert default_processor.parallel_safe is False
        assert default_processor.channel_wise is False
        assert channel_processor.parallel_safe is False
        assert channel_processor.channel_wise is False


@pytest.mark.unit
class TestFlexAveragingMatrix:
    """Test candidate construction, selection, and distribution weights."""

    @pytest.mark.parametrize(
        ("target_idx", "expected"),
        [
            (0, [1, 2, 3]),
            (1, [2, 3, 4]),
            (2, [3, 4, 5]),
            (3, [2, 4, 5]),
            (4, [2, 3, 5]),
            (5, [2, 3, 4]),
        ],
    )
    def test_candidate_window_is_future_first_then_backfilled(self, target_idx, expected):
        """Only missing future positions should be filled from the recent past."""
        candidates = Flex._candidate_indices(target_idx=target_idx, n_epochs=6, window_size=3)

        np.testing.assert_array_equal(candidates, expected)
        assert target_idx not in candidates

    def test_candidate_window_shrinks_to_all_available_non_target_epochs(self):
        """A requested window larger than the recording should remain self-free."""
        for target_idx in range(4):
            candidates = Flex._candidate_indices(target_idx=target_idx, n_epochs=4, window_size=30)

            np.testing.assert_array_equal(np.sort(candidates), np.delete(np.arange(4), target_idx))

    def test_pearson_correlations_are_signed_and_constant_safe(self):
        """Pearson values should preserve sign and mark undefined inputs unusable."""
        target = np.array([-1.0, -1.0, 1.0, 1.0])
        candidates = np.array(
            [
                3.0 * target + 7.0,
                -2.0 * target,
                [-1.0, 1.0, -1.0, 1.0],
                np.ones_like(target),
            ]
        )

        correlations = Flex._pearson_correlations(target, candidates)

        np.testing.assert_allclose(correlations[:3], [1.0, -1.0, 0.0], atol=1e-15)
        assert correlations[3] == -np.inf

    def test_threshold_keeps_every_match_instead_of_capping_at_minimum(self):
        """``min_accepted`` is a floor, not a maximum selection count."""
        processor = Flex(window_size=5, threshold=0.95, min_accepted=2)
        candidate_indices = np.array([1, 2, 3, 4, 5])
        correlations = np.array([0.99, 0.98, 0.97, 0.40, -0.50])

        selected = processor._select_epoch_indices(
            target_idx=0,
            candidate_indices=candidate_indices,
            correlations=correlations,
            threshold=processor.threshold,
        )

        np.testing.assert_array_equal(selected, [1, 2, 3])

    def test_minimum_fallback_ranks_signed_not_absolute_correlation(self):
        """Below-threshold positive matches should beat strong anticorrelation."""
        processor = Flex(window_size=4, threshold=0.95, min_accepted=3)
        candidate_indices = np.array([1, 2, 3, 4])
        correlations = np.array([-0.90, 0.80, -0.10, 0.40])

        selected = processor._select_epoch_indices(
            target_idx=0,
            candidate_indices=candidate_indices,
            correlations=correlations,
            threshold=processor.threshold,
        )

        np.testing.assert_array_equal(selected, [2, 3, 4])

    def test_minimum_fallback_supplements_existing_threshold_matches(self):
        """Fallback candidates should be added without recounting accepted ones."""
        processor = Flex(window_size=4, threshold=0.95, min_accepted=3)
        candidate_indices = np.array([1, 2, 3, 4])
        correlations = np.array([0.99, 0.80, 0.70, -0.90])

        selected = processor._select_epoch_indices(
            target_idx=0,
            candidate_indices=candidate_indices,
            correlations=correlations,
            threshold=processor.threshold,
        )

        np.testing.assert_array_equal(selected, [1, 2, 3])

    def test_equal_distribution_uses_all_threshold_matches_uniformly(self):
        """Every highly correlated candidate should receive equal mass."""
        base_epoch = np.array([-2.0, -1.0, 1.0, 2.0])
        epochs = np.vstack([(epoch_idx + 1.0) * base_epoch for epoch_idx in range(5)])
        processor = Flex(window_size=3, threshold=0.99, min_accepted=1, N_distribution="equal")

        matrix = _averaging_matrix(processor, epochs)

        for target_idx in range(len(epochs)):
            candidates = Flex._candidate_indices(target_idx, n_epochs=len(epochs), window_size=3)
            expected = np.zeros(len(epochs))
            expected[candidates] = 1.0 / 3.0
            np.testing.assert_allclose(matrix[target_idx], expected)
        np.testing.assert_allclose(np.diag(matrix), 0.0)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)

    def test_normal_distribution_matches_temporal_gaussian_kernel(self):
        """Normal weighting should favor selected epochs nearest the target."""
        base_epoch = np.array([-2.0, -1.0, 1.0, 2.0])
        epochs = np.vstack([(epoch_idx + 1.0) * base_epoch for epoch_idx in range(5)])
        processor = Flex(window_size=3, threshold=0.99, min_accepted=1, N_distribution="normal")

        matrix = _averaging_matrix(processor, epochs)

        # Target 3 uses candidates [1, 2, 4], at temporal distances [2, 1, 1].
        unnormalized = np.exp(-0.5 * (np.array([2.0, 1.0, 1.0]) / 1.0) ** 2)
        expected_weights = unnormalized / unnormalized.sum()
        np.testing.assert_allclose(matrix[3, [1, 2, 4]], expected_weights)
        assert matrix[3, 2] == pytest.approx(matrix[3, 4])
        assert matrix[3, 2] > matrix[3, 1]
        np.testing.assert_allclose(np.diag(matrix), 0.0)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)

    def test_constant_epochs_leave_empty_rows_without_invalid_supplementation(self):
        """Undefined correlations must not pass or supplement the minimum."""
        epochs = np.ones((4, 8), dtype=float)
        processor = Flex(window_size=3, threshold=0.9, min_accepted=2, N_distribution="equal")

        matrix = _averaging_matrix(processor, epochs)

        assert np.all(np.isfinite(matrix))
        np.testing.assert_allclose(np.diag(matrix), 0.0)
        np.testing.assert_array_equal(matrix, np.zeros((4, 4)))

    def test_nonfinite_candidate_correlation_cannot_be_selected(self):
        """A non-finite candidate receives no weight during correlation selection."""
        epochs = np.arange(32, dtype=float).reshape(4, 8)
        epochs[2, 3] = np.nan
        processor = Flex(window_size=3, threshold=0.9, min_accepted=2)

        matrix = _averaging_matrix(processor, epochs)

        assert np.all(np.isfinite(matrix))
        np.testing.assert_array_equal(matrix[:, 2], np.zeros(4))

    def test_empty_and_single_epoch_matrices_have_defined_zero_shape(self):
        """Private matrix construction should remain total at extraction edges."""
        processor = Flex(window_size=3, threshold=0.9, min_accepted=2)

        empty = _averaging_matrix(processor, np.empty((0, 6)))
        single = _averaging_matrix(processor, np.ones((1, 6)))

        assert empty.shape == (0, 0)
        np.testing.assert_array_equal(single, np.zeros((1, 1)))


@pytest.mark.unit
class TestFlexValidation:
    """Test invalid configurations and short recordings."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"window_size": 0, "min_accepted": 1}, "window_size must be >= 1"),
            ({"window_size": 2.5, "min_accepted": 1}, "window_size must be an integer"),
            ({"window_size": np.nan, "min_accepted": 1}, "window_size must be an integer"),
            ({"threshold": 0.0}, "threshold must be in"),
            ({"threshold": 1.01}, "threshold must be in"),
            ({"threshold": np.nan}, "threshold must be in"),
            ({"window_size": 3, "min_accepted": 0}, "min_accepted must be >= 1"),
            ({"min_accepted": 2.5}, "min_accepted must be an integer"),
            ({"min_accepted": np.nan}, "min_accepted must be an integer"),
            ({"window_size": 3, "min_accepted": 4}, "min_accepted cannot exceed window_size"),
            ({"N_distribution": "triangular"}, "N_distribution must be either"),
            ({"search_window_factor": 0.0}, "search_window_factor must be positive"),
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs, message):
        """All public parameter bounds should fail before data is modified."""
        context = _make_flex_context()
        processor = Flex(realign_after_averaging=False, **kwargs)

        with pytest.raises(ProcessorValidationError, match=message):
            processor.validate(context)

    def test_single_trigger_is_rejected_because_self_averaging_is_forbidden(self):
        """A target needs at least one distinct epoch to form its template."""
        context = _make_flex_context(n_triggers=1, n_channels=1)
        processor = Flex(window_size=3, threshold=0.9, min_accepted=2)

        with pytest.raises(ProcessorValidationError, match="at least two triggers"):
            processor.execute(context)

    def test_two_triggers_use_each_other_when_minimum_cannot_be_met(self):
        """Short recordings should use every available non-target epoch."""
        context = _make_flex_context(n_triggers=2, n_channels=1)
        processor = Flex(
            window_size=30,
            threshold=0.99,
            min_accepted=5,
            N_distribution="normal",
            realign_after_averaging=False,
        )

        result = processor.execute(context)
        report = result.metadata.custom["artifact_template_matrices"][0]
        matrix = np.asarray(report["channels"][0]["averaging_matrix_A"]["matrix"])

        np.testing.assert_allclose(matrix, [[0.0, 1.0], [1.0, 0.0]])
        assert np.all(np.isfinite(result.get_raw()._data))
        assert np.all(np.isfinite(result.get_estimated_noise()))

    def test_eog_channels_use_the_same_validated_template_path(self):
        """Validation and processing should agree that EOG is supported."""
        context = _make_flex_context(n_triggers=4, n_channels=1)
        context.get_raw().set_channel_types({"EEG001": "eog"})
        processor = Flex(
            window_size=3,
            threshold=0.9,
            min_accepted=2,
            realign_after_averaging=False,
        )

        result = processor.execute(context)
        report = result.metadata.custom["artifact_template_matrices"][0]

        assert [channel["channel_name"] for channel in report["channels"]] == ["EEG001"]
        assert result.has_estimated_noise()


@pytest.mark.unit
class TestFlexEndToEnd:
    """Test the shared correction path, reporting, and execution modes."""

    def test_execute_is_immutable_and_records_noise_report_and_history(self):
        """Flex should subtract ``N`` on a copy and expose full provenance."""
        context = _make_flex_context(n_triggers=5, n_channels=2)
        input_data = context.get_raw()._data.copy()
        input_metadata = context.metadata.to_dict()
        processor = Flex(
            window_size=3,
            threshold=0.99,
            min_accepted=2,
            N_distribution="equal",
            realign_after_averaging=False,
        )

        result = processor.execute(context)

        assert result is not context
        assert result.get_raw() is not context.get_raw()
        np.testing.assert_array_equal(context.get_raw()._data, input_data)
        assert context.metadata.to_dict() == input_metadata
        assert context.get_history() == []
        assert not context.has_estimated_noise()

        corrected = result.get_raw()._data
        noise = result.get_estimated_noise()
        assert noise.shape == input_data.shape
        assert np.all(np.isfinite(noise))
        assert np.any(np.abs(noise) > 0.0)
        assert not np.array_equal(corrected, input_data)
        np.testing.assert_allclose(corrected + noise, input_data, atol=1e-20)

        history = result.get_history()
        assert len(history) == 1
        assert history[0].name == "flex_correction"
        assert history[0].processor_type == "Flex"
        assert history[0].parameters == processor._get_parameters()

        reports = result.metadata.custom["artifact_template_matrices"]
        assert len(reports) == 1
        report = reports[0]
        assert report["processor_name"] == "flex_correction"
        assert report["processor_type"] == "Flex"
        assert report["parameters"] == processor._get_parameters()
        assert report["matrix_equation"]["equation"] == "N = A @ D"
        assert report["num_triggers"] == 5
        assert report["artifact_length_samples"] == context.get_artifact_length()
        assert [channel["channel_name"] for channel in report["channels"]] == ["EEG001", "EEG002"]

        for channel in report["channels"]:
            averaging_payload = channel["averaging_matrix_A"]
            matrix = np.asarray(averaging_payload["matrix"])
            assert channel["data_matrix_D"]["shape"] == [5, context.get_artifact_length()]
            assert channel["artifact_template_matrix_N"]["shape"] == [5, context.get_artifact_length()]
            assert averaging_payload["storage"] == "dense"
            assert averaging_payload["shape"] == [5, 5]
            np.testing.assert_allclose(np.diag(matrix), 0.0)
            np.testing.assert_allclose(matrix.sum(axis=1), 1.0)

    @pytest.mark.parametrize("interpolate_volume_gaps", [False, True])
    def test_disabling_noise_tracking_preserves_corrected_signal(
        self,
        interpolate_volume_gaps,
    ):
        """Untracked Flex must subtract and interpolate the identical signal."""
        tracked = Flex(
            window_size=3,
            threshold=0.99,
            min_accepted=2,
            realign_after_averaging=False,
            interpolate_volume_gaps=interpolate_volume_gaps,
            track_estimated_noise=True,
        ).execute(_make_flex_context(n_triggers=5, n_channels=2))
        untracked = Flex(
            window_size=3,
            threshold=0.99,
            min_accepted=2,
            realign_after_averaging=False,
            interpolate_volume_gaps=interpolate_volume_gaps,
            track_estimated_noise=False,
        ).execute(_make_flex_context(n_triggers=5, n_channels=2))

        np.testing.assert_array_equal(
            untracked.get_raw()._data,
            tracked.get_raw()._data,
        )
        assert tracked.has_estimated_noise()
        assert not untracked.has_estimated_noise()

    def test_channel_sequential_request_falls_back_to_equivalent_serial_execution(self):
        """Pipeline channel mode should leave full-context Flex results intact."""
        serial_context = _make_flex_context(n_triggers=5, n_channels=2)
        sequential_context = _make_flex_context(n_triggers=5, n_channels=2)
        serial_processor = Flex(
            window_size=3,
            threshold=0.99,
            min_accepted=2,
            N_distribution="normal",
            realign_after_averaging=False,
        )
        sequential_processor = Flex(
            window_size=3,
            threshold=0.99,
            min_accepted=2,
            N_distribution="normal",
            realign_after_averaging=False,
        )

        serial = (
            Pipeline([serial_processor])
            .run(
                initial_context=serial_context,
                channel_sequential=False,
                show_progress=False,
            )
            .context
        )
        sequential = (
            Pipeline([sequential_processor])
            .run(
                initial_context=sequential_context,
                channel_sequential=True,
                show_progress=False,
            )
            .context
        )

        np.testing.assert_allclose(sequential.get_raw()._data, serial.get_raw()._data, atol=1e-20)
        np.testing.assert_allclose(sequential.get_estimated_noise(), serial.get_estimated_noise(), atol=1e-20)
        np.testing.assert_array_equal(sequential.get_triggers(), serial.get_triggers())

        serial_reports = serial.metadata.custom["artifact_template_matrices"]
        sequential_reports = sequential.metadata.custom["artifact_template_matrices"]
        serial_report_channels = [channel for report in serial_reports for channel in report["channels"]]
        sequential_report_channels = [channel for report in sequential_reports for channel in report["channels"]]
        assert [channel["channel_name"] for channel in serial_report_channels] == ["EEG001", "EEG002"]
        assert [channel["channel_name"] for channel in sequential_report_channels] == ["EEG001", "EEG002"]
        assert [channel["channel_index"] for channel in sequential_report_channels] == [0, 1]
