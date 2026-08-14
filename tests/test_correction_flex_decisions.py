"""Tests for the composable Flex averaging-matrix decision graph."""

from __future__ import annotations

import mne
import numpy as np
import pytest

import facet
from facet.core import ProcessingContext, ProcessingMetadata, ProcessorValidationError
from facet.correction import (
    AveragingMatrixBuilder,
    CandidateScoringPolicy,
    CorrelationMode,
    CorrelationPolicy,
    DirectionalQuota,
    Flex,
    MatrixDecisionError,
    MatrixDecisions,
    MatrixMetadata,
    MotionEligibility,
    MotionEpochMetadata,
    SamplingPolicy,
    TargetPolicy,
    TemplateSizePolicy,
    TemporalDistanceUnit,
    WeightingBasis,
    WeightingPolicy,
)
from facet.correction.flex.decisions import (
    pearson_scores,
    select_by_correlation,
    select_scored_candidates,
)
from facet.correction.presets import (
    AAS_PER_TARGET,
    FARM_PER_TARGET_K10,
    FLEX_DEFAULT,
    MOOSMANN_COST,
    STRUCTURAL_SLICE,
    STRUCTURAL_VOLUME,
    build_flex_preset,
)


def _decisions(
    *,
    quota: DirectionalQuota,
    sampling: SamplingPolicy | None = None,
    motion: MotionEligibility | None = None,
    target_policy: TargetPolicy = TargetPolicy.EXCLUDE,
    correlation: CorrelationPolicy | None = None,
    weighting: WeightingPolicy | None = None,
) -> MatrixDecisions:
    """Build a concise recipe with selection gates disabled by default."""
    return MatrixDecisions(
        quota=quota,
        sampling=sampling or SamplingPolicy.consecutive(),
        motion=motion or MotionEligibility(),
        target_policy=target_policy,
        correlation=correlation or CorrelationPolicy.none(),
        weighting=weighting or WeightingPolicy.equal(),
    )


def _pool(
    decisions: MatrixDecisions,
    *,
    target: int,
    n_epochs: int,
    metadata: MatrixMetadata | None = None,
) -> np.ndarray:
    """Return one public candidate-pool result."""
    return AveragingMatrixBuilder(decisions).candidate_pool(
        target_idx=target,
        n_epochs=n_epochs,
        metadata=metadata,
    )


def _scaled_epochs(n_epochs: int) -> np.ndarray:
    """Return non-constant epochs with perfect positive correlations."""
    base = np.array([-2.0, -1.0, 1.0, 2.0])
    return np.vstack([(index + 1.0) * base for index in range(n_epochs)])


@pytest.mark.unit
class TestNamedLegacyPresets:
    """Lock the decision manifests promised by the CLI and HTML report."""

    @staticmethod
    def _base_manifest() -> dict:
        return {
            "motion": {
                "max_motion_distance": None,
                "motion_stable_only": False,
                "same_motion_segment": False,
            },
            "quota": {"future": 10, "global_mode": False, "past": 0, "window_size": 10},
            "sampling": {"mode": "consecutive", "start_offset": 1, "stride": 1},
            "scoring": {
                "mode": "signed_pearson",
                "motion_weight": None,
                "temporal_unit": None,
                "temporal_weight": None,
                "threshold": 0.975,
            },
            "target_policy": "exclude_target",
            "template_size": {"k": 5, "mode": "minimum_k"},
            "weighting": {
                "basis": None,
                "degrees_of_freedom": None,
                "kernel": "equal",
                "scale": None,
                "sigma": None,
                "temporal_unit": None,
            },
        }

    def test_flex_and_aas_manifests(self):
        """AAS differs from default Flex only by including the target."""
        expected = self._base_manifest()
        assert build_flex_preset(FLEX_DEFAULT).to_dict() == expected

        aas_expected = {**expected, "target_policy": "include_target"}
        assert build_flex_preset(AAS_PER_TARGET).to_dict() == aas_expected

    def test_farm_manifest(self):
        """FARM uses a symmetric pool, absolute scores, and at most ten rows."""
        expected = self._base_manifest()
        expected.update(
            quota={"future": 15, "global_mode": False, "past": 15, "window_size": 30},
            scoring={
                "mode": "absolute_pearson",
                "motion_weight": None,
                "temporal_unit": None,
                "temporal_weight": None,
                "threshold": 0.9,
            },
            template_size={"k": 10, "mode": "maximum_k"},
        )
        assert build_flex_preset(FARM_PER_TARGET_K10).to_dict() == expected

    @pytest.mark.parametrize(
        ("preset", "quota", "sampling", "target_policy"),
        [
            (
                STRUCTURAL_VOLUME,
                {"future": 4, "global_mode": False, "past": 6, "window_size": 10},
                {"mode": "consecutive", "start_offset": 1, "stride": 1},
                "include_target",
            ),
            (
                STRUCTURAL_SLICE,
                {"future": 10, "global_mode": False, "past": 0, "window_size": 10},
                {"mode": "alternating", "start_offset": 1, "stride": 2},
                "exclude_target",
            ),
        ],
    )
    def test_structural_manifests(self, preset, quota, sampling, target_policy):
        """Structural presets select every candidate without correlation scoring."""
        expected = self._base_manifest()
        expected.update(
            quota=quota,
            sampling=sampling,
            scoring={
                "mode": "none",
                "motion_weight": None,
                "temporal_unit": None,
                "temporal_weight": None,
                "threshold": None,
            },
            target_policy=target_policy,
            template_size={"k": None, "mode": "select_all"},
        )
        assert build_flex_preset(preset).to_dict() == expected

    def test_moosmann_manifest_is_motion_aware(self):
        """The Moosmann approximation should rank a global stable-motion pool."""
        manifest = build_flex_preset(MOOSMANN_COST).to_dict()

        assert manifest["quota"]["global_mode"] is True
        assert manifest["motion"]["motion_stable_only"] is True
        assert manifest["scoring"] == {
            "mode": "temporal_motion_cost",
            "motion_weight": 1.0,
            "temporal_unit": "epoch_index",
            "temporal_weight": 1.0,
            "threshold": None,
        }
        assert manifest["template_size"] == {"k": 60, "mode": "exactly_k"}


@pytest.mark.unit
class TestDirectionalCandidatePools:
    """Exercise quota, completion, and structural sampling independently."""

    @pytest.mark.parametrize(
        ("quota", "expected"),
        [
            (DirectionalQuota.future_only(4), [5, 6, 7, 8]),
            (DirectionalQuota.past_only(4), [0, 1, 2, 3]),
            (DirectionalQuota.symmetric(4), [2, 3, 5, 6]),
            (DirectionalQuota.custom(past=1, future=3, window_size=4), [3, 5, 6, 7]),
            (DirectionalQuota.custom(past=3, future=1, window_size=4), [1, 2, 3, 5]),
        ],
    )
    def test_finite_directional_quotas(self, quota, expected):
        """P and F should count sampled candidates from their own directions."""
        decisions = _decisions(quota=quota)

        np.testing.assert_array_equal(_pool(decisions, target=4, n_epochs=9), expected)

    def test_one_third_helpers_are_complementary_and_sum_to_window(self):
        """The two default asymmetric options should preserve the total quota."""
        past_heavy = DirectionalQuota.past_heavy(10)
        future_heavy = DirectionalQuota.future_heavy(10)

        assert (past_heavy.past, past_heavy.future) == (7, 3)
        assert (future_heavy.past, future_heavy.future) == (3, 7)

    def test_boundary_completion_fills_from_the_opposite_side(self):
        """A side deficit should be filled without changing the total finite quota."""
        decisions = _decisions(quota=DirectionalQuota.symmetric(4))

        np.testing.assert_array_equal(_pool(decisions, target=1, n_epochs=8), [0, 2, 3, 4])

    def test_global_mode_ignores_window_and_returns_all_sampled_candidates(self):
        """Global consecutive sampling should include every non-target epoch."""
        decisions = _decisions(quota=DirectionalQuota.global_pool())

        np.testing.assert_array_equal(_pool(decisions, target=3, n_epochs=7), [0, 1, 2, 4, 5, 6])

    def test_alternating_quota_counts_alternating_epochs_not_raw_offsets(self):
        """(0, 3) plus stride two means three forward offsets: 1, 3, and 5."""
        decisions = _decisions(
            quota=DirectionalQuota.future_only(3),
            sampling=SamplingPolicy.alternating(),
        )

        np.testing.assert_array_equal(_pool(decisions, target=2, n_epochs=10), [3, 5, 7])

    def test_alternating_global_mode_retains_the_full_sampling_lattice(self):
        """Global changes the quota, not the chosen alternating topology."""
        decisions = _decisions(
            quota=DirectionalQuota.global_pool(),
            sampling=SamplingPolicy.alternating(),
        )

        np.testing.assert_array_equal(_pool(decisions, target=4, n_epochs=12), [1, 3, 5, 7, 9, 11])

    def test_same_slice_quota_counts_matching_phases_not_adjacent_epochs(self):
        """Three future candidates should span three volumes at one slice phase."""
        decisions = _decisions(
            quota=DirectionalQuota.future_only(3),
            sampling=SamplingPolicy.same_slice_phase(),
        )
        metadata = MatrixMetadata(slices_per_volume=3)

        np.testing.assert_array_equal(
            _pool(decisions, target=1, n_epochs=14, metadata=metadata),
            [4, 7, 10],
        )

    @pytest.mark.parametrize(
        ("sampling", "expected"),
        [
            (SamplingPolicy.consecutive(), [3, 4, 5]),
            (SamplingPolicy.alternating(), [4, 6]),
            (SamplingPolicy.same_slice_phase(), [0, 3, 6]),
        ],
    )
    def test_target_is_included_only_when_sampling_naturally_contains_it(self, sampling, expected):
        """Inclusion must not force offset zero into the alternating lattice."""
        decisions = _decisions(
            quota=DirectionalQuota.future_only(2),
            sampling=sampling,
            target_policy=TargetPolicy.INCLUDE,
        )
        metadata = MatrixMetadata(slices_per_volume=3) if sampling == SamplingPolicy.same_slice_phase() else None

        np.testing.assert_array_equal(
            _pool(decisions, target=3, n_epochs=9, metadata=metadata),
            expected,
        )


@pytest.mark.unit
class TestMotionEligibility:
    """Verify that motion constraints filter before directional quota counting."""

    def test_directional_quota_is_counted_after_stability_filtering(self):
        """Eligible epochs farther away should fill skipped unstable positions."""
        decisions = _decisions(
            quota=DirectionalQuota.future_only(3),
            motion=MotionEligibility(motion_stable_only=True),
        )
        metadata = MatrixMetadata(
            motion=MotionEpochMetadata(stable=np.array([True, True, True, True, False, True, False, True]))
        )

        np.testing.assert_array_equal(
            _pool(decisions, target=3, n_epochs=8, metadata=metadata),
            [2, 5, 7],
        )

    def test_all_enabled_motion_conditions_are_intersected(self):
        """Segment, stability, and distance are independent AND constraints."""
        parameters = np.column_stack((np.arange(6, dtype=float) / 10.0, np.zeros((6, 2))))
        decisions = _decisions(
            quota=DirectionalQuota.global_pool(),
            motion=MotionEligibility(
                same_motion_segment=True,
                motion_stable_only=True,
                max_motion_distance=0.25,
            ),
        )
        metadata = MatrixMetadata(
            motion=MotionEpochMetadata(
                parameters=parameters,
                segment_ids=np.array([0, 0, 0, 0, 1, 1]),
                stable=np.array([True, False, True, True, True, True]),
            )
        )

        np.testing.assert_array_equal(
            _pool(decisions, target=2, n_epochs=6, metadata=metadata),
            [0, 3],
        )

    def test_volume_motion_rows_require_and_honor_explicit_epoch_mapping(self):
        """Volume parameters should map predictably to their slice-artifact epochs."""
        motion = MotionEpochMetadata.from_volume_parameters(
            parameters=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            n_artifact_epochs=6,
            slices_per_volume=3,
            segment_ids=np.array([0, 1]),
        )

        resolved = motion.resolve(6)

        np.testing.assert_array_equal(resolved.segment_ids, [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(resolved.parameters[:, 0], [0, 0, 0, 1, 1, 1])

    def test_motion_constraint_requires_corresponding_metadata(self):
        """Motion incompatibility must fail clearly rather than mimic low correlation."""
        decisions = _decisions(
            quota=DirectionalQuota.future_only(2),
            motion=MotionEligibility(same_motion_segment=True),
        )

        with pytest.raises(MatrixDecisionError, match="motion decisions require"):
            _pool(decisions, target=0, n_epochs=4)


@pytest.mark.unit
class TestCorrelationSelection:
    """Exercise correlation features and minimum supplementation semantics."""

    def test_signed_and_absolute_pearson_open_distinct_candidate_sets(self):
        """Absolute correlation may retain a strong inverted artifact waveform."""
        target = np.array([-1.0, -1.0, 1.0, 1.0])
        epochs = np.vstack((target, 2.0 * target, -3.0 * target, [-1.0, 1.0, -1.0, 1.0]))
        quota = DirectionalQuota.future_only(3)
        signed = _decisions(
            quota=quota,
            correlation=CorrelationPolicy(CorrelationMode.SIGNED, threshold=0.9, min_accepted=1),
        )
        absolute = _decisions(
            quota=quota,
            correlation=CorrelationPolicy(CorrelationMode.ABSOLUTE, threshold=0.9, min_accepted=1),
        )

        signed_matrix = AveragingMatrixBuilder(signed).build(epochs)
        absolute_matrix = AveragingMatrixBuilder(absolute).build(epochs)

        np.testing.assert_array_equal(np.flatnonzero(signed_matrix[0]), [1])
        np.testing.assert_array_equal(np.flatnonzero(absolute_matrix[0]), [1, 2])

    def test_threshold_acceptance_is_not_capped_by_minimum(self):
        """Every threshold match should survive when more than ma are accepted."""
        decisions = _decisions(
            quota=DirectionalQuota.future_only(4),
            correlation=CorrelationPolicy(CorrelationMode.SIGNED, threshold=0.99, min_accepted=2),
        )

        matrix = AveragingMatrixBuilder(decisions).build(_scaled_epochs(5))

        np.testing.assert_array_equal(np.flatnonzero(matrix[0]), [1, 2, 3, 4])

    def test_invalid_scores_never_pass_or_supplement_minimum(self):
        """NaN and constant epochs should receive the unusable -inf sentinel."""
        target = np.array([-1.0, 0.0, 1.0])
        candidates = np.array(
            [
                target,
                np.ones(3),
                [np.nan, 0.0, 1.0],
                -target,
            ]
        )
        scores = pearson_scores(target, candidates, CorrelationMode.SIGNED)
        selected = select_by_correlation(
            np.array([7, 2, 4, 6]),
            scores,
            CorrelationPolicy(CorrelationMode.SIGNED, threshold=0.95, min_accepted=3),
        )

        np.testing.assert_allclose(scores[[0, 3]], [1.0, -1.0])
        assert scores[1] == scores[2] == -np.inf
        np.testing.assert_array_equal(selected, [7, 6])

    def test_score_ties_are_supplemented_by_chronological_index(self):
        """Equal rejected scores should use epoch index as the stable tie-break."""
        selected = select_by_correlation(
            np.array([5, 2, 4]),
            np.array([0.5, 0.5, 0.5]),
            CorrelationPolicy(CorrelationMode.SIGNED, threshold=0.9, min_accepted=2),
        )

        np.testing.assert_array_equal(np.sort(selected), [2, 4])

    def test_no_correlation_selects_all_candidates_without_gate(self):
        """The no-feature branch should not apply ct or ma implicitly."""
        decisions = _decisions(quota=DirectionalQuota.future_only(3))
        epochs = np.ones((4, 5))

        matrix = AveragingMatrixBuilder(decisions).build(epochs)

        np.testing.assert_array_equal(np.flatnonzero(matrix[0]), [1, 2, 3])


@pytest.mark.unit
class TestRevisedScoringAndTemplateSize:
    """Exercise every valid scoring/cardinality edge in the revised graph."""

    @staticmethod
    def _select(
        scores: np.ndarray,
        template_size: TemplateSizePolicy,
        *,
        scoring: CandidateScoringPolicy | None = None,
        candidates: np.ndarray | None = None,
        target_idx: int = 0,
    ) -> np.ndarray:
        """Select from a small deterministic pool without constructing D."""
        if candidates is None:
            candidates = np.arange(1, len(scores) + 1)
        if scoring is None:
            scoring = CandidateScoringPolicy.signed_pearson(0.9)
        return select_scored_candidates(
            candidates,
            scores,
            target_idx=target_idx,
            scoring=scoring,
            template_size=template_size,
        )

    def test_minimum_k_keeps_all_accepted_and_supplements_only_when_needed(self):
        """minimum_k is a floor and never truncates threshold matches."""
        all_accepted = self._select(
            np.array([0.99, 0.98, 0.97, 0.96]),
            TemplateSizePolicy.minimum(2),
        )
        supplemented = self._select(
            np.array([0.99, 0.89, 0.70, 0.80]),
            TemplateSizePolicy.minimum(3),
        )

        np.testing.assert_array_equal(all_accepted, [1, 2, 3, 4])
        np.testing.assert_array_equal(supplemented, [1, 2, 4])

    def test_maximum_k_caps_accepted_without_supplementing_rejected(self):
        """maximum_k is a ceiling over correlation-threshold matches."""
        capped = self._select(
            np.array([0.91, 0.99, 0.95]),
            TemplateSizePolicy.maximum(2),
        )
        below_cap = self._select(
            np.array([0.99, 0.70, 0.80]),
            TemplateSizePolicy.maximum(5),
        )

        np.testing.assert_array_equal(capped, [2, 3])
        np.testing.assert_array_equal(below_cap, [1])

    def test_exactly_k_caps_or_supplements_in_score_order(self):
        """exactly_k should approach k from either side of the threshold."""
        selected = self._select(
            np.array([0.99, 0.50, 0.80, 0.40]),
            TemplateSizePolicy.exactly(3),
        )

        np.testing.assert_array_equal(selected, [1, 3, 2])

    def test_ranking_ties_use_temporal_distance_then_candidate_index(self):
        """The graph's two explicit tie-breakers should be stable."""
        selected = self._select(
            np.full(4, 0.99),
            TemplateSizePolicy.exactly(4),
            candidates=np.array([1, 8, 2, 9]),
            target_idx=5,
        )

        np.testing.assert_array_equal(selected, [2, 8, 1, 9])

    @pytest.mark.parametrize("k", [5, 10])
    def test_configured_k_values_select_up_to_available_pool(self, k):
        """Both requested experimental template sizes should be first-class."""
        scores = np.linspace(1.0, 0.0, 12)

        selected = self._select(scores, TemplateSizePolicy.exactly(k))

        assert len(selected) == k

    @pytest.mark.parametrize(
        ("template_size", "expected"),
        [
            (TemplateSizePolicy.maximum(2), [2, 8]),
            (TemplateSizePolicy.exactly(3), [2, 8, 0]),
        ],
    )
    def test_temporal_motion_cost_selects_lowest_finite_costs(self, template_size, expected):
        """Cost ranking should use its own direction and exclude invalid costs."""
        selected = self._select(
            np.array([4.0, 1.0, 1.0, np.inf]),
            template_size,
            scoring=CandidateScoringPolicy.temporal_motion_cost(),
            candidates=np.array([0, 2, 8, 9]),
            target_idx=5,
        )

        np.testing.assert_array_equal(selected, expected)

    @pytest.mark.parametrize(
        ("scoring", "template_size"),
        [
            (CandidateScoringPolicy.signed_pearson(), TemplateSizePolicy.select_all()),
            (CandidateScoringPolicy.absolute_pearson(), TemplateSizePolicy.select_all()),
            (CandidateScoringPolicy.temporal_motion_cost(), TemplateSizePolicy.minimum()),
            (CandidateScoringPolicy.temporal_motion_cost(), TemplateSizePolicy.select_all()),
            (CandidateScoringPolicy.none(), TemplateSizePolicy.minimum()),
            (CandidateScoringPolicy.none(), TemplateSizePolicy.maximum()),
            (CandidateScoringPolicy.none(), TemplateSizePolicy.exactly()),
        ],
    )
    def test_invalid_graph_edges_raise_configuration_error(self, scoring, template_size):
        """Unsupported scoring/cardinality pairs must never be reinterpreted."""
        with pytest.raises(MatrixDecisionError, match="supports template-size modes"):
            MatrixDecisions(scoring=scoring, template_size=template_size)

    def test_none_requires_select_all_and_preserves_chronological_order(self):
        """The ungated path should select all without ct or k."""
        selected = select_scored_candidates(
            np.array([8, 2, 5]),
            None,
            target_idx=4,
            scoring=CandidateScoringPolicy.none(),
            template_size=TemplateSizePolicy.select_all(),
        )

        np.testing.assert_array_equal(selected, [2, 5, 8])

    def test_cumulative_motion_and_temporal_terms_are_independently_configurable(self):
        """Motion ranking should integrate the path, not endpoint displacement."""
        metadata = MatrixMetadata(
            motion=MotionEpochMetadata(
                parameters=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [10.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                )
            )
        )
        common = {
            "quota": DirectionalQuota.global_pool(),
            "template_size": TemplateSizePolicy.exactly(1),
            "weighting": WeightingPolicy.equal(),
        }
        temporal_only = MatrixDecisions(
            **common,
            scoring=CandidateScoringPolicy.temporal_motion_cost(
                temporal_weight=1.0,
                motion_weight=0.0,
            ),
        )
        motion_only = MatrixDecisions(
            **common,
            scoring=CandidateScoringPolicy.temporal_motion_cost(
                temporal_weight=0.0,
                motion_weight=1.0,
            ),
        )

        temporal_matrix = AveragingMatrixBuilder(temporal_only).build(_scaled_epochs(4))
        motion_matrix = AveragingMatrixBuilder(motion_only).build(_scaled_epochs(4), metadata)

        # At target 2, epochs 1 and 3 are equally near in time, so index 1
        # wins. Motion instead favors epoch 3: its path motion is zero, while
        # the return-to-pose epoch 0 still has cumulative path motion 20.
        np.testing.assert_array_equal(np.flatnonzero(temporal_matrix[2]), [1])
        np.testing.assert_array_equal(np.flatnonzero(motion_matrix[2]), [3])

    def test_nonfinite_cost_candidate_is_not_used_to_fill_exactly_k(self):
        """An invalid motion path should remain unavailable when k is unmet."""
        decisions = MatrixDecisions(
            quota=DirectionalQuota.global_pool(),
            scoring=CandidateScoringPolicy.temporal_motion_cost(
                temporal_weight=0.0,
                motion_weight=1.0,
            ),
            template_size=TemplateSizePolicy.exactly(3),
        )
        metadata = MatrixMetadata(
            motion=MotionEpochMetadata(
                parameters=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [np.nan, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ]
                )
            )
        )

        matrix = AveragingMatrixBuilder(decisions).build(_scaled_epochs(4), metadata)

        np.testing.assert_array_equal(np.flatnonzero(matrix[2]), [3])

    def test_selection_score_is_not_reused_as_weight(self):
        """Contributor ranking and row coefficients must remain separate stages."""
        decisions = MatrixDecisions(
            quota=DirectionalQuota.future_only(2),
            scoring=CandidateScoringPolicy.signed_pearson(0.5),
            template_size=TemplateSizePolicy.exactly(2),
            weighting=WeightingPolicy.equal(),
        )
        target = np.array([-1.0, 0.0, 1.0])
        epochs = np.vstack((target, target, [-1.0, 0.2, 0.8]))

        matrix = AveragingMatrixBuilder(decisions).build(epochs)

        np.testing.assert_allclose(matrix[0, 1:], [0.5, 0.5])


@pytest.mark.unit
class TestWeightingKernels:
    """Verify every distance/kernel composition and normalized row invariant."""

    @pytest.mark.parametrize(
        ("weighting", "raw_weights"),
        [
            (WeightingPolicy.equal(), np.ones(3)),
            (
                WeightingPolicy.gaussian(sigma=2.0),
                np.exp(-(np.array([1.0, 2.0, 3.0]) ** 2) / (2.0 * 2.0**2)),
            ),
            (
                WeightingPolicy.laplace(scale=2.0),
                np.exp(-np.array([1.0, 2.0, 3.0]) / 2.0),
            ),
            (
                WeightingPolicy.student_t(scale=2.0, degrees_of_freedom=3.0),
                (1.0 + np.array([1.0, 2.0, 3.0]) ** 2 / (3.0 * 2.0**2)) ** -2.0,
            ),
        ],
    )
    def test_temporal_index_kernels_match_their_definitions(self, weighting, raw_weights):
        """Each kernel should normalize its stated formula over selected epochs."""
        decisions = _decisions(
            quota=DirectionalQuota.future_only(3),
            weighting=weighting,
        )

        matrix = AveragingMatrixBuilder(decisions).build(_scaled_epochs(4))

        np.testing.assert_allclose(matrix[0, 1:4], raw_weights / raw_weights.sum())
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
        assert np.all(np.isfinite(matrix))
        assert np.all(matrix >= 0.0)

    def test_trigger_time_can_replace_index_distance_explicitly(self):
        """Reliable trigger times should be opt-in and preserve index defaults."""
        weighting = WeightingPolicy.gaussian(
            sigma=1.0,
            temporal_unit=TemporalDistanceUnit.TIME,
        )
        decisions = _decisions(
            quota=DirectionalQuota.future_only(2),
            weighting=weighting,
        )
        times = np.array([0.0, 0.5, 2.0])

        matrix = AveragingMatrixBuilder(decisions).build(
            _scaled_epochs(3),
            MatrixMetadata(trigger_times=times),
        )

        raw = np.exp(-(np.array([0.5, 2.0]) ** 2) / 2.0)
        np.testing.assert_allclose(matrix[0, 1:], raw / raw.sum())

    def test_motion_distance_can_drive_a_nonuniform_kernel(self):
        """Motion weighting should use candidate-to-target parameter distance."""
        weighting = WeightingPolicy.laplace(
            basis=WeightingBasis.MOTION,
            scale=1.0,
        )
        decisions = _decisions(
            quota=DirectionalQuota.future_only(2),
            weighting=weighting,
        )
        metadata = MatrixMetadata(
            motion=MotionEpochMetadata(
                parameters=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0],
                    ]
                )
            )
        )

        matrix = AveragingMatrixBuilder(decisions).build(_scaled_epochs(3), metadata)

        raw = np.exp(-np.array([1.0, 3.0]))
        np.testing.assert_allclose(matrix[0, 1:], raw / raw.sum())


@pytest.mark.unit
class TestDecisionValidationAndIntegration:
    """Check inactive parameters, serialization, and Flex-only delegation."""

    @pytest.mark.parametrize(
        ("factory", "message"),
        [
            (
                lambda: DirectionalQuota.custom(past=2, future=2, window_size=5),
                r"past \+ future == window_size",
            ),
            (
                lambda: CorrelationPolicy(CorrelationMode.NONE, threshold=0.9, min_accepted=2),
                "inactive when correlation mode is 'none'",
            ),
            (
                lambda: WeightingPolicy(kernel="equal", sigma=1.0),
                "equal weighting ignores distance",
            ),
            (
                lambda: WeightingPolicy.gaussian(sigma=0.0),
                "sigma must be > 0",
            ),
            (
                lambda: WeightingPolicy.laplace(scale=0.0),
                "scale must be > 0",
            ),
            (
                lambda: WeightingPolicy.student_t(scale=1.0, degrees_of_freedom=0.0),
                "degrees_of_freedom must be > 0",
            ),
        ],
    )
    def test_incompatible_and_inactive_parameters_fail_at_construction(self, factory, message):
        """Invalid graph configurations should fail before matrix construction."""
        with pytest.raises(MatrixDecisionError, match=message):
            factory()

    def test_sampling_and_weighting_metadata_requirements_are_explicit(self):
        """Metadata-dependent decisions must not silently fall back to other axes."""
        same_slice = _decisions(
            quota=DirectionalQuota.future_only(2),
            sampling=SamplingPolicy.same_slice_phase(),
        )
        motion_weighted = _decisions(
            quota=DirectionalQuota.future_only(2),
            weighting=WeightingPolicy.gaussian(
                basis=WeightingBasis.MOTION,
                sigma=1.0,
            ),
        )

        with pytest.raises(MatrixDecisionError, match="requires slices_per_volume"):
            AveragingMatrixBuilder(same_slice).build(_scaled_epochs(4))
        with pytest.raises(MatrixDecisionError, match="motion decisions require"):
            AveragingMatrixBuilder(motion_weighted).build(_scaled_epochs(4))

    def test_global_recipe_ignores_legacy_window_size_during_validation(self):
        """The inactive compatibility window must not constrain global mode."""
        raw = mne.io.RawArray(
            np.zeros((1, 20)),
            mne.create_info(["EEG001"], sfreq=100.0, ch_types=["eeg"]),
            verbose=False,
        )
        metadata = ProcessingMetadata(
            triggers=np.array([1, 5, 9]),
            artifact_length=3,
            artifact_to_trigger_offset=0.0,
            upsampling_factor=1,
        )
        context = ProcessingContext(raw=raw, raw_original=raw.copy(), metadata=metadata)
        decisions = _decisions(quota=DirectionalQuota.global_pool())

        Flex(window_size=0, matrix_decisions=decisions).validate(context)

    def test_complete_recipe_round_trips_through_public_manifest(self):
        """History/config serialization should recreate every active decision."""
        decisions = MatrixDecisions(
            quota=DirectionalQuota.future_heavy(9),
            sampling=SamplingPolicy.alternating(),
            motion=MotionEligibility(max_motion_distance=2.5),
            target_policy=TargetPolicy.INCLUDE,
            correlation=CorrelationPolicy(CorrelationMode.ABSOLUTE, threshold=0.8, min_accepted=3),
            weighting=WeightingPolicy.student_t(
                basis=WeightingBasis.MOTION,
                scale=1.5,
                degrees_of_freedom=4.0,
            ),
        )

        assert MatrixDecisions.from_dict(decisions.to_dict()) == decisions
        assert facet.MatrixDecisions is MatrixDecisions
        assert facet.MOTION_METADATA_KEY == "artifact_epoch_motion"

    def test_first_version_correlation_manifest_migrates_to_revised_stages(self):
        """Stored recipes should retain meaning after scoring/size separation."""
        decisions = MatrixDecisions.from_dict(
            {
                "correlation": {
                    "mode": "absolute",
                    "threshold": 0.8,
                    "min_accepted": 10,
                }
            }
        )

        assert decisions.scoring == CandidateScoringPolicy.absolute_pearson(0.8)
        assert decisions.template_size == TemplateSizePolicy.minimum(10)
        assert "correlation" not in decisions.to_dict()

    def test_flex_delegates_only_matrix_construction_and_reports_recipe(self):
        """The configurable builder should run through the unchanged Flex lifecycle."""
        artifact_length = 4
        triggers = np.array([2, 8, 14, 20])
        data = np.zeros((1, 26))
        waveform = np.array([-1.0, 0.5, 1.0, -0.5]) * 1e-5
        for index, trigger in enumerate(triggers):
            data[0, trigger : trigger + artifact_length] = (index + 1.0) * waveform
        raw = mne.io.RawArray(
            data,
            mne.create_info(["EEG001"], sfreq=100.0, ch_types=["eeg"]),
            verbose=False,
        )
        metadata = ProcessingMetadata(
            triggers=triggers,
            artifact_length=artifact_length,
            artifact_to_trigger_offset=0.0,
            upsampling_factor=1,
        )
        context = ProcessingContext(raw=raw, raw_original=raw.copy(), metadata=metadata)
        decisions = _decisions(quota=DirectionalQuota.future_only(3))
        processor = Flex(
            window_size=99,
            matrix_decisions=decisions,
            realign_after_averaging=False,
        )

        result = processor.execute(context)
        report = result.metadata.custom["artifact_template_matrices"][0]

        assert processor.window_size == 3
        assert processor._get_parameters()["matrix_decisions"] == decisions.to_dict()
        assert report["matrix_equation"]["equation"] == "N = A @ D"
        assert report["matrix_decisions"] == decisions.to_dict()
        np.testing.assert_allclose(
            np.asarray(report["channels"][0]["averaging_matrix_A"]["matrix"]).sum(axis=1),
            1.0,
        )

    def test_flex_validation_translates_missing_metadata_to_processor_error(self):
        """Context validation should expose graph errors through the processor API."""
        raw = mne.io.RawArray(
            np.zeros((1, 20)),
            mne.create_info(["EEG001"], sfreq=100.0, ch_types=["eeg"]),
            verbose=False,
        )
        metadata = ProcessingMetadata(
            triggers=np.array([1, 5, 9]),
            artifact_length=3,
            artifact_to_trigger_offset=0.0,
            upsampling_factor=1,
        )
        context = ProcessingContext(raw=raw, raw_original=raw.copy(), metadata=metadata)
        decisions = _decisions(
            quota=DirectionalQuota.future_only(2),
            sampling=SamplingPolicy.same_slice_phase(),
        )

        with pytest.raises(ProcessorValidationError, match="requires slices_per_volume"):
            Flex(matrix_decisions=decisions).validate(context)
