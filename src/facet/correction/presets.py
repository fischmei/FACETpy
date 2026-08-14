"""Named Flex decision recipes for established correction strategies.

The command line uses only :class:`~facet.correction.flex.Flex` for scanner
template subtraction.  These recipes describe the closest representation of
the former hard-coded averaging-matrix strategies within Flex's composable
decision space.  Keeping the recipes in one module makes the approximation
explicit, serializable, and easy to report.
"""

from __future__ import annotations

from collections.abc import Callable

from .flex import (
    CandidateScoringPolicy,
    DirectionalQuota,
    MatrixDecisions,
    MotionEligibility,
    SamplingPolicy,
    TargetPolicy,
    TemplateSizePolicy,
    TemporalDistanceUnit,
    WeightingPolicy,
)

FLEX_DEFAULT = "flex_default"
AAS_PER_TARGET = "aas_per_target"
FARM_PER_TARGET_K10 = "farm_per_target_k10"
STRUCTURAL_VOLUME = "structural_volume"
STRUCTURAL_SLICE = "structural_slice"
CORRESPONDING_SLICE = "corresponding_slice"
MOOSMANN_COST = "moosmann_cost"

CLI_CORRECTION_PRESETS = {
    "flex": FLEX_DEFAULT,
    "aas": AAS_PER_TARGET,
    "farm": FARM_PER_TARGET_K10,
    "volume-trigger": STRUCTURAL_VOLUME,
    "slice-trigger": STRUCTURAL_SLICE,
    "corresponding-slice": CORRESPONDING_SLICE,
    "moosmann": MOOSMANN_COST,
}

LEGACY_RESEMBLANCE = {
    FLEX_DEFAULT: "default Flex",
    AAS_PER_TARGET: "AAS (per-target approximation)",
    FARM_PER_TARGET_K10: "FARM (per-target, maximum 10 templates)",
    STRUCTURAL_VOLUME: "volume-trigger structural averaging",
    STRUCTURAL_SLICE: "slice-trigger odd/even structural averaging",
    CORRESPONDING_SLICE: "corresponding-slice averaging across volumes",
    MOOSMANN_COST: "Moosmann motion-informed averaging",
}


def flex_default(
    *,
    window_size: int = 10,
    threshold: float = 0.975,
    template_size: int = 5,
    weighting: WeightingPolicy | None = None,
) -> MatrixDecisions:
    """Return the default future-first Flex recipe."""
    return MatrixDecisions(
        quota=DirectionalQuota.future_only(window_size),
        sampling=SamplingPolicy.consecutive(),
        motion=MotionEligibility(),
        target_policy=TargetPolicy.EXCLUDE,
        scoring=CandidateScoringPolicy.signed_pearson(threshold),
        template_size=TemplateSizePolicy.minimum(template_size),
        weighting=weighting or WeightingPolicy.equal(),
    )


def aas_per_target(
    *,
    window_size: int = 10,
    threshold: float = 0.975,
    template_size: int = 5,
) -> MatrixDecisions:
    """Return the closest per-target Flex representation of legacy AAS."""
    return MatrixDecisions(
        quota=DirectionalQuota.future_only(window_size),
        sampling=SamplingPolicy.consecutive(),
        motion=MotionEligibility(),
        target_policy=TargetPolicy.INCLUDE,
        scoring=CandidateScoringPolicy.signed_pearson(threshold),
        template_size=TemplateSizePolicy.minimum(template_size),
        weighting=WeightingPolicy.equal(),
    )


def farm_per_target_k10(
    *,
    window_size: int = 30,
    threshold: float = 0.9,
    template_size: int = 10,
) -> MatrixDecisions:
    """Return the symmetric absolute-correlation FARM approximation."""
    past = window_size // 2
    return MatrixDecisions(
        quota=DirectionalQuota.custom(
            window_size=window_size,
            past=past,
            future=window_size - past,
        ),
        sampling=SamplingPolicy.consecutive(),
        motion=MotionEligibility(),
        target_policy=TargetPolicy.EXCLUDE,
        scoring=CandidateScoringPolicy.absolute_pearson(threshold),
        template_size=TemplateSizePolicy.maximum(template_size),
        weighting=WeightingPolicy.equal(),
    )


def structural_volume(*, window_size: int = 10) -> MatrixDecisions:
    """Return fixed neighboring-volume structural averaging decisions."""
    # The established ten-candidate recipe is slightly past-heavy (6/4).
    # Preserve that proportion when a caller explicitly changes the window.
    past = min(window_size, int(round(window_size * 0.6)))
    return MatrixDecisions(
        quota=DirectionalQuota.custom(
            window_size=window_size,
            past=past,
            future=window_size - past,
        ),
        sampling=SamplingPolicy.consecutive(),
        motion=MotionEligibility(),
        target_policy=TargetPolicy.INCLUDE,
        scoring=CandidateScoringPolicy.none(),
        template_size=TemplateSizePolicy.select_all(),
        weighting=WeightingPolicy.equal(),
    )


def structural_slice(*, window_size: int = 10) -> MatrixDecisions:
    """Return odd/even slice-trigger structural averaging decisions."""
    return MatrixDecisions(
        quota=DirectionalQuota.future_only(window_size),
        sampling=SamplingPolicy.alternating(),
        motion=MotionEligibility(),
        target_policy=TargetPolicy.EXCLUDE,
        scoring=CandidateScoringPolicy.none(),
        template_size=TemplateSizePolicy.select_all(),
        weighting=WeightingPolicy.equal(),
    )


def corresponding_slice(*, window_size: int = 10) -> MatrixDecisions:
    """Return same-slice-phase averaging across neighboring volumes."""
    past = window_size // 2
    return MatrixDecisions(
        quota=DirectionalQuota.custom(
            window_size=window_size,
            past=past,
            future=window_size - past,
        ),
        sampling=SamplingPolicy.same_slice_phase(),
        motion=MotionEligibility(),
        target_policy=TargetPolicy.INCLUDE,
        scoring=CandidateScoringPolicy.none(),
        template_size=TemplateSizePolicy.select_all(),
        weighting=WeightingPolicy.equal(),
    )


def moosmann_cost(*, template_size: int = 60) -> MatrixDecisions:
    """Return a global motion-path-cost approximation of Moosmann weighting.

    Motion metadata is attached by the CLI adapter.  Candidates marked as a
    motion transition are excluded, and the remaining candidates are ranked
    by temporal distance plus cumulative movement along the acquisition path.
    """
    return MatrixDecisions(
        quota=DirectionalQuota.global_pool(),
        sampling=SamplingPolicy.consecutive(),
        motion=MotionEligibility(motion_stable_only=True),
        target_policy=TargetPolicy.INCLUDE,
        scoring=CandidateScoringPolicy.temporal_motion_cost(
            temporal_weight=1.0,
            motion_weight=1.0,
            temporal_unit=TemporalDistanceUnit.INDEX,
        ),
        template_size=TemplateSizePolicy.exactly(template_size),
        weighting=WeightingPolicy.equal(),
    )


PRESET_BUILDERS: dict[str, Callable[..., MatrixDecisions]] = {
    FLEX_DEFAULT: flex_default,
    AAS_PER_TARGET: aas_per_target,
    FARM_PER_TARGET_K10: farm_per_target_k10,
    STRUCTURAL_VOLUME: structural_volume,
    STRUCTURAL_SLICE: structural_slice,
    CORRESPONDING_SLICE: corresponding_slice,
    MOOSMANN_COST: moosmann_cost,
}


def build_flex_preset(name: str, **overrides) -> MatrixDecisions:
    """Build one named recipe, applying only preset-specific overrides."""
    try:
        builder = PRESET_BUILDERS[name]
    except KeyError as exc:
        choices = ", ".join(PRESET_BUILDERS)
        raise ValueError(f"Unknown Flex preset {name!r}; choose one of: {choices}") from exc
    return builder(**overrides)


__all__ = [
    "AAS_PER_TARGET",
    "CLI_CORRECTION_PRESETS",
    "CORRESPONDING_SLICE",
    "FARM_PER_TARGET_K10",
    "FLEX_DEFAULT",
    "LEGACY_RESEMBLANCE",
    "MOOSMANN_COST",
    "PRESET_BUILDERS",
    "STRUCTURAL_SLICE",
    "STRUCTURAL_VOLUME",
    "aas_per_target",
    "build_flex_preset",
    "corresponding_slice",
    "farm_per_target_k10",
    "flex_default",
    "moosmann_cost",
    "structural_slice",
    "structural_volume",
]
