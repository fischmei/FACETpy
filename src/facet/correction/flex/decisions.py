"""Composable theoretical decisions for constructing averaging matrix ``A``.

This module deliberately knows nothing about named correction algorithms.  It
builds each row of ``A`` from independent decisions about directional quota,
sampling, motion eligibility, self inclusion, candidate scoring, template
size, and distance-based weighting. ``Flex`` owns the surrounding ``D -> A -> N``
correction lifecycle and delegates only matrix construction here.

The decision axes are motivated by the methods decomposed by FACETpy:

* Allen et al. (2000), NeuroImage 12, 230-239,
  https://doi.org/10.1006/nimg.2000.0599.
* Niazy et al. (2005), NeuroImage 28, 720-737,
  https://doi.org/10.1016/j.neuroimage.2005.06.067.
* Moosmann et al. (2009), NeuroImage 45, 1144-1150,
  https://doi.org/10.1016/j.neuroimage.2009.01.024.
* Van der Meer et al. (2010), Clinical Neurophysiology 121, 766-776,
  https://doi.org/10.1016/j.clinph.2009.12.035.
* Glaser et al. (2013), BMC Neuroscience 14, 138,
  https://doi.org/10.1186/1471-2202-14-138.

The citations identify the provenance of the theoretical axes; there are no
named-algorithm branches in the implementation below.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral, Real
from typing import Any

import numpy as np

MOTION_METADATA_KEY = "artifact_epoch_motion"


class MatrixDecisionError(ValueError):
    """Raised when matrix decisions or required epoch metadata are invalid."""


class SamplingMode(StrEnum):
    """Structural relation used while scanning away from a target epoch."""

    CONSECUTIVE = "consecutive"
    ALTERNATING = "alternating"
    SAME_SLICE_PHASE = "same_slice_phase"


class TargetPolicy(StrEnum):
    """Whether a naturally sampled target may contribute to its template."""

    INCLUDE = "include_target"
    EXCLUDE = "exclude_target"


class CandidateScoringMode(StrEnum):
    """Feature used to accept or rank candidate epochs."""

    SIGNED_PEARSON = "signed_pearson"
    ABSOLUTE_PEARSON = "absolute_pearson"
    TEMPORAL_MOTION_COST = "temporal_motion_cost"
    NONE = "none"

    # Compatibility aliases for the first version of the decision framework.
    SIGNED = SIGNED_PEARSON
    ABSOLUTE = ABSOLUTE_PEARSON


# Public compatibility name retained for serialized recipes and callers that
# configured the correlation-only version of this framework.
CorrelationMode = CandidateScoringMode


class TemplateSizeMode(StrEnum):
    """Cardinality rule applied after candidate scoring or acceptance."""

    MINIMUM_K = "minimum_k"
    MAXIMUM_K = "maximum_k"
    EXACTLY_K = "exactly_k"
    SELECT_ALL = "select_all"


class WeightingBasis(StrEnum):
    """Distance source supplied to a non-uniform weighting kernel."""

    TEMPORAL = "temporal_distance"
    MOTION = "motion_distance"


class TemporalDistanceUnit(StrEnum):
    """Coordinate used to calculate temporal candidate-to-target distance."""

    INDEX = "epoch_index"
    TIME = "trigger_time"


class WeightingKernel(StrEnum):
    """Kernel applied to selected candidate distances."""

    EQUAL = "equal"
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    STUDENT_T = "student_t"


def _coerce_enum(value: Any, enum_type: type[StrEnum], field_name: str) -> StrEnum:
    """Return an enum member with a concise error for invalid public input."""
    try:
        return value if isinstance(value, enum_type) else enum_type(str(value))
    except ValueError as exc:
        choices = ", ".join(repr(member.value) for member in enum_type)
        raise MatrixDecisionError(f"{field_name} must be one of {choices}, got {value!r}") from exc


def _coerce_candidate_scoring_mode(value: Any) -> CandidateScoringMode:
    """Coerce revised names while accepting first-version recipe values."""
    legacy_values = {
        "signed": CandidateScoringMode.SIGNED_PEARSON.value,
        "absolute": CandidateScoringMode.ABSOLUTE_PEARSON.value,
    }
    if isinstance(value, str):
        value = legacy_values.get(value, value)
    return _coerce_enum(
        value,
        CandidateScoringMode,
        "candidate scoring mode",
    )


def _require_integer(value: Any, field_name: str, *, minimum: int = 0) -> int:
    """Validate and return a non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise MatrixDecisionError(f"{field_name} must be an integer, got {value!r}")
    resolved = int(value)
    if resolved < minimum:
        raise MatrixDecisionError(f"{field_name} must be >= {minimum}, got {resolved}")
    return resolved


def _require_positive(value: Any, field_name: str, *, allow_zero: bool = False) -> float:
    """Validate and return a finite positive scalar."""
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise MatrixDecisionError(f"{field_name} must be a finite number, got {value!r}")
    resolved = float(value)
    valid = resolved >= 0.0 if allow_zero else resolved > 0.0
    if not valid:
        comparison = ">= 0" if allow_zero else "> 0"
        raise MatrixDecisionError(f"{field_name} must be {comparison}, got {resolved}")
    return resolved


@dataclass(frozen=True)
class DirectionalQuota:
    """Requested non-target candidates from the past and future.

    ``past`` and ``future`` count candidates *after* structural sampling and
    motion eligibility.  If one side cannot meet its quota, the builder takes
    additional eligible candidates from the other side.  Global mode ignores
    all finite quota values and returns every eligible sampled candidate.
    """

    window_size: int | None = 10
    past: int = 0
    future: int = 10
    global_mode: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.global_mode, bool):
            raise MatrixDecisionError(f"global_mode must be a boolean, got {self.global_mode!r}")
        if self.global_mode:
            return

        window_size = _require_integer(self.window_size, "window_size", minimum=1)
        past = _require_integer(self.past, "past", minimum=0)
        future = _require_integer(self.future, "future", minimum=0)
        if past + future != window_size:
            raise MatrixDecisionError(
                "Directional quota must satisfy past + future == window_size, "
                f"got {past} + {future} != {window_size}"
            )

    @classmethod
    def future_only(cls, window_size: int = 10) -> DirectionalQuota:
        """Request every finite candidate from the future first."""
        return cls(window_size=window_size, past=0, future=window_size)

    @classmethod
    def past_only(cls, window_size: int = 10) -> DirectionalQuota:
        """Request every finite candidate from the past first."""
        return cls(window_size=window_size, past=window_size, future=0)

    @classmethod
    def symmetric(cls, window_size: int = 10) -> DirectionalQuota:
        """Split an even candidate quota equally across both directions."""
        window_size = _require_integer(window_size, "window_size", minimum=1)
        if window_size % 2:
            raise MatrixDecisionError("symmetric directional quota requires an even window_size")
        half = window_size // 2
        return cls(window_size=window_size, past=half, future=half)

    @classmethod
    def past_heavy(cls, window_size: int = 10) -> DirectionalQuota:
        """Allocate approximately two thirds of the quota to the past."""
        window_size = _require_integer(window_size, "window_size", minimum=1)
        future = window_size // 3
        return cls(window_size=window_size, past=window_size - future, future=future)

    @classmethod
    def future_heavy(cls, window_size: int = 10) -> DirectionalQuota:
        """Allocate approximately two thirds of the quota to the future."""
        window_size = _require_integer(window_size, "window_size", minimum=1)
        past = window_size // 3
        return cls(window_size=window_size, past=past, future=window_size - past)

    @classmethod
    def custom(cls, *, past: int, future: int, window_size: int = 10) -> DirectionalQuota:
        """Create an arbitrary finite directional allocation."""
        return cls(window_size=window_size, past=past, future=future)

    @classmethod
    def global_pool(cls) -> DirectionalQuota:
        """Return every eligible sampled candidate in the recording."""
        return cls(window_size=None, past=0, future=0, global_mode=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly manifest."""
        return {
            "window_size": self.window_size,
            "past": self.past,
            "future": self.future,
            "global_mode": self.global_mode,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DirectionalQuota:
        """Reconstruct a quota from a serialized manifest."""
        global_mode = value.get("global_mode", False)
        if global_mode:
            window_size = value.get("window_size")
            past = value.get("past", 0)
            future = value.get("future", 0)
        else:
            window_size = _require_integer(value.get("window_size", 10), "window_size", minimum=1)
            past = _require_integer(value.get("past", 0), "past", minimum=0)
            future = value.get("future", window_size - past)
        return cls(
            window_size=window_size,
            past=past,
            future=future,
            global_mode=global_mode,
        )


@dataclass(frozen=True)
class SamplingPolicy:
    """Structural candidate pattern applied independently in each direction."""

    mode: SamplingMode = SamplingMode.CONSECUTIVE
    stride: int | None = 1
    start_offset: int | None = 1

    def __post_init__(self) -> None:
        mode = _coerce_enum(self.mode, SamplingMode, "sampling mode")
        object.__setattr__(self, "mode", mode)

        if mode is SamplingMode.SAME_SLICE_PHASE:
            if self.stride is not None or self.start_offset is not None:
                raise MatrixDecisionError(
                    "same_slice_phase derives stride from slices_per_volume; stride and start_offset must be None"
                )
            return

        stride = _require_integer(self.stride, "stride", minimum=1)
        start_offset = _require_integer(self.start_offset, "start_offset", minimum=1)
        if mode is SamplingMode.CONSECUTIVE and (stride != 1 or start_offset != 1):
            raise MatrixDecisionError("consecutive sampling requires stride=1 and start_offset=1")
        if mode is SamplingMode.ALTERNATING and (stride != 2 or start_offset != 1):
            raise MatrixDecisionError("alternating sampling requires stride=2 and start_offset=1")

    @classmethod
    def consecutive(cls) -> SamplingPolicy:
        """Visit adjacent epochs in each temporal direction."""
        return cls(mode=SamplingMode.CONSECUTIVE, stride=1, start_offset=1)

    @classmethod
    def alternating(cls) -> SamplingPolicy:
        """Visit offsets ``1, 3, 5, ...`` in each direction."""
        return cls(mode=SamplingMode.ALTERNATING, stride=2, start_offset=1)

    @classmethod
    def same_slice_phase(cls) -> SamplingPolicy:
        """Visit the same acquisition phase in neighboring volumes."""
        return cls(mode=SamplingMode.SAME_SLICE_PHASE, stride=None, start_offset=None)

    def naturally_contains_target(self) -> bool:
        """Return whether offset zero belongs to this sampling lattice."""
        return self.mode is not SamplingMode.ALTERNATING

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly manifest."""
        return {
            "mode": self.mode.value,
            "stride": self.stride,
            "start_offset": self.start_offset,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SamplingPolicy:
        """Reconstruct a sampling policy from a serialized manifest."""
        return cls(
            mode=value.get("mode", SamplingMode.CONSECUTIVE.value),
            stride=value.get("stride", 1),
            start_offset=value.get("start_offset", 1),
        )


@dataclass(frozen=True)
class MotionEligibility:
    """Optional motion-derived Boolean constraints on candidate epochs."""

    same_motion_segment: bool = False
    motion_stable_only: bool = False
    max_motion_distance: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.same_motion_segment, bool):
            raise MatrixDecisionError("same_motion_segment must be a boolean")
        if not isinstance(self.motion_stable_only, bool):
            raise MatrixDecisionError("motion_stable_only must be a boolean")
        if self.max_motion_distance is not None:
            resolved = _require_positive(
                self.max_motion_distance,
                "max_motion_distance",
                allow_zero=True,
            )
            object.__setattr__(self, "max_motion_distance", resolved)

    @property
    def enabled(self) -> bool:
        """Return whether any motion eligibility condition is active."""
        return self.same_motion_segment or self.motion_stable_only or self.max_motion_distance is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly manifest."""
        return {
            "same_motion_segment": self.same_motion_segment,
            "motion_stable_only": self.motion_stable_only,
            "max_motion_distance": self.max_motion_distance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MotionEligibility:
        """Reconstruct motion conditions from a serialized manifest."""
        return cls(
            same_motion_segment=value.get("same_motion_segment", False),
            motion_stable_only=value.get("motion_stable_only", False),
            max_motion_distance=value.get("max_motion_distance"),
        )


@dataclass(frozen=True)
class CandidateScoringPolicy:
    """Candidate feature, acceptance threshold, and cost composition.

    Pearson modes use ``threshold`` and rank larger scores first. The
    temporal-motion mode has no acceptance gate and ranks the lowest combined
    cost first. Its two nonnegative component weights remain independent so
    experiments can vary their relative contribution without changing the
    later template-weighting policy.
    """

    mode: CandidateScoringMode = CandidateScoringMode.SIGNED_PEARSON
    threshold: float | None = 0.975
    temporal_weight: float | None = None
    motion_weight: float | None = None
    temporal_unit: TemporalDistanceUnit | None = None

    def __post_init__(self) -> None:
        mode = _coerce_candidate_scoring_mode(self.mode)
        object.__setattr__(self, "mode", mode)

        if mode in {
            CandidateScoringMode.SIGNED_PEARSON,
            CandidateScoringMode.ABSOLUTE_PEARSON,
        }:
            if any(
                value is not None
                for value in (self.temporal_weight, self.motion_weight, self.temporal_unit)
            ):
                raise MatrixDecisionError(
                    "Pearson scoring does not use temporal_weight, motion_weight, or temporal_unit"
                )
            if isinstance(self.threshold, bool) or not isinstance(self.threshold, Real):
                raise MatrixDecisionError(f"threshold must be a finite number, got {self.threshold!r}")
            threshold = float(self.threshold)
            if not np.isfinite(threshold):
                raise MatrixDecisionError(f"threshold must be finite, got {self.threshold!r}")
            lower = 0.0 if mode is CandidateScoringMode.ABSOLUTE_PEARSON else -1.0
            if not lower <= threshold <= 1.0:
                raise MatrixDecisionError(
                    f"threshold for {mode.value} must be in [{lower}, 1], got {threshold}"
                )
            object.__setattr__(self, "threshold", threshold)
            return

        if self.threshold is not None:
            raise MatrixDecisionError(
                f"threshold is inactive when candidate scoring mode is {mode.value!r}; set it to None"
            )

        if mode is CandidateScoringMode.NONE:
            if any(
                value is not None
                for value in (self.temporal_weight, self.motion_weight, self.temporal_unit)
            ):
                raise MatrixDecisionError(
                    "scoring mode 'none' does not use temporal_weight, motion_weight, or temporal_unit"
                )
            return

        temporal_weight = _require_positive(
            self.temporal_weight,
            "temporal_weight",
            allow_zero=True,
        )
        motion_weight = _require_positive(
            self.motion_weight,
            "motion_weight",
            allow_zero=True,
        )
        if temporal_weight == 0.0 and motion_weight == 0.0:
            raise MatrixDecisionError(
                "temporal_motion_cost requires temporal_weight or motion_weight to be greater than zero"
            )
        temporal_unit = _coerce_enum(
            self.temporal_unit,
            TemporalDistanceUnit,
            "temporal distance unit",
        )
        object.__setattr__(self, "temporal_weight", temporal_weight)
        object.__setattr__(self, "motion_weight", motion_weight)
        object.__setattr__(self, "temporal_unit", temporal_unit)

    @classmethod
    def signed_pearson(cls, threshold: float = 0.975) -> CandidateScoringPolicy:
        """Rank signed Pearson scores from highest to lowest."""
        return cls(mode=CandidateScoringMode.SIGNED_PEARSON, threshold=threshold)

    @classmethod
    def absolute_pearson(cls, threshold: float = 0.975) -> CandidateScoringPolicy:
        """Rank absolute Pearson scores from highest to lowest."""
        return cls(mode=CandidateScoringMode.ABSOLUTE_PEARSON, threshold=threshold)

    @classmethod
    def temporal_motion_cost(
        cls,
        *,
        temporal_weight: float = 1.0,
        motion_weight: float = 1.0,
        temporal_unit: TemporalDistanceUnit = TemporalDistanceUnit.INDEX,
    ) -> CandidateScoringPolicy:
        """Rank the lowest temporal plus cumulative-motion costs first."""
        return cls(
            mode=CandidateScoringMode.TEMPORAL_MOTION_COST,
            threshold=None,
            temporal_weight=temporal_weight,
            motion_weight=motion_weight,
            temporal_unit=temporal_unit,
        )

    @classmethod
    def none(cls) -> CandidateScoringPolicy:
        """Preserve eligible candidates without calculating a score."""
        return cls(mode=CandidateScoringMode.NONE, threshold=None)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly manifest."""
        return {
            "mode": self.mode.value,
            "threshold": self.threshold,
            "temporal_weight": self.temporal_weight,
            "motion_weight": self.motion_weight,
            "temporal_unit": self.temporal_unit.value if self.temporal_unit is not None else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateScoringPolicy:
        """Reconstruct candidate scoring from a serialized manifest."""
        mode = value.get("mode", CandidateScoringMode.SIGNED_PEARSON.value)
        if mode == CandidateScoringMode.NONE.value:
            return cls.none()
        if mode == CandidateScoringMode.TEMPORAL_MOTION_COST.value:
            return cls.temporal_motion_cost(
                temporal_weight=value.get("temporal_weight", 1.0),
                motion_weight=value.get("motion_weight", 1.0),
                temporal_unit=value.get("temporal_unit", TemporalDistanceUnit.INDEX.value),
            )
        return cls(mode=mode, threshold=value.get("threshold", 0.975))


@dataclass(frozen=True)
class TemplateSizePolicy:
    """Minimum, maximum, exact, or unrestricted template cardinality."""

    mode: TemplateSizeMode = TemplateSizeMode.MINIMUM_K
    k: int | None = 5

    def __post_init__(self) -> None:
        mode = _coerce_enum(self.mode, TemplateSizeMode, "template-size mode")
        object.__setattr__(self, "mode", mode)
        if mode is TemplateSizeMode.SELECT_ALL:
            if self.k is not None:
                raise MatrixDecisionError("k is inactive when template-size mode is 'select_all'; set it to None")
            return
        object.__setattr__(self, "k", _require_integer(self.k, "k", minimum=1))

    @classmethod
    def minimum(cls, k: int = 5) -> TemplateSizePolicy:
        """Keep all accepted candidates and supplement until at least ``k``."""
        return cls(mode=TemplateSizeMode.MINIMUM_K, k=k)

    @classmethod
    def maximum(cls, k: int = 5) -> TemplateSizePolicy:
        """Retain at most the best ``k`` accepted or cost-ranked candidates."""
        return cls(mode=TemplateSizeMode.MAXIMUM_K, k=k)

    @classmethod
    def exactly(cls, k: int = 5) -> TemplateSizePolicy:
        """Retain exactly the best ``k`` valid candidates when available."""
        return cls(mode=TemplateSizeMode.EXACTLY_K, k=k)

    @classmethod
    def select_all(cls) -> TemplateSizePolicy:
        """Select every eligible candidate without scoring or truncation."""
        return cls(mode=TemplateSizeMode.SELECT_ALL, k=None)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly manifest."""
        return {"mode": self.mode.value, "k": self.k}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TemplateSizePolicy:
        """Reconstruct a template-size policy from a serialized manifest."""
        mode = value.get("mode", TemplateSizeMode.MINIMUM_K.value)
        if mode == TemplateSizeMode.SELECT_ALL.value:
            return cls.select_all()
        return cls(mode=mode, k=value.get("k", 5))


@dataclass(frozen=True)
class CorrelationPolicy:
    """Compatibility adapter for correlation plus ``minimum_k`` selection.

    New recipes should use :class:`CandidateScoringPolicy` and
    :class:`TemplateSizePolicy` separately. This adapter preserves manifests
    and call sites created by the first composable-framework version.
    """

    mode: CandidateScoringMode = CandidateScoringMode.SIGNED_PEARSON
    threshold: float | None = 0.975
    min_accepted: int | None = 5

    def __post_init__(self) -> None:
        mode = _coerce_candidate_scoring_mode(self.mode)
        if mode is CandidateScoringMode.TEMPORAL_MOTION_COST:
            raise MatrixDecisionError("CorrelationPolicy cannot represent temporal_motion_cost")
        object.__setattr__(self, "mode", mode)

        if mode is CandidateScoringMode.NONE:
            if self.threshold is not None or self.min_accepted is not None:
                raise MatrixDecisionError(
                    "threshold and min_accepted are inactive when correlation mode is 'none'; set both to None"
                )
            return

        scoring = CandidateScoringPolicy(mode=mode, threshold=self.threshold)
        object.__setattr__(self, "threshold", scoring.threshold)
        object.__setattr__(
            self,
            "min_accepted",
            _require_integer(self.min_accepted, "min_accepted", minimum=1),
        )

    @classmethod
    def none(cls) -> CorrelationPolicy:
        """Create the legacy no-correlation/select-all combination."""
        return cls(mode=CandidateScoringMode.NONE, threshold=None, min_accepted=None)

    def to_decisions(self) -> tuple[CandidateScoringPolicy, TemplateSizePolicy]:
        """Translate the compatibility policy into the two revised stages."""
        if self.mode is CandidateScoringMode.NONE:
            return CandidateScoringPolicy.none(), TemplateSizePolicy.select_all()
        return (
            CandidateScoringPolicy(mode=self.mode, threshold=self.threshold),
            TemplateSizePolicy.minimum(int(self.min_accepted)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the legacy JSON-friendly manifest."""
        return {
            "mode": self.mode.value,
            "threshold": self.threshold,
            "min_accepted": self.min_accepted,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CorrelationPolicy:
        """Reconstruct a legacy correlation policy."""
        mode = value.get("mode", CandidateScoringMode.SIGNED_PEARSON.value)
        if mode == CandidateScoringMode.NONE.value:
            return cls.none()
        return cls(
            mode=mode,
            threshold=value.get("threshold", 0.975),
            min_accepted=value.get("min_accepted", 5),
        )


@dataclass(frozen=True)
class WeightingPolicy:
    """Distance basis and normalized kernel used after candidate selection."""

    kernel: WeightingKernel = WeightingKernel.EQUAL
    basis: WeightingBasis | None = None
    temporal_unit: TemporalDistanceUnit | None = None
    sigma: float | None = None
    scale: float | None = None
    degrees_of_freedom: float | None = None

    def __post_init__(self) -> None:
        kernel = _coerce_enum(self.kernel, WeightingKernel, "weighting kernel")
        object.__setattr__(self, "kernel", kernel)

        if kernel is WeightingKernel.EQUAL:
            inactive = (self.basis, self.temporal_unit, self.sigma, self.scale, self.degrees_of_freedom)
            if any(value is not None for value in inactive):
                raise MatrixDecisionError(
                    "equal weighting ignores distance and kernel parameters; basis, temporal_unit, sigma, "
                    "scale, and degrees_of_freedom must be None"
                )
            return

        if self.basis is None:
            raise MatrixDecisionError(f"{kernel.value} weighting requires a distance basis")
        basis = _coerce_enum(self.basis, WeightingBasis, "weighting basis")
        object.__setattr__(self, "basis", basis)

        if basis is WeightingBasis.TEMPORAL:
            if self.temporal_unit is None:
                raise MatrixDecisionError("temporal weighting requires temporal_unit")
            temporal_unit = _coerce_enum(self.temporal_unit, TemporalDistanceUnit, "temporal distance unit")
            object.__setattr__(self, "temporal_unit", temporal_unit)
        elif self.temporal_unit is not None:
            raise MatrixDecisionError("temporal_unit is inactive when weighting basis is motion_distance")

        if kernel is WeightingKernel.GAUSSIAN:
            object.__setattr__(self, "sigma", _require_positive(self.sigma, "sigma"))
            if self.scale is not None or self.degrees_of_freedom is not None:
                raise MatrixDecisionError("Gaussian weighting uses sigma only; scale and degrees_of_freedom are inactive")
        elif kernel is WeightingKernel.LAPLACE:
            object.__setattr__(self, "scale", _require_positive(self.scale, "scale"))
            if self.sigma is not None or self.degrees_of_freedom is not None:
                raise MatrixDecisionError("Laplace weighting uses scale only; sigma and degrees_of_freedom are inactive")
        else:
            object.__setattr__(self, "scale", _require_positive(self.scale, "scale"))
            object.__setattr__(
                self,
                "degrees_of_freedom",
                _require_positive(self.degrees_of_freedom, "degrees_of_freedom"),
            )
            if self.sigma is not None:
                raise MatrixDecisionError("Student-t weighting uses scale and degrees_of_freedom; sigma is inactive")

    @classmethod
    def equal(cls) -> WeightingPolicy:
        """Assign every selected candidate equal raw weight."""
        return cls(kernel=WeightingKernel.EQUAL)

    @classmethod
    def gaussian(
        cls,
        *,
        basis: WeightingBasis = WeightingBasis.TEMPORAL,
        sigma: float,
        temporal_unit: TemporalDistanceUnit | None = TemporalDistanceUnit.INDEX,
    ) -> WeightingPolicy:
        """Create a Gaussian distance kernel."""
        if basis == WeightingBasis.MOTION:
            temporal_unit = None
        return cls(kernel=WeightingKernel.GAUSSIAN, basis=basis, temporal_unit=temporal_unit, sigma=sigma)

    @classmethod
    def laplace(
        cls,
        *,
        basis: WeightingBasis = WeightingBasis.TEMPORAL,
        scale: float,
        temporal_unit: TemporalDistanceUnit | None = TemporalDistanceUnit.INDEX,
    ) -> WeightingPolicy:
        """Create a Laplace distance kernel."""
        if basis == WeightingBasis.MOTION:
            temporal_unit = None
        return cls(kernel=WeightingKernel.LAPLACE, basis=basis, temporal_unit=temporal_unit, scale=scale)

    @classmethod
    def student_t(
        cls,
        *,
        basis: WeightingBasis = WeightingBasis.TEMPORAL,
        scale: float,
        degrees_of_freedom: float,
        temporal_unit: TemporalDistanceUnit | None = TemporalDistanceUnit.INDEX,
    ) -> WeightingPolicy:
        """Create a Student-t distance kernel."""
        if basis == WeightingBasis.MOTION:
            temporal_unit = None
        return cls(
            kernel=WeightingKernel.STUDENT_T,
            basis=basis,
            temporal_unit=temporal_unit,
            scale=scale,
            degrees_of_freedom=degrees_of_freedom,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly manifest."""
        return {
            "kernel": self.kernel.value,
            "basis": self.basis.value if self.basis is not None else None,
            "temporal_unit": self.temporal_unit.value if self.temporal_unit is not None else None,
            "sigma": self.sigma,
            "scale": self.scale,
            "degrees_of_freedom": self.degrees_of_freedom,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WeightingPolicy:
        """Reconstruct a weighting policy from a serialized manifest."""
        return cls(
            kernel=value.get("kernel", WeightingKernel.EQUAL.value),
            basis=value.get("basis"),
            temporal_unit=value.get("temporal_unit"),
            sigma=value.get("sigma"),
            scale=value.get("scale"),
            degrees_of_freedom=value.get("degrees_of_freedom"),
        )


@dataclass(frozen=True)
class MatrixDecisions:
    """Complete, independently configurable recipe for one matrix builder."""

    quota: DirectionalQuota = field(default_factory=DirectionalQuota.future_only)
    sampling: SamplingPolicy = field(default_factory=SamplingPolicy.consecutive)
    motion: MotionEligibility = field(default_factory=MotionEligibility)
    target_policy: TargetPolicy = TargetPolicy.EXCLUDE
    scoring: CandidateScoringPolicy | None = None
    template_size: TemplateSizePolicy | None = None
    weighting: WeightingPolicy = field(default_factory=WeightingPolicy.equal)
    # Constructor-only compatibility input. It is normalized into ``scoring``
    # and ``template_size`` during initialization and omitted from manifests.
    correlation: CorrelationPolicy | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        scoring = self.scoring
        template_size = self.template_size
        if self.correlation is not None:
            if scoring is not None or template_size is not None:
                raise MatrixDecisionError(
                    "correlation compatibility policy cannot be combined with scoring or template_size"
                )
            if not isinstance(self.correlation, CorrelationPolicy):
                raise MatrixDecisionError(
                    f"correlation must be CorrelationPolicy, got {type(self.correlation).__name__}"
                )
            scoring, template_size = self.correlation.to_decisions()
        elif scoring is None:
            scoring = CandidateScoringPolicy.signed_pearson()

        if not isinstance(scoring, CandidateScoringPolicy):
            raise MatrixDecisionError(
                f"scoring must be CandidateScoringPolicy, got {type(scoring).__name__}"
            )
        if template_size is None:
            if scoring.mode is CandidateScoringMode.NONE:
                template_size = TemplateSizePolicy.select_all()
            elif scoring.mode is CandidateScoringMode.TEMPORAL_MOTION_COST:
                template_size = TemplateSizePolicy.exactly()
            else:
                template_size = TemplateSizePolicy.minimum()
        elif not isinstance(template_size, TemplateSizePolicy):
            raise MatrixDecisionError(
                f"template_size must be TemplateSizePolicy, got {type(template_size).__name__}"
            )

        object.__setattr__(self, "scoring", scoring)
        object.__setattr__(self, "template_size", template_size)
        object.__setattr__(self, "correlation", None)

        policies = (
            ("quota", self.quota, DirectionalQuota),
            ("sampling", self.sampling, SamplingPolicy),
            ("motion", self.motion, MotionEligibility),
            ("scoring", self.scoring, CandidateScoringPolicy),
            ("template_size", self.template_size, TemplateSizePolicy),
            ("weighting", self.weighting, WeightingPolicy),
        )
        for field_name, value, expected_type in policies:
            if not isinstance(value, expected_type):
                raise MatrixDecisionError(
                    f"{field_name} must be {expected_type.__name__}, got {type(value).__name__}"
                )
        object.__setattr__(self, "target_policy", _coerce_enum(self.target_policy, TargetPolicy, "target policy"))
        self._validate_scoring_template_combination()

    def _validate_scoring_template_combination(self) -> None:
        """Reject graph edges that have no defined selection semantics."""
        valid_modes = {
            CandidateScoringMode.SIGNED_PEARSON: {
                TemplateSizeMode.MINIMUM_K,
                TemplateSizeMode.MAXIMUM_K,
                TemplateSizeMode.EXACTLY_K,
            },
            CandidateScoringMode.ABSOLUTE_PEARSON: {
                TemplateSizeMode.MINIMUM_K,
                TemplateSizeMode.MAXIMUM_K,
                TemplateSizeMode.EXACTLY_K,
            },
            CandidateScoringMode.TEMPORAL_MOTION_COST: {
                TemplateSizeMode.MAXIMUM_K,
                TemplateSizeMode.EXACTLY_K,
            },
            CandidateScoringMode.NONE: {TemplateSizeMode.SELECT_ALL},
        }
        if self.template_size.mode not in valid_modes[self.scoring.mode]:
            allowed = ", ".join(sorted(mode.value for mode in valid_modes[self.scoring.mode]))
            raise MatrixDecisionError(
                f"candidate scoring mode {self.scoring.mode.value!r} supports template-size modes "
                f"{allowed}; got {self.template_size.mode.value!r}"
            )

    @classmethod
    def legacy_flex(
        cls,
        *,
        window_size: int,
        threshold: float,
        min_accepted: int,
        distribution: str,
        effective_window_size: int | None = None,
    ) -> MatrixDecisions:
        """Map the established four-argument Flex API onto the decision graph."""
        distribution = str(distribution).strip().lower()
        if distribution == "equal":
            weighting = WeightingPolicy.equal()
        elif distribution == "normal":
            effective = window_size if effective_window_size is None else effective_window_size
            weighting = WeightingPolicy.gaussian(sigma=max(float(effective) / 3.0, 1.0))
        else:
            raise MatrixDecisionError(f"Unsupported legacy N_distribution {distribution!r}")

        return cls(
            quota=DirectionalQuota.future_only(window_size),
            sampling=SamplingPolicy.consecutive(),
            target_policy=TargetPolicy.EXCLUDE,
            scoring=CandidateScoringPolicy.signed_pearson(threshold),
            template_size=TemplateSizePolicy.minimum(min_accepted),
            weighting=weighting,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the full JSON-friendly decision manifest."""
        return {
            "quota": self.quota.to_dict(),
            "sampling": self.sampling.to_dict(),
            "motion": self.motion.to_dict(),
            "target_policy": self.target_policy.value,
            "scoring": self.scoring.to_dict(),
            "template_size": self.template_size.to_dict(),
            "weighting": self.weighting.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MatrixDecisions:
        """Reconstruct a decision recipe from a serialized manifest."""
        common = {
            "quota": DirectionalQuota.from_dict(value.get("quota", {})),
            "sampling": SamplingPolicy.from_dict(value.get("sampling", {})),
            "motion": MotionEligibility.from_dict(value.get("motion", {})),
            "target_policy": value.get("target_policy", TargetPolicy.EXCLUDE.value),
            "weighting": WeightingPolicy.from_dict(value.get("weighting", {})),
        }
        if "scoring" in value or "template_size" in value:
            return cls(
                **common,
                scoring=(
                    CandidateScoringPolicy.from_dict(value["scoring"])
                    if "scoring" in value
                    else None
                ),
                template_size=(
                    TemplateSizePolicy.from_dict(value["template_size"])
                    if "template_size" in value
                    else None
                ),
            )
        return cls(
            **common,
            correlation=CorrelationPolicy.from_dict(value.get("correlation", {})),
        )


@dataclass(frozen=True)
class ResolvedMotionMetadata:
    """Motion arrays expanded or mapped to one row per artifact epoch."""

    parameters: np.ndarray | None
    segment_ids: np.ndarray | None
    stable: np.ndarray | None
    rotation_scale: float


@dataclass(frozen=True)
class MotionEpochMetadata:
    """Motion information and an optional mapping to artifact epochs.

    The source arrays may contain one row per artifact epoch or one row per
    fMRI volume.  In the latter case ``epoch_to_motion_index`` must explicitly
    map each artifact epoch to the appropriate source row.  Six-column SPM
    parameters are interpreted as three translations followed by three
    rotations; ``rotation_scale=0`` preserves FACETpy's established
    translation-only distance convention.
    """

    parameters: np.ndarray | None = None
    segment_ids: np.ndarray | None = None
    stable: np.ndarray | None = None
    epoch_to_motion_index: np.ndarray | None = None
    rotation_scale: float = 0.0

    def __post_init__(self) -> None:
        parameters = None if self.parameters is None else np.asarray(self.parameters, dtype=float)
        segment_ids = None if self.segment_ids is None else np.asarray(self.segment_ids)
        stable = None if self.stable is None else np.asarray(self.stable, dtype=bool)
        mapping = None if self.epoch_to_motion_index is None else np.asarray(self.epoch_to_motion_index)

        if parameters is not None and (parameters.ndim != 2 or parameters.shape[1] not in {3, 6}):
            raise MatrixDecisionError("motion parameters must have shape (n_motion_rows, 3 or 6)")
        if segment_ids is not None and segment_ids.ndim != 1:
            raise MatrixDecisionError("motion segment_ids must be one-dimensional")
        if stable is not None and stable.ndim != 1:
            raise MatrixDecisionError("motion stable mask must be one-dimensional")
        if mapping is not None:
            if mapping.ndim != 1 or not np.issubdtype(mapping.dtype, np.integer):
                raise MatrixDecisionError("epoch_to_motion_index must be a one-dimensional integer array")
            mapping = mapping.astype(int, copy=False)

        source_lengths = {
            len(value)
            for value in (parameters, segment_ids, stable)
            if value is not None
        }
        if len(source_lengths) > 1:
            raise MatrixDecisionError("motion parameters, segment_ids, and stable mask must share one source length")
        if not source_lengths:
            raise MatrixDecisionError("motion metadata must provide parameters, segment_ids, or a stable mask")

        rotation_scale = _require_positive(self.rotation_scale, "rotation_scale", allow_zero=True)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "stable", stable)
        object.__setattr__(self, "epoch_to_motion_index", mapping)
        object.__setattr__(self, "rotation_scale", rotation_scale)

    @classmethod
    def from_volume_parameters(
        cls,
        *,
        parameters: np.ndarray,
        n_artifact_epochs: int,
        slices_per_volume: int,
        segment_ids: np.ndarray | None = None,
        stable: np.ndarray | None = None,
        rotation_scale: float = 0.0,
    ) -> MotionEpochMetadata:
        """Map consecutive slice-artifact epochs to volume-level motion rows."""
        slices_per_volume = _require_integer(slices_per_volume, "slices_per_volume", minimum=1)
        n_artifact_epochs = _require_integer(n_artifact_epochs, "n_artifact_epochs", minimum=1)
        mapping = np.arange(n_artifact_epochs, dtype=int) // slices_per_volume
        return cls(
            parameters=parameters,
            segment_ids=segment_ids,
            stable=stable,
            epoch_to_motion_index=mapping,
            rotation_scale=rotation_scale,
        )

    @classmethod
    def from_value(cls, value: MotionEpochMetadata | Mapping[str, Any]) -> MotionEpochMetadata:
        """Coerce a typed object or metadata dictionary."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise MatrixDecisionError(
                f"{MOTION_METADATA_KEY!r} must contain MotionEpochMetadata or a mapping"
            )
        return cls(
            parameters=value.get("parameters"),
            segment_ids=value.get("segment_ids"),
            stable=value.get("stable"),
            epoch_to_motion_index=value.get("epoch_to_motion_index"),
            rotation_scale=value.get("rotation_scale", 0.0),
        )

    def resolve(self, n_epochs: int) -> ResolvedMotionMetadata:
        """Return arrays aligned exactly to the artifact-epoch rows of ``D``."""
        n_epochs = _require_integer(n_epochs, "n_epochs", minimum=0)
        source = next(
            value
            for value in (self.parameters, self.segment_ids, self.stable)
            if value is not None
        )
        source_length = len(source)

        if self.epoch_to_motion_index is None:
            if source_length != n_epochs:
                raise MatrixDecisionError(
                    "motion metadata is not epoch-aligned: provide epoch_to_motion_index when motion rows "
                    f"({source_length}) differ from artifact epochs ({n_epochs})"
                )
            mapping = np.arange(n_epochs, dtype=int)
        else:
            mapping = self.epoch_to_motion_index
            if len(mapping) != n_epochs:
                raise MatrixDecisionError(
                    f"epoch_to_motion_index has {len(mapping)} entries but D contains {n_epochs} epochs"
                )
            if np.any(mapping < 0) or np.any(mapping >= source_length):
                raise MatrixDecisionError("epoch_to_motion_index contains an out-of-range motion row")

        parameters = None if self.parameters is None else self.parameters[mapping]
        segment_ids = None if self.segment_ids is None else self.segment_ids[mapping]
        stable = None if self.stable is None else self.stable[mapping]
        return ResolvedMotionMetadata(parameters, segment_ids, stable, self.rotation_scale)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly motion metadata."""
        return {
            "parameters": None if self.parameters is None else self.parameters.tolist(),
            "segment_ids": None if self.segment_ids is None else self.segment_ids.tolist(),
            "stable": None if self.stable is None else self.stable.tolist(),
            "epoch_to_motion_index": (
                None if self.epoch_to_motion_index is None else self.epoch_to_motion_index.tolist()
            ),
            "rotation_scale": self.rotation_scale,
        }


@dataclass(frozen=True)
class MatrixMetadata:
    """Optional acquisition metadata available during construction of ``A``."""

    slices_per_volume: int | None = None
    trigger_times: np.ndarray | None = None
    motion: MotionEpochMetadata | None = None

    def __post_init__(self) -> None:
        if self.slices_per_volume is not None:
            object.__setattr__(
                self,
                "slices_per_volume",
                _require_integer(self.slices_per_volume, "slices_per_volume", minimum=1),
            )
        if self.trigger_times is not None:
            trigger_times = np.asarray(self.trigger_times, dtype=float)
            if trigger_times.ndim != 1:
                raise MatrixDecisionError("trigger_times must be one-dimensional")
            object.__setattr__(self, "trigger_times", trigger_times)
        if self.motion is not None and not isinstance(self.motion, MotionEpochMetadata):
            object.__setattr__(self, "motion", MotionEpochMetadata.from_value(self.motion))

    @classmethod
    def from_processing_context(cls, context: Any) -> MatrixMetadata:
        """Resolve typed matrix metadata from a ``ProcessingContext``."""
        triggers = np.asarray(context.get_triggers(), dtype=float)
        sfreq = float(context.get_sfreq())
        motion_value = context.metadata.custom.get(MOTION_METADATA_KEY)
        motion = None if motion_value is None else MotionEpochMetadata.from_value(motion_value)
        return cls(
            slices_per_volume=context.metadata.slices_per_volume,
            trigger_times=triggers / sfreq,
            motion=motion,
        )


def pearson_scores(
    target_epoch: np.ndarray,
    candidate_epochs: np.ndarray,
    mode: CandidateScoringMode = CandidateScoringMode.SIGNED_PEARSON,
) -> np.ndarray:
    """Calculate deterministic signed or absolute Pearson candidate scores.

    Constant or non-finite target/candidate epochs receive ``-inf``.  This
    sentinel cannot pass a threshold and is explicitly excluded from minimum
    supplementation.
    """
    mode = _coerce_candidate_scoring_mode(mode)
    if mode not in {
        CandidateScoringMode.SIGNED_PEARSON,
        CandidateScoringMode.ABSOLUTE_PEARSON,
    }:
        raise MatrixDecisionError(f"pearson_scores cannot be used with scoring mode {mode.value!r}")

    target = np.asarray(target_epoch, dtype=float)
    candidates = np.asarray(candidate_epochs, dtype=float)
    if candidates.ndim != 2:
        raise MatrixDecisionError("candidate_epochs must be two-dimensional")

    scores = np.full(len(candidates), -np.inf, dtype=float)
    if target.ndim != 1 or candidates.shape[1] != target.shape[0] or not np.all(np.isfinite(target)):
        return scores

    target_centered = target - np.mean(target)
    target_norm = np.linalg.norm(target_centered)
    if not np.isfinite(target_norm) or target_norm <= 0.0:
        return scores

    finite_candidates = np.all(np.isfinite(candidates), axis=1)
    if not np.any(finite_candidates):
        return scores

    centered = candidates[finite_candidates] - np.mean(candidates[finite_candidates], axis=1, keepdims=True)
    numerators = centered @ target_centered
    denominators = np.linalg.norm(centered, axis=1) * target_norm
    valid_local = np.isfinite(numerators) & np.isfinite(denominators) & (denominators > 0.0)
    local_scores = np.full(len(centered), -np.inf, dtype=float)
    np.divide(numerators, denominators, out=local_scores, where=valid_local)
    finite_local = np.isfinite(local_scores)
    local_scores[finite_local] = np.clip(local_scores[finite_local], -1.0, 1.0)
    if mode is CandidateScoringMode.ABSOLUTE_PEARSON:
        local_scores[finite_local] = np.abs(local_scores[finite_local])
    scores[np.flatnonzero(finite_candidates)] = local_scores
    return scores


def _rank_positions(
    candidate_indices: np.ndarray,
    values: np.ndarray,
    positions: np.ndarray,
    *,
    target_idx: int,
    lowest_first: bool,
) -> np.ndarray:
    """Rank valid positions by value, temporal distance, and epoch index."""
    if positions.size == 0:
        return positions
    candidates = candidate_indices[positions]
    primary = values[positions] if lowest_first else -values[positions]
    temporal_distance = np.abs(candidates - target_idx)
    order = np.lexsort((candidates, temporal_distance, primary))
    return positions[order]


def select_scored_candidates(
    candidate_indices: np.ndarray,
    values: np.ndarray | None,
    *,
    target_idx: int,
    scoring: CandidateScoringPolicy,
    template_size: TemplateSizePolicy,
) -> np.ndarray:
    """Apply the configured acceptance, ranking, and cardinality stages."""
    # MatrixDecisions performs the same validation for normal builder use.
    # Keep this public helper strict as well so it cannot silently reinterpret
    # an edge that is absent from the decision graph.
    valid_modes = {
        CandidateScoringMode.SIGNED_PEARSON: {
            TemplateSizeMode.MINIMUM_K,
            TemplateSizeMode.MAXIMUM_K,
            TemplateSizeMode.EXACTLY_K,
        },
        CandidateScoringMode.ABSOLUTE_PEARSON: {
            TemplateSizeMode.MINIMUM_K,
            TemplateSizeMode.MAXIMUM_K,
            TemplateSizeMode.EXACTLY_K,
        },
        CandidateScoringMode.TEMPORAL_MOTION_COST: {
            TemplateSizeMode.MAXIMUM_K,
            TemplateSizeMode.EXACTLY_K,
        },
        CandidateScoringMode.NONE: {TemplateSizeMode.SELECT_ALL},
    }
    if template_size.mode not in valid_modes[scoring.mode]:
        raise MatrixDecisionError(
            f"candidate scoring mode {scoring.mode.value!r} cannot be combined with "
            f"template-size mode {template_size.mode.value!r}"
        )

    candidates = np.asarray(candidate_indices, dtype=int)
    if candidates.ndim != 1:
        raise MatrixDecisionError("candidate indices must be one-dimensional")

    if scoring.mode is CandidateScoringMode.NONE:
        if values is not None:
            raise MatrixDecisionError("scoring mode 'none' must not receive candidate values")
        return np.sort(candidates.copy())

    ranked_values = np.asarray(values, dtype=float)
    if ranked_values.shape != candidates.shape:
        raise MatrixDecisionError("candidate indices and scoring values must be matching vectors")
    valid_positions = np.flatnonzero(np.isfinite(ranked_values))

    if scoring.mode is CandidateScoringMode.TEMPORAL_MOTION_COST:
        ranked = _rank_positions(
            candidates,
            ranked_values,
            valid_positions,
            target_idx=target_idx,
            lowest_first=True,
        )
        selected = ranked[: int(template_size.k)]
        return candidates[selected]

    accepted_positions = valid_positions[
        ranked_values[valid_positions] >= float(scoring.threshold)
    ]
    rejected_positions = valid_positions[
        ranked_values[valid_positions] < float(scoring.threshold)
    ]
    accepted = _rank_positions(
        candidates,
        ranked_values,
        accepted_positions,
        target_idx=target_idx,
        lowest_first=False,
    )
    rejected = _rank_positions(
        candidates,
        ranked_values,
        rejected_positions,
        target_idx=target_idx,
        lowest_first=False,
    )

    k = int(template_size.k)
    if template_size.mode is TemplateSizeMode.MINIMUM_K:
        missing = max(0, k - len(accepted))
        selected = np.concatenate((accepted, rejected[:missing]))
    elif template_size.mode is TemplateSizeMode.MAXIMUM_K:
        selected = accepted[:k]
    else:
        selected = accepted[:k]
        if len(selected) < k:
            selected = np.concatenate((selected, rejected[: k - len(selected)]))
    return candidates[selected]


def select_by_correlation(
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    policy: CorrelationPolicy,
    *,
    target_idx: int = 0,
) -> np.ndarray:
    """Compatibility wrapper for correlation plus ``minimum_k`` selection."""
    scoring, template_size = policy.to_decisions()
    values = None if scoring.mode is CandidateScoringMode.NONE else scores
    return select_scored_candidates(
        candidate_indices,
        values,
        target_idx=target_idx,
        scoring=scoring,
        template_size=template_size,
    )


class AveragingMatrixBuilder:
    """Construct one normalized row of ``A`` for every target epoch."""

    def __init__(self, decisions: MatrixDecisions | Mapping[str, Any] | None = None) -> None:
        if decisions is None:
            decisions = MatrixDecisions()
        elif isinstance(decisions, Mapping):
            decisions = MatrixDecisions.from_dict(decisions)
        if not isinstance(decisions, MatrixDecisions):
            raise MatrixDecisionError("decisions must be MatrixDecisions or a serialized mapping")
        self.decisions = decisions

    def build(self, epochs: np.ndarray, metadata: MatrixMetadata | None = None) -> np.ndarray:
        """Build a square, finite, nonnegative averaging matrix."""
        epoch_data = np.asarray(epochs, dtype=float)
        if epoch_data.ndim != 2:
            raise MatrixDecisionError("epochs must have shape (n_epochs, n_samples)")
        metadata = MatrixMetadata() if metadata is None else metadata
        if not isinstance(metadata, MatrixMetadata):
            raise MatrixDecisionError("metadata must be MatrixMetadata")

        n_epochs = int(epoch_data.shape[0])
        matrix = np.zeros((n_epochs, n_epochs), dtype=float)
        resolved_motion = self._validate_and_resolve_metadata(metadata, n_epochs)

        for target_idx in range(n_epochs):
            candidates = self._candidate_pool(
                target_idx=target_idx,
                n_epochs=n_epochs,
                metadata=metadata,
                motion=resolved_motion,
            )
            selected = self._selected_candidates(
                epochs=epoch_data,
                target_idx=target_idx,
                candidate_indices=candidates,
                metadata=metadata,
                motion=resolved_motion,
            )
            if selected.size == 0:
                continue
            matrix[target_idx, selected] = self._normalized_weights(
                target_idx=target_idx,
                selected_indices=selected,
                metadata=metadata,
                motion=resolved_motion,
            )

        self._validate_matrix(matrix)
        return matrix

    def candidate_pool(
        self,
        *,
        target_idx: int,
        n_epochs: int,
        metadata: MatrixMetadata | None = None,
    ) -> np.ndarray:
        """Expose deterministic pool construction for testing and inspection."""
        metadata = MatrixMetadata() if metadata is None else metadata
        if not isinstance(metadata, MatrixMetadata):
            raise MatrixDecisionError("metadata must be MatrixMetadata")
        motion = self._validate_and_resolve_metadata(metadata, n_epochs)
        return self._candidate_pool(
            target_idx=target_idx,
            n_epochs=n_epochs,
            metadata=metadata,
            motion=motion,
        )

    def _validate_and_resolve_metadata(
        self,
        metadata: MatrixMetadata,
        n_epochs: int,
    ) -> ResolvedMotionMetadata | None:
        sampling = self.decisions.sampling
        if sampling.mode is SamplingMode.SAME_SLICE_PHASE and metadata.slices_per_volume is None:
            raise MatrixDecisionError("same_slice_phase sampling requires slices_per_volume metadata")

        scoring = self.decisions.scoring
        weighting = self.decisions.weighting
        time_scoring = (
            scoring.mode is CandidateScoringMode.TEMPORAL_MOTION_COST
            and float(scoring.temporal_weight) > 0.0
            and scoring.temporal_unit is TemporalDistanceUnit.TIME
        )
        time_weighting = (
            weighting.kernel is not WeightingKernel.EQUAL
            and weighting.basis is WeightingBasis.TEMPORAL
            and weighting.temporal_unit is TemporalDistanceUnit.TIME
        )
        if time_scoring or time_weighting:
            times = metadata.trigger_times
            if times is None or len(times) != n_epochs:
                purpose = "scoring" if time_scoring else "weighting"
                raise MatrixDecisionError(
                    f"trigger-time {purpose} requires one trigger time per artifact epoch"
                )
            # A non-finite scoring distance invalidates only the affected
            # candidate. Weighting cannot normalize such a distance, so its
            # metadata remains a strict all-finite requirement.
            if time_weighting and not np.all(np.isfinite(times)):
                raise MatrixDecisionError(
                    "trigger-time weighting requires one finite trigger time per artifact epoch"
                )

        motion_scoring = (
            scoring.mode is CandidateScoringMode.TEMPORAL_MOTION_COST
            and float(scoring.motion_weight) > 0.0
        )
        motion_weighting = (
            weighting.kernel is not WeightingKernel.EQUAL
            and weighting.basis is WeightingBasis.MOTION
        )
        motion_required = self.decisions.motion.enabled or motion_weighting or motion_scoring
        if not motion_required:
            return None
        if metadata.motion is None:
            raise MatrixDecisionError(
                f"motion decisions require metadata.custom[{MOTION_METADATA_KEY!r}] or explicit "
                "MatrixMetadata.motion; this includes motion eligibility, scoring, and weighting"
            )

        motion = metadata.motion.resolve(n_epochs)
        eligibility = self.decisions.motion
        if eligibility.same_motion_segment and motion.segment_ids is None:
            raise MatrixDecisionError("same_motion_segment requires motion segment_ids")
        if eligibility.motion_stable_only and motion.stable is None:
            raise MatrixDecisionError("motion_stable_only requires a motion stable mask")
        needs_parameters = eligibility.max_motion_distance is not None or motion_weighting or motion_scoring
        if needs_parameters:
            if motion.parameters is None:
                raise MatrixDecisionError("motion distance requires motion parameters")
            if (eligibility.max_motion_distance is not None or motion_weighting) and not np.all(
                np.isfinite(motion.parameters)
            ):
                raise MatrixDecisionError("motion parameters used for distance must be finite")
        return motion

    def _sampling_offsets(self, metadata: MatrixMetadata) -> tuple[int, int]:
        sampling = self.decisions.sampling
        if sampling.mode is SamplingMode.SAME_SLICE_PHASE:
            period = int(metadata.slices_per_volume)
            return period, period
        return int(sampling.start_offset), int(sampling.stride)

    def _candidate_pool(
        self,
        *,
        target_idx: int,
        n_epochs: int,
        metadata: MatrixMetadata,
        motion: ResolvedMotionMetadata | None,
    ) -> np.ndarray:
        if target_idx < 0 or target_idx >= n_epochs:
            raise MatrixDecisionError(f"target_idx {target_idx} is outside 0..{n_epochs - 1}")

        start_offset, stride = self._sampling_offsets(metadata)
        past = np.arange(target_idx - start_offset, -1, -stride, dtype=int)
        future = np.arange(target_idx + start_offset, n_epochs, stride, dtype=int)
        past = self._motion_eligible_indices(target_idx, past, motion)
        future = self._motion_eligible_indices(target_idx, future, motion)

        quota = self.decisions.quota
        if quota.global_mode:
            selected = np.concatenate((past, future))
        else:
            requested_past = int(quota.past)
            requested_future = int(quota.future)
            selected_past = past[:requested_past]
            selected_future = future[:requested_future]

            past_deficit = requested_past - len(selected_past)
            future_deficit = requested_future - len(selected_future)
            if past_deficit > 0:
                selected_future = np.concatenate(
                    (selected_future, future[requested_future : requested_future + past_deficit])
                )
            if future_deficit > 0:
                selected_past = np.concatenate(
                    (selected_past, past[requested_past : requested_past + future_deficit])
                )
            selected = np.concatenate((selected_past, selected_future))

        if (
            self.decisions.target_policy is TargetPolicy.INCLUDE
            and self.decisions.sampling.naturally_contains_target()
            and self._motion_candidate_is_eligible(target_idx, target_idx, motion)
        ):
            selected = np.append(selected, target_idx)

        return np.sort(np.unique(selected.astype(int, copy=False)))

    def _motion_eligible_indices(
        self,
        target_idx: int,
        indices: np.ndarray,
        motion: ResolvedMotionMetadata | None,
    ) -> np.ndarray:
        if motion is None or indices.size == 0:
            return indices
        eligible = np.fromiter(
            (self._motion_candidate_is_eligible(target_idx, int(candidate), motion) for candidate in indices),
            dtype=bool,
            count=len(indices),
        )
        return indices[eligible]

    def _motion_candidate_is_eligible(
        self,
        target_idx: int,
        candidate_idx: int,
        motion: ResolvedMotionMetadata | None,
    ) -> bool:
        eligibility = self.decisions.motion
        if not eligibility.enabled:
            return True
        if motion is None:  # pragma: no cover - guarded by metadata validation
            return False
        if eligibility.same_motion_segment and motion.segment_ids[candidate_idx] != motion.segment_ids[target_idx]:
            return False
        if eligibility.motion_stable_only and not bool(motion.stable[candidate_idx]):
            return False
        if eligibility.max_motion_distance is not None:
            distance = self._motion_distances(target_idx, np.array([candidate_idx]), motion)[0]
            if distance > eligibility.max_motion_distance:
                return False
        return True

    def _selected_candidates(
        self,
        *,
        epochs: np.ndarray,
        target_idx: int,
        candidate_indices: np.ndarray,
        metadata: MatrixMetadata,
        motion: ResolvedMotionMetadata | None,
    ) -> np.ndarray:
        scoring = self.decisions.scoring
        if scoring.mode is CandidateScoringMode.NONE:
            values = None
        elif scoring.mode is CandidateScoringMode.TEMPORAL_MOTION_COST:
            values = self._temporal_motion_costs(
                target_idx=target_idx,
                candidate_indices=candidate_indices,
                metadata=metadata,
                motion=motion,
            )
        else:
            values = pearson_scores(
                target_epoch=epochs[target_idx],
                candidate_epochs=epochs[candidate_indices],
                mode=scoring.mode,
            )
        return select_scored_candidates(
            candidate_indices,
            values,
            target_idx=target_idx,
            scoring=scoring,
            template_size=self.decisions.template_size,
        )

    def _temporal_motion_costs(
        self,
        *,
        target_idx: int,
        candidate_indices: np.ndarray,
        metadata: MatrixMetadata,
        motion: ResolvedMotionMetadata | None,
    ) -> np.ndarray:
        """Return independently scaled temporal plus cumulative-motion cost.

        The cumulative term integrates motion increments along the acquisition
        path between the target and candidate. This preserves the important
        Moosmann distinction between two epochs that have similar poses and
        two epochs separated by substantial intervening movement.
        """
        scoring = self.decisions.scoring
        costs = np.zeros(len(candidate_indices), dtype=float)

        with np.errstate(over="ignore", invalid="ignore"):
            if float(scoring.temporal_weight) > 0.0:
                if scoring.temporal_unit is TemporalDistanceUnit.TIME:
                    temporal = np.abs(
                        metadata.trigger_times[candidate_indices]
                        - metadata.trigger_times[target_idx]
                    )
                else:
                    temporal = np.abs(candidate_indices - target_idx).astype(float)
                costs += float(scoring.temporal_weight) * temporal

            if float(scoring.motion_weight) > 0.0:
                if motion is None:  # pragma: no cover - guarded by metadata validation
                    raise MatrixDecisionError("temporal_motion_cost requires resolved motion metadata")
                cumulative_motion = self._cumulative_motion_distances(
                    target_idx,
                    candidate_indices,
                    motion,
                )
                costs += float(scoring.motion_weight) * cumulative_motion

        # Overflow and non-finite metadata deliberately remain non-finite;
        # selection excludes those candidates rather than treating them as a
        # good fallback merely because a finite template size was requested.
        costs[~np.isfinite(costs)] = np.inf
        return costs

    @classmethod
    def _cumulative_motion_distances(
        cls,
        target_idx: int,
        candidate_indices: np.ndarray,
        motion: ResolvedMotionMetadata,
    ) -> np.ndarray:
        """Integrate per-step motion magnitude between each epoch pair."""
        vectors = cls._motion_vectors(motion)
        with np.errstate(over="ignore", invalid="ignore"):
            increments = np.linalg.norm(np.diff(vectors, axis=0), axis=1)

        finite_increments = np.isfinite(increments)
        cumulative_distance = np.concatenate(
            ([0.0], np.cumsum(np.where(finite_increments, increments, 0.0)))
        )
        cumulative_invalid = np.concatenate(
            ([0], np.cumsum(~finite_increments, dtype=int))
        )
        starts = np.minimum(target_idx, candidate_indices)
        stops = np.maximum(target_idx, candidate_indices)
        path_is_finite = cumulative_invalid[stops] == cumulative_invalid[starts]
        endpoints_are_finite = (
            np.all(np.isfinite(vectors[target_idx]))
            & np.all(np.isfinite(vectors[candidate_indices]), axis=1)
        )
        valid = path_is_finite & endpoints_are_finite
        distances = np.full(len(candidate_indices), np.inf, dtype=float)
        distances[valid] = cumulative_distance[stops[valid]] - cumulative_distance[starts[valid]]
        return distances

    def _normalized_weights(
        self,
        *,
        target_idx: int,
        selected_indices: np.ndarray,
        metadata: MatrixMetadata,
        motion: ResolvedMotionMetadata | None,
    ) -> np.ndarray:
        policy = self.decisions.weighting
        if policy.kernel is WeightingKernel.EQUAL:
            raw_weights = np.ones(len(selected_indices), dtype=float)
        else:
            distances = self._distances(target_idx, selected_indices, metadata, motion)
            if policy.kernel is WeightingKernel.GAUSSIAN:
                log_weights = -(distances**2) / (2.0 * float(policy.sigma) ** 2)
            elif policy.kernel is WeightingKernel.LAPLACE:
                log_weights = -distances / float(policy.scale)
            else:
                degrees = float(policy.degrees_of_freedom)
                scale = float(policy.scale)
                log_weights = -((degrees + 1.0) / 2.0) * np.log1p((distances**2) / (degrees * scale**2))
            # Shifting in log space prevents complete underflow for distant
            # candidates without changing the normalized kernel.
            raw_weights = np.exp(log_weights - np.max(log_weights))

        if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights < 0.0):
            raise MatrixDecisionError("weighting kernel produced non-finite or negative weights")
        total = float(np.sum(raw_weights))
        if not np.isfinite(total) or total <= 0.0:
            raise MatrixDecisionError("weighting kernel produced a non-positive row sum")
        return raw_weights / total

    def _distances(
        self,
        target_idx: int,
        selected_indices: np.ndarray,
        metadata: MatrixMetadata,
        motion: ResolvedMotionMetadata | None,
    ) -> np.ndarray:
        policy = self.decisions.weighting
        if policy.basis is WeightingBasis.MOTION:
            if motion is None:  # pragma: no cover - guarded by metadata validation
                raise MatrixDecisionError("motion weighting requires resolved motion metadata")
            return self._motion_distances(target_idx, selected_indices, motion)
        if policy.temporal_unit is TemporalDistanceUnit.TIME:
            return np.abs(metadata.trigger_times[selected_indices] - metadata.trigger_times[target_idx])
        return np.abs(selected_indices - target_idx).astype(float)

    @staticmethod
    def _motion_vectors(motion: ResolvedMotionMetadata) -> np.ndarray:
        """Return FACETpy translation/rotation vectors for motion distance."""
        parameters = motion.parameters
        if parameters.shape[1] == 6:
            translations = parameters[:, :3]
            rotations = parameters[:, 3:] * motion.rotation_scale
            return np.concatenate((translations, rotations), axis=1)
        return parameters

    @classmethod
    def _motion_distances(
        cls,
        target_idx: int,
        candidate_indices: np.ndarray,
        motion: ResolvedMotionMetadata,
    ) -> np.ndarray:
        vectors = cls._motion_vectors(motion)
        return np.linalg.norm(vectors[candidate_indices] - vectors[target_idx], axis=1)

    @staticmethod
    def _validate_matrix(matrix: np.ndarray) -> None:
        if not np.all(np.isfinite(matrix)):
            raise MatrixDecisionError("averaging matrix contains non-finite weights")
        if np.any(matrix < 0.0):
            raise MatrixDecisionError("averaging matrix contains negative weights")

        row_nonzero = np.count_nonzero(matrix, axis=1) > 0
        row_sums = np.sum(matrix, axis=1)
        if np.any(row_nonzero & ~np.isclose(row_sums, 1.0, rtol=1e-10, atol=1e-12)):
            raise MatrixDecisionError("every non-empty averaging-matrix row must sum to one")
        if np.any(~row_nonzero & (row_sums != 0.0)):
            raise MatrixDecisionError("empty averaging-matrix rows must remain zero")


def future_first_candidate_indices(target_idx: int, n_epochs: int, window_size: int) -> np.ndarray:
    """Compatibility helper for Flex's established future-first candidate pool."""
    decisions = MatrixDecisions(
        quota=DirectionalQuota.future_only(window_size),
        sampling=SamplingPolicy.consecutive(),
        target_policy=TargetPolicy.EXCLUDE,
        correlation=CorrelationPolicy.none(),
        weighting=WeightingPolicy.equal(),
    )
    return AveragingMatrixBuilder(decisions).candidate_pool(target_idx=target_idx, n_epochs=n_epochs)
