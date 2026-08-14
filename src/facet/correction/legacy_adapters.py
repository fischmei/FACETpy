"""Backwards-compatible processor names backed by Flex decision presets.

The former hand-written averaging-matrix implementations are retained under
``correction/archived_algos`` for historical inspection.  Public imports keep
working through these thin adapters, but all active matrix construction is
delegated to the same composable Flex engine used by the CLI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core import ProcessingContext, ProcessorValidationError, register_processor
from .flex import MOTION_METADATA_KEY, Flex, MotionEpochMetadata
from .presets import (
    AAS_PER_TARGET,
    CORRESPONDING_SLICE,
    FARM_PER_TARGET_K10,
    LEGACY_RESEMBLANCE,
    MOOSMANN_COST,
    STRUCTURAL_SLICE,
    STRUCTURAL_VOLUME,
    build_flex_preset,
)


def _label_preset(processor: Flex, preset_name: str) -> None:
    """Attach report-only provenance without expanding Flex's public API."""
    processor.flex_preset_name = preset_name
    processor.legacy_algorithm_resemblance = LEGACY_RESEMBLANCE[preset_name]


@register_processor
class AASCorrection(Flex):
    """Compatibility name for the per-target AAS-like Flex preset."""

    name = "aas_correction"
    description = "Flex configured as the closest per-target AAS recipe"

    def __init__(
        self,
        window_size: int = 10,
        rel_window_position: float = 0.0,
        correlation_threshold: float = 0.975,
        plot_artifacts: bool = False,
        realign_after_averaging: bool = True,
        search_window_factor: float = 3.0,
        interpolate_volume_gaps: bool = False,
        apply_epoch_alpha_scaling: bool = False,
        track_estimated_noise: bool = True,
    ) -> None:
        self.rel_window_position = rel_window_position
        self.correlation_threshold = correlation_threshold
        super().__init__(
            plot_artifacts=plot_artifacts,
            realign_after_averaging=realign_after_averaging,
            search_window_factor=search_window_factor,
            interpolate_volume_gaps=interpolate_volume_gaps,
            apply_epoch_alpha_scaling=apply_epoch_alpha_scaling,
            track_estimated_noise=track_estimated_noise,
            matrix_decisions=build_flex_preset(
                AAS_PER_TARGET,
                window_size=window_size,
                threshold=correlation_threshold,
            ),
        )
        _label_preset(self, AAS_PER_TARGET)

    def _get_parameters(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "rel_window_position": self.rel_window_position,
            "correlation_threshold": self.correlation_threshold,
            "plot_artifacts": self.plot_artifacts,
            "realign_after_averaging": self.realign_after_averaging,
            "search_window_factor": self.search_window_factor,
            "interpolate_volume_gaps": self.interpolate_volume_gaps,
            "apply_epoch_alpha_scaling": self.apply_epoch_alpha_scaling,
            "track_estimated_noise": self.track_estimated_noise,
        }


@register_processor
class FARMCorrection(Flex):
    """Compatibility name for the per-target FARM-like Flex preset."""

    name = "farm_correction"
    description = "Flex configured as the FARM-like correlation recipe"

    def __init__(
        self,
        window_size: int = 30,
        correlation_threshold: float = 0.9,
        search_half_window: int | None = None,
        search_half_window_factor: float = 3.0,
        plot_artifacts: bool = False,
        realign_after_averaging: bool = True,
        search_window_factor: float = 3.0,
        interpolate_volume_gaps: bool = False,
        apply_epoch_alpha_scaling: bool = False,
        track_estimated_noise: bool = True,
        *,
        rel_window_position: float = 0.0,
    ) -> None:
        self.search_half_window = search_half_window
        self.search_half_window_factor = search_half_window_factor
        self.rel_window_position = rel_window_position
        self.correlation_threshold = correlation_threshold
        decision_window = 2 * search_half_window if search_half_window is not None else window_size
        super().__init__(
            plot_artifacts=plot_artifacts,
            realign_after_averaging=realign_after_averaging,
            search_window_factor=search_window_factor,
            interpolate_volume_gaps=interpolate_volume_gaps,
            apply_epoch_alpha_scaling=apply_epoch_alpha_scaling,
            track_estimated_noise=track_estimated_noise,
            matrix_decisions=build_flex_preset(
                FARM_PER_TARGET_K10,
                window_size=decision_window,
                threshold=correlation_threshold,
            ),
        )
        _label_preset(self, FARM_PER_TARGET_K10)

    def _get_parameters(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "correlation_threshold": self.correlation_threshold,
            "search_half_window": self.search_half_window,
            "search_half_window_factor": self.search_half_window_factor,
            "plot_artifacts": self.plot_artifacts,
            "realign_after_averaging": self.realign_after_averaging,
            "search_window_factor": self.search_window_factor,
            "interpolate_volume_gaps": self.interpolate_volume_gaps,
            "apply_epoch_alpha_scaling": self.apply_epoch_alpha_scaling,
            "track_estimated_noise": self.track_estimated_noise,
            "rel_window_position": self.rel_window_position,
        }


class _StructuralFlexAdapter(Flex):
    """Shared constructor behavior for structural compatibility names."""

    preset_name: str

    def __init__(
        self,
        *,
        window_size: int = 10,
        plot_artifacts: bool = False,
        realign_after_averaging: bool = True,
        search_window_factor: float = 3.0,
        apply_epoch_alpha_scaling: bool = False,
        track_estimated_noise: bool = True,
    ) -> None:
        super().__init__(
            plot_artifacts=plot_artifacts,
            realign_after_averaging=realign_after_averaging,
            search_window_factor=search_window_factor,
            apply_epoch_alpha_scaling=apply_epoch_alpha_scaling,
            track_estimated_noise=track_estimated_noise,
            matrix_decisions=build_flex_preset(self.preset_name, window_size=window_size),
        )
        _label_preset(self, self.preset_name)

    def _structural_parameters(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "plot_artifacts": self.plot_artifacts,
            "realign_after_averaging": self.realign_after_averaging,
            "search_window_factor": self.search_window_factor,
            "apply_epoch_alpha_scaling": self.apply_epoch_alpha_scaling,
            "track_estimated_noise": self.track_estimated_noise,
        }


@register_processor
class VolumeTriggerCorrection(_StructuralFlexAdapter):
    """Compatibility name for structural volume decisions."""

    name = "volume_trigger_correction"
    description = "Flex configured for neighboring-volume structure"
    preset_name = STRUCTURAL_VOLUME

    def __init__(self, window_size: int = 10, **kwargs) -> None:
        super().__init__(window_size=window_size, **kwargs)

    def _get_parameters(self) -> dict[str, object]:
        return self._structural_parameters()


@register_processor
class SliceTriggerCorrection(_StructuralFlexAdapter):
    """Compatibility name for structural odd/even slice decisions."""

    name = "slice_trigger_correction"
    description = "Flex configured for odd/even slice-trigger structure"
    preset_name = STRUCTURAL_SLICE

    def __init__(self, window_size: int = 10, **kwargs) -> None:
        super().__init__(window_size=window_size, **kwargs)

    def _get_parameters(self) -> dict[str, object]:
        return self._structural_parameters()


@register_processor
class CorrespondingSliceCorrection(_StructuralFlexAdapter):
    """Compatibility name for same-slice-phase Flex decisions."""

    name = "corresponding_slice_correction"
    description = "Flex configured for corresponding slices across volumes"
    preset_name = CORRESPONDING_SLICE

    def __init__(
        self,
        slices_per_volume: int | None = None,
        window_size: int = 10,
        **kwargs,
    ) -> None:
        self.slices_per_volume = slices_per_volume
        super().__init__(window_size=window_size, **kwargs)

    def process(self, context: ProcessingContext) -> ProcessingContext:
        if self.slices_per_volume is None:
            return super().process(context)
        metadata = context.metadata.copy()
        metadata.slices_per_volume = int(self.slices_per_volume)
        return super().process(context.with_metadata(metadata, copy_estimated_noise=False))

    def _get_parameters(self) -> dict[str, object]:
        return {"slices_per_volume": self.slices_per_volume, **self._structural_parameters()}


@register_processor
class MoosmannCorrection(Flex):
    """Compatibility name for the motion-path-cost Flex preset."""

    name = "moosmann_correction"
    description = "Flex configured with motion-informed candidate costs"

    def __init__(
        self,
        rp_file: str,
        window_size: int = 30,
        motion_threshold: float = 5.0,
        motion_window_size: int | None = None,
        plot_artifacts: bool = False,
        realign_after_averaging: bool = True,
        search_window_factor: float = 3.0,
        apply_epoch_alpha_scaling: bool = False,
        track_estimated_noise: bool = True,
    ) -> None:
        self.rp_file = rp_file
        self.motion_threshold = motion_threshold
        self.motion_window_size = motion_window_size
        super().__init__(
            window_size=window_size,
            plot_artifacts=plot_artifacts,
            realign_after_averaging=realign_after_averaging,
            search_window_factor=search_window_factor,
            apply_epoch_alpha_scaling=apply_epoch_alpha_scaling,
            track_estimated_noise=track_estimated_noise,
            matrix_decisions=build_flex_preset(
                MOOSMANN_COST,
                template_size=motion_window_size or 2 * window_size,
            ),
        )
        _label_preset(self, MOOSMANN_COST)

    def validate(self, context: ProcessingContext) -> None:
        if self.motion_threshold <= 0:
            raise ProcessorValidationError(f"motion_threshold must be positive, got {self.motion_threshold}")
        if not Path(self.rp_file).is_file():
            raise ProcessorValidationError(f"rp_file was not found: {self.rp_file}")
        super().validate(self._with_motion_metadata(context))

    def process(self, context: ProcessingContext) -> ProcessingContext:
        result = super().process(self._with_motion_metadata(context))
        metadata = result.metadata.copy()
        metadata.custom["moosmann"] = {
            "rp_file": self.rp_file,
            "motion_threshold": self.motion_threshold,
            "motion_window_size": self.motion_window_size or 2 * self.window_size,
            "flex_preset": MOOSMANN_COST,
        }
        return result.with_metadata(metadata, copy_estimated_noise=False)

    def _with_motion_metadata(self, context: ProcessingContext) -> ProcessingContext:
        existing = context.metadata.custom.get(MOTION_METADATA_KEY)
        if existing is not None:
            return context
        try:
            parameters = np.asarray(np.loadtxt(self.rp_file, comments="#", ndmin=2), dtype=float)
        except ValueError:
            try:
                parameters = np.asarray(
                    np.loadtxt(self.rp_file, comments="#", skiprows=1, ndmin=2),
                    dtype=float,
                )
            except (OSError, ValueError) as exc:
                raise ProcessorValidationError(f"Could not read rp_file {self.rp_file}: {exc}") from exc
        except OSError as exc:
            raise ProcessorValidationError(f"Could not read rp_file {self.rp_file}: {exc}") from exc
        if (
            parameters.ndim != 2
            or parameters.shape[1] not in {3, 6}
            or not np.all(np.isfinite(parameters))
        ):
            raise ProcessorValidationError(
                "rp_file must contain three or six finite numeric columns"
            )

        n_epochs = len(context.get_triggers())
        slices_per_volume = context.metadata.slices_per_volume
        if slices_per_volume:
            mapping = np.arange(n_epochs, dtype=int) // int(slices_per_volume)
            required = int(np.max(mapping)) + 1 if mapping.size else 0
        else:
            mapping = np.arange(n_epochs, dtype=int)
            required = n_epochs
        if len(parameters) < required:
            parameters = np.vstack((np.zeros((required - len(parameters), parameters.shape[1])), parameters))
        if len(parameters) > required and slices_per_volume is None:
            parameters = parameters[-required:]

        increments = np.zeros(len(parameters), dtype=float)
        if len(parameters) > 1:
            increments[1:] = np.linalg.norm(np.diff(parameters[:, :3], axis=0), axis=1)
        motion = MotionEpochMetadata(
            parameters=parameters,
            stable=increments <= self.motion_threshold,
            epoch_to_motion_index=mapping,
            rotation_scale=0.0,
        )
        metadata = context.metadata.copy()
        metadata.custom[MOTION_METADATA_KEY] = motion
        return context.with_metadata(metadata, copy_estimated_noise=False)

    def _get_parameters(self) -> dict[str, object]:
        return {
            "rp_file": self.rp_file,
            "window_size": self.window_size,
            "motion_threshold": self.motion_threshold,
            "motion_window_size": self.motion_window_size,
            "plot_artifacts": self.plot_artifacts,
            "realign_after_averaging": self.realign_after_averaging,
            "search_window_factor": self.search_window_factor,
            "apply_epoch_alpha_scaling": self.apply_epoch_alpha_scaling,
            "track_estimated_noise": self.track_estimated_noise,
        }


AveragedArtifactSubtraction = AASCorrection
FARMArtifactCorrection = FARMCorrection
AvgArtWghtCorrespondingSliceCorrection = CorrespondingSliceCorrection
AvgArtWghtVolumeTriggerCorrection = VolumeTriggerCorrection
AvgArtWghtSliceTriggerCorrection = SliceTriggerCorrection
AvgArtWghtMoosmannCorrection = MoosmannCorrection


__all__ = [
    "AASCorrection",
    "AveragedArtifactSubtraction",
    "AvgArtWghtCorrespondingSliceCorrection",
    "AvgArtWghtMoosmannCorrection",
    "AvgArtWghtSliceTriggerCorrection",
    "AvgArtWghtVolumeTriggerCorrection",
    "CorrespondingSliceCorrection",
    "FARMArtifactCorrection",
    "FARMCorrection",
    "MoosmannCorrection",
    "SliceTriggerCorrection",
    "VolumeTriggerCorrection",
]
