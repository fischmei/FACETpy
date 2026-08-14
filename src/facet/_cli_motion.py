"""Motion-sidecar adaptation for the CLI's Moosmann-like Flex preset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from .core import ProcessingContext, Processor, ProcessorValidationError
from .correction.flex import MOTION_METADATA_KEY, MotionEpochMetadata
from .io import Loader
from .preprocessing import TriggerDetector


class CLISlicesPerVolumeInjector(Processor):
    """Apply an explicit CLI slice count before Flex resolves its metadata."""

    name = "cli_slices_per_volume_injector"
    description = "Set the slice count used by structural Flex sampling"
    version = "1.0.0"
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, slices_per_volume: int) -> None:
        self.slices_per_volume = int(slices_per_volume)
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if self.slices_per_volume < 1:
            raise ProcessorValidationError(f"slices_per_volume must be at least one, got {self.slices_per_volume}")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        metadata = context.metadata.copy()
        metadata.slices_per_volume = self.slices_per_volume
        return context.with_metadata(metadata, copy_estimated_noise=False)


class CLIMotionMetadataInjector(Processor):
    """Attach SPM realignment rows to the artifact epochs in one CLI chunk.

    Realignment sidecars normally contain one row per fMRI volume, whereas
    slice-trigger correction has several artifact epochs per volume.  The
    injector resolves each chunk's triggers back to their full-recording
    ordinal and stores an explicit epoch-to-motion mapping for Flex.
    """

    name = "cli_motion_metadata_injector"
    description = "Map an SPM motion sidecar onto Flex artifact epochs"
    version = "1.0.0"
    requires_raw = True
    requires_triggers = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        *,
        rp_file: str,
        trigger_regex: str,
        motion_threshold: float = 5.0,
    ) -> None:
        self.rp_file = str(rp_file)
        self.trigger_regex = str(trigger_regex)
        self.motion_threshold = float(motion_threshold)
        self._full_trigger_cache: dict[str, np.ndarray] = {}
        super().__init__()

    def _get_parameters(self) -> dict[str, object]:
        return {
            "rp_file": self.rp_file,
            "trigger_regex": self.trigger_regex,
            "motion_threshold": self.motion_threshold,
        }

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if self.motion_threshold <= 0.0:
            raise ProcessorValidationError(f"motion_threshold must be positive, got {self.motion_threshold}")
        path = Path(self.rp_file)
        if not path.is_file():
            raise ProcessorValidationError(f"Motion realignment parameter file not found: {path}")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        parameters = self._load_parameters(Path(self.rp_file))
        triggers = np.asarray(context.get_triggers(), dtype=int)
        global_indices = self._global_trigger_indices(context, triggers)
        slices_per_volume = self._resolve_slices_per_volume(
            context,
            n_motion_rows=len(parameters),
            n_global_triggers=(int(np.max(global_indices)) + 1 if global_indices.size else len(triggers)),
        )

        motion_indices = global_indices if slices_per_volume is None else global_indices // slices_per_volume

        required_rows = int(np.max(motion_indices)) + 1 if motion_indices.size else 0
        if required_rows > len(parameters):
            # Match the former Moosmann helper's dummy-scan convention by
            # padding missing early motion rows with stationary zeros.
            missing = required_rows - len(parameters)
            parameters = np.vstack((np.zeros((missing, parameters.shape[1])), parameters))
            logger.info("Padded motion sidecar with {} stationary dummy row(s)", missing)

        if motion_indices.size and np.max(motion_indices) >= len(parameters):
            raise ProcessorValidationError("Motion sidecar could not be aligned to every artifact epoch in this chunk.")

        stable = self._stable_motion_rows(parameters)
        motion = MotionEpochMetadata(
            parameters=parameters,
            stable=stable,
            epoch_to_motion_index=motion_indices,
            # The archived implementation used translation-only distances.
            rotation_scale=0.0,
        )
        metadata = context.metadata.copy()
        metadata.custom[MOTION_METADATA_KEY] = motion
        metadata.custom["moosmann_motion_mapping"] = {
            "rp_file": str(Path(self.rp_file).resolve()),
            "motion_threshold": self.motion_threshold,
            "motion_rows": len(parameters),
            "artifact_epochs": len(triggers),
            "slices_per_volume": slices_per_volume,
        }
        logger.info(
            "Mapped {} artifact epoch(s) to {} motion row(s) for the Moosmann-like Flex preset",
            len(triggers),
            len(parameters),
        )
        return context.with_metadata(metadata, copy_estimated_noise=False)

    @staticmethod
    def _load_parameters(path: Path) -> np.ndarray:
        try:
            values = np.loadtxt(path, comments="#", ndmin=2)
        except ValueError:
            # Some realignment exports include one un-commented column-name
            # row. SPM's numeric files remain the primary format, but accepting
            # that common header keeps the sidecar option practical.
            try:
                values = np.loadtxt(path, comments="#", skiprows=1, ndmin=2)
            except (OSError, ValueError) as exc:
                raise ProcessorValidationError(f"Could not read motion sidecar {path}: {exc}") from exc
        except OSError as exc:
            raise ProcessorValidationError(f"Could not read motion sidecar {path}: {exc}") from exc
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] not in {3, 6} or not np.all(np.isfinite(values)):
            raise ProcessorValidationError("Motion sidecar must contain finite numeric rows with three or six columns.")
        if len(values) == 0:
            raise ProcessorValidationError("Motion sidecar did not contain any parameter rows.")
        return values

    def _global_trigger_indices(
        self,
        context: ProcessingContext,
        local_triggers: np.ndarray,
    ) -> np.ndarray:
        chunk = context.metadata.custom.get("chunk", {})
        source_path = str(chunk.get("source_path", "")) if isinstance(chunk, dict) else ""
        if not source_path:
            return np.arange(len(local_triggers), dtype=int)

        full_triggers = self._full_recording_triggers(source_path)
        if len(full_triggers) == 0:
            raise ProcessorValidationError("No full-recording triggers were available for motion alignment.")
        factor = max(int(context.metadata.upsampling_factor), 1)
        chunk_start = int(chunk.get("start_sample", 0))
        absolute = chunk_start + np.rint(local_triggers / factor).astype(int)
        indices = np.searchsorted(full_triggers, absolute)
        indices = np.clip(indices, 0, max(len(full_triggers) - 1, 0))
        previous = np.maximum(indices - 1, 0)
        choose_previous = np.abs(full_triggers[previous] - absolute) < np.abs(full_triggers[indices] - absolute)
        indices[choose_previous] = previous[choose_previous]
        tolerance = max(1, int(round(context.get_sfreq() / factor * 0.01)))
        if np.any(np.abs(full_triggers[indices] - absolute) > tolerance):
            raise ProcessorValidationError(
                "Chunk triggers could not be matched reliably to the full recording for motion alignment."
            )
        return indices.astype(int, copy=False)

    def _full_recording_triggers(self, source_path: str) -> np.ndarray:
        cached = self._full_trigger_cache.get(source_path)
        if cached is not None:
            return cached
        probe = Loader(path=source_path, preload=False).execute(None)
        probe = TriggerDetector(regex=self.trigger_regex).execute(probe)
        triggers = np.asarray(probe.get_triggers(), dtype=int)
        self._full_trigger_cache[source_path] = triggers
        return triggers

    @staticmethod
    def _resolve_slices_per_volume(
        context: ProcessingContext,
        *,
        n_motion_rows: int,
        n_global_triggers: int,
    ) -> int | None:
        value = context.metadata.slices_per_volume
        if value is None:
            value = context.metadata.custom.get("slices_per_volume")
        if value is not None:
            return max(1, int(value))
        if n_motion_rows < n_global_triggers:
            inferred = max(1, int(round(n_global_triggers / n_motion_rows)))
            logger.warning(
                "slices_per_volume was unavailable; inferred {} from {} triggers and {} motion rows",
                inferred,
                n_global_triggers,
                n_motion_rows,
            )
            return inferred
        return None

    def _stable_motion_rows(self, parameters: np.ndarray) -> np.ndarray:
        translations = parameters[:, :3]
        increments = np.zeros(len(parameters), dtype=float)
        if len(parameters) > 1:
            increments[1:] = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        return increments <= self.motion_threshold


__all__ = ["CLIMotionMetadataInjector", "CLISlicesPerVolumeInjector"]
