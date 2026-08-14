"""Nested multi-objective optimization of composable Flex matrix decisions.

This runner searches only the construction of averaging matrix ``A``.  The
surrounding Flex lifecycle remains fixed::

    D -> build A -> N = A @ D -> realign -> alpha scale -> subtract

The search is deliberately nested and multi-fidelity.  For every outer
leave-one-dataset-out fold, multi-objective TPE first screens recipes on a
training-only subset.  Screening-Pareto recipes are then evaluated on every
training dataset, one recipe is selected from that full-training Pareto front,
and the held-out dataset is evaluated exactly once.  The held-out metrics
never affect optimization or recipe selection.

The optimization objectives are:

* minimize residual power at automatically detected scanner spectral peaks;
* minimize physiological-band deviation outside those scanner peaks.

Theta, alpha, and non-scanner beta preservation are constrained relative to
the matching uncorrected scanner-on signal. This prevents broadband artifact
removal from being mistaken for physiological signal preservation.

Optuna's multi-objective TPE is used because this decision graph mixes
categorical, conditional, integer, and continuous choices.  A dense grid grows
combinatorially and spends most evaluations on uninformative combinations.

References for the named anchor recipes are recorded in
``facet.correction.flex.decisions``.  The anchors in this file are graph-space
representations used to seed optimization; the historical blockwise AAS rule
is not falsely claimed to be identical to a per-target recipe.

Example
-------
Install the optional optimizer and start a resumable run::

    uv sync --extra optimization
    uv run python examples/run_matrix_optimization.py \
        /path/to/eeg /path/to/results --recursive --trials 120
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

# Keep one optimization trial from multiplying BLAS threads internally.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from scipy.signal import find_peaks, welch  # noqa: E402

from facet import (  # noqa: E402
    MOTION_METADATA_KEY,
    CandidateScoringMode,
    CandidateScoringPolicy,
    DirectionalQuota,
    DownSample,
    Flex,
    Loader,
    MatrixDecisionError,
    MatrixDecisions,
    MedianArtifactCalculator,
    MotionEligibility,
    MotionEpochMetadata,
    Pipeline,
    ProcessingContext,
    Processor,
    ProcessorValidationError,
    RMSResidualCalculator,
    SamplingMode,
    SamplingPolicy,
    SNRCalculator,
    TargetPolicy,
    TemplateSizeMode,
    TemplateSizePolicy,
    TemporalDistanceUnit,
    TriggerDetector,
    UpSample,
    WeightingBasis,
    WeightingKernel,
    WeightingPolicy,
)

OBJECTIVE_COLUMNS = ("scanner_peak_residual", "nonpeak_eeg_log_deviation")
OBJECTIVE_DIRECTIONS = ("minimize", "minimize")
SPECTRAL_EVALUATION_COLUMNS = (
    "scanner_peak_residual",
    "theta_preservation",
    "alpha_preservation",
    "nonpeak_beta_preservation",
    "nonpeak_eeg_log_deviation",
)
RAW_EVALUATION_COLUMNS = (*SPECTRAL_EVALUATION_COLUMNS, "snr", "rms_log_deviation", "median_artifact_log_deviation")
BAD_OBJECTIVES = (1.0e6, 1.0e6)
WINDOW_SIZE_CHOICES = (10, 20, 30)
K_CHOICES = (5, 10)
SEARCH_SPACE_VERSION = 3
EVALUATION_CACHE_VERSION = 2


def _load_optuna() -> Any:
    """Import the optional optimization dependency with an actionable error."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise RuntimeError("This runner requires Optuna. Install it with `uv sync --extra optimization`.") from exc
    return optuna


def _python_scalar(value: Any) -> Any:
    """Convert NumPy scalar values before JSON serialization."""
    return value.item() if isinstance(value, np.generic) else value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a deterministic JSON report."""
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=_python_scalar))


def _atomic_write_text(path: Path, contents: str) -> None:
    """Replace a report without leaving a partially written checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically persist a tabular checkpoint."""
    _atomic_write_text(path, frame.to_csv(index=False))


def release_process_memory() -> None:
    """Release Python objects and return unused glibc arenas when possible."""
    gc.collect()
    try:
        import ctypes

        malloc_trim = getattr(ctypes.CDLL("libc.so.6"), "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except (OSError, AttributeError):
        pass


@dataclass(frozen=True)
class MotionSource:
    """Full-recording motion arrays and artifact-epoch mapping."""

    parameters: np.ndarray | None
    segment_ids: np.ndarray | None
    stable: np.ndarray | None
    epoch_to_motion_index: np.ndarray
    rotation_scale: float

    @property
    def has_parameters(self) -> bool:
        return self.parameters is not None

    @property
    def has_segments(self) -> bool:
        return self.segment_ids is not None

    @property
    def has_stable_mask(self) -> bool:
        return self.stable is not None

    def fingerprint(self) -> str:
        """Hash motion content so changed sidecars invalidate cached metrics."""
        digest = hashlib.sha256()
        for value in (
            self.parameters,
            self.segment_ids,
            self.stable,
            self.epoch_to_motion_index,
        ):
            if value is None:
                digest.update(b"none")
                continue
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        digest.update(str(self.rotation_scale).encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class DatasetSpec:
    """Preflight metadata needed for valid conditional search choices."""

    dataset_id: str
    path: Path
    trigger_samples: np.ndarray
    slices_per_volume: int | None
    motion: MotionSource | None = None

    def signature_payload(self) -> dict[str, Any]:
        """Return provenance fields that make metric-cache reuse safe."""
        stat = self.path.stat()
        return {
            "path": str(self.path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "trigger_hash": hashlib.sha256(self.trigger_samples.tobytes()).hexdigest(),
            "slices_per_volume": self.slices_per_volume,
            "motion_hash": None if self.motion is None else self.motion.fingerprint(),
        }


@dataclass(frozen=True)
class SearchCapabilities:
    """Features guaranteed to exist for every dataset in one training fold."""

    same_slice_phase: bool
    motion_parameters: bool
    motion_segments: bool
    motion_stable_mask: bool

    @classmethod
    def from_datasets(cls, datasets: Sequence[DatasetSpec]) -> SearchCapabilities:
        return cls(
            same_slice_phase=all(item.slices_per_volume is not None for item in datasets),
            motion_parameters=all(item.motion is not None and item.motion.has_parameters for item in datasets),
            motion_segments=all(item.motion is not None and item.motion.has_segments for item in datasets),
            motion_stable_mask=all(item.motion is not None and item.motion.has_stable_mask for item in datasets),
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "same_slice_phase": self.same_slice_phase,
            "motion_parameters": self.motion_parameters,
            "motion_segments": self.motion_segments,
            "motion_stable_mask": self.motion_stable_mask,
        }


@dataclass(frozen=True)
class RunSettings:
    """Fixed correction, chunking, and optimization settings."""

    trigger_regex: str
    upsample_factor: int
    chunk_padding_seconds: float
    chunk_min_triggers: int
    chunk_gap_seconds: float | None
    preservation_ratio_bound: float
    beta_preservation_minimum: float
    beta_preservation_maximum: float
    scanner_peak_minimum_hz: float
    scanner_peak_maximum_hz: float
    scanner_peak_prominence_db: float
    scanner_peak_half_width_hz: float
    max_motion_distance_low: float
    max_motion_distance_high: float

    @property
    def log_ratio_tolerance(self) -> float:
        return math.log(self.preservation_ratio_bound)

    def signature_payload(self) -> dict[str, Any]:
        return {
            "search_space_version": SEARCH_SPACE_VERSION,
            "trigger_regex": self.trigger_regex,
            "upsample_factor": self.upsample_factor,
            "chunk_padding_seconds": self.chunk_padding_seconds,
            "chunk_min_triggers": self.chunk_min_triggers,
            "chunk_gap_seconds": self.chunk_gap_seconds,
            "preservation_ratio_bound": self.preservation_ratio_bound,
            "beta_preservation": [self.beta_preservation_minimum, self.beta_preservation_maximum],
            "scanner_peak_detection": {
                "minimum_hz": self.scanner_peak_minimum_hz,
                "maximum_hz": self.scanner_peak_maximum_hz,
                "prominence_db": self.scanner_peak_prominence_db,
                "half_width_hz": self.scanner_peak_half_width_hz,
            },
            "max_motion_distance": [self.max_motion_distance_low, self.max_motion_distance_high],
            # These lifecycle settings are fixed rather than optimized.
            "realign_after_averaging": True,
            "interpolate_volume_gaps": True,
            "apply_epoch_alpha_scaling": True,
            "objective_definition": (
                "scanner_peak_residual|min;mean_abs_log_nonpeak_theta_alpha_beta|min;band_preservation_constraints"
            ),
        }

    def evaluation_signature_payload(self) -> dict[str, Any]:
        """Return only settings that can change raw per-recipe metrics."""
        return {
            "evaluation_cache_version": EVALUATION_CACHE_VERSION,
            "trigger_regex": self.trigger_regex,
            "upsample_factor": self.upsample_factor,
            "chunk_padding_seconds": self.chunk_padding_seconds,
            "chunk_min_triggers": self.chunk_min_triggers,
            "chunk_gap_seconds": self.chunk_gap_seconds,
            "realign_after_averaging": True,
            "interpolate_volume_gaps": True,
            "apply_epoch_alpha_scaling": True,
            "scanner_peak_detection": {
                "minimum_hz": self.scanner_peak_minimum_hz,
                "maximum_hz": self.scanner_peak_maximum_hz,
                "prominence_db": self.scanner_peak_prominence_db,
                "half_width_hz": self.scanner_peak_half_width_hz,
            },
        }


def find_input_datasets(input_directory: Path, *, recursive: bool) -> list[Path]:
    """Find supported recording paths in deterministic order."""
    iterator = input_directory.rglob("*") if recursive else input_directory.glob("*")
    supported = {".edf", ".bdf", ".gdf", ".vhdr", ".set", ".fif", ".mff"}
    datasets = sorted(
        path
        for path in iterator
        if path.suffix.lower() in supported and (path.is_file() or path.suffix.lower() == ".mff")
    )
    if len(datasets) < 3:
        raise ValueError("Nested leave-one-dataset-out optimization requires at least three datasets.")
    return datasets


def unique_dataset_ids(paths: Sequence[Path]) -> dict[str, Path]:
    """Build stable, collision-free dataset identifiers."""
    resolved: dict[str, Path] = {}
    for path in paths:
        candidate = path.stem
        if candidate in resolved:
            candidate = f"{path.parent.name}__{path.stem}"
        base = candidate
        suffix = 2
        while candidate in resolved:
            candidate = f"{base}__{suffix}"
            suffix += 1
        resolved[candidate] = path
    return resolved


def _load_motion_sidecar(
    sidecar: Path,
    *,
    n_epochs: int,
    slices_per_volume: int | None,
    rotation_scale: float,
) -> MotionSource:
    """Load an ``.npz``, ``.npy``, or text motion sidecar.

    ``.npz`` files may contain ``parameters``, ``segment_ids``, ``stable``,
    and ``epoch_to_motion_index``.  Plain arrays are interpreted as motion
    parameters.  When no explicit mapping is supplied, rows are accepted only
    when they map unambiguously one-to-one to artifact epochs or volumes.
    """
    suffix = sidecar.suffix.lower()
    if suffix == ".npz":
        with np.load(sidecar, allow_pickle=False) as archive:
            parameters = archive.get("parameters")
            segment_ids = archive.get("segment_ids")
            stable = archive.get("stable")
            mapping = archive.get("epoch_to_motion_index")
    else:
        parameters = np.load(sidecar, allow_pickle=False) if suffix == ".npy" else np.loadtxt(sidecar)
        segment_ids = None
        stable = None
        mapping = None

    arrays = [value for value in (parameters, segment_ids, stable) if value is not None]
    if not arrays:
        raise ValueError(f"Motion sidecar {sidecar} contains no motion arrays.")
    source_length = len(arrays[0])
    if any(len(value) != source_length for value in arrays):
        raise ValueError(f"Motion arrays in {sidecar} must have the same number of rows.")

    if mapping is None:
        if source_length == n_epochs:
            mapping = np.arange(n_epochs, dtype=int)
        elif slices_per_volume is not None and source_length >= math.ceil(n_epochs / slices_per_volume):
            mapping = np.arange(n_epochs, dtype=int) // slices_per_volume
        else:
            raise ValueError(
                f"Cannot map {source_length} motion rows in {sidecar} to {n_epochs} artifact epochs. "
                "Provide epoch_to_motion_index in an .npz sidecar."
            )

    mapping = np.asarray(mapping)
    if mapping.ndim != 1 or len(mapping) != n_epochs or not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("epoch_to_motion_index must contain one integer for every full-recording artifact epoch.")
    mapping = mapping.astype(int, copy=False)
    if np.any(mapping < 0) or np.any(mapping >= source_length):
        raise ValueError("epoch_to_motion_index contains an out-of-range motion row.")

    # Let MotionEpochMetadata perform the remaining shape and type validation.
    validated = MotionEpochMetadata(
        parameters=parameters,
        segment_ids=segment_ids,
        stable=stable,
        epoch_to_motion_index=mapping,
        rotation_scale=rotation_scale,
    )
    return MotionSource(
        parameters=validated.parameters,
        segment_ids=validated.segment_ids,
        stable=validated.stable,
        epoch_to_motion_index=validated.epoch_to_motion_index,
        rotation_scale=validated.rotation_scale,
    )


def _find_motion_sidecar(motion_directory: Path, dataset_id: str, path: Path) -> Path | None:
    """Return the first deterministic sidecar matching a dataset."""
    for stem in (dataset_id, path.stem):
        for suffix in (".npz", ".npy", ".txt", ".tsv"):
            candidate = motion_directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def preflight_datasets(
    paths: Mapping[str, Path],
    *,
    trigger_regex: str,
    motion_directory: Path | None,
    rotation_scale: float,
) -> dict[str, DatasetSpec]:
    """Detect full-recording triggers and validate optional motion mappings."""
    output: dict[str, DatasetSpec] = {}
    for dataset_id, path in paths.items():
        logger.info("Preflighting dataset '{}': {}", dataset_id, path)
        context = Loader(path=str(path), preload=False).execute(None)
        try:
            context = TriggerDetector(regex=trigger_regex).execute(context)
            if not context.has_triggers():
                raise ValueError(f"No triggers matching {trigger_regex!r} were found in {path}.")
            triggers = np.asarray(context.get_triggers(), dtype=int)
            slices_per_volume = context.metadata.slices_per_volume
        finally:
            del context  # Release memory before loading the next dataset. CHECK HERE

        motion = None
        if motion_directory is not None:
            sidecar = _find_motion_sidecar(motion_directory, dataset_id, path)
            if sidecar is None:
                logger.warning(
                    "No motion sidecar found for '{}'; motion branches are disabled where required.", dataset_id
                )
            else:
                motion = _load_motion_sidecar(
                    sidecar,
                    n_epochs=len(triggers),
                    slices_per_volume=slices_per_volume,
                    rotation_scale=rotation_scale,
                )

        output[dataset_id] = DatasetSpec(
            dataset_id=dataset_id,
            path=path,
            trigger_samples=triggers,
            slices_per_volume=slices_per_volume,
            motion=motion,
        )
    return output


class MotionMetadataInjector(Processor):
    """Map full-recording motion metadata onto triggers in one chunk."""

    name = "motion_metadata_injector"
    description = "Attach artifact-epoch motion metadata for matrix construction"
    requires_triggers = True
    requires_raw = False
    modifies_raw = False
    parallel_safe = False

    def __init__(self, dataset: DatasetSpec) -> None:
        if dataset.motion is None:
            raise ValueError("MotionMetadataInjector requires a motion sidecar.")
        self.dataset = dataset
        super().__init__()

    def _get_parameters(self) -> dict[str, Any]:
        return {"dataset_id": self.dataset.dataset_id}

    def process(self, context: ProcessingContext) -> ProcessingContext:
        chunk = context.metadata.custom.get("chunk", {})
        chunk_start = int(chunk.get("start_sample", 0)) if isinstance(chunk, Mapping) else 0
        global_triggers = np.asarray(context.get_triggers(), dtype=int) + chunk_start
        full_lookup = {int(sample): index for index, sample in enumerate(self.dataset.trigger_samples)}
        try:
            full_epoch_indices = np.asarray([full_lookup[int(sample)] for sample in global_triggers], dtype=int)
        except KeyError as exc:
            raise ProcessorValidationError(
                "A chunk trigger could not be mapped to the full-recording trigger list. "
                "Motion-dependent decisions cannot be evaluated safely."
            ) from exc

        source = self.dataset.motion
        motion = MotionEpochMetadata(
            parameters=source.parameters,
            segment_ids=source.segment_ids,
            stable=source.stable,
            epoch_to_motion_index=source.epoch_to_motion_index[full_epoch_indices],
            rotation_scale=source.rotation_scale,
        )
        metadata = context.metadata.copy()
        metadata.custom[MOTION_METADATA_KEY] = motion
        return context.with_metadata(metadata, copy_estimated_noise=False)


def _suggest_quota(trial: Any) -> DirectionalQuota:
    """Suggest a finite topology or the global pool."""
    mode = trial.suggest_categorical(
        "quota_mode",
        ["future", "past", "symmetric", "past_heavy", "future_heavy", "custom", "global"],
    )
    if mode == "global":
        return DirectionalQuota.global_pool()

    window_size = trial.suggest_categorical("window_size", list(WINDOW_SIZE_CHOICES))
    if mode == "future":
        return DirectionalQuota.future_only(window_size)
    if mode == "past":
        return DirectionalQuota.past_only(window_size)
    if mode == "symmetric":
        return DirectionalQuota.symmetric(window_size)
    if mode == "past_heavy":
        return DirectionalQuota.past_heavy(window_size)
    if mode == "future_heavy":
        return DirectionalQuota.future_heavy(window_size)

    past_fraction = trial.suggest_int("custom_past_percent", 0, 100, step=10)
    past = int(round(window_size * past_fraction / 100.0))
    return DirectionalQuota.custom(past=past, future=window_size - past, window_size=window_size)


def _suggest_sampling(trial: Any, capabilities: SearchCapabilities) -> SamplingPolicy:
    modes = [SamplingMode.CONSECUTIVE.value, SamplingMode.ALTERNATING.value]
    if capabilities.same_slice_phase:
        modes.append(SamplingMode.SAME_SLICE_PHASE.value)
    selected = trial.suggest_categorical("sampling_mode", modes)
    if selected == SamplingMode.ALTERNATING.value:
        return SamplingPolicy.alternating()
    if selected == SamplingMode.SAME_SLICE_PHASE.value:
        return SamplingPolicy.same_slice_phase()
    return SamplingPolicy.consecutive()


def _suggest_motion_eligibility(
    trial: Any,
    capabilities: SearchCapabilities,
    settings: RunSettings,
) -> MotionEligibility:
    modes = ["none"]
    if capabilities.motion_parameters:
        modes.append("max_distance")
    if capabilities.motion_segments:
        modes.append("same_segment")
    if capabilities.motion_stable_mask:
        modes.append("stable_only")
    if capabilities.motion_segments and capabilities.motion_stable_mask:
        modes.append("same_segment_and_stable")

    mode = trial.suggest_categorical("motion_eligibility", modes)
    maximum = None
    if mode == "max_distance":
        maximum = trial.suggest_float(
            "max_motion_distance",
            settings.max_motion_distance_low,
            settings.max_motion_distance_high,
            log=True,
        )
    return MotionEligibility(
        same_motion_segment=mode in {"same_segment", "same_segment_and_stable"},
        motion_stable_only=mode in {"stable_only", "same_segment_and_stable"},
        max_motion_distance=maximum,
    )


def _suggest_scoring_and_size(
    trial: Any,
    capabilities: SearchCapabilities,
) -> tuple[CandidateScoringPolicy, TemplateSizePolicy]:
    scoring_choices = [
        CandidateScoringMode.SIGNED_PEARSON.value,
        CandidateScoringMode.ABSOLUTE_PEARSON.value,
        CandidateScoringMode.TEMPORAL_MOTION_COST.value,
        CandidateScoringMode.NONE.value,
    ]
    mode = trial.suggest_categorical("scoring_mode", scoring_choices)

    if mode == CandidateScoringMode.NONE.value:
        return CandidateScoringPolicy.none(), TemplateSizePolicy.select_all()

    k = trial.suggest_categorical("k", list(K_CHOICES))
    if mode == CandidateScoringMode.TEMPORAL_MOTION_COST.value:
        size_mode = trial.suggest_categorical(
            "cost_template_size_mode",
            [TemplateSizeMode.MAXIMUM_K.value, TemplateSizeMode.EXACTLY_K.value],
        )
        motion_fraction = (
            trial.suggest_float("cost_motion_fraction", 0.0, 1.0, step=0.1) if capabilities.motion_parameters else 0.0
        )
        scoring = CandidateScoringPolicy.temporal_motion_cost(
            temporal_weight=1.0 - motion_fraction,
            motion_weight=motion_fraction,
            temporal_unit=TemporalDistanceUnit.INDEX,
        )
    else:
        size_mode = trial.suggest_categorical(
            "pearson_template_size_mode",
            [
                TemplateSizeMode.MINIMUM_K.value,
                TemplateSizeMode.MAXIMUM_K.value,
                TemplateSizeMode.EXACTLY_K.value,
            ],
        )
        threshold = trial.suggest_float("correlation_threshold", 0.90, 0.995, step=0.005)
        scoring = (
            CandidateScoringPolicy.signed_pearson(threshold)
            if mode == CandidateScoringMode.SIGNED_PEARSON.value
            else CandidateScoringPolicy.absolute_pearson(threshold)
        )

    if size_mode == TemplateSizeMode.MINIMUM_K.value:
        size = TemplateSizePolicy.minimum(k)
    elif size_mode == TemplateSizeMode.MAXIMUM_K.value:
        size = TemplateSizePolicy.maximum(k)
    else:
        size = TemplateSizePolicy.exactly(k)
    return scoring, size


def _suggest_weighting(trial: Any, capabilities: SearchCapabilities) -> WeightingPolicy:
    kernel = trial.suggest_categorical(
        "weighting_kernel",
        [
            WeightingKernel.EQUAL.value,
            WeightingKernel.GAUSSIAN.value,
            WeightingKernel.LAPLACE.value,
            WeightingKernel.STUDENT_T.value,
        ],
    )
    if kernel == WeightingKernel.EQUAL.value:
        return WeightingPolicy.equal()

    bases = [WeightingBasis.TEMPORAL.value]
    if capabilities.motion_parameters:
        bases.append(WeightingBasis.MOTION.value)
    basis = WeightingBasis(trial.suggest_categorical("weighting_basis", bases))
    temporal_unit = TemporalDistanceUnit.INDEX if basis is WeightingBasis.TEMPORAL else None
    width = trial.suggest_float("kernel_width", 0.5, 30.0, log=True)
    if kernel == WeightingKernel.GAUSSIAN.value:
        return WeightingPolicy.gaussian(basis=basis, temporal_unit=temporal_unit, sigma=width)
    if kernel == WeightingKernel.LAPLACE.value:
        return WeightingPolicy.laplace(basis=basis, temporal_unit=temporal_unit, scale=width)
    degrees = trial.suggest_float("student_degrees_of_freedom", 1.0, 30.0, log=True)
    return WeightingPolicy.student_t(
        basis=basis,
        temporal_unit=temporal_unit,
        scale=width,
        degrees_of_freedom=degrees,
    )


def suggest_matrix_decisions(
    trial: Any,
    capabilities: SearchCapabilities,
    settings: RunSettings,
) -> MatrixDecisions:
    """Construct one valid conditional recipe without named-algorithm branches."""
    scoring, template_size = _suggest_scoring_and_size(trial, capabilities)
    return MatrixDecisions(
        quota=_suggest_quota(trial),
        sampling=_suggest_sampling(trial, capabilities),
        motion=_suggest_motion_eligibility(trial, capabilities, settings),
        target_policy=TargetPolicy(
            trial.suggest_categorical(
                "target_policy",
                [TargetPolicy.INCLUDE.value, TargetPolicy.EXCLUDE.value],
            )
        ),
        scoring=scoring,
        template_size=template_size,
        weighting=_suggest_weighting(trial, capabilities),
    )


def configuration_id(decisions: MatrixDecisions) -> str:
    """Hash the complete serialized A-matrix recipe."""
    encoded = json.dumps(decisions.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _fixed_parameters(**updates: Any) -> dict[str, Any]:
    """Return a complete valid parameter set for an enqueued equal-weight trial."""
    parameters: dict[str, Any] = {
        "quota_mode": "future",
        "window_size": 10,
        "sampling_mode": SamplingMode.CONSECUTIVE.value,
        "motion_eligibility": "none",
        "target_policy": TargetPolicy.EXCLUDE.value,
        "scoring_mode": CandidateScoringMode.SIGNED_PEARSON.value,
        "k": 5,
        "pearson_template_size_mode": TemplateSizeMode.MINIMUM_K.value,
        "correlation_threshold": 0.975,
        "weighting_kernel": WeightingKernel.EQUAL.value,
    }
    parameters.update(updates)
    return parameters


def anchor_parameter_sets(capabilities: SearchCapabilities) -> list[tuple[str, dict[str, Any]]]:
    """Return legacy-inspired reference points represented by the graph.

    AAS is explicitly labelled ``aas_per_target`` because Allen's historical
    blockwise evolving-average construction is not identical to any
    independently per-target recipe.  This distinction is retained in every
    report instead of overstating legacy recovery.
    """
    anchors = [
        ("flex_default", _fixed_parameters()),
        ("aas_per_target", _fixed_parameters(target_policy=TargetPolicy.INCLUDE.value)),
        (
            "farm_per_target_k10",
            _fixed_parameters(
                quota_mode="symmetric",
                window_size=30,
                scoring_mode=CandidateScoringMode.ABSOLUTE_PEARSON.value,
                pearson_template_size_mode=TemplateSizeMode.MAXIMUM_K.value,
                correlation_threshold=0.90,
                k=10,
            ),
        ),
        (
            "structural_volume",
            _fixed_parameters(
                quota_mode="custom",
                window_size=10,
                custom_past_percent=60,
                target_policy=TargetPolicy.INCLUDE.value,
                scoring_mode=CandidateScoringMode.NONE.value,
            ),
        ),
        (
            "structural_slice",
            _fixed_parameters(
                sampling_mode=SamplingMode.ALTERNATING.value,
                scoring_mode=CandidateScoringMode.NONE.value,
            ),
        ),
    ]
    if capabilities.same_slice_phase:
        anchors.append(
            (
                "corresponding_slice",
                _fixed_parameters(
                    quota_mode="global",
                    sampling_mode=SamplingMode.SAME_SLICE_PHASE.value,
                    target_policy=TargetPolicy.INCLUDE.value,
                    scoring_mode=CandidateScoringMode.NONE.value,
                ),
            )
        )
    if capabilities.motion_parameters:
        anchors.append(
            (
                "moosmann_cost",
                _fixed_parameters(
                    quota_mode="global",
                    target_policy=TargetPolicy.INCLUDE.value,
                    scoring_mode=CandidateScoringMode.TEMPORAL_MOTION_COST.value,
                    cost_template_size_mode=TemplateSizeMode.EXACTLY_K.value,
                    cost_motion_fraction=0.5,
                    k=5,
                ),
            )
        )
    return anchors


def balanced_seed_parameter_sets(
    capabilities: SearchCapabilities,
    *,
    count: int,
) -> list[dict[str, Any]]:
    """Seed categorical coverage before TPE concentrates around good regions."""
    quota_modes = ["future", "past", "symmetric", "past_heavy", "future_heavy", "custom", "global"]
    sampling_modes = [SamplingMode.CONSECUTIVE.value, SamplingMode.ALTERNATING.value]
    if capabilities.same_slice_phase:
        sampling_modes.append(SamplingMode.SAME_SLICE_PHASE.value)
    scoring_modes = [
        CandidateScoringMode.SIGNED_PEARSON.value,
        CandidateScoringMode.ABSOLUTE_PEARSON.value,
        CandidateScoringMode.TEMPORAL_MOTION_COST.value,
        CandidateScoringMode.NONE.value,
    ]
    kernels = [member.value for member in WeightingKernel]
    seeds: list[dict[str, Any]] = []
    for index in range(count):
        scoring = scoring_modes[index % len(scoring_modes)]
        kernel = kernels[(index // len(scoring_modes)) % len(kernels)]
        parameters = _fixed_parameters(
            quota_mode=quota_modes[index % len(quota_modes)],
            window_size=WINDOW_SIZE_CHOICES[index % len(WINDOW_SIZE_CHOICES)],
            sampling_mode=sampling_modes[index % len(sampling_modes)],
            target_policy=(TargetPolicy.INCLUDE.value if index % 2 else TargetPolicy.EXCLUDE.value),
            scoring_mode=scoring,
            k=K_CHOICES[index % len(K_CHOICES)],
            weighting_kernel=kernel,
        )
        if parameters["quota_mode"] == "custom":
            parameters["custom_past_percent"] = (index * 30) % 110
        if scoring in {
            CandidateScoringMode.SIGNED_PEARSON.value,
            CandidateScoringMode.ABSOLUTE_PEARSON.value,
        }:
            parameters["pearson_template_size_mode"] = [
                TemplateSizeMode.MINIMUM_K.value,
                TemplateSizeMode.MAXIMUM_K.value,
                TemplateSizeMode.EXACTLY_K.value,
            ][index % 3]
            parameters["correlation_threshold"] = (0.95, 0.975, 0.99)[index % 3]
        elif scoring == CandidateScoringMode.TEMPORAL_MOTION_COST.value:
            parameters["cost_template_size_mode"] = (
                TemplateSizeMode.MAXIMUM_K.value if index % 2 else TemplateSizeMode.EXACTLY_K.value
            )
            if capabilities.motion_parameters:
                parameters["cost_motion_fraction"] = (index % 11) / 10.0
        if kernel != WeightingKernel.EQUAL.value:
            parameters["weighting_basis"] = (
                WeightingBasis.MOTION.value
                if capabilities.motion_parameters and index % 3 == 0
                else WeightingBasis.TEMPORAL.value
            )
            parameters["kernel_width"] = (1.0, 3.0, 10.0)[index % 3]
            if kernel == WeightingKernel.STUDENT_T.value:
                parameters["student_degrees_of_freedom"] = (2.0, 5.0, 10.0)[index % 3]
        seeds.append(parameters)
    return seeds


def recipe_requires_motion(decisions: MatrixDecisions) -> bool:
    """Return whether evaluation must attach a motion sidecar."""
    return (
        decisions.motion.enabled
        or (
            decisions.scoring.mode is CandidateScoringMode.TEMPORAL_MOTION_COST
            and float(decisions.scoring.motion_weight) > 0.0
        )
        or decisions.weighting.basis is WeightingBasis.MOTION
    )


def _integrated_power(frequencies: np.ndarray, psd: np.ndarray, mask: np.ndarray) -> float:
    """Integrate PSD bins selected by a frequency mask."""
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.trapezoid(psd[mask], frequencies[mask]))


def detect_scanner_peak_mask(
    frequencies: np.ndarray,
    psd: np.ndarray,
    *,
    minimum_hz: float,
    maximum_hz: float,
    prominence_db: float,
    half_width_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect prominent scanner-contaminated peaks in channel-median PSD.

    Detection starts at 13 Hz by default so physiological theta and alpha
    peaks cannot be classified as scanner artifact. The returned evaluation
    mask uses fixed narrow neighborhoods around each detected peak.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    psd = np.asarray(psd, dtype=float)
    search = (frequencies >= minimum_hz) & (frequencies <= maximum_hz)
    search_indices = np.flatnonzero(search)
    mask = np.zeros_like(frequencies, dtype=bool)
    if len(search_indices) < 3:
        return mask, np.array([], dtype=float)
    safe_psd = np.maximum(psd[search], np.finfo(float).tiny)
    psd_db = 10.0 * np.log10(safe_psd)
    frequency_step = float(np.median(np.diff(frequencies)))
    minimum_distance = max(1, int(round(0.5 / frequency_step)))
    local_peaks, _ = find_peaks(
        psd_db,
        prominence=prominence_db,
        # Numerical spectral leakage can have large formal prominence near
        # machine zero. Keep only peaks within a meaningful dynamic range of
        # the strongest scanner-on peak.
        height=float(np.max(psd_db) - 40.0),
        distance=minimum_distance,
    )
    peak_indices = search_indices[local_peaks]
    peak_frequencies = frequencies[peak_indices]
    for peak_frequency in peak_frequencies:
        mask |= np.abs(frequencies - peak_frequency) <= half_width_hz
    return mask, peak_frequencies


def calculate_spectral_optimization_metrics(
    original: np.ndarray,
    corrected: np.ndarray,
    *,
    sfreq: float,
    settings: RunSettings,
) -> dict[str, Any]:
    """Calculate artifact suppression and scanner-on preservation metrics."""
    common_samples = min(original.shape[1], corrected.shape[1])
    if common_samples < 16 or not len(original):
        raise ProcessorValidationError("Insufficient EEG samples for spectral optimization metrics.")
    original = original[:, :common_samples]
    corrected = corrected[:, :common_samples]
    nperseg = min(4096, common_samples)
    frequencies, original_psd = welch(
        original,
        fs=sfreq,
        axis=1,
        nperseg=nperseg,
        detrend="constant",
        scaling="density",
    )
    _, corrected_psd = welch(
        corrected,
        fs=sfreq,
        axis=1,
        nperseg=nperseg,
        detrend="constant",
        scaling="density",
    )
    original_median = np.median(original_psd, axis=0)
    corrected_median = np.median(corrected_psd, axis=0)
    scanner_mask, scanner_peaks = detect_scanner_peak_mask(
        frequencies,
        original_median,
        minimum_hz=settings.scanner_peak_minimum_hz,
        maximum_hz=min(settings.scanner_peak_maximum_hz, sfreq / 2.0),
        prominence_db=settings.scanner_peak_prominence_db,
        half_width_hz=settings.scanner_peak_half_width_hz,
    )
    if np.count_nonzero(scanner_mask) < 2:
        raise ProcessorValidationError(
            "No scanner spectral peaks passed the configured detection rule; "
            "adjust --scanner-peak-prominence-db or inspect the recording PSD."
        )

    scanner_peak_residual = _safe_positive_ratio(
        _integrated_power(frequencies, corrected_median, scanner_mask),
        _integrated_power(frequencies, original_median, scanner_mask),
    )
    preservations = {}
    for name, low, high in (
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("nonpeak_beta", 13.0, 30.0),
    ):
        band_mask = (frequencies >= low) & (frequencies < min(high, sfreq / 2.0))
        if name == "nonpeak_beta":
            band_mask &= ~scanner_mask
        preservations[name] = _safe_positive_ratio(
            _integrated_power(frequencies, corrected_median, band_mask),
            _integrated_power(frequencies, original_median, band_mask),
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in preservations.values()):
        raise ProcessorValidationError("A physiological preservation band contained insufficient finite power.")
    return {
        "scanner_peak_residual": scanner_peak_residual,
        "theta_preservation": preservations["theta"],
        "alpha_preservation": preservations["alpha"],
        "nonpeak_beta_preservation": preservations["nonpeak_beta"],
        "nonpeak_eeg_log_deviation": float(np.mean([abs(math.log(value)) for value in preservations.values()])),
        "scanner_peak_count": int(len(scanner_peaks)),
        "scanner_peak_frequencies_hz": scanner_peaks.tolist(),
    }


def _safe_positive_ratio(numerator: float, denominator: float) -> float:
    """Return a positive finite ratio or NaN for an unusable denominator."""
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


class SpectralOptimizationCalculator(Processor):
    """Measure scanner-peak suppression and non-peak EEG preservation."""

    name = "spectral_optimization_calculator"
    description = "Calculate artifact-focused optimization metrics"
    version = "1.0.0"
    requires_raw = True
    requires_triggers = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, settings: RunSettings) -> None:
        self.settings = settings
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw()
        original_raw = context.get_raw_original()
        if original_raw is None:
            raise ProcessorValidationError("Original EEG is required for scanner-on normalization.")
        triggers = np.asarray(context.get_triggers(), dtype=int)
        artifact_length = context.get_artifact_length()
        if not len(triggers) or artifact_length is None:
            raise ProcessorValidationError("Triggers and artifact length are required for spectral metrics.")
        start = max(0, int(triggers[0]) - int(artifact_length * 0.5))
        stop = min(raw.n_times, int(triggers[-1]) + int(artifact_length * 1.5))
        if stop <= start:
            raise ProcessorValidationError("The inferred scanner-on evaluation interval is empty.")
        picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude="bads")
        if not len(picks):
            raise ProcessorValidationError("No good EEG channels are available for spectral metrics.")
        metrics = calculate_spectral_optimization_metrics(
            original_raw.get_data(picks=picks, start=start, stop=stop),
            raw.get_data(picks=picks, start=start, stop=stop),
            sfreq=float(raw.info["sfreq"]),
            settings=self.settings,
        )
        metadata = context.metadata.copy()
        metadata.custom.setdefault("metrics", {}).update(metrics)
        return context.with_metadata(metadata, copy_estimated_noise=False)


def build_pipeline(decisions: MatrixDecisions, dataset: DatasetSpec, settings: RunSettings) -> Pipeline:
    """Build the fixed Flex lifecycle around one variable A-matrix recipe."""
    processors: list[Processor] = [TriggerDetector(regex=settings.trigger_regex)]
    if recipe_requires_motion(decisions):
        processors.append(MotionMetadataInjector(dataset))
    processors.extend(
        [
            UpSample(factor=settings.upsample_factor),
            Flex(
                matrix_decisions=decisions,
                plot_artifacts=False,
                realign_after_averaging=True,
                interpolate_volume_gaps=True,
                apply_epoch_alpha_scaling=True,
                track_estimated_noise=False,
            ),
            DownSample(factor=settings.upsample_factor),
            SpectralOptimizationCalculator(settings),
            SNRCalculator(),
            RMSResidualCalculator(),
            MedianArtifactCalculator(),
        ]
    )
    return Pipeline(processors, name="Composable matrix optimization")


def _iter_result_items(chunked_result: Any) -> Iterable[Any]:
    """Yield results across supported ChunkedPipelineResult versions."""
    for name in ("results", "chunk_results", "pipeline_results", "_results"):
        value = getattr(chunked_result, name, None)
        if value is not None:
            yield from value.values() if isinstance(value, Mapping) else value
            return
    yield from iter(chunked_result)


def _result_context(item: Any) -> ProcessingContext | None:
    if isinstance(item, ProcessingContext):
        return item
    for name in ("context", "final_context", "result"):
        value = getattr(item, name, None)
        if isinstance(value, ProcessingContext):
            return value
    return None


def _chunk_weight(context: ProcessingContext) -> float:
    if context.has_triggers():
        return float(max(len(context.get_triggers()), 1))
    return 1.0


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Calculate a deterministic weighted median for robust chunk aggregation."""
    data = np.asarray(values, dtype=float)
    mass = np.asarray(weights, dtype=float)
    valid = np.isfinite(data) & np.isfinite(mass) & (mass > 0.0)
    if not np.any(valid):
        return float("nan")
    data = data[valid]
    mass = mass[valid]
    order = np.argsort(data, kind="stable")
    data = data[order]
    mass = mass[order]
    cutoff = 0.5 * float(np.sum(mass))
    return float(data[np.searchsorted(np.cumsum(mass), cutoff, side="left")])


def safe_log_deviation(ratio: float) -> float:
    """Return symmetric distance from the ideal ratio of one."""
    if not np.isfinite(ratio) or ratio <= 0.0:
        return float("nan")
    return float(abs(math.log(ratio)))


def normalize_snr_row(
    row: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one SNR against the same dataset's fixed Flex baseline.

    ``log(candidate_snr / baseline_snr)`` is zero for the baseline, positive
    for improvement, and negative for degradation. Both values originate from
    the same recording, channels, trigger sections, and reference definition,
    which removes the recording-specific SNR scale before cross-dataset
    aggregation.
    """
    normalized = dict(row)
    candidate_snr = float(row.get("snr", np.nan))
    baseline_snr = float(baseline_row.get("snr", np.nan))
    valid = (
        bool(row.get("success"))
        and bool(baseline_row.get("success"))
        and np.isfinite(candidate_snr)
        and np.isfinite(baseline_snr)
        and candidate_snr > 0.0
        and baseline_snr > 0.0
    )
    normalized["baseline_snr"] = baseline_snr
    normalized["normalized_snr"] = float(math.log(candidate_snr / baseline_snr)) if valid else float("nan")
    if not valid:
        normalized["success"] = False
        baseline_error = str(baseline_row.get("error", "")).strip()
        suffix = "SNR normalization baseline was unavailable or non-positive."
        normalized["error"] = " | ".join(
            part for part in (str(normalized.get("error", "")).strip(), baseline_error, suffix) if part
        )
    return normalized


def aggregate_chunk_results(chunked_result: Any) -> dict[str, Any]:
    """Robustly combine spectral objectives and legacy diagnostic metrics."""
    values: dict[str, list[float]] = {
        **{name: [] for name in SPECTRAL_EVALUATION_COLUMNS},
        "snr": [],
        "rms_residual": [],
        "median_artifact_ratio": [],
        "scanner_peak_count": [],
    }
    weights: list[float] = []
    errors: list[str] = []
    scanner_peak_frequencies: set[float] = set()
    total = 0
    successful = 0
    execution_time = 0.0

    for item in _iter_result_items(chunked_result):
        total += 1
        execution_time += float(getattr(item, "execution_time", 0.0) or 0.0)
        if not bool(getattr(item, "success", True)):
            errors.append(str(getattr(item, "error", "unknown chunk failure")))
            continue
        context = _result_context(item)
        if context is None:
            errors.append("Successful chunk did not retain a ProcessingContext.")
            continue
        metrics = context.metadata.custom.get("metrics", {})
        scanner_peak_frequencies.update(
            round(float(value), 6) for value in metrics.get("scanner_peak_frequencies_hz", [])
        )
        row = {name: metrics.get(name) for name in values}
        if any(not isinstance(value, (int, float, np.number)) or not np.isfinite(value) for value in row.values()):
            errors.append("Chunk did not produce every finite spectral and legacy diagnostic metric.")
            continue
        successful += 1
        weights.append(_chunk_weight(context))
        for name, value in row.items():
            values[name].append(float(value))

    metrics = {name: weighted_median(items, weights) for name, items in values.items()}
    metrics["rms_log_deviation"] = safe_log_deviation(metrics["rms_residual"])
    metrics["median_artifact_log_deviation"] = safe_log_deviation(metrics["median_artifact_ratio"])
    success = total > 0 and successful == total and all(np.isfinite(metrics[name]) for name in RAW_EVALUATION_COLUMNS)
    return {
        **metrics,
        "success": success,
        "n_chunks": total,
        "successful_chunks": successful,
        "execution_time": execution_time,
        "scanner_peak_frequencies_hz": json.dumps(sorted(scanner_peak_frequencies)),
        "error": " | ".join(error for error in errors if error),
    }


class EvaluationCache:
    """Dataset-by-recipe cache shared safely across all outer folds."""

    def __init__(
        self,
        path: Path,
        *,
        signature: str,
        rebuild: bool,
        compatible_signatures: Sequence[str] = (),
    ) -> None:
        self.path = path
        self.signature = signature
        self.rows: list[dict[str, Any]] = []
        if path.exists() and not rebuild:
            frame = pd.read_csv(path)
            observed = set(frame.get("evaluation_signature", pd.Series(dtype=str)).astype(str))
            accepted = {signature, *compatible_signatures}
            if not observed or not observed.issubset(accepted):
                raise RuntimeError(
                    f"Cache {path} was created with different run settings. "
                    "Use --rebuild-cache or a new output directory."
                )
            self.rows = frame.to_dict(orient="records")
            if observed != {signature}:
                # Objective-only changes do not invalidate raw correction
                # metrics, but Optuna studies still need to restart.
                for row in self.rows:
                    row["evaluation_signature"] = signature
                _atomic_write_csv(pd.DataFrame(self.rows), path)
        self._index = {(str(row["configuration_id"]), str(row["dataset_id"])): row for row in self.rows}

    def get(self, configuration: str, dataset_id: str) -> dict[str, Any] | None:
        return self._index.get((configuration, dataset_id))

    def add(self, row: dict[str, Any]) -> None:
        key = (str(row["configuration_id"]), str(row["dataset_id"]))
        if key in self._index:
            return
        row = {"evaluation_signature": self.signature, **row}
        self.rows.append(row)
        self._index[key] = row
        _atomic_write_csv(pd.DataFrame(self.rows), self.path)


def evaluate_configuration(
    decisions: MatrixDecisions,
    dataset: DatasetSpec,
    settings: RunSettings,
    cache: EvaluationCache,
) -> dict[str, Any]:
    """Evaluate and checkpoint one recipe on one dataset."""
    identifier = configuration_id(decisions)
    cached = cache.get(identifier, dataset.dataset_id)
    if cached is not None:
        return cached

    started = time.perf_counter()
    logger.info("Evaluating recipe {} on dataset '{}'", identifier, dataset.dataset_id)
    try:
        pipeline = build_pipeline(decisions, dataset, settings)
        with tempfile.TemporaryDirectory(prefix="facetpy_matrix_optimization_") as temporary:
            result = pipeline.run_chunked(
                input_path=str(dataset.path),
                output_dir=temporary,
                output_extension=".edf",
                overwrite=True,
                channel_sequential=True,
                on_error="continue",
                keep_raw=False,
                chunk_by_trigger_sections=True,
                trigger_section_padding_seconds=settings.chunk_padding_seconds,
                trigger_section_min_triggers=settings.chunk_min_triggers,
                trigger_section_gap_seconds=settings.chunk_gap_seconds,
            )
            aggregated = aggregate_chunk_results(result)
            del result
        row = {
            "dataset_id": dataset.dataset_id,
            "configuration_id": identifier,
            "recipe_json": json.dumps(decisions.to_dict(), sort_keys=True),
            **aggregated,
        }
    except Exception as exc:  # A failed recipe becomes infeasible, not fatal to a long study.
        row = {
            "dataset_id": dataset.dataset_id,
            "configuration_id": identifier,
            "recipe_json": json.dumps(decisions.to_dict(), sort_keys=True),
            "success": False,
            "n_chunks": 0,
            "successful_chunks": 0,
            "execution_time": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }
    cache.add(row)
    release_process_memory()
    return row


def aggregate_dataset_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate across datasets while retaining dispersion and failures."""
    successful = [
        row
        for row in rows
        if bool(row.get("success")) and all(np.isfinite(float(row.get(column, np.nan))) for column in OBJECTIVE_COLUMNS)
    ]
    success_rate = len(successful) / len(rows) if rows else 0.0
    output: dict[str, Any] = {
        "success": bool(rows) and len(successful) == len(rows),
        "success_rate": success_rate,
        "n_datasets": len(rows),
        "successful_datasets": len(successful),
        "mean_execution_time": float(np.mean([float(row.get("execution_time", 0.0)) for row in rows]))
        if rows
        else float("nan"),
    }
    for column in (
        *OBJECTIVE_COLUMNS,
        "theta_preservation",
        "alpha_preservation",
        "nonpeak_beta_preservation",
        "scanner_peak_count",
        "snr",
        "rms_residual",
        "median_artifact_ratio",
    ):
        values = np.asarray([float(row[column]) for row in successful if column in row], dtype=float)
        if values.size:
            output[column] = float(np.median(values))
            output[f"{column}_q25"] = float(np.quantile(values, 0.25))
            output[f"{column}_q75"] = float(np.quantile(values, 0.75))
        else:
            output[column] = float("nan")
            output[f"{column}_q25"] = float("nan")
            output[f"{column}_q75"] = float("nan")
    return output


def _trial_constraints(trial: Any) -> Sequence[float]:
    """Optuna constraint callback; feasible values are non-positive."""
    return trial.user_attrs.get("constraints", [1.0] * 7)


def preservation_constraints(aggregated: Mapping[str, Any], settings: RunSettings) -> list[float]:
    """Return lower/upper Q25/Q75 constraints for preserved EEG bands."""
    reciprocal_bound = 1.0 / settings.preservation_ratio_bound
    constraints = []
    for metric, lower, upper in (
        ("theta_preservation", reciprocal_bound, settings.preservation_ratio_bound),
        ("alpha_preservation", reciprocal_bound, settings.preservation_ratio_bound),
        (
            "nonpeak_beta_preservation",
            settings.beta_preservation_minimum,
            settings.beta_preservation_maximum,
        ),
    ):
        q25 = float(aggregated.get(f"{metric}_q25", np.nan))
        q75 = float(aggregated.get(f"{metric}_q75", np.nan))
        constraints.extend([lower - q25 if np.isfinite(q25) else 1.0, q75 - upper if np.isfinite(q75) else 1.0])
    return constraints


def make_screening_objective(
    *,
    datasets: Sequence[DatasetSpec],
    capabilities: SearchCapabilities,
    settings: RunSettings,
    cache: EvaluationCache,
    anchor_ids: Mapping[str, str],
) -> Any:
    """Create the training-only low-fidelity objective closure."""

    def objective(trial: Any) -> tuple[float, float]:
        try:
            decisions = suggest_matrix_decisions(trial, capabilities, settings)
        except (MatrixDecisionError, ValueError) as exc:
            trial.set_user_attr("configuration_error", str(exc))
            trial.set_user_attr("constraints", [1.0] * 7)
            return BAD_OBJECTIVES

        identifier = configuration_id(decisions)
        trial.set_user_attr("configuration_id", identifier)
        trial.set_user_attr("recipe", decisions.to_dict())
        if identifier in anchor_ids:
            trial.set_user_attr("anchor_name", anchor_ids[identifier])
        rows = []
        for dataset in datasets:
            rows.append(evaluate_configuration(decisions, dataset, settings, cache))
        aggregated = aggregate_dataset_rows(rows)
        failure_constraint = 1.0 - float(aggregated["success_rate"])
        trial.set_user_attr(
            "constraints",
            [failure_constraint, *preservation_constraints(aggregated, settings)],
        )
        trial.set_user_attr("screening_aggregate", aggregated)
        if not aggregated["success"]:
            return BAD_OBJECTIVES
        return tuple(float(aggregated[column]) for column in OBJECTIVE_COLUMNS)

    return objective


def pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    """Return a non-dominated mask for the configured objective directions."""
    if frame.empty:
        return np.zeros(0, dtype=bool)
    values = frame.loc[:, OBJECTIVE_COLUMNS].to_numpy(dtype=float)
    transformed = np.column_stack(
        [
            -values[:, index] if direction == "maximize" else values[:, index]
            for index, direction in enumerate(OBJECTIVE_DIRECTIONS)
        ]
    )
    keep = np.ones(len(frame), dtype=bool)
    for index, point in enumerate(transformed):
        if not np.all(np.isfinite(point)):
            keep[index] = False
            continue
        dominates = np.all(transformed <= point, axis=1) & np.any(transformed < point, axis=1)
        if np.any(dominates):
            keep[index] = False
    return keep


def select_ideal_recipe(frame: pd.DataFrame, settings: RunSettings) -> pd.Series:
    """Select a balanced point from the feasible full-training Pareto front."""
    successful = frame[frame["success"].astype(bool)].copy()
    if successful.empty:
        raise RuntimeError("No promoted recipe succeeded on every training dataset.")
    feasible = successful[
        (successful["theta_preservation_q25"] >= 1.0 / settings.preservation_ratio_bound)
        & (successful["theta_preservation_q75"] <= settings.preservation_ratio_bound)
        & (successful["alpha_preservation_q25"] >= 1.0 / settings.preservation_ratio_bound)
        & (successful["alpha_preservation_q75"] <= settings.preservation_ratio_bound)
        & (successful["nonpeak_beta_preservation_q25"] >= settings.beta_preservation_minimum)
        & (successful["nonpeak_beta_preservation_q75"] <= settings.beta_preservation_maximum)
    ].copy()
    feasible_set_used = not feasible.empty
    candidates = feasible if feasible_set_used else successful
    candidates = candidates.loc[pareto_mask(candidates)].copy()
    distance_squared = np.zeros(len(candidates), dtype=float)
    for column, direction in zip(OBJECTIVE_COLUMNS, OBJECTIVE_DIRECTIONS, strict=True):
        values = candidates[column].to_numpy(dtype=float)
        span = float(np.max(values) - np.min(values))
        if span <= np.finfo(float).eps:
            normalized = np.zeros_like(values)
        elif direction == "maximize":
            normalized = (np.max(values) - values) / span
        else:
            normalized = (values - np.min(values)) / span
        distance_squared += normalized**2
    candidates["ideal_point_distance"] = np.sqrt(distance_squared)
    selected = (
        candidates.sort_values(
            ["ideal_point_distance", "mean_execution_time", "configuration_id"],
            ascending=[True, True, True],
        )
        .iloc[0]
        .copy()
    )
    selected["selection_pool"] = "feasible_pareto" if feasible_set_used else "successful_pareto_fallback"
    return selected


def _trial_frame(study: Any) -> pd.DataFrame:
    """Return Optuna's table with JSON recipe fields made explicit."""
    frame = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs", "state"))
    return frame


def _candidate_recipes_from_study(
    study: Any,
    *,
    anchors: Sequence[tuple[str, MatrixDecisions]],
    promotion_count: int,
) -> list[tuple[str | None, MatrixDecisions]]:
    """Promote anchors plus the strongest unique screening-Pareto recipes."""
    output: list[tuple[str | None, MatrixDecisions]] = []
    seen: set[str] = set()
    for name, decisions in anchors:
        identifier = configuration_id(decisions)
        if identifier not in seen:
            output.append((name, decisions))
            seen.add(identifier)

    trials = sorted(
        study.best_trials,
        key=lambda item: (
            sum(float(value) for value in item.values),
            item.number,
        ),
    )
    target = max(promotion_count, len(output))
    for trial in trials:
        manifest = trial.user_attrs.get("recipe")
        if not isinstance(manifest, Mapping):
            continue
        decisions = MatrixDecisions.from_dict(manifest)
        identifier = configuration_id(decisions)
        if identifier in seen:
            continue
        output.append((trial.user_attrs.get("anchor_name"), decisions))
        seen.add(identifier)
        if len(output) >= target:
            break
    return output


def _fixed_trial_recipes(
    optuna: Any,
    parameters: Sequence[tuple[str, Mapping[str, Any]]],
    capabilities: SearchCapabilities,
    settings: RunSettings,
) -> list[tuple[str, MatrixDecisions]]:
    """Resolve enqueued anchor parameters through the same conditional parser."""
    output = []
    for name, values in parameters:
        decisions = suggest_matrix_decisions(optuna.trial.FixedTrial(dict(values)), capabilities, settings)
        output.append((name, decisions))
    return output


def _screening_ids(training: Sequence[DatasetSpec], *, fraction: float, seed: int) -> list[str]:
    """Choose a deterministic rotating low-fidelity dataset subset."""
    rng = np.random.default_rng(seed)
    identifiers = np.asarray([item.dataset_id for item in training], dtype=object)
    rng.shuffle(identifiers)
    count = min(len(identifiers), max(2, math.ceil(len(identifiers) * fraction)))
    return sorted(str(value) for value in identifiers[:count])


def run_outer_fold(
    *,
    held_out: DatasetSpec,
    training: Sequence[DatasetSpec],
    fold_directory: Path,
    settings: RunSettings,
    cache: EvaluationCache,
    trials: int,
    promotion_count: int,
    screening_fraction: float,
    seed: int,
    timeout_seconds: float | None,
    rebuild_study: bool,
) -> dict[str, Any]:
    """Run one complete nested leave-one-dataset-out fold."""
    optuna = _load_optuna()
    fold_directory.mkdir(parents=True, exist_ok=True)
    database = fold_directory / "screening_study.sqlite3"
    if rebuild_study and database.exists():
        database.unlink()

    capabilities = SearchCapabilities.from_datasets(training)
    screening_ids = _screening_ids(training, fraction=screening_fraction, seed=seed)
    screening = [item for item in training if item.dataset_id in screening_ids]
    anchor_parameters = anchor_parameter_sets(capabilities)
    anchors = _fixed_trial_recipes(optuna, anchor_parameters, capabilities, settings)
    anchor_ids = {configuration_id(recipe): name for name, recipe in anchors}
    write_json(
        fold_directory / "anchor_recipes.json",
        {
            name: {
                "configuration_id": configuration_id(recipe),
                "recipe": recipe.to_dict(),
            }
            for name, recipe in anchors
        },
    )

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        group=True,
        constraints_func=_trial_constraints,
        n_startup_trials=max(10, len(anchor_parameters)),
    )
    study = optuna.create_study(
        study_name=f"matrix_screening_{held_out.dataset_id}",
        storage=f"sqlite:///{database.resolve()}",
        directions=list(OBJECTIVE_DIRECTIONS),
        sampler=sampler,
        load_if_exists=True,
    )
    study_signature_payload = {
        "settings": settings.signature_payload(),
        "training_datasets": [item.signature_payload() for item in training],
        "screening_datasets": screening_ids,
        "capabilities": capabilities.to_dict(),
        "seed": seed,
    }
    study_signature = hashlib.sha256(json.dumps(study_signature_payload, sort_keys=True).encode("utf-8")).hexdigest()
    existing_signature = study.user_attrs.get("study_signature")
    if existing_signature is not None and existing_signature != study_signature:
        raise RuntimeError(
            f"Study {database} was created for a different search space or training fold. "
            "Use --rebuild-studies or a new output directory."
        )
    study.set_user_attr("study_signature", study_signature)

    queued = [values for _, values in anchor_parameters]
    queued.extend(balanced_seed_parameter_sets(capabilities, count=max(0, min(16, trials - len(queued)))))
    for values in queued[:trials]:
        study.enqueue_trial(dict(values), skip_if_exists=True)

    completed = sum(trial.state.is_finished() for trial in study.trials)
    remaining = max(0, trials - completed)
    if remaining:
        objective = make_screening_objective(
            datasets=screening,
            capabilities=capabilities,
            settings=settings,
            cache=cache,
            anchor_ids=anchor_ids,
        )
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=timeout_seconds,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    trial_frame = _trial_frame(study)
    pareto_trial_numbers = {trial.number for trial in study.best_trials}
    trial_frame["is_screening_pareto"] = trial_frame["number"].isin(pareto_trial_numbers)
    _atomic_write_csv(trial_frame, fold_directory / "screening_trials.csv")
    _atomic_write_csv(
        trial_frame[trial_frame["is_screening_pareto"]],
        fold_directory / "screening_pareto.csv",
    )

    promoted = _candidate_recipes_from_study(
        study,
        anchors=anchors,
        promotion_count=promotion_count,
    )
    training_rows: list[dict[str, Any]] = []
    for anchor_name, decisions in promoted:
        rows = []
        for dataset in training:
            rows.append(evaluate_configuration(decisions, dataset, settings, cache))
        aggregate = aggregate_dataset_rows(rows)
        training_rows.append(
            {
                "configuration_id": configuration_id(decisions),
                "anchor_name": anchor_name,
                "recipe_json": json.dumps(decisions.to_dict(), sort_keys=True),
                **aggregate,
            }
        )
    training_frame = pd.DataFrame(training_rows)
    training_frame["is_pareto"] = False
    successful_mask = training_frame["success"].astype(bool)
    training_frame.loc[successful_mask, "is_pareto"] = pareto_mask(training_frame.loc[successful_mask])
    _atomic_write_csv(training_frame, fold_directory / "promoted_training_results.csv")
    _atomic_write_csv(training_frame[training_frame["is_pareto"]], fold_directory / "training_pareto.csv")

    selected = select_ideal_recipe(training_frame, settings)
    selected_decisions = MatrixDecisions.from_dict(json.loads(selected["recipe_json"]))
    held_out_row = evaluate_configuration(selected_decisions, held_out, settings, cache)
    held_out_row = {
        "held_out_dataset": held_out.dataset_id,
        "configuration_id": configuration_id(selected_decisions),
        "anchor_name": selected.get("anchor_name"),
        "selection_pool": selected["selection_pool"],
        "training_scanner_peak_residual": selected["scanner_peak_residual"],
        "training_nonpeak_eeg_log_deviation": selected["nonpeak_eeg_log_deviation"],
        "training_theta_preservation": selected["theta_preservation"],
        "training_alpha_preservation": selected["alpha_preservation"],
        "training_nonpeak_beta_preservation": selected["nonpeak_beta_preservation"],
        **held_out_row,
    }
    _atomic_write_csv(pd.DataFrame([held_out_row]), fold_directory / "held_out_result.csv")
    write_json(
        fold_directory / "selected_recipe.json",
        {
            "held_out_dataset": held_out.dataset_id,
            "training_datasets": [item.dataset_id for item in training],
            "screening_datasets": screening_ids,
            "capabilities": capabilities.to_dict(),
            "configuration_id": configuration_id(selected_decisions),
            "anchor_name": selected.get("anchor_name"),
            "selection_pool": selected["selection_pool"],
            "recipe": selected_decisions.to_dict(),
            "training_metrics": {column: selected[column] for column in OBJECTIVE_COLUMNS},
        },
    )
    return held_out_row


def bootstrap_summary(frame: pd.DataFrame, *, samples: int, seed: int) -> pd.DataFrame:
    """Bootstrap cross-fold medians without treating chunks as subjects."""
    rng = np.random.default_rng(seed)
    rows = []
    metrics = (
        *OBJECTIVE_COLUMNS,
        "theta_preservation",
        "alpha_preservation",
        "nonpeak_beta_preservation",
        "snr",
        "rms_residual",
        "median_artifact_ratio",
    )
    successful = frame[frame["success"].astype(bool)]
    for metric in metrics:
        values = pd.to_numeric(successful[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        bootstraps = np.empty(samples, dtype=float)
        for index in range(samples):
            bootstraps[index] = np.median(rng.choice(values, size=len(values), replace=True))
        rows.append(
            {
                "metric": metric,
                "median": float(np.median(values)),
                "ci95_low": float(np.quantile(bootstraps, 0.025)),
                "ci95_high": float(np.quantile(bootstraps, 0.975)),
                "n_folds": len(values),
            }
        )
    return pd.DataFrame(rows)


def parameter_stability(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize how often each selected decision value recurs across folds."""
    records: list[dict[str, Any]] = []
    manifests = [json.loads(value) for value in frame["recipe_json"]]

    def flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, Mapping):
                output.update(flatten(item, name))
            else:
                output[name] = item
        return output

    flattened = [flatten(value) for value in manifests]
    for parameter in sorted({key for row in flattened for key in row}):
        values = [json.dumps(row.get(parameter), sort_keys=True) for row in flattened]
        counts = pd.Series(values).value_counts(dropna=False)
        for encoded, count in counts.items():
            records.append(
                {
                    "parameter": parameter,
                    "value": json.loads(encoded),
                    "count": int(count),
                    "fraction": float(count / len(flattened)),
                }
            )
    return pd.DataFrame(records)


def plot_held_out_metrics(frame: pd.DataFrame, path: Path) -> None:
    """Plot the independent held-out objectives for quick inspection."""
    successful = frame[frame["success"].astype(bool)].copy()
    if successful.empty:
        return
    labels = successful["held_out_dataset"].astype(str).tolist()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, column, title in zip(
        axes,
        OBJECTIVE_COLUMNS,
        (
            "scanner-peak residual (lower)",
            "non-peak EEG deviation (lower)",
        ),
        strict=True,
    ):
        axis.bar(labels, successful[column].to_numpy(dtype=float))
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=60)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Nested leave-one-dataset-out performance")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.trials < 1:
        raise ValueError("--trials must be at least 1.")
    if args.promotion_count < 1:
        raise ValueError("--promotion-count must be at least 1.")
    if not 0.0 < args.screening_fraction <= 1.0:
        raise ValueError("--screening-fraction must be in (0, 1].")
    if args.preservation_ratio_bound <= 1.0:
        raise ValueError("--preservation-ratio-bound must be greater than 1.")
    if not 0.0 < args.beta_preservation_minimum <= 1.0:
        raise ValueError("--beta-preservation-minimum must be in (0, 1].")
    if args.beta_preservation_maximum < 1.0:
        raise ValueError("--beta-preservation-maximum must be at least 1.")
    if not 0.0 <= args.scanner_peak_minimum_hz < args.scanner_peak_maximum_hz:
        raise ValueError("Scanner peak frequency bounds must satisfy 0 <= minimum < maximum.")
    if args.scanner_peak_prominence_db <= 0.0:
        raise ValueError("--scanner-peak-prominence-db must be positive.")
    if args.scanner_peak_half_width_hz <= 0.0:
        raise ValueError("--scanner-peak-half-width-hz must be positive.")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100.")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0.0:
        raise ValueError("--timeout-seconds must be positive when supplied.")
    if args.upsample_factor < 1:
        raise ValueError("--upsample-factor must be at least 1.")
    if args.chunk_padding_seconds < 0.0:
        raise ValueError("--chunk-padding-seconds cannot be negative.")
    if args.chunk_min_triggers < 1:
        raise ValueError("--chunk-min-triggers must be at least 1.")
    if args.chunk_gap_seconds is not None and args.chunk_gap_seconds <= 0.0:
        raise ValueError("--chunk-gap-seconds must be positive when supplied.")
    if args.rotation_scale < 0.0:
        raise ValueError("--rotation-scale cannot be negative.")
    if args.max_motion_distance_low <= 0 or args.max_motion_distance_high <= args.max_motion_distance_low:
        raise ValueError("Motion-distance bounds must satisfy 0 < low < high.")
    if args.motion_directory is not None and not args.motion_directory.is_dir():
        raise ValueError("--motion-directory must identify an existing directory.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--recursive", action="store_true", help="Search input subdirectories.")
    parser.add_argument("--trigger-regex", default=r"^R128$")
    parser.add_argument("--trials", type=int, default=100, help="Total screening trials per outer fold.")
    parser.add_argument("--promotion-count", type=int, default=20)
    parser.add_argument("--screening-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--upsample-factor", type=int, default=10)
    parser.add_argument("--chunk-padding-seconds", type=float, default=5.0)
    parser.add_argument("--chunk-min-triggers", type=int, default=20)
    parser.add_argument("--chunk-gap-seconds", type=float)
    parser.add_argument(
        "--preservation-ratio-bound",
        type=float,
        default=1.25,
        help="Theta/alpha Q25 and Q75 must lie within reciprocal bounds (default 0.8-1.25).",
    )
    parser.add_argument(
        "--beta-preservation-minimum",
        type=float,
        default=0.5,
        help="Minimum Q25 corrected/original non-scanner beta power (default 0.5).",
    )
    parser.add_argument(
        "--beta-preservation-maximum",
        type=float,
        default=1.5,
        help="Maximum Q75 corrected/original non-scanner beta power (default 1.5).",
    )
    parser.add_argument("--scanner-peak-minimum-hz", type=float, default=13.0)
    parser.add_argument("--scanner-peak-maximum-hz", type=float, default=80.0)
    parser.add_argument("--scanner-peak-prominence-db", type=float, default=6.0)
    parser.add_argument("--scanner-peak-half-width-hz", type=float, default=0.5)
    parser.add_argument("--motion-directory", type=Path)
    parser.add_argument("--rotation-scale", type=float, default=0.0)
    parser.add_argument("--max-motion-distance-low", type=float, default=0.05)
    parser.add_argument("--max-motion-distance-high", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-studies", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run nested optimization and write resumable reports."""
    args = build_argument_parser().parse_args(argv)
    _validate_arguments(args)
    _load_optuna()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    paths = unique_dataset_ids(find_input_datasets(args.input_directory, recursive=args.recursive))
    datasets = preflight_datasets(
        paths,
        trigger_regex=args.trigger_regex,
        motion_directory=args.motion_directory,
        rotation_scale=args.rotation_scale,
    )
    settings = RunSettings(
        trigger_regex=args.trigger_regex,
        upsample_factor=args.upsample_factor,
        chunk_padding_seconds=args.chunk_padding_seconds,
        chunk_min_triggers=args.chunk_min_triggers,
        chunk_gap_seconds=args.chunk_gap_seconds,
        preservation_ratio_bound=args.preservation_ratio_bound,
        beta_preservation_minimum=args.beta_preservation_minimum,
        beta_preservation_maximum=args.beta_preservation_maximum,
        scanner_peak_minimum_hz=args.scanner_peak_minimum_hz,
        scanner_peak_maximum_hz=args.scanner_peak_maximum_hz,
        scanner_peak_prominence_db=args.scanner_peak_prominence_db,
        scanner_peak_half_width_hz=args.scanner_peak_half_width_hz,
        max_motion_distance_low=args.max_motion_distance_low,
        max_motion_distance_high=args.max_motion_distance_high,
    )
    dataset_signature_payload = {key: value.signature_payload() for key, value in datasets.items()}
    signature_payload = {
        "settings": settings.signature_payload(),
        "datasets": dataset_signature_payload,
    }
    run_signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()
    evaluation_signature_payload = {
        "settings": settings.evaluation_signature_payload(),
        "datasets": dataset_signature_payload,
    }
    evaluation_signature = hashlib.sha256(
        json.dumps(evaluation_signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache = EvaluationCache(
        args.output_directory / "evaluation_cache.csv",
        signature=evaluation_signature,
        rebuild=args.rebuild_cache,
    )
    write_json(
        args.output_directory / "run_manifest.json",
        {
            "signature": run_signature,
            "evaluation_signature": evaluation_signature,
            **signature_payload,
        },
    )

    held_out_rows = []
    identifiers = sorted(datasets)
    for fold_index, held_out_id in enumerate(identifiers):
        logger.info("Starting outer fold {}/{}; held out '{}'", fold_index + 1, len(identifiers), held_out_id)
        training = [datasets[item] for item in identifiers if item != held_out_id]
        row = run_outer_fold(
            held_out=datasets[held_out_id],
            training=training,
            fold_directory=args.output_directory / "folds" / held_out_id,
            settings=settings,
            cache=cache,
            trials=args.trials,
            promotion_count=args.promotion_count,
            screening_fraction=args.screening_fraction,
            seed=args.seed + fold_index,
            timeout_seconds=args.timeout_seconds,
            rebuild_study=args.rebuild_studies,
        )
        held_out_rows.append(row)
        _atomic_write_csv(pd.DataFrame(held_out_rows), args.output_directory / "held_out_results.csv")

    held_out_frame = pd.DataFrame(held_out_rows)
    summary = bootstrap_summary(held_out_frame, samples=args.bootstrap_samples, seed=args.seed)
    stability = parameter_stability(held_out_frame)
    anchor_summary = (
        held_out_frame["anchor_name"]
        .fillna("unnamed_combination")
        .value_counts()
        .rename_axis("anchor_name")
        .reset_index(name="selected_folds")
    )
    anchor_summary["selection_fraction"] = anchor_summary["selected_folds"] / len(held_out_frame)
    _atomic_write_csv(summary, args.output_directory / "held_out_bootstrap_summary.csv")
    _atomic_write_csv(stability, args.output_directory / "selected_parameter_stability.csv")
    _atomic_write_csv(anchor_summary, args.output_directory / "legacy_anchor_selection.csv")
    plot_held_out_metrics(held_out_frame, args.output_directory / "held_out_objectives.png")
    logger.info(
        "Optimization complete. Independent held-out results: {}", args.output_directory / "held_out_results.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
