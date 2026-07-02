"""Pipeline pattern construction for the FACETpy processing CLI."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from facet import (
    AASCorrection,
    CutAcquisitionWindow,
    DownSample,
    DropChannelsMatching,
    HighPassFilter,
    LowPassFilter,
    PasteAcquisitionWindow,
    Pipeline,
    QRSTriggerDetector,
    SliceAligner,
    SubsampleAligner,
    TriggerDetector,
    UpSample,
)
from facet.correction import (
    CorrespondingSliceCorrection,
    FARMCorrection,
    MoosmannCorrection,
    SliceTriggerCorrection,
    VolumeArtifactCorrection,
    VolumeTriggerCorrection,
)

try:
    from facet.correction import ANCCorrection
except ImportError:  # pragma: no cover - depends on optional scientific stack
    ANCCorrection = None

try:
    from facet.correction import PCACorrection
except ImportError:  # pragma: no cover - depends on optional scientific stack
    PCACorrection = None

DEFAULT_EGI_DROP_REGEX = r"^E(?:[1-9]|[1-9]\d|1[01]\d|12[0-8])$"

CORRECTION_MODE_DESCRIPTIONS = {
    "aas": "Baseline Averaged Artifact Subtraction.",
    "farm": "FACET FARM-style template weighting for similar artifact epochs.",
    "volume-trigger": "FACET volume-trigger template weighting.",
    "slice-trigger": "FACET slice-trigger odd/even template weighting.",
    "corresponding-slice": "Average corresponding slice positions across volumes.",
    "moosmann": "Motion-informed template weighting from an SPM realignment-parameter file.",
}
ADD_ON_MODE_DESCRIPTIONS = {
    "volume-artifact": "Correct transition artifacts around slice-trigger volume gaps before template subtraction.",
    "pca": "Apply PCA residual cleanup after template subtraction.",
    "anc": "Apply adaptive noise cancellation after downsampling, using the accumulated noise estimate.",
}
CORRECTION_MATRIX_DESCRIPTIONS = {
    "aas": "AAS builds A with correlation-selected epochs from sliding windows.",
    "farm": "FARM builds A from the most correlated neighboring epochs above threshold.",
    "volume-trigger": "Volume-trigger correction builds A from fixed neighboring volume-trigger epochs.",
    "slice-trigger": "Slice-trigger correction builds A from alternating odd/even slice-trigger epochs.",
    "corresponding-slice": "Corresponding-slice correction builds A from the same slice position across volumes.",
    "moosmann": "Moosmann correction builds A from motion-informed realignment-parameter weights.",
}
PROCESS_PATTERN_DESCRIPTIONS = {
    "quickstart": "Memory-light trigger-section chunks: trigger detection, upsample, correction, downsample.",
    "standard": "Docs standard pipeline: cut, high-pass, align, correction, PCA, downsample, paste, low-pass, ANC.",
    "bcg": "Ballistocardiogram pattern: QRS trigger detection plus AAS correction.",
}
PATTERN_DESCRIPTIONS = {
    **PROCESS_PATTERN_DESCRIPTIONS,
    "custom": "Python pattern for manually assembling Pipeline([...]) with chosen processors.",
    "step-by-step": "Python pattern for executing processors one at a time against a ProcessingContext.",
    "pipe": "Python pattern for chaining processors with the ProcessingContext pipe operator.",
    "batch": "CLI/input pattern using --input-list or --input-dir to process many recordings.",
}


def _parse_pca_components(value: str) -> int | float | str:
    """Parse PCA component settings from the CLI."""
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"

    try:
        if re.fullmatch(r"[+-]?\d+", normalized):
            return int(normalized)
        return float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "PCA components must be an integer, a 0-1 fraction, or 'auto'."
        ) from exc


def _unique_modes(modes: Sequence[str] | None) -> list[str]:
    """Return add-on modes in user order without duplicates."""
    selected: list[str] = []
    for mode in modes or ():
        if mode not in selected:
            selected.append(mode)
    return selected


def _selected_add_on_modes(args: argparse.Namespace) -> tuple[list[str], bool]:
    """Return add-on modes and whether they were selected by a pattern."""
    if args.mode is not None:
        return _unique_modes(args.mode), False
    if args.pattern == "standard":
        return ["pca", "anc"], True
    return [], False


def _common_template_kwargs(args: argparse.Namespace) -> dict:
    """Build shared options for AAS-style template subtraction processors."""
    return {
        "window_size": args.window_size,
        "plot_artifacts": args.plot_artifacts,
        "realign_after_averaging": args.realign_after_averaging,
        "search_window_factor": args.search_window_factor,
        "apply_epoch_alpha_scaling": args.apply_epoch_alpha_scaling,
    }


def _build_template_correction(args: argparse.Namespace):
    """Create the selected template-subtraction correction processor."""
    mode = args.correction_mode
    common = _common_template_kwargs(args)

    if mode == "aas":
        return AASCorrection(
            **common,
            correlation_threshold=args.aas_correlation_threshold,
            interpolate_volume_gaps=args.interpolate_volume_gaps,
        )
    if mode == "farm":
        return FARMCorrection(
            **common,
            correlation_threshold=args.farm_correlation_threshold,
            search_half_window=args.farm_search_half_window,
            search_half_window_factor=args.farm_search_half_window_factor,
            interpolate_volume_gaps=args.interpolate_volume_gaps,
        )
    if mode == "volume-trigger":
        return VolumeTriggerCorrection(**common)
    if mode == "slice-trigger":
        return SliceTriggerCorrection(**common)
    if mode == "corresponding-slice":
        return CorrespondingSliceCorrection(slices_per_volume=args.slices_per_volume, **common)
    if mode == "moosmann":
        if args.motion_rp_file is None:
            raise ValueError("--motion-rp-file is required when --correction-mode=moosmann")
        rp_file = Path(args.motion_rp_file).expanduser().resolve()
        if not rp_file.exists():
            raise FileNotFoundError(f"Motion realignment parameter file not found: {rp_file}")
        return MoosmannCorrection(
            rp_file=str(rp_file),
            motion_threshold=args.motion_threshold,
            motion_window_size=args.motion_window_size,
            **common,
        )

    raise ValueError(f"Unsupported correction mode: {mode}")


def _build_mode_processors(args: argparse.Namespace) -> tuple[list, list, list]:
    """Build pre-template, post-template, and post-downsample mode processors."""
    pre_template = []
    post_template = []
    post_downsample = []
    modes, from_pattern = _selected_add_on_modes(args)

    for mode in modes:
        if mode == "volume-artifact":
            pre_template.append(
                VolumeArtifactCorrection(
                    template_count=args.volume_template_count,
                    weighting_position=args.volume_weighting_position,
                    weighting_slope=args.volume_weighting_slope,
                )
            )
        elif mode == "pca":
            if PCACorrection is None:
                if from_pattern:
                    continue
                raise ImportError("PCACorrection is not available in this installation.")
            post_template.append(
                PCACorrection(
                    n_components=args.pca_components,
                    hp_freq=args.pca_hp_freq,
                )
            )
        elif mode == "anc":
            if ANCCorrection is None:
                if from_pattern:
                    continue
                raise ImportError("ANCCorrection is not available in this installation.")
            post_downsample.append(
                ANCCorrection(
                    filter_order=args.anc_filter_order,
                    hp_freq=args.anc_hp_freq,
                    use_c_extension=args.anc_c_extension,
                    mu_factor=args.anc_mu_factor,
                    max_gain=args.anc_max_gain,
                )
            )
        else:
            raise ValueError(f"Unsupported add-on mode: {mode}")

    return pre_template, post_template, post_downsample


def _drop_channel_processors(args: argparse.Namespace) -> list:
    """Return optional channel-dropping processors for every process pattern."""
    processors = []
    # EGI channel removal is opt-in. By default, trigger-section processing keeps
    # every channel in the cut so AAS correction runs over the full segment.
    if args.drop_egi_e_channels:
        processors.append(DropChannelsMatching(regex=args.drop_channel_regex))
    return processors


def _build_quickstart_pattern(args: argparse.Namespace) -> list:
    """Build the memory-light quickstart processing pattern."""
    pre_template, post_template, post_downsample = _build_mode_processors(args)
    return [
        TriggerDetector(regex=args.trigger_regex),
        UpSample(factor=args.upsample_factor),
        *pre_template,
        _build_template_correction(args),
        *post_template,
        DownSample(factor=args.upsample_factor),
        *post_downsample,
    ]


def _build_standard_pattern(args: argparse.Namespace) -> list:
    """Build the docs standard pattern without loader/exporter steps."""
    pre_template, post_template, post_downsample = _build_mode_processors(args)
    return [
        TriggerDetector(regex=args.trigger_regex),
        CutAcquisitionWindow(),
        HighPassFilter(freq=args.highpass_freq),
        UpSample(factor=args.upsample_factor),
        SliceAligner(ref_trigger_index=0),
        SubsampleAligner(ref_trigger_index=0),
        *pre_template,
        _build_template_correction(args),
        *post_template,
        DownSample(factor=args.upsample_factor),
        PasteAcquisitionWindow(),
        LowPassFilter(freq=args.lowpass_freq),
        *post_downsample,
    ]


def _build_bcg_pattern(args: argparse.Namespace) -> list:
    """Build the BCG/QRS pattern from the quickstart documentation."""
    return [
        QRSTriggerDetector(),
        AASCorrection(
            window_size=args.bcg_window_size,
            correlation_threshold=args.aas_correlation_threshold,
            plot_artifacts=args.plot_artifacts,
            realign_after_averaging=args.realign_after_averaging,
            search_window_factor=args.search_window_factor,
            apply_epoch_alpha_scaling=args.apply_epoch_alpha_scaling,
        ),
    ]


def _build_processing_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build the selected FACET correction pipeline used by the CLI."""
    processors = _drop_channel_processors(args)

    if args.pattern == "standard":
        processors.extend(_build_standard_pattern(args))
    elif args.pattern == "bcg":
        processors.extend(_build_bcg_pattern(args))
    else:
        processors.extend(_build_quickstart_pattern(args))

    return Pipeline(processors, name="FACETpy CLI Pipeline")
