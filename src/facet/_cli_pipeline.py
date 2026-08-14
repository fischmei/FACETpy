"""Pipeline pattern construction for the FACETpy processing CLI."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from facet import (
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
    Flex,
    VolumeArtifactCorrection,
    WeightingPolicy,
)
from facet.correction.presets import (
    AAS_PER_TARGET,
    CLI_CORRECTION_PRESETS,
    CORRESPONDING_SLICE,
    FARM_PER_TARGET_K10,
    FLEX_DEFAULT,
    LEGACY_RESEMBLANCE,
    MOOSMANN_COST,
    STRUCTURAL_SLICE,
    STRUCTURAL_VOLUME,
    build_flex_preset,
)

from ._cli_motion import CLIMotionMetadataInjector, CLISlicesPerVolumeInjector

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
    "aas": "Flex configured as the closest per-target AAS recipe.",
    "farm": "Flex configured as the FARM-like absolute-correlation recipe.",
    "flex": "Default composable Flex correlation recipe.",
    "volume-trigger": "Flex configured for fixed neighboring-volume structure.",
    "slice-trigger": "Flex configured for odd/even slice-trigger structure.",
    "corresponding-slice": "Flex configured for matching slice phases across volumes.",
    "moosmann": "Flex configured with motion-path costs from an SPM realignment file.",
}
ADD_ON_MODE_DESCRIPTIONS = {
    "volume-artifact": "Correct transition artifacts around slice-trigger volume gaps before template subtraction.",
    "pca": "Apply PCA residual cleanup after template subtraction.",
    "anc": "Apply adaptive noise cancellation after downsampling, using the accumulated noise estimate.",
    "bcg": "Apply QRS-triggered BCG artifact correction after scanner-template correction.",
}
CORRECTION_MATRIX_DESCRIPTIONS = {
    "aas": "Flex uses future candidates, includes the target, and enforces at least five signed-correlation matches.",
    "farm": "Flex keeps at most ten absolute-correlation matches from a symmetric thirty-candidate window.",
    "flex": (
        "Flex builds each row of A from a future-first correlation window, backfilled with preceding epochs, "
        "then supplements threshold matches to a configurable minimum and applies equal or normal weights."
    ),
    "volume-trigger": "Flex selects all candidates in a past-heavy structural volume window, including the target.",
    "slice-trigger": "Flex selects alternating future slice-trigger epochs with no correlation scoring.",
    "corresponding-slice": "Flex samples the same slice phase in neighboring volumes and includes the target.",
    "moosmann": "Flex ranks stable candidates by temporal plus cumulative-motion path cost.",
}
PROCESS_PATTERN_DESCRIPTIONS = {
    "quickstart": "Memory-light scanner correction. Add --mode bcg for QRS-triggered BCG cleanup.",
    "standard": "Docs standard scanner correction with PCA and ANC add-ons by default. Add --mode bcg for BCG cleanup.",
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
        raise argparse.ArgumentTypeError("PCA components must be an integer, a 0-1 fraction, or 'auto'.") from exc


def _unique_modes(modes: Sequence[str] | None) -> list[str]:
    """Return add-on modes in user order without duplicates."""
    selected: list[str] = []
    for mode in modes or ():
        if mode not in selected:
            selected.append(mode)
    return selected


def _selected_add_on_modes(args: argparse.Namespace) -> tuple[list[str], bool]:
    """Return add-on modes and whether they were selected by a pattern."""
    if args.pattern == "standard":
        return _unique_modes(["pca", "anc", *(args.mode or [])]), True
    if args.mode is not None:
        return _unique_modes(args.mode), False
    return [], False


def _common_template_kwargs(args: argparse.Namespace) -> dict:
    """Build options shared by Flex template-matrix processors."""
    return {
        "plot_artifacts": args.plot_artifacts,
        "realign_after_averaging": args.realign_after_averaging,
        "search_window_factor": args.search_window_factor,
        "apply_epoch_alpha_scaling": args.apply_epoch_alpha_scaling,
        "track_estimated_noise": args.track_estimated_noise,
    }


def _build_template_correction(args: argparse.Namespace):
    """Create one Flex processor from the selected named decision recipe."""
    mode = args.correction_mode
    common = _common_template_kwargs(args)
    preset_name = CLI_CORRECTION_PRESETS.get(mode)
    if preset_name is None:
        raise ValueError(f"Unsupported correction mode: {mode}")

    if preset_name == FLEX_DEFAULT:
        window_size = 10 if args.window_size is None else args.window_size
        weighting = (
            WeightingPolicy.equal()
            if args.flex_distribution == "equal"
            else WeightingPolicy.gaussian(sigma=max(window_size / 3.0, 1.0))
        )
        decisions = build_flex_preset(
            preset_name,
            window_size=window_size,
            threshold=args.flex_threshold,
            template_size=args.flex_min_accepted,
            weighting=weighting,
        )
    elif preset_name == AAS_PER_TARGET:
        decisions = build_flex_preset(
            preset_name,
            window_size=10 if args.window_size is None else args.window_size,
            threshold=args.aas_correlation_threshold,
        )
    elif preset_name == FARM_PER_TARGET_K10:
        farm_window = 30 if args.window_size is None else args.window_size
        if args.farm_search_half_window is not None:
            farm_window = 2 * args.farm_search_half_window
        decisions = build_flex_preset(
            preset_name,
            window_size=farm_window,
            threshold=args.farm_correlation_threshold,
            template_size=args.farm_template_size,
        )
    elif preset_name in {STRUCTURAL_VOLUME, STRUCTURAL_SLICE, CORRESPONDING_SLICE}:
        decisions = build_flex_preset(
            preset_name,
            window_size=10 if args.window_size is None else args.window_size,
        )
    else:
        legacy_window = 30 if args.window_size is None else args.window_size
        decisions = build_flex_preset(
            MOOSMANN_COST,
            template_size=(
                args.motion_window_size
                if args.motion_window_size is not None
                else 2 * legacy_window
            ),
        )

    processor = Flex(
        **common,
        N_distribution=args.flex_distribution if preset_name == FLEX_DEFAULT else "equal",
        interpolate_volume_gaps=args.interpolate_volume_gaps,
        matrix_decisions=decisions,
    )
    processor.name = "flex_correction" if mode == "flex" else f"{mode.replace('-', '_')}_flex_correction"
    processor.flex_preset_name = preset_name
    processor.legacy_algorithm_resemblance = LEGACY_RESEMBLANCE[preset_name]
    return processor


def _build_mode_processors(args: argparse.Namespace) -> tuple[list, list, list]:
    """Build pre-template, post-template, and post-downsample mode processors."""
    pre_template = []
    post_template = []
    post_downsample = []
    modes, from_pattern = _selected_add_on_modes(args)
    if args.correction_mode == "corresponding-slice" and args.slices_per_volume is not None:
        pre_template.append(CLISlicesPerVolumeInjector(args.slices_per_volume))
    if args.correction_mode == "moosmann":
        if args.motion_rp_file is None:
            raise ValueError("--motion-rp-file is required when --correction-mode=moosmann")
        rp_file = Path(args.motion_rp_file).expanduser().resolve()
        if not rp_file.is_file():
            raise FileNotFoundError(f"Motion realignment parameter file not found: {rp_file}")
        pre_template.append(
            CLIMotionMetadataInjector(
                rp_file=str(rp_file),
                trigger_regex=args.trigger_regex,
                motion_threshold=args.motion_threshold,
            )
        )
    if "anc" in modes and not args.track_estimated_noise:
        raise ValueError(
            "--no-track-estimated-noise cannot be combined with ANC, which requires the retained artifact estimate."
        )

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
                    track_estimated_noise=args.track_estimated_noise,
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
        elif mode == "bcg":
            post_downsample.extend(_build_bcg_pattern(args))
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
    correction = Flex(
        matrix_decisions=build_flex_preset(
            AAS_PER_TARGET,
            window_size=args.bcg_window_size,
            threshold=args.aas_correlation_threshold,
        ),
        plot_artifacts=args.plot_artifacts,
        realign_after_averaging=args.realign_after_averaging,
        search_window_factor=args.search_window_factor,
        apply_epoch_alpha_scaling=args.apply_epoch_alpha_scaling,
        track_estimated_noise=args.track_estimated_noise,
    )
    correction.name = "bcg_aas_flex_correction"
    correction.flex_preset_name = AAS_PER_TARGET
    correction.legacy_algorithm_resemblance = LEGACY_RESEMBLANCE[AAS_PER_TARGET]
    return [QRSTriggerDetector(), correction]


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
