"""Argument parser construction for FACETpy command line entry points."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from facet.io.loaders import SUPPORTED_EXTENSIONS

from ._cli_pipeline import (
    ADD_ON_MODE_DESCRIPTIONS,
    CORRECTION_MODE_DESCRIPTIONS,
    DEFAULT_EGI_DROP_REGEX,
    PROCESS_PATTERN_DESCRIPTIONS,
    _parse_pca_components,
)

ANALYSIS_METRIC_DESCRIPTIONS = {
    "snr": "Signal-to-noise ratio using original and corrected data.",
    "legacy-snr": "Legacy SNR calculation for original/corrected pairs.",
    "rms": "RMS ratio between original and corrected data.",
    "rms-residual": "Residual RMS over artifact/reference intervals.",
    "median-artifact": "Median artifact summary around detected triggers.",
    "fft-allen": "Allen-style FFT artifact-frequency comparison.",
    "fft-niazy": "Niazy-style FFT improvement estimate.",
    "report": "Aggregate the metrics stored in processing metadata.",
}


def _add_input_arguments(parser: argparse.ArgumentParser, *, default_extensions: Sequence[str]) -> None:
    """Add shared single/list/folder input arguments."""
    parser.add_argument("--input", action="append", help="Input EEG file or MFF folder. May be passed multiple times.")
    parser.add_argument("--input-list", help="Text file containing one input path per line.")
    parser.add_argument("--input-dir", help="Folder containing inputs to process.")
    parser.add_argument("--recursive", action="store_true", help="Search input folders recursively.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(default_extensions),
        help="Input extensions to include when scanning folders.",
    )


def _build_parser(
    *,
    run_process: Callable[[argparse.Namespace], int],
    run_to_bids: Callable[[argparse.Namespace], int],
    run_modes: Callable[[argparse.Namespace], int],
    run_patterns: Callable[[argparse.Namespace], int],
    run_viewer: Callable[[argparse.Namespace], int],
    run_analysis: Callable[[argparse.Namespace], int],
) -> argparse.ArgumentParser:
    """Build the FACETpy command line parser."""
    parser = argparse.ArgumentParser(
        description="Run FACETpy correction and convert corrected outputs to BIDS.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser(
        "process",
        help="Run AAS correction over one file, a path list, or an input folder.",
    )
    _add_input_arguments(process, default_extensions=SUPPORTED_EXTENSIONS)
    process.add_argument(
        "--output-dir",
        "--output",
        required=True,
        dest="output_dir",
        help="Output folder for corrected segmented files.",
    )
    process.add_argument("--output-extension", default=".edf", help="Corrected output extension.")
    process.add_argument("--trigger-regex", default=r"\b1\b", help="Regex used to detect scanner triggers.")
    process.add_argument(
        "--upsample-factor", type=int, default=10, help="Upsampling factor before template correction."
    )
    process.add_argument("--window-size", type=int, default=30, help="Window size for template correction.")
    process.add_argument(
        "--pattern",
        choices=tuple(PROCESS_PATTERN_DESCRIPTIONS),
        default="quickstart",
        help="Whole-pipeline pattern. Use the 'patterns' command to list details.",
    )
    process.add_argument(
        "--highpass-freq",
        type=float,
        default=1.0,
        help="High-pass frequency used by --pattern=standard.",
    )
    process.add_argument(
        "--lowpass-freq",
        type=float,
        default=70.0,
        help="Low-pass frequency used by --pattern=standard.",
    )
    process.add_argument(
        "--bcg-window-size",
        type=int,
        default=20,
        help="AAS window size used by --pattern=bcg.",
    )
    process.add_argument(
        "--correction-mode",
        choices=tuple(CORRECTION_MODE_DESCRIPTIONS),
        default="aas",
        help="Template-subtraction strategy. Use 'modes' to list details.",
    )
    process.add_argument(
        "--mode",
        action="append",
        choices=tuple(ADD_ON_MODE_DESCRIPTIONS),
        default=None,
        help="Optional add-on correction mode. Repeat to combine modes, for example --mode pca --mode anc.",
    )
    process.add_argument(
        "--aas-correlation-threshold",
        type=float,
        default=0.975,
        help="Correlation threshold for the baseline AAS template strategy.",
    )
    process.add_argument(
        "--farm-correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for --correction-mode=farm.",
    )
    process.add_argument(
        "--farm-search-half-window",
        type=int,
        default=None,
        help="Explicit FARM candidate search half-window in epochs.",
    )
    process.add_argument(
        "--farm-search-half-window-factor",
        type=float,
        default=3.0,
        help="FARM search half-window multiplier when no explicit half-window is set.",
    )
    process.add_argument(
        "--slices-per-volume",
        type=int,
        default=None,
        help="Slice count used by --correction-mode=corresponding-slice when it cannot be inferred.",
    )
    process.add_argument(
        "--motion-rp-file",
        default=None,
        help="SPM-style realignment parameter file required by --correction-mode=moosmann.",
    )
    process.add_argument(
        "--motion-threshold",
        type=float,
        default=5.0,
        help="Motion threshold for --correction-mode=moosmann.",
    )
    process.add_argument(
        "--motion-window-size",
        type=int,
        default=None,
        help="Motion weighting window size for --correction-mode=moosmann.",
    )
    process.add_argument(
        "--plot-artifacts",
        action="store_true",
        help="Plot one representative averaged artifact for template-correction modes.",
    )
    process.add_argument(
        "--realign-after-averaging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Realign triggers to averaged artifact templates after template calculation.",
    )
    process.add_argument(
        "--search-window-factor",
        type=float,
        default=3.0,
        help="Trigger realignment search-window factor for AAS-style modes.",
    )
    process.add_argument(
        "--interpolate-volume-gaps",
        action="store_true",
        help="Interpolate estimated noise in volume gaps for AAS/FARM modes.",
    )
    process.add_argument(
        "--apply-epoch-alpha-scaling",
        action="store_true",
        help="Scale each epoch template before subtraction with a least-squares alpha factor.",
    )
    process.add_argument(
        "--volume-template-count",
        type=int,
        default=5,
        help="Neighboring slice count for --mode=volume-artifact.",
    )
    process.add_argument(
        "--volume-weighting-position",
        type=float,
        default=0.8,
        help="Logistic midpoint inside one artifact epoch for --mode=volume-artifact.",
    )
    process.add_argument(
        "--volume-weighting-slope",
        type=float,
        default=20.0,
        help="Logistic slope for --mode=volume-artifact.",
    )
    process.add_argument(
        "--pca-components",
        type=_parse_pca_components,
        default=0.95,
        help="PCA components for --mode=pca: integer, 0-1 variance fraction, or 'auto'.",
    )
    process.add_argument(
        "--pca-hp-freq",
        type=float,
        default=1.0,
        help="High-pass cutoff before --mode=pca. Use 0 to disable.",
    )
    process.add_argument(
        "--anc-filter-order",
        type=int,
        default=None,
        help="Adaptive filter order for --mode=anc. Defaults to artifact length.",
    )
    process.add_argument(
        "--anc-hp-freq",
        type=float,
        default=None,
        help="High-pass cutoff for --mode=anc. Defaults to trigger-rate-derived value.",
    )
    process.add_argument(
        "--anc-c-extension",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the FastRANC C extension for --mode=anc when available.",
    )
    process.add_argument(
        "--anc-mu-factor",
        type=float,
        default=0.05,
        help="Learning-rate numerator for --mode=anc.",
    )
    process.add_argument(
        "--anc-max-gain",
        type=float,
        default=50.0,
        help="Maximum stable filtered-noise gain for --mode=anc.",
    )
    process.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    process.add_argument("--flat-output", action="store_true", help="Write all batch outputs directly into output-dir.")
    process.add_argument(
        "--channel-sequential",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the high-memory correction steps one channel at a time.",
    )
    process.add_argument(
        "--fixed-length-chunks",
        action="store_true",
        help="Use memory-estimated fixed-length chunks instead of trigger-section chunks.",
    )
    process.add_argument("--min-chunks", type=int, default=2, help="Minimum fixed-length chunk count.")
    process.add_argument("--max-chunks", type=int, default=128, help="Maximum fixed-length chunk count.")
    process.add_argument("--memory-budget-mb", type=float, default=None, help="Explicit per-chunk memory budget.")
    process.add_argument("--memory-fraction", type=float, default=0.5, help="Fraction of available memory to use.")
    process.add_argument(
        "--trigger-section-padding-seconds",
        type=float,
        default=10.0,
        help="Seconds kept before first trigger and after last trigger in each section.",
    )
    process.add_argument(
        "--trigger-section-min-triggers",
        type=int,
        default=16,
        help="Minimum trigger count for a trigger section.",
    )
    process.add_argument(
        "--trigger-section-gap-seconds",
        type=float,
        default=None,
        help="Explicit no-trigger gap used to split trigger sections.",
    )
    process.add_argument(
        "--trigger-section-max-sections",
        type=int,
        default=None,
        help="Maximum trigger sections exported per input. Current: None",
    )
    process.add_argument(
        "--drop-egi-e-channels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop EGI E1-E128 channels on a copy before trigger detection and correction. Disabled by default.",
    )
    process.add_argument(
        "--drop-channel-regex",
        default=DEFAULT_EGI_DROP_REGEX,
        help="Regex used when --drop-egi-e-channels is enabled.",
    )
    process.add_argument("--on-error", choices=("raise", "continue"), default="raise", help="Batch error behavior.")
    process.set_defaults(func=run_process)

    bids = subparsers.add_parser(
        "to-bids",
        help="Convert corrected output files into a BIDS dataset.",
    )
    _add_input_arguments(bids, default_extensions=(".edf", ".bdf", ".gdf", ".vhdr", ".set", ".fif"))
    bids.add_argument(
        "--output-dir",
        "--bids-dir",
        required=True,
        dest="output_dir",
        help="Destination BIDS root folder.",
    )
    bids.add_argument("--task", default="facetcorrected", help="BIDS task label.")
    bids.add_argument("--subject", default=None, help="BIDS subject label. Defaults to source filename.")
    bids.add_argument("--session", default=None, help="Optional BIDS session label.")
    bids.add_argument("--trigger-regex", default=r"\b1\b", help="Regex used to recover event markers.")
    bids.add_argument(
        "--detect-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally detect trigger events before BIDS export.",
    )
    bids.add_argument("--overwrite", action="store_true", help="Overwrite existing BIDS files.")
    bids.add_argument("--on-error", choices=("raise", "continue"), default="raise", help="Batch error behavior.")
    bids.set_defaults(func=run_to_bids)

    modes = subparsers.add_parser(
        "modes",
        help="List correction and add-on modes available to the process command.",
    )
    modes.set_defaults(func=run_modes)

    patterns = subparsers.add_parser(
        "patterns",
        help="List whole-pipeline patterns from the quickstart documentation.",
    )
    patterns.set_defaults(func=run_patterns)

    viewer = subparsers.add_parser(
        "viewer",
        aliases=("view",),
        help="View or save a raw EEG plot using FACETpy RawPlotter.",
    )
    viewer.add_argument("--input", required=True, help="Input EEG file or MFF folder.")
    viewer.add_argument("--output", default=None, help="Optional plot image path.")
    viewer.add_argument("--viewer-mode", choices=("matplotlib", "mne"), default="matplotlib", help="Viewer backend.")
    viewer.add_argument("--channel", default=None, help="Channel name or index for matplotlib mode.")
    viewer.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    viewer.add_argument("--duration", type=float, default=10.0, help="Duration in seconds. Use 0 for full span.")
    viewer.add_argument("--show", action="store_true", help="Show the plot interactively.")
    viewer.add_argument("--title", default=None, help="Optional plot title.")
    viewer.add_argument("--n-channels", type=int, default=None, help="MNE viewer channel count.")
    viewer.add_argument("--scalings", default=None, help="MNE viewer scaling mode, e.g. 'auto'.")
    viewer.set_defaults(func=run_viewer)

    analysis = subparsers.add_parser(
        "analysis",
        help="Run FACETpy recording checks and optional quality metrics.",
    )
    analysis.add_argument("--input", default=None, help="Input EEG file or MFF folder.")
    analysis.add_argument("--trigger-regex", default=r"\b1\b", help="Regex used when detecting events.")
    analysis.add_argument(
        "--detect-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Detect trigger events before running reports.",
    )
    analysis.add_argument(
        "--require-triggers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat missing triggers as a failed data check.",
    )
    analysis.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Raise an error when CheckDataReport finds problems.",
    )
    analysis.add_argument("--metrics", action="store_true", help="Run the default quality metric suite.")
    analysis.add_argument(
        "--metric",
        action="append",
        choices=tuple(ANALYSIS_METRIC_DESCRIPTIONS),
        default=None,
        help="Run one selected metric. May be passed multiple times.",
    )
    analysis.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available metric names and exit.",
    )
    analysis.add_argument(
        "--skip-inapplicable-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip metric processors that do not apply to this recording.",
    )
    analysis.add_argument("--output-json", default=None, help="Optional path for structured analysis metadata.")
    analysis.set_defaults(func=run_analysis)

    return parser
