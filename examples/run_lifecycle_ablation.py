"""Diagnose Flex lifecycle over-correction on one EEG recording.

This script deliberately holds averaging-matrix construction constant while
changing only the three post-matrix lifecycle switches most likely to affect
signal preservation:

* template realignment;
* epoch-wise least-squares alpha scaling;
* interpolation of the artifact estimate between epochs.

An uncorrected control is evaluated through the same trigger-section chunking
and metric processors. The resulting table therefore shows whether an
abnormal RMS or median-artifact ratio already exists before correction, or is
introduced by one of the shared Flex lifecycle stages.

By default, the fixed matrix recipe is ``MatrixDecisions()`` (the current Flex
per-target default). Pass a fold's ``selected_recipe.json`` with
``--recipe-json`` to diagnose the exact recipe selected by optimization.

Example
-------
::

    uv run python examples/run_lifecycle_ablation.py \
        /path/to/example.edf /path/to/ablation_results \
        --recipe-json output/OP_test1_results/folds/<subject>/selected_recipe.json

Results from diagnosing rest sub1 from NATVEIW: 
                               name  success  snr_log_vs_uncorrected  rms_residual  median_artifact_ratio  both_preservation_feasible
                        uncorrected     True                0.000000      1.305737               1.003433                       False
      alpha_off_gaps_off_realign_on     True               10.081024      0.087050               0.011878                       False
       alpha_off_gaps_on_realign_on     True               10.081024      0.087050               0.011878                       False
     alpha_off_gaps_off_realign_off     True               10.080982      0.087052               0.011878                       False
       alpha_on_gaps_off_realign_on     True               10.081024      0.086978               0.011939                       False
current_alpha_on_gaps_on_realign_on     True               10.081024      0.086978               0.011939                       False

Explanation:


1. 
Key measurements
  artifact_length_seconds_median: 2.1
  template_fraction_of_acquisition_interval_median: 0.9965274482382777
  estimate_to_original_template_rms_median: 0.9983048265967098
  corrected_to_original_template_rms_median: 0.056977559100687816
  template_original_estimate_correlation_median: 0.9854014718531838
  subtraction_identity_relative_error_median: 0.15842845604876776
  estimate_energy_outside_template_fraction_median: 5.777390724015212e-10

Findings
  [CRITICAL] Corrected data does not equal original minus the retained Flex estimate.
    Next: Inspect noise resampling and trigger realignment before changing A.
  [CRITICAL] The template reproduces most target-epoch energy and leaves less than 20% RMS.
    Next: Inspect D/N overlays and target self-weight; constrain A or template scaling only if masks are correct.
2026-08-10 15:09:59.917 | INFO     | __main__:main:433 - Diagnostic written to output/OP_test1_alignment_diagnostic

2. 
Key measurements
  artifact_length_seconds_median: 2.1
  template_fraction_of_acquisition_interval_median: 0.9965274482382777
  estimate_to_original_template_rms_median: 0.9983048575003656
  corrected_to_original_template_rms_median: 0.05697756084521831
  template_original_estimate_correlation_median: 0.9993407239174933
  subtraction_identity_relative_error_median: 5.158067464765207e-16
  resampling_roundtrip_relative_error_median: 0.15842845604876776
  estimate_energy_outside_template_fraction_median: 5.777390724015212e-10

Findings
  [CRITICAL] The template reproduces most target-epoch energy and leaves less than 20% RMS.
    Next: Inspect D/N overlays and target self-weight; constrain A or template scaling only if masks are correct.

3. 
Key measurements
  artifact_length_seconds_median: 2.1
  template_fraction_of_acquisition_interval_median: 0.9965274482382777
  estimate_to_original_template_rms_median: 0.9983048575003656
  corrected_to_original_template_rms_median: 0.05697756084521831
  template_original_estimate_correlation_median: 0.9993407239174933
  subtraction_identity_relative_error_median: 5.158067464765207e-16
  resampling_roundtrip_relative_error_median: 0.15842845604876776
  estimate_energy_outside_template_fraction_median: 5.777390724015212e-10

Findings
  [CRITICAL] The template reproduces most target-epoch energy and leaves less than 20% RMS.
    Next: Inspect D/N overlays and target self-weight; constrain A or template scaling only if masks are correct.
  [CRITICAL] Corrected beta power is below 25% of scanner-off reference power.
    Next: Do not optimize on time-domain SNR alone; inspect whether this band is present in N and add a band-preservation constraint.

Median band preservation (corrected acquisition / scanner-off reference)
  delta: 0.9592135928825141
  theta: 0.9199361939175332
  alpha: 0.724740555028649
  beta: 0.02464866116605488
  high_frequency: 0.0032525594359758755
  broadband: 0.0053545655269692715

A-matrix target self-weight audit
  nonzero_fraction_median: 0.0
  mean_median: 0.0
  maximum: 0.0
  any_truncated_audit: False


"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

# The optimization runner owns the robust chunk-result aggregation used by
# the main experiment. Reusing it keeps diagnostic and optimization metrics
# directly comparable.
from run_matrix_optimization import aggregate_chunk_results, release_process_memory

from facet import (
    DownSample,
    Flex,
    MatrixDecisions,
    MedianArtifactCalculator,
    Pipeline,
    Processor,
    RMSResidualCalculator,
    SNRCalculator,
    TriggerDetector,
    UpSample,
)


@dataclass(frozen=True)
class LifecycleVariant:
    """One fixed setting of Flex's post-matrix lifecycle decisions."""

    name: str
    realign_after_averaging: bool | None
    apply_epoch_alpha_scaling: bool | None
    interpolate_volume_gaps: bool | None
    description: str

    @property
    def corrected(self) -> bool:
        return self.realign_after_averaging is not None


QUICK_VARIANTS = (
    LifecycleVariant(
        name="uncorrected",
        realign_after_averaging=None,
        apply_epoch_alpha_scaling=None,
        interpolate_volume_gaps=None,
        description="Metrics without Flex correction",
    ),
    LifecycleVariant(
        name="alpha_off_gaps_off_realign_on",
        realign_after_averaging=True,
        apply_epoch_alpha_scaling=False,
        interpolate_volume_gaps=False,
        description="Recommended conservative starting point",
    ),
    LifecycleVariant(
        name="alpha_off_gaps_on_realign_on",
        realign_after_averaging=True,
        apply_epoch_alpha_scaling=False,
        interpolate_volume_gaps=True,
        description="Isolate gap interpolation with alpha disabled",
    ),
    LifecycleVariant(
        name="alpha_off_gaps_off_realign_off",
        realign_after_averaging=False,
        apply_epoch_alpha_scaling=False,
        interpolate_volume_gaps=False,
        description="Isolate realignment with alpha and gaps disabled",
    ),
    LifecycleVariant(
        name="alpha_on_gaps_off_realign_on",
        realign_after_averaging=True,
        apply_epoch_alpha_scaling=True,
        interpolate_volume_gaps=False,
        description="Isolate alpha scaling without gap interpolation",
    ),
    LifecycleVariant(
        name="current_alpha_on_gaps_on_realign_on",
        realign_after_averaging=True,
        apply_epoch_alpha_scaling=True,
        interpolate_volume_gaps=True,
        description="Lifecycle used by the original optimization run",
    ),
)


def full_factorial_variants() -> tuple[LifecycleVariant, ...]:
    """Return the control and every Boolean lifecycle combination."""
    variants = [QUICK_VARIANTS[0]]
    for realign in (False, True):
        for alpha in (False, True):
            for gaps in (False, True):
                variants.append(
                    LifecycleVariant(
                        name=(
                            f"realign_{'on' if realign else 'off'}__"
                            f"alpha_{'on' if alpha else 'off'}__"
                            f"gaps_{'on' if gaps else 'off'}"
                        ),
                        realign_after_averaging=realign,
                        apply_epoch_alpha_scaling=alpha,
                        interpolate_volume_gaps=gaps,
                        description="Full-factorial lifecycle combination",
                    )
                )
    return tuple(variants)


def load_matrix_recipe(path: Path | None) -> MatrixDecisions:
    """Load a raw manifest or an optimizer ``selected_recipe.json`` file."""
    if path is None:
        return MatrixDecisions()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Recipe JSON must contain an object.")
    manifest = payload.get("recipe", payload)
    if not isinstance(manifest, Mapping):
        raise ValueError("The JSON 'recipe' field must contain a MatrixDecisions object.")
    return MatrixDecisions.from_dict(manifest)


def metric_processors() -> list[Processor]:
    """Create fresh metric processors for one pipeline."""
    return [SNRCalculator(), RMSResidualCalculator(), MedianArtifactCalculator()]


def build_variant_pipeline(
    variant: LifecycleVariant,
    *,
    decisions: MatrixDecisions,
    trigger_regex: str,
    upsample_factor: int,
) -> Pipeline:
    """Build a control or corrected pipeline for one lifecycle variant."""
    processors: list[Processor] = [TriggerDetector(regex=trigger_regex)]
    if variant.corrected:
        processors.extend(
            [
                UpSample(factor=upsample_factor),
                Flex(
                    matrix_decisions=decisions,
                    plot_artifacts=False,
                    realign_after_averaging=bool(variant.realign_after_averaging),
                    interpolate_volume_gaps=bool(variant.interpolate_volume_gaps),
                    apply_epoch_alpha_scaling=bool(variant.apply_epoch_alpha_scaling),
                    track_estimated_noise=False,
                ),
                DownSample(factor=upsample_factor),
            ]
        )
    processors.extend(metric_processors())
    return Pipeline(processors, name=f"Lifecycle ablation: {variant.name}")


def evaluate_variant(
    variant: LifecycleVariant,
    *,
    input_path: Path,
    output_directory: Path,
    decisions: MatrixDecisions,
    trigger_regex: str,
    upsample_factor: int,
    chunk_padding_seconds: float,
    chunk_min_triggers: int,
    chunk_gap_seconds: float | None,
    keep_corrected: bool,
) -> dict[str, Any]:
    """Run and summarize one lifecycle setting."""
    logger.info("Evaluating lifecycle variant '{}': {}", variant.name, variant.description)
    started = time.perf_counter()
    pipeline = build_variant_pipeline(
        variant,
        decisions=decisions,
        trigger_regex=trigger_regex,
        upsample_factor=upsample_factor,
    )

    try:
        if keep_corrected:
            variant_output = output_directory / "corrected_eeg" / variant.name
            variant_output.mkdir(parents=True, exist_ok=True)
            temporary_context = None
        else:
            temporary_context = tempfile.TemporaryDirectory(prefix="facetpy_lifecycle_ablation_")
            variant_output = Path(temporary_context.name)

        try:
            result = pipeline.run_chunked(
                input_path=str(input_path),
                output_dir=str(variant_output),
                output_extension=".edf",
                overwrite=True,
                channel_sequential=True,
                on_error="continue",
                keep_raw=False,
                chunk_by_trigger_sections=True,
                trigger_section_padding_seconds=chunk_padding_seconds,
                trigger_section_min_triggers=chunk_min_triggers,
                trigger_section_gap_seconds=chunk_gap_seconds,
            )
            metrics = aggregate_chunk_results(result)
            del result
        finally:
            if temporary_context is not None:
                temporary_context.cleanup()

        return {
            **asdict(variant),
            "corrected": variant.corrected,
            **metrics,
            "wall_time": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            **asdict(variant),
            "corrected": variant.corrected,
            "success": False,
            "n_chunks": 0,
            "successful_chunks": 0,
            "execution_time": 0.0,
            "wall_time": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }
    finally:
        release_process_memory()


def add_control_comparisons(frame: pd.DataFrame, *, ratio_bound: float) -> pd.DataFrame:
    """Add interpretable changes relative to the uncorrected control."""
    output = frame.copy()
    for metric in ("snr", "rms_residual", "median_artifact_ratio"):
        if metric not in output:
            output[metric] = np.nan
    controls = output[output["name"] == "uncorrected"]
    if controls.empty or not bool(controls.iloc[0].get("success", False)):
        output["snr_log_vs_uncorrected"] = np.nan
        output["rms_fraction_of_uncorrected"] = np.nan
        output["median_artifact_fraction_of_uncorrected"] = np.nan
    else:
        control = controls.iloc[0]
        control_snr = float(control["snr"])
        control_rms = float(control["rms_residual"])
        control_median = float(control["median_artifact_ratio"])
        output["snr_log_vs_uncorrected"] = output["snr"].map(
            lambda value: (
                math.log(float(value) / control_snr)
                if np.isfinite(value) and value > 0.0 and control_snr > 0.0
                else np.nan
            )
        )
        output["rms_fraction_of_uncorrected"] = output["rms_residual"] / control_rms
        output["median_artifact_fraction_of_uncorrected"] = output["median_artifact_ratio"] / control_median

    lower = 1.0 / ratio_bound
    output["rms_preservation_feasible"] = output["rms_residual"].between(lower, ratio_bound)
    output["median_artifact_preservation_feasible"] = output["median_artifact_ratio"].between(
        lower,
        ratio_bound,
    )
    output["both_preservation_feasible"] = (
        output["rms_preservation_feasible"] & output["median_artifact_preservation_feasible"]
    )
    return output


def lifecycle_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple main-effect summaries from a full factorial run."""
    corrected = frame[frame["corrected"].astype(bool) & frame["success"].astype(bool)]
    rows: list[dict[str, Any]] = []
    lifecycle_columns = (
        "realign_after_averaging",
        "apply_epoch_alpha_scaling",
        "interpolate_volume_gaps",
    )
    # An unbalanced quick diagnostic confounds the switches. Main effects are
    # therefore emitted only when all eight corrected combinations succeeded.
    observed_settings = corrected[list(lifecycle_columns)].drop_duplicates()
    if len(observed_settings) != 8:
        return pd.DataFrame()
    for parameter in lifecycle_columns:
        for metric in (
            "snr_log_vs_uncorrected",
            "rms_log_deviation",
            "median_artifact_log_deviation",
        ):
            grouped = corrected.groupby(parameter, dropna=False)[metric].mean()
            if True not in grouped.index or False not in grouped.index:
                continue
            rows.append(
                {
                    "parameter": parameter,
                    "metric": metric,
                    "mean_when_off": float(grouped.loc[False]),
                    "mean_when_on": float(grouped.loc[True]),
                    "on_minus_off": float(grouped.loc[True] - grouped.loc[False]),
                }
            )
    return pd.DataFrame(rows)


def plot_results(frame: pd.DataFrame, path: Path, *, ratio_bound: float) -> None:
    """Plot raw metrics with ratio targets and preservation bounds."""
    successful = frame[frame["success"].astype(bool)].copy()
    if successful.empty:
        return
    labels = successful["name"].str.replace("_", " ").tolist()
    positions = np.arange(len(successful))
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].bar(positions, successful["snr_log_vs_uncorrected"])
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("log SNR change vs uncorrected")
    axes[0].set_ylabel("Higher is better")

    lower = 1.0 / ratio_bound
    for axis, metric, title in (
        (axes[1], "rms_residual", "RMS residual ratio"),
        (axes[2], "median_artifact_ratio", "Median artifact ratio"),
    ):
        axis.bar(positions, successful[metric])
        axis.axhspan(lower, ratio_bound, color="green", alpha=0.15, label="acceptable")
        axis.axhline(1.0, color="green", linewidth=1.5, label="ideal")
        axis.set_yscale("log")
        axis.set_title(title)
        axis.legend(loc="best")

    for axis in axes:
        axis.set_xticks(positions, labels, rotation=55, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("One-recording Flex lifecycle ablation")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    frame: pd.DataFrame,
    *,
    output_directory: Path,
    input_path: Path,
    decisions: MatrixDecisions,
    variants: Sequence[LifecycleVariant],
    ratio_bound: float,
    settings: Mapping[str, Any],
) -> None:
    """Write complete diagnostic tables, plot, and provenance manifest."""
    output_directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_directory / "lifecycle_ablation_results.csv", index=False)
    effects = lifecycle_effects(frame)
    effects.to_csv(output_directory / "lifecycle_main_effects.csv", index=False)
    plot_results(
        frame,
        output_directory / "lifecycle_ablation_metrics.png",
        ratio_bound=ratio_bound,
    )
    manifest = {
        "input_path": str(input_path.resolve()),
        "matrix_recipe": decisions.to_dict(),
        "ratio_target": 1.0,
        "acceptable_ratio_interval": [1.0 / ratio_bound, ratio_bound],
        "variants": [asdict(item) for item in variants],
        "settings": dict(settings),
        "interpretation": {
            "rms_residual": "Ideal 1; below interval suggests over-correction.",
            "median_artifact_ratio": "Ideal 1; below interval suggests over-correction.",
            "snr_log_vs_uncorrected": "Positive improves on the same recording's uncorrected SNR.",
        },
    }
    (output_directory / "lifecycle_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _validate_arguments(args: argparse.Namespace) -> None:
    if not args.input_path.exists():
        raise ValueError(f"Input recording does not exist: {args.input_path}")
    if args.upsample_factor < 1:
        raise ValueError("--upsample-factor must be at least 1.")
    if args.chunk_padding_seconds < 0.0:
        raise ValueError("--chunk-padding-seconds cannot be negative.")
    if args.chunk_min_triggers < 1:
        raise ValueError("--chunk-min-triggers must be at least 1.")
    if args.chunk_gap_seconds is not None and args.chunk_gap_seconds <= 0.0:
        raise ValueError("--chunk-gap-seconds must be positive when supplied.")
    if args.ratio_bound <= 1.0:
        raise ValueError("--ratio-bound must be greater than 1.")
    if args.recipe_json is not None and not args.recipe_json.is_file():
        raise ValueError("--recipe-json must identify an existing JSON file.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="One EEG recording to diagnose.")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--recipe-json",
        type=Path,
        help="Optimizer selected_recipe.json or a raw MatrixDecisions manifest.",
    )
    parser.add_argument("--trigger-regex", default=r"^R128$")
    parser.add_argument("--upsample-factor", type=int, default=10)
    parser.add_argument("--chunk-padding-seconds", type=float, default=5.0)
    parser.add_argument("--chunk-min-triggers", type=int, default=20)
    parser.add_argument("--chunk-gap-seconds", type=float)
    parser.add_argument("--ratio-bound", type=float, default=1.25)
    parser.add_argument(
        "--full-factorial",
        action="store_true",
        help="Run all eight corrected combinations instead of the six diagnostic variants.",
    )
    parser.add_argument(
        "--keep-corrected",
        action="store_true",
        help="Keep each variant's corrected EDF chunks under corrected_eeg/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the one-recording lifecycle diagnostic."""
    args = build_argument_parser().parse_args(argv)
    _validate_arguments(args)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    decisions = load_matrix_recipe(args.recipe_json)
    variants = full_factorial_variants() if args.full_factorial else QUICK_VARIANTS
    settings = {
        "trigger_regex": args.trigger_regex,
        "upsample_factor": args.upsample_factor,
        "chunk_padding_seconds": args.chunk_padding_seconds,
        "chunk_min_triggers": args.chunk_min_triggers,
        "chunk_gap_seconds": args.chunk_gap_seconds,
        "full_factorial": args.full_factorial,
        "keep_corrected": args.keep_corrected,
    }

    rows = []
    for variant in variants:
        row = evaluate_variant(
            variant,
            input_path=args.input_path,
            output_directory=args.output_directory,
            decisions=decisions,
            trigger_regex=args.trigger_regex,
            upsample_factor=args.upsample_factor,
            chunk_padding_seconds=args.chunk_padding_seconds,
            chunk_min_triggers=args.chunk_min_triggers,
            chunk_gap_seconds=args.chunk_gap_seconds,
            keep_corrected=args.keep_corrected,
        )
        rows.append(row)
        # Checkpoint after every expensive lifecycle run.
        partial = add_control_comparisons(pd.DataFrame(rows), ratio_bound=args.ratio_bound)
        partial.to_csv(args.output_directory / "lifecycle_ablation_results.csv", index=False)

    frame = add_control_comparisons(pd.DataFrame(rows), ratio_bound=args.ratio_bound)
    write_outputs(
        frame,
        output_directory=args.output_directory,
        input_path=args.input_path,
        decisions=decisions,
        variants=variants,
        ratio_bound=args.ratio_bound,
        settings=settings,
    )
    display_columns = [
        "name",
        "success",
        "snr_log_vs_uncorrected",
        "rms_residual",
        "median_artifact_ratio",
        "both_preservation_feasible",
    ]
    print(frame[display_columns].to_string(index=False))
    logger.info("Lifecycle ablation complete: {}", args.output_directory / "lifecycle_ablation_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
