"""Run a FACETpy AAS/FARM correction grid search.

This script is intentionally configured with a small starter grid by default.
Use ``--full-grid`` only when you are ready for the full 12,000-combination
search.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from facet.pipelines import create_correction_grid_search_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the grid-search runner."""
    parser = argparse.ArgumentParser(description="Run an AAS/FARM correction grid search.")
    parser.add_argument(
        "--input",
        default="examples/datasets/NiazyFMRI.edf",
        help="Input EEG file path.",
    )
    parser.add_argument(
        "--output-dir",
        default="grid_search_outputs",
        help="Directory for CSV and diagram outputs.",
    )
    parser.add_argument(
        "--trigger-regex",
        default=r"\b1\b",
        help="Regex used by TriggerDetector to find scanner triggers.",
    )
    parser.add_argument(
        "--upsample-factor",
        type=int,
        default=10,
        help="Upsampling/downsampling factor used around the correction model.",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Run the full default 12,000-combination grid instead of the starter grid.",
    )
    parser.add_argument(
        "--all-channels-at-once",
        action="store_true",
        help="Disable channel-sequential correction. Faster on small files, but much higher memory use.",
    )
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace):
    """Build the configured grid-search pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_kwargs = {"channel_sequential": not args.all_channels_at_once}
    if not args.full_grid:
        grid_kwargs.update(
            {
                "models": ["aas", "farm"],
                "correlation_thresholds": [0.90, 0.95, 0.99],
                "rel_window_positions": [0.0],
                "search_window_factors": [1.0, 2.0, 3.0],
                "window_sizes": [10, 20, 30],
            }
        )

    return create_correction_grid_search_pipeline(
        input_path=args.input,
        output_csv=output_dir / "grid_results.csv",
        output_score_grid_csv=output_dir / "grid_score_grid.csv",
        output_diagram=output_dir / "grid_score_grid.png",
        trigger_regex=args.trigger_regex,
        upsample_factor=args.upsample_factor,
        **grid_kwargs,
    )


def main() -> None:
    """Run the grid search and print the best parameter set."""
    args = parse_args()
    pipeline = build_pipeline(args)

    result = pipeline.run(show_progress=True)
    if not result.success:
        raise SystemExit(f"Grid search failed: {result.error}")

    summary = result.context.metadata.custom["grid_search"]
    print("Best params:", summary["best_params"])
    print("Best score:", summary["best_score"])
    print("Results CSV:", summary["csv_path"])
    print("Score grid CSV:", summary["score_grid_csv_path"])
    print("Diagram:", summary["diagram_path"])


if __name__ == "__main__":
    main()
