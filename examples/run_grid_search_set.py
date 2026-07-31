"""Run the Flex grid search over every EDF file in a directory.
    Grid search only!
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
from loguru import logger

from facet.core import ProcessingContext

# from facet.correction.grid_search import CorrectionGridSearch
from facet.correction.grid_search_pareto import CorrectionGridSearch


def find_edf_files(
    input_directory: Path,
    recursive: bool = True,
) -> list[Path]:
    """Return sorted EDF files from an input directory."""
    pattern = "**/*.edf" if recursive else "*.edf"

    files = sorted(path for path in input_directory.glob(pattern) if path.is_file())

    if not files:
        raise FileNotFoundError(f"No EDF files found in {input_directory}")

    return files


def load_dataset(
    path: Path,
    dataset_id: str,
) -> ProcessingContext:
    """Load one EDF file for streamed grid-search processing."""
    logger.info("Loading dataset '{}': {}", dataset_id, path)

    raw = mne.io.read_raw_edf(
        path,
        preload=True,
        verbose="ERROR",
    )

    context = ProcessingContext(
        raw=raw,
        # Every combination receives an independent working Raw object.  The
        # source recording is read-only reference data, so it can safely serve
        # as both the initial and original Raw without a second full copy.
        raw_original=raw,
    )

    metadata = context.metadata.copy()
    metadata.custom["dataset_id"] = dataset_id
    metadata.custom["input_path"] = str(path.resolve())

    return context.with_metadata(metadata)


def build_search(output_directory: Path) -> CorrectionGridSearch:
    """Construct the Flex-only grid search."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return CorrectionGridSearch(
        trigger_regex=r"\b1\b",
        upsample_factor=10,
        output_csv=(output_directory / "facet_grid_search_results.csv"),
        output_aggregate_csv=(output_directory / "facet_grid_search_aggregate.csv"),
        output_parameter_effects_csv=(output_directory / "facet_grid_search_parameter_effects.csv"),
        output_combination_grid_csv=(output_directory / "facet_grid_search_combinations.csv"),
        output_pareto_csv=(output_directory / "facet_grid_search_pareto.csv"),
        output_pareto_2d=(output_directory / "facet_grid_search_pareto_2d.png"),
        output_pareto_3d=(output_directory / "facet_grid_search_pareto_3d.png"),
        output_pareto_matrix=(output_directory / "facet_grid_search_pareto_matrix.png"),
        output_combination_heatmap=(output_directory / "facet_grid_search_combination_heatmap.png"),
        output_parameter_effects_plot=(output_directory / "facet_grid_search_parameter_effects.png"),
        # Starter grid
        window_sizes=[10, 20, 30],
        thresholds=[0.50, 0.75, 0.90, 0.95],
        min_accepted_values=[3, 5],
        N_distributions=["equal", "normal"],
        realign_after_averaging_values=[True],
        search_window_factors=[1.0, 3.0],
        interpolate_volume_gaps_values=[
            False,
            True,
        ],
        apply_epoch_alpha_scaling_values=[
            False,
            True,
        ],
        pareto_objectives={
            "snr": "max",
            "rms_residual": "min",
            "fft_niazy_*": "min",
        },
        continue_on_error=True,
        show_progress=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=("Run the Flex parameter grid over all EDF files in a directory."))

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing EDF files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV files and plots.",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help=("Only search directly inside input-dir, not its subdirectories."),
    )

    args = parser.parse_args()

    input_directory = args.input_dir.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()

    if not input_directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")

    files = find_edf_files(
        input_directory,
        recursive=not args.no_recursive,
    )

    logger.info(
        "Found {} EDF dataset(s)",
        len(files),
    )

    dataset_paths: dict[str, Path] = {}

    for path in files:
        dataset_id = path.stem

        if dataset_id in dataset_paths:
            dataset_id = f"{path.parent.name}__{path.stem}"

        dataset_paths[dataset_id] = path

    search = build_search(output_directory)

    logger.info(
        "Running {} Flex configurations over {} datasets",
        search.n_combinations,
        len(dataset_paths),
    )

    logger.info(
        "Total pipeline executions: {}",
        search.n_combinations * len(dataset_paths),
    )

    result = search.run_many_from_paths(
        dataset_paths,
        load_dataset,
    )

    print()
    print("Grid search complete")
    print("====================")
    print(f"Datasets: {len(dataset_paths)}")
    print(f"Configurations: {search.n_combinations}")
    print(f"Total runs: {search.n_combinations * len(dataset_paths)}")
    print()
    print(
        "Detailed results:",
        result.csv_path,
    )
    print(
        "Aggregate results:",
        result.aggregate_csv_path,
    )
    print(
        "Parameter effects:",
        result.parameter_effects_csv_path,
    )
    print("Combination grid:", result.combination_grid_csv_path)
    print(
        "Pareto results:",
        result.pareto_csv_path,
    )
    print(
        "Pareto 2D plot:",
        result.pareto_2d_path,
    )
    print(
        "Pareto 3D plot:",
        result.pareto_3d_path,
    )
    print("Pareto matrix:", result.pareto_matrix_path)
    print("Combination heatmap:", result.combination_heatmap_path)
    print(
        "Parameter-effects plot:",
        result.parameter_effects_plot_path,
    )
    print("Selected Pareto parameters:", search.select_pareto_configuration(result))


if __name__ == "__main__":
    main()
