"""Grid-search utilities for Flex correction parameters.

This module searches only the public parameters of :class:`Flex`. It can run
one parameter grid on one ``ProcessingContext`` or the same grid across many
datasets. The detailed output contains one row per dataset and parameter
combination. Additional CSV files summarize cross-dataset performance, the
best configuration for each dataset, and descriptive parameter effects.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid

from ..core import (
    Pipeline,
    ProcessingContext,
    Processor,
    ProcessorValidationError,
    register_processor,
)
from ..evaluation import (
    FFTAllenCalculator,
    FFTNiazyCalculator,
    LegacySNRCalculator,
    MedianArtifactCalculator,
    RMSCalculator,
    RMSResidualCalculator,
    SNRCalculator,
)
from ..preprocessing import DownSample, TriggerDetector, UpSample
from .flex import Flex

# ---------------------------------------------------------------------------
# Default Flex search space
# ---------------------------------------------------------------------------
# These defaults deliberately form a moderate starter grid. Expand them only
# after confirming that the complete pipeline and output files work correctly.
DEFAULT_WINDOW_SIZES = (10, 20, 30)
DEFAULT_THRESHOLDS = (0.50, 0.75, 0.90, 0.95)
DEFAULT_MIN_ACCEPTED_VALUES = (3, 5)
DEFAULT_N_DISTRIBUTIONS = ("equal", "normal")
DEFAULT_REALIGN_AFTER_AVERAGING_VALUES = (True,)
DEFAULT_SEARCH_WINDOW_FACTORS = (1.0, 3.0)
DEFAULT_INTERPOLATE_VOLUME_GAPS_VALUES = (False, True)
DEFAULT_APPLY_EPOCH_ALPHA_SCALING_VALUES = (False, True)

FLEX_PARAMETER_NAMES = (
    "window_size",
    "threshold",
    "min_accepted",
    "N_distribution",
    "realign_after_averaging",
    "search_window_factor",
    "interpolate_volume_gaps",
    "apply_epoch_alpha_scaling",
)

DEFAULT_METRIC_FACTORIES: tuple[Callable[[], Processor], ...] = (
    SNRCalculator,
    LegacySNRCalculator,
    RMSCalculator,
    RMSResidualCalculator,
    MedianArtifactCalculator,
    FFTAllenCalculator,
    FFTNiazyCalculator,
)


@dataclass(frozen=True)
class CorrectionGridSearchResult:
    """Results from a Flex grid search on one or more datasets."""

    results: pd.DataFrame
    aggregate_results: pd.DataFrame
    parameter_effects: pd.DataFrame
    best_per_dataset: pd.DataFrame
    best_params: dict[str, Any] | None
    best_metrics: dict[str, float]
    best_score: float | None
    best_index: int | None
    csv_path: Path | None = None
    aggregate_csv_path: Path | None = None
    parameter_effects_csv_path: Path | None = None
    best_per_dataset_csv_path: Path | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-friendly summary for ``ProcessingContext`` metadata."""
        return {
            "n_results": int(len(self.results)),
            "n_configurations": int(len(self.aggregate_results)),
            "n_datasets": (int(self.results["dataset_id"].nunique()) if "dataset_id" in self.results else 0),
            "best_params": deepcopy(self.best_params),
            "best_metrics": deepcopy(self.best_metrics),
            "best_score": self.best_score,
            "best_index": self.best_index,
            "csv_path": str(self.csv_path) if self.csv_path is not None else None,
            "aggregate_csv_path": (str(self.aggregate_csv_path) if self.aggregate_csv_path is not None else None),
            "parameter_effects_csv_path": (
                str(self.parameter_effects_csv_path) if self.parameter_effects_csv_path is not None else None
            ),
            "best_per_dataset_csv_path": (
                str(self.best_per_dataset_csv_path) if self.best_per_dataset_csv_path is not None else None
            ),
        }


@register_processor
class CorrectionGridSearch(Processor):
    """Exhaustively search Flex parameters.

    Every valid parameter combination is evaluated using this pipeline order::

        TriggerDetector -> UpSample -> Flex -> DownSample -> metrics

    ``run_search`` evaluates one dataset. ``run_many`` evaluates the same grid
    across multiple datasets and ranks configurations by their mean score.

    Notes
    -----
    This class searches only ``Flex``. It does not import or instantiate AAS,
    FARM, or any other correction class.
    """

    name = "correction_grid_search"
    description = "Grid search Flex correction parameters"
    version = "2.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        trigger_regex: str = r"\b1\b",
        upsample_factor: int = 10,
        output_csv: str | Path | None = "facet_grid_search_results.csv",
        output_aggregate_csv: str | Path | None = "facet_grid_search_aggregate.csv",
        output_parameter_effects_csv: str | Path | None = "facet_grid_search_parameter_effects.csv",
        output_best_per_dataset_csv: str | Path | None = "facet_grid_search_best_per_dataset.csv",
        window_sizes: Sequence[int] | None = None,
        thresholds: Sequence[float] | None = None,
        min_accepted_values: Sequence[int] | None = None,
        N_distributions: Sequence[str] | None = None,
        realign_after_averaging_values: Sequence[bool] | None = None,
        search_window_factors: Sequence[float] | None = None,
        interpolate_volume_gaps_values: Sequence[bool] | None = None,
        apply_epoch_alpha_scaling_values: Sequence[bool] | None = None,
        metric_processors: Sequence[Processor | Callable[[], Processor]] | None = None,
        scoring: str | Callable[[dict[str, float]], float] = "auto",
        greater_is_better: bool = True,
        track_estimated_noise: bool = False,
        continue_on_error: bool = True,
        show_progress: bool = False,
    ) -> None:
        self.trigger_regex = trigger_regex
        self.upsample_factor = int(upsample_factor)

        self.output_csv = Path(output_csv) if output_csv is not None else None
        self.output_aggregate_csv = Path(output_aggregate_csv) if output_aggregate_csv is not None else None
        self.output_parameter_effects_csv = (
            Path(output_parameter_effects_csv) if output_parameter_effects_csv is not None else None
        )
        self.output_best_per_dataset_csv = (
            Path(output_best_per_dataset_csv) if output_best_per_dataset_csv is not None else None
        )

        self.window_sizes = self._as_int_tuple(
            DEFAULT_WINDOW_SIZES if window_sizes is None else window_sizes,
            name="window_sizes",
        )
        self.thresholds = self._as_float_tuple(
            DEFAULT_THRESHOLDS if thresholds is None else thresholds,
            name="thresholds",
        )
        self.min_accepted_values = self._as_int_tuple(
            (DEFAULT_MIN_ACCEPTED_VALUES if min_accepted_values is None else min_accepted_values),
            name="min_accepted_values",
        )
        self.N_distributions = self._as_string_tuple(
            DEFAULT_N_DISTRIBUTIONS if N_distributions is None else N_distributions,
            name="N_distributions",
        )
        self.realign_after_averaging_values = self._as_bool_tuple(
            (
                DEFAULT_REALIGN_AFTER_AVERAGING_VALUES
                if realign_after_averaging_values is None
                else realign_after_averaging_values
            ),
            name="realign_after_averaging_values",
        )
        self.search_window_factors = self._as_float_tuple(
            (DEFAULT_SEARCH_WINDOW_FACTORS if search_window_factors is None else search_window_factors),
            name="search_window_factors",
        )
        self.interpolate_volume_gaps_values = self._as_bool_tuple(
            (
                DEFAULT_INTERPOLATE_VOLUME_GAPS_VALUES
                if interpolate_volume_gaps_values is None
                else interpolate_volume_gaps_values
            ),
            name="interpolate_volume_gaps_values",
        )
        self.apply_epoch_alpha_scaling_values = self._as_bool_tuple(
            (
                DEFAULT_APPLY_EPOCH_ALPHA_SCALING_VALUES
                if apply_epoch_alpha_scaling_values is None
                else apply_epoch_alpha_scaling_values
            ),
            name="apply_epoch_alpha_scaling_values",
        )

        self.scoring = scoring
        self.greater_is_better = bool(greater_is_better)
        self.track_estimated_noise = track_estimated_noise
        self.continue_on_error = bool(continue_on_error)
        self.show_progress = bool(show_progress)
        self._metric_processor_specs = tuple(metric_processors) if metric_processors is not None else None
        self.last_result: CorrectionGridSearchResult | None = None

        super().__init__()
        self._validate_search_space()

    def validate(self, context: ProcessingContext) -> None:
        """Validate the input context and configured Flex grid."""
        super().validate(context)
        if self.upsample_factor < 1:
            raise ProcessorValidationError(f"upsample_factor must be >= 1, got {self.upsample_factor}")
        self._validate_search_space()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Run the grid on one context and attach its summary to metadata."""
        dataset_id = self._infer_dataset_id(context)
        result = self.run_search(context, dataset_id=dataset_id)

        metadata = context.metadata.copy()
        metadata.custom["grid_search"] = result.to_metadata()
        return context.with_metadata(
            metadata,
            copy_estimated_noise=False,
        )

    @property
    def n_combinations(self) -> int:
        """Return the number of valid Flex configurations."""
        return len(self.iter_parameter_grid())

    def iter_parameter_grid(self) -> list[dict[str, Any]]:
        """Return every valid Flex parameter combination.

        Combinations with ``min_accepted > window_size`` are excluded because
        Flex rejects them during validation.
        """
        grid = {
            "window_size": list(self.window_sizes),
            "threshold": list(self.thresholds),
            "min_accepted": list(self.min_accepted_values),
            "N_distribution": list(self.N_distributions),
            "realign_after_averaging": list(self.realign_after_averaging_values),
            "search_window_factor": list(self.search_window_factors),
            "interpolate_volume_gaps": list(self.interpolate_volume_gaps_values),
            "apply_epoch_alpha_scaling": list(self.apply_epoch_alpha_scaling_values),
        }

        return [
            dict(params) for params in ParameterGrid(grid) if int(params["min_accepted"]) <= int(params["window_size"])
        ]

    def build_pipeline(self, params: dict[str, Any]) -> Pipeline:
        """Build the correction-and-metrics pipeline for one parameter set."""
        correction = self._build_flex(params)
        processors: list[Processor] = [
            TriggerDetector(regex=self.trigger_regex),
            UpSample(factor=self.upsample_factor),
            correction,
            DownSample(factor=self.upsample_factor),
        ]
        processors.extend(self._make_metric_processors())
        return Pipeline(processors, name="Flex Grid Search")

    def run_search(
        self,
        context: ProcessingContext,
        *,
        dataset_id: str | None = None,
    ) -> CorrectionGridSearchResult:
        """Evaluate every Flex configuration on one dataset."""
        resolved_dataset_id = dataset_id or self._infer_dataset_id(context)
        rows = self._run_dataset(
            context=context,
            dataset_id=str(resolved_dataset_id),
        )
        return self._finalize(pd.DataFrame(rows))

    def run_many(
        self,
        datasets: Mapping[str, ProcessingContext],
    ) -> CorrectionGridSearchResult:
        """Evaluate the same Flex grid across multiple datasets.

        Parameters
        ----------
        datasets : Mapping[str, ProcessingContext]
            Mapping from a stable dataset identifier to its input context.
        """
        if not datasets:
            raise ValueError("datasets must contain at least one dataset")

        rows: list[dict[str, Any]] = []
        for dataset_number, (dataset_id, context) in enumerate(
            datasets.items(),
            start=1,
        ):
            logger.info(
                "Starting dataset {}/{}: {}",
                dataset_number,
                len(datasets),
                dataset_id,
            )
            rows.extend(
                self._run_dataset(
                    context=context,
                    dataset_id=str(dataset_id),
                )
            )

        return self._finalize(pd.DataFrame(rows))

    def _run_dataset(
        self,
        *,
        context: ProcessingContext,
        dataset_id: str,
    ) -> list[dict[str, Any]]:
        """Run all Flex configurations on one dataset."""
        rows: list[dict[str, Any]] = []
        parameter_grid = self.iter_parameter_grid()

        logger.info(
            "Starting Flex grid search for dataset '{}' with {} combination(s)",
            dataset_id,
            len(parameter_grid),
        )

        for index, params in enumerate(parameter_grid, start=1):
            logger.info(
                "Dataset '{}', combination {}/{}: {}",
                dataset_id,
                index,
                len(parameter_grid),
                params,
            )

            row = self._run_one_combination(
                context=context,
                dataset_id=dataset_id,
                params=params,
            )
            rows.append(row)

            if not row["success"] and not self.continue_on_error:
                break

        return rows

    def _run_one_combination(
        self,
        *,
        context: ProcessingContext,
        dataset_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one Flex configuration and return one result-table row."""
        run_context = self._fresh_context(context)
        pipeline = self.build_pipeline(params)
        result = pipeline.run(
            initial_context=run_context,
            show_progress=self.show_progress,
        )

        metrics: dict[str, float] = {}
        if result.success and result.context is not None:
            metrics = self._flatten_metrics(result.context.metadata.custom.get("metrics", {}))

        score = self.score_metrics(metrics) if metrics else np.nan
        row = {
            "dataset_id": dataset_id,
            "configuration_id": self._configuration_id(params),
            **params,
            "success": bool(result.success),
            "execution_time": float(result.execution_time),
            "score": score,
            "error": "" if result.success else str(result.error),
        }
        row.update(metrics)
        result.release_raw()
        return row

    def _build_flex(self, params: dict[str, Any]) -> Flex:
        """Instantiate Flex for one grid-search combination."""
        return Flex(
            window_size=int(params["window_size"]),
            threshold=float(params["threshold"]),
            min_accepted=int(params["min_accepted"]),
            N_distribution=str(params["N_distribution"]),
            realign_after_averaging=bool(params["realign_after_averaging"]),
            search_window_factor=float(params["search_window_factor"]),
            interpolate_volume_gaps=bool(params["interpolate_volume_gaps"]),
            apply_epoch_alpha_scaling=bool(params["apply_epoch_alpha_scaling"]),
            track_estimated_noise=self.track_estimated_noise,
        )

    def score_metrics(self, metrics: dict[str, float]) -> float:
        """Convert flattened metrics into one model-selection score."""
        if callable(self.scoring):
            return float(self.scoring(metrics))

        if self.scoring == "auto":
            return self._auto_score(metrics)

        metric_name = str(self.scoring)
        sign = 1.0
        if metric_name.startswith("-"):
            sign = -1.0
            metric_name = metric_name[1:]

        return sign * float(metrics.get(metric_name, np.nan))

    def _finalize(self, results: pd.DataFrame) -> CorrectionGridSearchResult:
        """Create summary tables, identify the best configuration, and export."""
        aggregate = self._aggregate_results(results)
        parameter_effects = self._calculate_parameter_effects(results)
        best_per_dataset = self._best_per_dataset(results)

        best_index: int | None = None
        best_score: float | None = None
        best_params: dict[str, Any] | None = None
        best_metrics: dict[str, float] = {}

        ranked = self._rank_aggregate_results(aggregate)
        if not ranked.empty:
            best_row = ranked.iloc[0]
            best_index = int(best_row.name)
            best_score = float(best_row["mean_score"])
            best_params = {
                parameter: self._python_scalar(best_row[parameter])
                for parameter in FLEX_PARAMETER_NAMES
                if parameter in best_row and not pd.isna(best_row[parameter])
            }

            configuration_id = str(best_row["configuration_id"])
            matching = results[results["configuration_id"] == configuration_id]
            for metric_name in self._metric_columns(results):
                numeric = pd.to_numeric(
                    matching[metric_name],
                    errors="coerce",
                )
                if numeric.notna().any():
                    best_metrics[metric_name] = float(numeric.mean())

            logger.info(
                "Best cross-dataset Flex parameters: {} (mean score={})",
                best_params,
                best_score,
            )
        else:
            logger.warning("Grid search completed without a finite best score")

        csv_path = self._write_csv(results, self.output_csv)
        aggregate_csv_path = self._write_csv(
            ranked,
            self.output_aggregate_csv,
        )
        parameter_effects_csv_path = self._write_csv(
            parameter_effects,
            self.output_parameter_effects_csv,
        )
        best_per_dataset_csv_path = self._write_csv(
            best_per_dataset,
            self.output_best_per_dataset_csv,
        )

        grid_result = CorrectionGridSearchResult(
            results=results,
            aggregate_results=ranked,
            parameter_effects=parameter_effects,
            best_per_dataset=best_per_dataset,
            best_params=best_params,
            best_metrics=best_metrics,
            best_score=best_score,
            best_index=best_index,
            csv_path=csv_path,
            aggregate_csv_path=aggregate_csv_path,
            parameter_effects_csv_path=parameter_effects_csv_path,
            best_per_dataset_csv_path=best_per_dataset_csv_path,
        )
        self.last_result = grid_result
        return grid_result

    def _aggregate_results(self, results: pd.DataFrame) -> pd.DataFrame:
        """Summarize each Flex configuration across datasets."""
        if results.empty:
            return pd.DataFrame()

        grouped = results.groupby("configuration_id", dropna=False)
        aggregate = grouped.agg(
            n_datasets=("dataset_id", "nunique"),
            n_runs=("dataset_id", "size"),
            successful_runs=("success", "sum"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            score_std=("score", "std"),
            minimum_score=("score", "min"),
            maximum_score=("score", "max"),
            mean_execution_time=("execution_time", "mean"),
        ).reset_index()
        aggregate["success_rate"] = aggregate["successful_runs"] / aggregate["n_runs"]

        parameters = results.groupby("configuration_id", dropna=False)[list(FLEX_PARAMETER_NAMES)].first().reset_index()
        return parameters.merge(
            aggregate,
            on="configuration_id",
            how="left",
        )

    def _rank_aggregate_results(
        self,
        aggregate: pd.DataFrame,
    ) -> pd.DataFrame:
        """Rank configurations by score, reliability, and variability."""
        if aggregate.empty:
            return aggregate

        finite = aggregate[pd.to_numeric(aggregate["mean_score"], errors="coerce").notna()].copy()
        if finite.empty:
            return finite

        return finite.sort_values(
            ["mean_score", "success_rate", "score_std"],
            ascending=[not self.greater_is_better, False, True],
            na_position="last",
        ).reset_index(drop=True)

    def _calculate_parameter_effects(
        self,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Summarize the marginal score distribution for each parameter value.

        This table is descriptive. It does not remove interactions between
        parameters, so the full and aggregate result tables remain necessary.
        """
        if results.empty:
            return pd.DataFrame()

        numeric_scores = pd.to_numeric(results["score"], errors="coerce")
        successful = results[results["success"] & numeric_scores.notna()].copy()
        if successful.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for parameter in FLEX_PARAMETER_NAMES:
            for value, subset in successful.groupby(parameter, dropna=True):
                rows.append(
                    {
                        "parameter": parameter,
                        "value": self._python_scalar(value),
                        "n_runs": int(len(subset)),
                        "n_datasets": int(subset["dataset_id"].nunique()),
                        "mean_score": float(subset["score"].mean()),
                        "median_score": float(subset["score"].median()),
                        "score_std": float(subset["score"].std()),
                        "minimum_score": float(subset["score"].min()),
                        "maximum_score": float(subset["score"].max()),
                    }
                )

        effects = pd.DataFrame(rows)
        if effects.empty:
            return effects

        return effects.sort_values(
            ["parameter", "mean_score"],
            ascending=[True, not self.greater_is_better],
        ).reset_index(drop=True)

    def _best_per_dataset(self, results: pd.DataFrame) -> pd.DataFrame:
        """Return the best successful Flex configuration for each dataset."""
        if results.empty:
            return pd.DataFrame()

        numeric_scores = pd.to_numeric(results["score"], errors="coerce")
        successful = results[results["success"] & numeric_scores.notna()].copy()
        if successful.empty:
            return pd.DataFrame()

        ordered = successful.sort_values(
            ["dataset_id", "score"],
            ascending=[True, not self.greater_is_better],
        )
        return ordered.groupby("dataset_id", as_index=False).first()

    def _make_metric_processors(self) -> list[Processor]:
        """Create fresh metric processor instances for one grid row."""
        if self._metric_processor_specs is None:
            return [factory() for factory in DEFAULT_METRIC_FACTORIES]

        processors: list[Processor] = []
        for spec in self._metric_processor_specs:
            if isinstance(spec, Processor):
                processors.append(deepcopy(spec))
                continue

            processor = spec()
            if not isinstance(processor, Processor):
                raise ProcessorValidationError(
                    f"metric processor factory returned {type(processor)}, expected Processor"
                )
            processors.append(processor)

        return processors

    @staticmethod
    def _fresh_context(context: ProcessingContext) -> ProcessingContext:
        """Isolate one run so configurations cannot mutate each other."""
        metadata = context.metadata.copy()
        metadata.custom.pop("metrics", None)
        metadata.custom.pop("grid_search", None)
        metadata.custom.pop("artifact_template_matrices", None)

        raw = context.get_raw().copy()
        raw_original = context.get_raw_original()

        return ProcessingContext(
            raw=raw,
            # Metrics only read the reference recording. Sharing it avoids a
            # second full signal copy for every parameter combination while
            # the independent working ``raw`` still isolates correction runs.
            raw_original=raw_original if raw_original is not None else context.get_raw(),
            metadata=metadata,
        )

    def _auto_score(self, metrics: dict[str, float]) -> float:
        """Compute the existing FACETpy composite model-selection score."""
        rewards: list[float] = []
        for key, weight in (
            ("snr", 1.0),
            ("legacy_snr", 0.5),
            ("rms_ratio", 1.0),
        ):
            value = metrics.get(key)
            if self._is_finite_number(value):
                rewards.append(weight * np.log1p(max(float(value), 0.0)))

        targets: list[float] = []
        predictions: list[float] = []
        for key in ("rms_residual", "median_artifact_ratio"):
            value = metrics.get(key)
            if self._is_finite_number(value):
                targets.append(1.0)
                predictions.append(float(value))

        if not rewards and not predictions:
            return float("nan")

        score = float(np.sum(rewards))
        if predictions:
            score -= float(mean_absolute_error(targets, predictions))

        fft_allen_values = [
            value for key, value in metrics.items() if key.startswith("fft_allen_") and self._is_finite_number(value)
        ]
        if fft_allen_values:
            score -= 0.01 * float(np.mean(np.log1p(np.maximum(fft_allen_values, 0.0))))

        return score

    def _validate_search_space(self) -> None:
        """Validate all configured Flex parameter values."""
        if not isinstance(self.track_estimated_noise, bool):
            raise ProcessorValidationError("track_estimated_noise must be a boolean")

        if any(value < 1 for value in self.window_sizes):
            raise ProcessorValidationError("window_sizes must all be >= 1")

        if any(value <= 0 or value > 1 for value in self.thresholds):
            raise ProcessorValidationError("thresholds must all be in (0, 1]")

        if any(value < 1 for value in self.min_accepted_values):
            raise ProcessorValidationError("min_accepted_values must all be >= 1")

        if any(value not in {"equal", "normal"} for value in self.N_distributions):
            raise ProcessorValidationError("N_distributions must contain only 'equal' or 'normal'")

        if any(value <= 0 for value in self.search_window_factors):
            raise ProcessorValidationError("search_window_factors must all be positive")

        if not self.iter_parameter_grid():
            raise ProcessorValidationError(
                "The configured values produce no valid combinations. "
                "At least one min_accepted value must be <= one window_size."
            )

    @staticmethod
    def _configuration_id(params: Mapping[str, Any]) -> str:
        """Return a deterministic identifier for one parameter combination."""
        serializable = {key: CorrectionGridSearch._python_scalar(value) for key, value in sorted(params.items())}
        return json.dumps(
            serializable,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _metric_columns(results: pd.DataFrame) -> list[str]:
        """Return result columns that contain evaluation metrics."""
        excluded = {
            "dataset_id",
            "configuration_id",
            *FLEX_PARAMETER_NAMES,
            "success",
            "execution_time",
            "score",
            "error",
        }
        return [column for column in results.columns if column not in excluded]

    @staticmethod
    def _write_csv(
        frame: pd.DataFrame,
        path: Path | None,
    ) -> Path | None:
        """Write a result table when an output path is configured."""
        if path is None:
            return None

        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        logger.info("Saved grid-search output to {}", path)
        return path

    @staticmethod
    def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        """Flatten nested scalar metrics and summarize numeric arrays."""
        flat: dict[str, float] = {}

        def walk(prefix: str, value: Any) -> None:
            if CorrectionGridSearch._is_finite_number(value):
                flat[prefix] = float(value)
                return

            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    walk(f"{prefix}_{sub_key}", sub_value)
                return

            if isinstance(value, (list, tuple, np.ndarray)):
                array = np.asarray(value, dtype=float)
                if array.size and np.all(np.isfinite(array)):
                    flat[f"{prefix}_mean"] = float(np.mean(array))
                    flat[f"{prefix}_std"] = float(np.std(array))

        for key, value in metrics.items():
            walk(key, value)

        return flat

    @staticmethod
    def _infer_dataset_id(context: ProcessingContext) -> str:
        """Infer a stable label from context metadata when available."""
        custom = context.metadata.custom
        for key in (
            "dataset_id",
            "subject",
            "input_path",
            "source_path",
        ):
            value = custom.get(key)
            if value:
                return str(value)
        return "dataset"

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        """Return whether a value is a finite, non-boolean scalar number."""
        return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and np.isfinite(value)

    @staticmethod
    def _python_scalar(value: Any) -> Any:
        """Convert NumPy scalar objects into JSON-compatible Python values."""
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _as_float_tuple(
        values: Sequence[float],
        *,
        name: str,
    ) -> tuple[float, ...]:
        result = tuple(float(value) for value in values)
        if not result:
            raise ValueError(f"{name} must not be empty")
        return result

    @staticmethod
    def _as_int_tuple(
        values: Sequence[int],
        *,
        name: str,
    ) -> tuple[int, ...]:
        result = tuple(int(value) for value in values)
        if not result:
            raise ValueError(f"{name} must not be empty")
        return result

    @staticmethod
    def _as_string_tuple(
        values: Sequence[str],
        *,
        name: str,
    ) -> tuple[str, ...]:
        result = tuple(str(value).strip().lower() for value in values)
        if not result:
            raise ValueError(f"{name} must not be empty")
        return result

    @staticmethod
    def _as_bool_tuple(
        values: Sequence[bool],
        *,
        name: str,
    ) -> tuple[bool, ...]:
        result: list[bool] = []
        for value in values:
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must contain only booleans, got {value!r}")
            result.append(bool(value))

        if not result:
            raise ValueError(f"{name} must not be empty")
        return tuple(result)

    def _get_parameters(self) -> dict[str, Any]:
        """Expose serializable configuration for processing history."""
        if self._metric_processor_specs is None:
            metric_names = [factory().__class__.__name__ for factory in DEFAULT_METRIC_FACTORIES]
        else:
            metric_names = [
                (spec.__class__.__name__ if isinstance(spec, Processor) else getattr(spec, "__name__", repr(spec)))
                for spec in self._metric_processor_specs
            ]

        return {
            "trigger_regex": self.trigger_regex,
            "upsample_factor": self.upsample_factor,
            "output_csv": (str(self.output_csv) if self.output_csv is not None else None),
            "output_aggregate_csv": (str(self.output_aggregate_csv) if self.output_aggregate_csv is not None else None),
            "output_parameter_effects_csv": (
                str(self.output_parameter_effects_csv) if self.output_parameter_effects_csv is not None else None
            ),
            "output_best_per_dataset_csv": (
                str(self.output_best_per_dataset_csv) if self.output_best_per_dataset_csv is not None else None
            ),
            "window_sizes": list(self.window_sizes),
            "thresholds": list(self.thresholds),
            "min_accepted_values": list(self.min_accepted_values),
            "N_distributions": list(self.N_distributions),
            "realign_after_averaging_values": list(self.realign_after_averaging_values),
            "search_window_factors": list(self.search_window_factors),
            "interpolate_volume_gaps_values": list(self.interpolate_volume_gaps_values),
            "apply_epoch_alpha_scaling_values": list(self.apply_epoch_alpha_scaling_values),
            "metric_processors": metric_names,
            "scoring": getattr(self.scoring, "__name__", self.scoring),
            "greater_is_better": self.greater_is_better,
            "track_estimated_noise": self.track_estimated_noise,
            "continue_on_error": self.continue_on_error,
            "show_progress": self.show_progress,
        }
