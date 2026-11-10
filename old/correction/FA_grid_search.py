"""Grid search utilities for AAS and FARM correction parameters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid

from ..core import Pipeline, ProcessingContext, Processor, ProcessorValidationError, register_processor
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
# from .aas import AASCorrection
# from .farm import FARMCorrection

from .flex import Flex

###
#Different Parameter Values
###

#CORRELATION_THRESHOLD_GRID = tuple(round(value / 100.0, 2) for value in range(1, 101))
#CORRELATION_THRESHOLD_GRID = tuple(round(value / 100.0, 2) for value in range(5, 101, 5))
CORRELATION_THRESHOLD_GRID = (0.90, 0.95, 0.99)
DEFAULT_REL_WINDOW_POSITIONS = (-1.0, -0.5, 0.0, 0.5, 1.0)
DEFAULT_SEARCH_WINDOW_FACTORS = (1.0, 2.0, 3.0)
DEFAULT_WINDOW_SIZES = (10, 20, 30)
DEFAULT_MODELS = ("aas", "farm")


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
    """Container mirroring the useful parts of scikit-learn grid-search output."""

    results: pd.DataFrame
    best_params: dict[str, Any] | None
    best_metrics: dict[str, float]
    best_score: float | None
    best_index: int | None
    csv_path: Path | None = None
    score_grid_csv_path: Path | None = None
    diagram_path: Path | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-friendly summary for ``ProcessingContext`` metadata."""
        return {
            "n_results": int(len(self.results)),
            "best_params": deepcopy(self.best_params),
            "best_metrics": deepcopy(self.best_metrics),
            "best_score": self.best_score,
            "best_index": self.best_index,
            "csv_path": str(self.csv_path) if self.csv_path is not None else None,
            "score_grid_csv_path": str(self.score_grid_csv_path) if self.score_grid_csv_path is not None else None,
            "diagram_path": str(self.diagram_path) if self.diagram_path is not None else None,
        }


@register_processor
class CorrectionGridSearch(Processor):
    """Run a scikit-learn-style grid search over AAS/FARM correction settings.

    Each parameter combination is evaluated with a FACETpy pipeline in this
    exact order::

        TriggerDetector -> UpSample -> AASCorrection/FARMCorrection -> DownSample

    Metric processors are appended after downsampling so every combination
    receives the same evaluation pass. Results are exported as a flat CSV, and
    an optional score grid CSV / heatmap can summarize the best score per
    model, threshold, and window size.
    """

    name = "correction_grid_search"
    description = "Grid search AAS/FARM correction parameters"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        trigger_regex: str = r"\b1\b",
        upsample_factor: int = 10,
        output_csv: str | Path | None = "facet_grid_search_results.csv",
        output_score_grid_csv: str | Path | None = None,
        output_diagram: str | Path | None = None,
        correlation_thresholds: Sequence[float] | None = None,
        rel_window_positions: Sequence[float] | None = None,
        search_window_factors: Sequence[float] | None = None,
        window_sizes: Sequence[int] | None = None,
        models: Sequence[str] = DEFAULT_MODELS,
        metric_processors: Sequence[Processor | Callable[[], Processor]] | None = None,
        scoring: str | Callable[[dict[str, float]], float] = "auto",
        greater_is_better: bool = True,
        continue_on_error: bool = True,
        channel_sequential: bool = False,
        show_progress: bool = False,
        realign_after_averaging: bool = True,
    ) -> None:
        self.trigger_regex = trigger_regex
        self.upsample_factor = int(upsample_factor)
        self.output_csv = Path(output_csv) if output_csv is not None else None
        self.output_score_grid_csv = Path(output_score_grid_csv) if output_score_grid_csv is not None else None
        self.output_diagram = Path(output_diagram) if output_diagram is not None else None
        self.correlation_thresholds = self._as_float_tuple(
            CORRELATION_THRESHOLD_GRID if correlation_thresholds is None else correlation_thresholds,
            name="correlation_thresholds",
        )
        self.rel_window_positions = self._as_float_tuple(
            DEFAULT_REL_WINDOW_POSITIONS if rel_window_positions is None else rel_window_positions,
            name="rel_window_positions",
        )
        self.search_window_factors = self._as_float_tuple(
            DEFAULT_SEARCH_WINDOW_FACTORS if search_window_factors is None else search_window_factors,
            name="search_window_factors",
        )
        self.window_sizes = self._as_int_tuple(
            DEFAULT_WINDOW_SIZES if window_sizes is None else window_sizes,
            name="window_sizes",
        )
        self.models = self._normalize_models(models)
        self.scoring = scoring
        self.greater_is_better = bool(greater_is_better)
        self.continue_on_error = bool(continue_on_error)
        self.channel_sequential = bool(channel_sequential)
        self.show_progress = bool(show_progress)
        self.realign_after_averaging = bool(realign_after_averaging)
        self._metric_processor_specs = tuple(metric_processors) if metric_processors is not None else None
        self.last_result: CorrectionGridSearchResult | None = None
        super().__init__()
        self._validate_search_space()

    def validate(self, context: ProcessingContext) -> None:
        """Validate the input context and configured search grid."""
        super().validate(context)
        if self.upsample_factor < 1:
            raise ProcessorValidationError(f"upsample_factor must be >= 1, got {self.upsample_factor}")
        self._validate_search_space()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Run the grid search and attach the summary to context metadata."""
        result = self.run_search(context)

        new_metadata = context.metadata.copy()
        new_metadata.custom["grid_search"] = result.to_metadata()
        return context.with_metadata(new_metadata)

    @property
    def n_combinations(self) -> int:
        """Return the number of correction pipelines that will be evaluated."""
        return len(self.iter_parameter_grid())

    ###
    #Every possible Combination
    ###

    def iter_parameter_grid(self) -> list[dict[str, Any]]:
        """Return the concrete scikit-learn ``ParameterGrid`` combinations."""
        grid = {
            "model": list(self.models),
            "correlation_threshold": list(self.correlation_thresholds),
            "rel_window_position": list(self.rel_window_positions),
            "search_window_factor": list(self.search_window_factors),
            "window_size": list(self.window_sizes),
        }
        return [dict(params) for params in ParameterGrid(grid)]

    def build_pipeline(self, params: dict[str, Any]) -> Pipeline:
        """Build the correction-and-metrics pipeline for one parameter set."""
        model = self._build_model(params)
        processors: list[Processor] = [
            TriggerDetector(regex=self.trigger_regex),
            UpSample(factor=self.upsample_factor),
            model,
            DownSample(factor=self.upsample_factor),
        ]
        processors.extend(self._make_metric_processors())
        return Pipeline(processors, name=f"Grid Search {params['model'].upper()}")

    def run_search(self, context: ProcessingContext) -> CorrectionGridSearchResult:
        """Evaluate all parameter combinations and export the result tables."""
        rows: list[dict[str, Any]] = []
        best_row_index: int | None = None
        best_score: float | None = None

        parameter_grid = self.iter_parameter_grid()
        logger.info("Starting correction grid search with {} combination(s)", len(parameter_grid))

        for index, params in enumerate(parameter_grid):
            logger.info("Grid search combination {}/{}: {}", index + 1, len(parameter_grid), params)
            row = self._run_one_combination(context, params)
            rows.append(row)

            score = row.get("score")
            if self._is_better_score(score, best_score):
                best_score = float(score)
                best_row_index = index

            if not row["success"] and not self.continue_on_error:
                break

        results = pd.DataFrame(rows)
        best_params, best_metrics = self._extract_best(results, best_row_index)
        csv_path = self._export_results(results)
        score_grid_csv_path = self._export_score_grid(results)
        diagram_path = self._export_diagram(results)

        grid_result = CorrectionGridSearchResult(
            results=results,
            best_params=best_params,
            best_metrics=best_metrics,
            best_score=best_score,
            best_index=best_row_index,
            csv_path=csv_path,
            score_grid_csv_path=score_grid_csv_path,
            diagram_path=diagram_path,
        )
        self.last_result = grid_result

        if best_params is not None:
            logger.info("Best grid-search parameters: {} (score={})", best_params, best_score)
        else:
            logger.warning("Grid search completed without a finite best score")

        return grid_result

    def score_metrics(self, metrics: dict[str, float]) -> float:
        """Score flattened metrics for model selection."""
        if callable(self.scoring):
            return float(self.scoring(metrics))

        if self.scoring == "auto":
            return self._auto_score(metrics)

        metric_name = str(self.scoring)
        sign = 1.0
        if metric_name.startswith("-"):
            sign = -1.0
            metric_name = metric_name[1:]

        value = metrics.get(metric_name, np.nan)
        return sign * float(value)

    def _run_one_combination(self, context: ProcessingContext, params: dict[str, Any]) -> dict[str, Any]:
        """Run one correction pipeline and return one CSV row."""
        run_context = self._fresh_context(context)
        pipeline = self.build_pipeline(params)
        result = pipeline.run(
            initial_context=run_context,
            channel_sequential=self.channel_sequential,
            show_progress=self.show_progress,
        )

        metrics = {}
        if result.success and result.context is not None:
            metrics = self._flatten_metrics(result.context.metadata.custom.get("metrics", {}))

        score = self.score_metrics(metrics) if metrics else np.nan
        row = {
            **params,
            "success": bool(result.success),
            "execution_time": float(result.execution_time),
            "score": score,
            "error": "" if result.success else str(result.error),
        }
        row.update(metrics)
        result.release_raw()
        return row

    def _build_model(self, params: dict[str, Any]) -> Processor:
        """Instantiate AAS or FARM for one grid row."""
        shared = {
            "window_size": int(params["window_size"]),
            "rel_window_position": float(params["rel_window_position"]),
            "correlation_threshold": float(params["correlation_threshold"]),
            "search_window_factor": float(params["search_window_factor"]),
            "realign_after_averaging": self.realign_after_averaging,
        }

        if params["model"] == "aas":
            return AASCorrection(**shared)

        if params["model"] == "farm":
            return FARMCorrection(**shared)

        raise ProcessorValidationError(f"Unknown model '{params['model']}'")

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

    def _fresh_context(self, context: ProcessingContext) -> ProcessingContext:
        """Return an isolated context so one grid row cannot affect another."""
        metadata = context.metadata.copy()
        metadata.custom.pop("metrics", None)
        metadata.custom.pop("grid_search", None)

        raw = context.get_raw().copy()
        raw_original = context.get_raw_original()
        raw_original_copy = raw_original.copy() if raw_original is not None else raw.copy()
        return ProcessingContext(raw=raw, raw_original=raw_original_copy, metadata=metadata)

    def _auto_score(self, metrics: dict[str, float]) -> float:
        """Compute a practical default score from FACETpy quality metrics.

        Higher SNR/RMS improvement is rewarded. Metrics that have an ideal
        target of 1.0 are converted to a scikit-learn mean-absolute-error
        penalty.
        """
        rewards = []
        for key, weight in (("snr", 1.0), ("legacy_snr", 0.5), ("rms_ratio", 1.0)):
            value = metrics.get(key)
            if self._is_finite_number(value):
                rewards.append(weight * np.log1p(max(float(value), 0.0)))

        targets = []
        predictions = []
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

    def _is_better_score(self, candidate: Any, best: float | None) -> bool:
        """Return whether *candidate* improves on *best*."""
        if not self._is_finite_number(candidate):
            return False
        if best is None:
            return True
        candidate = float(candidate)
        return candidate > best if self.greater_is_better else candidate < best

    def _extract_best(
        self,
        results: pd.DataFrame,
        best_row_index: int | None,
    ) -> tuple[dict[str, Any] | None, dict[str, float]]:
        """Extract best params and scalar metrics from the result table."""
        if best_row_index is None or best_row_index >= len(results):
            return None, {}

        row = results.iloc[best_row_index].to_dict()
        param_names = {"model", "correlation_threshold", "rel_window_position", "search_window_factor", "window_size"}
        best_params = {key: row[key] for key in param_names if key in row}
        best_metrics = {
            key: float(value)
            for key, value in row.items()
            if key not in param_names
            and key not in {"success", "execution_time", "score", "error"}
            and self._is_finite_number(value)
        }
        return best_params, best_metrics

    def _export_results(self, results: pd.DataFrame) -> Path | None:
        """Write the flat all-combinations CSV."""
        if self.output_csv is None:
            return None

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.output_csv, index=False)
        logger.info("Saved grid-search results to {}", self.output_csv)
        return self.output_csv

    def _export_score_grid(self, results: pd.DataFrame) -> Path | None:
        """Write a compact score grid CSV for quick spreadsheet inspection."""
        output_path = self.output_score_grid_csv
        if output_path is None and self.output_csv is not None:
            output_path = self.output_csv.with_name(f"{self.output_csv.stem}_score_grid.csv")
        if output_path is None or results.empty or "score" not in results:
            return None

        scored = results[results["success"] & np.isfinite(results["score"])]
        if scored.empty:
            return None

        grid = scored.pivot_table(
            index="correlation_threshold",
            columns=["model", "window_size"],
            values="score",
            aggfunc="max",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.to_csv(output_path)
        logger.info("Saved grid-search score grid to {}", output_path)
        return output_path

    def _export_diagram(self, results: pd.DataFrame) -> Path | None:
        """Save an optional score heatmap diagram."""
        if self.output_diagram is None or results.empty or "score" not in results:
            return None

        scored = results[results["success"] & np.isfinite(results["score"])]
        if scored.empty:
            return None

        models = sorted(scored["model"].dropna().unique())
        if not models:
            return None

        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
        for ax, model in zip(axes[0], models, strict=False):
            model_df = scored[scored["model"] == model]
            pivot = model_df.pivot_table(
                index="correlation_threshold",
                columns="window_size",
                values="score",
                aggfunc="max",
            ).sort_index()

            image = ax.imshow(pivot.values, origin="lower", aspect="auto", interpolation="nearest")
            ax.set_title(model.upper())
            ax.set_xlabel("window_size")
            ax.set_ylabel("correlation_threshold")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([str(int(col)) for col in pivot.columns], rotation=45, ha="right")

            y_positions = self._nice_tick_positions(len(pivot.index), max_ticks=12)
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f"{pivot.index[pos]:.2f}" for pos in y_positions])
            fig.colorbar(image, ax=ax, label="score")

        fig.suptitle("FACETpy correction grid-search score")
        fig.tight_layout()
        self.output_diagram.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_diagram, dpi=150)
        plt.close(fig)
        logger.info("Saved grid-search diagram to {}", self.output_diagram)
        return self.output_diagram

    @staticmethod
    def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        """Flatten nested scalar metrics and summarize numeric per-channel arrays."""
        flat: dict[str, float] = {}

        def _walk(prefix: str, value: Any) -> None:
            if CorrectionGridSearch._is_finite_number(value):
                flat[prefix] = float(value)
                return
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    _walk(f"{prefix}_{sub_key}", sub_value)
                return
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value, dtype=float)
                if arr.size and np.all(np.isfinite(arr)):
                    flat[f"{prefix}_mean"] = float(np.mean(arr))
                    flat[f"{prefix}_std"] = float(np.std(arr))

        for key, value in metrics.items():
            _walk(key, value)
        return flat

    @staticmethod
    def _nice_tick_positions(length: int, max_ticks: int) -> np.ndarray:
        """Return readable axis tick positions for dense grids."""
        if length <= max_ticks:
            return np.arange(length)
        return np.unique(np.linspace(0, length - 1, max_ticks, dtype=int))

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        """Return True for finite scalar numeric values."""
        return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and np.isfinite(value)

    @staticmethod
    def _as_float_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
        """Coerce a non-empty numeric sequence to floats."""
        result = tuple(float(value) for value in values)
        if not result:
            raise ValueError(f"{name} must not be empty")
        return result

    @staticmethod
    def _as_int_tuple(values: Sequence[int], *, name: str) -> tuple[int, ...]:
        """Coerce a non-empty numeric sequence to ints."""
        result = tuple(int(value) for value in values)
        if not result:
            raise ValueError(f"{name} must not be empty")
        return result

    @staticmethod
    def _normalize_models(models: Sequence[str]) -> tuple[str, ...]:
        """Normalize and validate model names."""
        normalized = tuple(str(model).lower() for model in models)
        if not normalized:
            raise ValueError("models must not be empty")

        unknown = sorted(set(normalized) - {"aas", "farm"})
        if unknown:
            raise ValueError(f"Unknown correction model(s): {unknown}")
        return normalized

    def _validate_search_space(self) -> None:
        """Validate all grid values early for clear failures."""
        if any(value <= 0 or value > 1 for value in self.correlation_thresholds):
            raise ProcessorValidationError("correlation_thresholds must be in (0, 1]")
        if any(value < -1 or value > 1 for value in self.rel_window_positions):
            raise ProcessorValidationError("rel_window_positions must be in [-1, 1]")
        if any(value <= 0 for value in self.search_window_factors):
            raise ProcessorValidationError("search_window_factors must be positive")
        if any(value < 1 for value in self.window_sizes):
            raise ProcessorValidationError("window_sizes must be >= 1")

    def _get_parameters(self) -> dict[str, Any]:
        """Expose serializable configuration for processing history."""
        if self._metric_processor_specs is None:
            metric_names = [factory().__class__.__name__ for factory in DEFAULT_METRIC_FACTORIES]
        else:
            metric_names = [
                spec.__class__.__name__ if isinstance(spec, Processor) else getattr(spec, "__name__", repr(spec))
                for spec in self._metric_processor_specs
            ]

        return {
            "trigger_regex": self.trigger_regex,
            "upsample_factor": self.upsample_factor,
            "output_csv": str(self.output_csv) if self.output_csv is not None else None,
            "output_score_grid_csv": str(self.output_score_grid_csv)
            if self.output_score_grid_csv is not None
            else None,
            "output_diagram": str(self.output_diagram) if self.output_diagram is not None else None,
            "correlation_thresholds": list(self.correlation_thresholds),
            "rel_window_positions": list(self.rel_window_positions),
            "search_window_factors": list(self.search_window_factors),
            "window_sizes": list(self.window_sizes),
            "models": list(self.models),
            "metric_processors": metric_names,
            "scoring": getattr(self.scoring, "__name__", self.scoring),
            "greater_is_better": self.greater_is_better,
            "continue_on_error": self.continue_on_error,
            "channel_sequential": self.channel_sequential,
            "show_progress": self.show_progress,
            "realign_after_averaging": self.realign_after_averaging,
        }
