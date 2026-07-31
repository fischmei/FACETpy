"""Tests for Pareto grid-search reporting and visualization."""

from __future__ import annotations

import errno

import pandas as pd
import pytest
from examples import run_kfold
from examples.run_kfold import summarize_results

from facet.correction.grid_search import CorrectionGridSearch as ScoredCorrectionGridSearch
from facet.correction.grid_search_pareto import (
    FLEX_PARAMETER_NAMES,
    CorrectionGridSearch,
)

pytestmark = pytest.mark.unit


def _search(tmp_path) -> CorrectionGridSearch:
    """Return a small search with every report directed to a temporary path."""
    return CorrectionGridSearch(
        output_csv=tmp_path / "results.csv",
        output_aggregate_csv=tmp_path / "aggregate.csv",
        output_parameter_effects_csv=tmp_path / "effects.csv",
        output_pareto_csv=tmp_path / "pareto.csv",
        output_combination_grid_csv=tmp_path / "combinations.csv",
        output_pareto_2d=tmp_path / "pareto_2d.png",
        output_pareto_3d=tmp_path / "pareto_3d.png",
        output_pareto_matrix=tmp_path / "pareto_matrix.png",
        output_combination_heatmap=tmp_path / "combination_heatmap.png",
        output_parameter_effects_plot=tmp_path / "effects.png",
        window_sizes=[10, 20],
        thresholds=[0.90, 0.95],
        min_accepted_values=[5],
        N_distributions=["equal"],
        realign_after_averaging_values=[True],
        search_window_factors=[3.0],
        interpolate_volume_gaps_values=[True],
        apply_epoch_alpha_scaling_values=[True],
    )


def _synthetic_results(search: CorrectionGridSearch) -> pd.DataFrame:
    """Build deterministic metrics for each configured combination."""
    rows = []
    metrics = [
        (10.0, 1.40, 1.20, True),
        (12.0, 1.20, 1.10, True),
        (11.0, 1.00, 0.90, True),
        # This incomplete configuration would otherwise look unrealistically
        # attractive and must not enter the comparable Pareto set.
        (100.0, 0.10, 0.10, False),
    ]
    for params, (snr, residual, fft, success) in zip(
        search.iter_parameter_grid(),
        metrics,
        strict=True,
    ):
        row = {
            "dataset_id": "dataset_a",
            "configuration_id": search._configuration_id(params),
            **params,
            "success": success,
            "execution_time": 1.0,
            "error": "" if success else "synthetic failure",
        }
        if success:
            row.update(
                {
                    "snr": snr,
                    "rms_residual": residual,
                    "fft_niazy_alpha": fft,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def test_combination_grid_numbers_every_valid_configuration(tmp_path):
    """The upfront grid should be complete, numbered, and reproducible."""
    search = _search(tmp_path)

    frame = search.combination_grid_frame()
    path = search.write_combination_grid()

    assert frame["combination_number"].tolist() == [1, 2, 3, 4]
    assert list(frame.columns) == [
        "combination_number",
        "configuration_id",
        *FLEX_PARAMETER_NAMES,
    ]
    assert path == tmp_path / "combinations.csv"
    pd.testing.assert_frame_equal(pd.read_csv(path), frame, check_dtype=False)


@pytest.mark.parametrize(
    "search_class",
    [CorrectionGridSearch, ScoredCorrectionGridSearch],
)
def test_grid_runs_share_read_only_reference_raw(search_class, sample_context):
    """Each run should copy its working signal but not its reference signal."""
    source_raw = sample_context.get_raw()
    reference_raw = sample_context.get_raw_original()
    source_value = reference_raw._data[0, 0]

    fresh = search_class._fresh_context(sample_context)

    assert fresh.get_raw() is not source_raw
    assert fresh.get_raw_original() is reference_raw
    fresh.get_raw()._data[0, 0] += 1.0
    assert reference_raw._data[0, 0] == source_value


def test_finalize_writes_clear_pareto_and_combination_reports(tmp_path):
    """Finalization should identify one compromise and create every plot."""
    search = _search(tmp_path)

    result = search._finalize(_synthetic_results(search))

    assert len(result.pareto_results) == 4
    assert result.pareto_front["success_rate"].eq(1.0).all()
    assert result.pareto_results["is_selected_compromise"].sum() == 1
    assert not result.pareto_results.loc[
        result.pareto_results["success_rate"] < 1.0,
        "pareto_eligible",
    ].any()

    expected_paths = [
        result.csv_path,
        result.aggregate_csv_path,
        result.parameter_effects_csv_path,
        result.pareto_csv_path,
        result.combination_grid_csv_path,
        result.pareto_2d_path,
        result.pareto_3d_path,
        result.pareto_matrix_path,
        result.combination_heatmap_path,
        result.parameter_effects_plot_path,
    ]
    assert all(path is not None and path.exists() for path in expected_paths)

    metadata = result.to_metadata()
    assert metadata["combination_grid_csv_path"].endswith("combinations.csv")
    assert metadata["pareto_matrix_path"].endswith("pareto_matrix.png")
    assert metadata["combination_heatmap_path"].endswith("combination_heatmap.png")


def test_kfold_summary_adds_stability_metric_and_runtime_plots(tmp_path):
    """Held-out summaries should include complementary diagnostic figures."""
    parameters = _search(tmp_path / "search").iter_parameter_grid()[:2]
    selected_rows = [
        {
            "fold": fold,
            "test_dataset_id": f"dataset_{fold}",
            "n_training_datasets": 1,
            **params,
        }
        for fold, params in enumerate(parameters, start=1)
    ]
    test_rows = [
        {
            "fold": fold,
            "test_dataset_id": f"dataset_{fold}",
            "dataset_id": f"dataset_{fold}",
            "configuration_id": f"configuration_{fold}",
            **params,
            "success": True,
            "execution_time": 10.0 + fold,
            "n_chunks": 2,
            "successful_chunks": 2,
            "error": "",
            "snr": 5.0 + fold,
            "rms_residual": 1.5 - (0.1 * fold),
        }
        for fold, params in enumerate(parameters, start=1)
    ]

    summarize_results(tmp_path, test_rows, selected_rows)

    expected = [
        "all_held_out_test_metrics.csv",
        "selected_parameters_by_fold.csv",
        "parameter_stability.csv",
        "held_out_metric_summary.csv",
        "parameter_stability.png",
        "held_out_metrics_heatmap.png",
        "held_out_runtime.png",
    ]
    assert all((tmp_path / name).exists() for name in expected)


def test_parallel_dataset_evaluation_keeps_future_dicts_as_rows(monkeypatch, tmp_path):
    """Parallel configuration results must be appended, not extended as keys."""
    search = _search(tmp_path)

    class ImmediateFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class ImmediateExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def submit(self, function, *args, **kwargs):
            return ImmediateFuture(function(*args, **kwargs))

    def fake_evaluate(search, input_path, dataset_id, params, **kwargs):
        return {
            "dataset_id": dataset_id,
            "configuration_id": search._configuration_id(params),
            **params,
            "success": True,
        }

    monkeypatch.setattr(run_kfold, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(run_kfold, "as_completed", lambda futures: reversed(list(futures)))
    monkeypatch.setattr(run_kfold, "evaluate_configuration_chunked", fake_evaluate)

    rows = run_kfold.evaluate_dataset_chunked(
        search,
        tmp_path / "input.edf",
        "dataset-a",
        chunk_padding_seconds=10.0,
        chunk_min_triggers=31,
        chunk_gap_seconds=None,
        workers=2,
    )

    expected_ids = [search._configuration_id(params) for params in search.iter_parameter_grid()]
    assert all(isinstance(row, dict) for row in rows)
    assert [row["configuration_id"] for row in rows] == expected_ids


def test_kfold_retries_transient_remote_io(monkeypatch, tmp_path):
    """Errno 121 from a network mount should be retried with backoff."""
    calls = 0
    delays = []

    def flaky_operation():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError(getattr(errno, "EREMOTEIO", 121), "Remote I/O error")
        return "available"

    monkeypatch.setattr(run_kfold.time, "sleep", delays.append)

    result = run_kfold._retry_remote_io(
        flaky_operation,
        path=tmp_path / "network-cache.csv",
        description="Testing network cache",
    )

    assert result == "available"
    assert calls == 3
    assert delays == [1.0, 2.0]


def test_dataset_evaluation_resumes_and_checkpoints_each_missing_row(
    monkeypatch,
    tmp_path,
):
    """A restart should keep finished combinations and persist every new one."""
    search = _search(tmp_path)
    grid = search.iter_parameter_grid()
    first_params = grid[0]
    existing = [
        {
            "dataset_id": "dataset-a",
            "configuration_id": search._configuration_id(first_params),
            **first_params,
            "success": True,
        }
    ]
    evaluated_ids = []
    checkpoint_sizes = []

    def fake_evaluate(search, input_path, dataset_id, params, **kwargs):
        configuration_id = search._configuration_id(params)
        evaluated_ids.append(configuration_id)
        return {
            "dataset_id": dataset_id,
            "configuration_id": configuration_id,
            **params,
            "success": True,
        }

    monkeypatch.setattr(run_kfold, "evaluate_configuration_chunked", fake_evaluate)

    rows = run_kfold.evaluate_dataset_chunked(
        search,
        tmp_path / "input.edf",
        "dataset-a",
        chunk_padding_seconds=10.0,
        chunk_min_triggers=31,
        chunk_gap_seconds=None,
        workers=1,
        existing_rows=existing,
        checkpoint=lambda current: checkpoint_sizes.append(len(current)),
    )

    expected_ids = [search._configuration_id(params) for params in grid]
    assert evaluated_ids == expected_ids[1:]
    assert [row["configuration_id"] for row in rows] == expected_ids
    assert checkpoint_sizes == [2, 3, 4]


def test_partial_dataset_checkpoint_is_loaded_for_resume(tmp_path):
    """A valid partial CSV is data, not a falsely complete dataset cache."""
    search = _search(tmp_path)
    grid = search.iter_parameter_grid()
    rows = [
        {
            "dataset_id": "dataset-a",
            "configuration_id": search._configuration_id(params),
            **params,
            "success": True,
        }
        for params in grid[:2]
    ]
    path = tmp_path / "dataset-a.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    loaded = run_kfold._load_dataset_checkpoint(
        path,
        dataset_id="dataset-a",
        expected_configuration_ids={
            search._configuration_id(params) for params in grid
        },
    )

    assert len(loaded) == 2
    assert [row["configuration_id"] for row in loaded] == [
        row["configuration_id"] for row in rows
    ]
