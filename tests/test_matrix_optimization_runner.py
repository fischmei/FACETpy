"""Tests for the standalone composable-matrix optimization runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUNNER_PATH = Path(__file__).parents[1] / "examples" / "run_matrix_optimization.py"
SPEC = importlib.util.spec_from_file_location("run_matrix_optimization", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class FixedTrial:
    """Minimal fixed-parameter trial that does not require optional Optuna."""

    def __init__(self, parameters):
        self.parameters = parameters

    def suggest_categorical(self, name, choices):
        value = self.parameters[name]
        assert value in choices
        return value

    def suggest_int(self, name, low, high, **kwargs):
        del kwargs
        value = self.parameters[name]
        assert low <= value <= high
        return value

    def suggest_float(self, name, low, high, **kwargs):
        del kwargs
        value = self.parameters[name]
        assert low <= value <= high
        return value


def _settings() -> runner.RunSettings:
    return runner.RunSettings(
        trigger_regex=r"^R128$",
        upsample_factor=10,
        chunk_padding_seconds=5.0,
        chunk_min_triggers=20,
        chunk_gap_seconds=None,
        preservation_ratio_bound=1.25,
        beta_preservation_minimum=0.5,
        beta_preservation_maximum=1.5,
        scanner_peak_minimum_hz=13.0,
        scanner_peak_maximum_hz=80.0,
        scanner_peak_prominence_db=6.0,
        scanner_peak_half_width_hz=0.5,
        max_motion_distance_low=0.05,
        max_motion_distance_high=10.0,
    )


@pytest.mark.parametrize(
    ("capabilities", "expected_anchor"),
    [
        (runner.SearchCapabilities(False, False, False, False), "structural_slice"),
        (runner.SearchCapabilities(True, True, True, True), "moosmann_cost"),
    ],
)
def test_anchor_parameters_resolve_through_conditional_space(capabilities, expected_anchor):
    anchors = [
        (
            name,
            runner.suggest_matrix_decisions(FixedTrial(parameters), capabilities, _settings()),
        )
        for name, parameters in runner.anchor_parameter_sets(capabilities)
    ]

    names = {name for name, _ in anchors}
    assert expected_anchor in names
    assert len({runner.configuration_id(recipe) for _, recipe in anchors}) == len(anchors)
    for _, recipe in anchors:
        assert recipe == runner.MatrixDecisions.from_dict(recipe.to_dict())
        if not capabilities.motion_parameters:
            assert not runner.recipe_requires_motion(recipe)


def test_balanced_seed_parameters_only_create_valid_graph_edges():
    capabilities = runner.SearchCapabilities(True, True, True, True)
    seeds = runner.balanced_seed_parameter_sets(capabilities, count=32)

    decisions = [runner.suggest_matrix_decisions(FixedTrial(values), capabilities, _settings()) for values in seeds]

    assert {item.scoring.mode.value for item in decisions} == {
        "signed_pearson",
        "absolute_pearson",
        "temporal_motion_cost",
        "none",
    }
    assert {item.weighting.kernel.value for item in decisions} == {
        "equal",
        "gaussian",
        "laplace",
        "student_t",
    }
    for item in decisions:
        if item.scoring.mode.value == "none":
            assert item.template_size.mode.value == "select_all"
        elif item.scoring.mode.value == "temporal_motion_cost":
            assert item.template_size.mode.value in {"maximum_k", "exactly_k"}


def test_safe_log_deviation_penalizes_reciprocal_ratios_equally():
    assert runner.safe_log_deviation(1.0) == pytest.approx(0.0)
    assert runner.safe_log_deviation(0.8) == pytest.approx(runner.safe_log_deviation(1.25))
    assert np.isnan(runner.safe_log_deviation(0.0))


def test_snr_is_normalized_against_same_dataset_baseline():
    improved = runner.normalize_snr_row(
        {"success": True, "snr": 20.0},
        {"success": True, "snr": 10.0},
    )
    degraded = runner.normalize_snr_row(
        {"success": True, "snr": 5.0},
        {"success": True, "snr": 10.0},
    )

    assert improved["normalized_snr"] == pytest.approx(np.log(2.0))
    assert degraded["normalized_snr"] == pytest.approx(-np.log(2.0))
    assert improved["baseline_snr"] == 10.0


def test_evaluation_cache_migrates_objective_only_signature(tmp_path):
    path = tmp_path / "evaluation_cache.csv"
    pd.DataFrame(
        [
            {
                "evaluation_signature": "raw-objective-signature",
                "configuration_id": "recipe",
                "dataset_id": "dataset",
                "success": True,
                "snr": 10.0,
            }
        ]
    ).to_csv(path, index=False)

    cache = runner.EvaluationCache(
        path,
        signature="normalized-objective-signature",
        rebuild=False,
        compatible_signatures=("raw-objective-signature",),
    )

    assert cache.get("recipe", "dataset")["snr"] == 10.0
    migrated = pd.read_csv(path)
    assert set(migrated["evaluation_signature"]) == {"normalized-objective-signature"}


def test_weighted_median_uses_chunk_trigger_mass():
    assert runner.weighted_median([1.0, 9.0, 20.0], [10.0, 2.0, 1.0]) == 1.0
    assert np.isnan(runner.weighted_median([np.nan], [1.0]))


def test_spectral_objectives_reward_peak_removal_and_preserve_other_bands():
    sfreq = 200.0
    times = np.arange(int(20 * sfreq)) / sfreq
    rng = np.random.default_rng(42)
    physiological = (
        np.sin(2 * np.pi * 6.0 * times)
        + np.sin(2 * np.pi * 10.0 * times)
        + 0.2 * np.sin(2 * np.pi * 17.0 * times)
        + 0.1 * rng.standard_normal(len(times))
    )
    scanner_artifact = 10.0 * np.sin(2 * np.pi * 20.0 * times)
    original = np.vstack([physiological + scanner_artifact] * 2)
    corrected = np.vstack([physiological] * 2)

    metrics = runner.calculate_spectral_optimization_metrics(
        original,
        corrected,
        sfreq=sfreq,
        settings=_settings(),
    )

    assert metrics["scanner_peak_residual"] < 0.01
    assert metrics["theta_preservation"] == pytest.approx(1.0, rel=0.02)
    assert metrics["alpha_preservation"] == pytest.approx(1.0, rel=0.02)
    assert metrics["nonpeak_beta_preservation"] == pytest.approx(1.0, rel=0.1)


def test_preservation_constraints_use_training_quantiles():
    constraints = runner.preservation_constraints(
        {
            "theta_preservation_q25": 0.9,
            "theta_preservation_q75": 1.1,
            "alpha_preservation_q25": 0.7,
            "alpha_preservation_q75": 1.0,
            "nonpeak_beta_preservation_q25": 0.6,
            "nonpeak_beta_preservation_q75": 1.6,
        },
        _settings(),
    )

    assert constraints[0] < 0.0
    assert constraints[1] < 0.0
    assert constraints[2] > 0.0
    assert constraints[4] < 0.0
    assert constraints[5] > 0.0


def test_pareto_mask_handles_mixed_objective_directions():
    frame = pd.DataFrame(
        {
            "scanner_peak_residual": [0.1, 0.2, 0.3, 0.4],
            "nonpeak_eeg_log_deviation": [0.1, 0.2, 0.05, 0.5],
        }
    )

    assert runner.pareto_mask(frame).tolist() == [True, False, True, False]


def test_ideal_selection_prefers_feasible_training_pareto_point():
    frame = pd.DataFrame(
        {
            "configuration_id": ["balanced", "overcorrected"],
            "success": [True, True],
            "scanner_peak_residual": [0.2, 0.01],
            "nonpeak_eeg_log_deviation": [0.05, 2.0],
            "theta_preservation_q25": [0.9, 0.1],
            "theta_preservation_q75": [1.0, 0.2],
            "alpha_preservation_q25": [0.9, 0.1],
            "alpha_preservation_q75": [1.0, 0.2],
            "nonpeak_beta_preservation_q25": [0.8, 0.01],
            "nonpeak_beta_preservation_q75": [1.0, 0.02],
            "mean_execution_time": [2.0, 1.0],
        }
    )

    selected = runner.select_ideal_recipe(frame, _settings())

    assert selected["configuration_id"] == "balanced"


def test_volume_motion_sidecar_is_mapped_to_slice_epochs(tmp_path):
    path = tmp_path / "recording.npz"
    np.savez(
        path,
        parameters=np.zeros((3, 6)),
        segment_ids=np.array([0, 0, 1]),
        stable=np.array([True, True, False]),
    )

    source = runner._load_motion_sidecar(
        path,
        n_epochs=6,
        slices_per_volume=2,
        rotation_scale=50.0,
    )

    assert source.epoch_to_motion_index.tolist() == [0, 0, 1, 1, 2, 2]
    assert source.rotation_scale == 50.0


def test_parameter_stability_flattens_selected_recipe_manifests():
    recipe = runner.MatrixDecisions().to_dict()
    frame = pd.DataFrame({"recipe_json": [json.dumps(recipe), json.dumps(recipe)]})

    stability = runner.parameter_stability(frame)

    target = stability[stability["parameter"] == "target_policy"].iloc[0]
    assert target["value"] == "exclude_target"
    assert target["fraction"] == 1.0
