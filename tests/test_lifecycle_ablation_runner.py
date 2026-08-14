"""Tests for the one-recording Flex lifecycle ablation runner."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# The ablation runner deliberately reuses aggregation from the optimization
# runner. Load both scripts as importable modules in their command-line order.
EXAMPLES_DIRECTORY = Path(__file__).parents[1] / "examples"
for module_name in ("run_matrix_optimization", "run_lifecycle_ablation"):
    module_path = EXAMPLES_DIRECTORY / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

runner = sys.modules["run_lifecycle_ablation"]


def test_quick_and_factorial_variants_cover_expected_lifecycle_settings():
    quick = runner.QUICK_VARIANTS
    factorial = runner.full_factorial_variants()

    assert quick[0].name == "uncorrected"
    assert not quick[0].corrected
    assert len(quick) == 6
    assert len(factorial) == 9
    corrected_settings = {
        (
            item.realign_after_averaging,
            item.apply_epoch_alpha_scaling,
            item.interpolate_volume_gaps,
        )
        for item in factorial
        if item.corrected
    }
    assert len(corrected_settings) == 8


def test_load_matrix_recipe_accepts_optimizer_and_raw_manifests(tmp_path):
    decisions = runner.MatrixDecisions()
    raw_path = tmp_path / "raw.json"
    selected_path = tmp_path / "selected.json"
    raw_path.write_text(json.dumps(decisions.to_dict()), encoding="utf-8")
    selected_path.write_text(json.dumps({"recipe": decisions.to_dict()}), encoding="utf-8")

    assert runner.load_matrix_recipe(raw_path) == decisions
    assert runner.load_matrix_recipe(selected_path) == decisions


def test_control_comparisons_report_relative_changes_and_feasibility():
    frame = pd.DataFrame(
        [
            {
                "name": "uncorrected",
                "success": True,
                "snr": 10.0,
                "rms_residual": 4.0,
                "median_artifact_ratio": 5.0,
            },
            {
                "name": "corrected",
                "success": True,
                "snr": 20.0,
                "rms_residual": 1.1,
                "median_artifact_ratio": 0.9,
            },
        ]
    )

    compared = runner.add_control_comparisons(frame, ratio_bound=1.25)
    corrected = compared.iloc[1]

    assert corrected["snr_log_vs_uncorrected"] == pytest.approx(0.693147)
    assert corrected["rms_fraction_of_uncorrected"] == pytest.approx(0.275)
    assert corrected["median_artifact_fraction_of_uncorrected"] == pytest.approx(0.18)
    assert bool(corrected["both_preservation_feasible"])


def test_main_effects_compare_enabled_and_disabled_factorial_groups():
    frame = pd.DataFrame(
        [
            {
                "corrected": True,
                "success": True,
                "realign_after_averaging": realign,
                "apply_epoch_alpha_scaling": alpha,
                "interpolate_volume_gaps": gaps,
                "snr_log_vs_uncorrected": float(alpha),
                "rms_log_deviation": 2.0 * float(alpha),
                "median_artifact_log_deviation": 3.0 * float(alpha),
            }
            for realign, alpha, gaps in itertools.product((False, True), repeat=3)
        ]
    )

    effects = runner.lifecycle_effects(frame)
    alpha_snr = effects[
        (effects["parameter"] == "apply_epoch_alpha_scaling") & (effects["metric"] == "snr_log_vs_uncorrected")
    ].iloc[0]
    assert alpha_snr["on_minus_off"] == pytest.approx(1.0)
