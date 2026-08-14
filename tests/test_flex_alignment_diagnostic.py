"""Tests for the standalone Flex alignment diagnostic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIRECTORY = Path(__file__).parents[1] / "examples"
for module_name in ("run_matrix_optimization", "diagnose_flex_alignment"):
    module_path = EXAMPLES_DIRECTORY / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

diagnostic = sys.modules["diagnose_flex_alignment"]


def test_interval_masks_separate_template_acquisition_and_reference():
    masks = diagnostic.interval_masks(
        n_samples=100,
        sfreq=10.0,
        triggers=np.array([20, 40]),
        artifact_length=10,
        artifact_offset_seconds=0.0,
        reference_buffer_seconds=0.1,
    )

    assert np.flatnonzero(masks["template"]).tolist() == list(range(20, 30)) + list(range(40, 50))
    assert np.flatnonzero(masks["acquisition"])[[0, -1]].tolist() == [15, 54]
    assert masks["reference"][:19].all()
    assert not masks["reference"][19:51].any()
    assert masks["reference"][51:].all()


def test_channel_diagnostics_detect_exact_subtraction_and_cancellation():
    original = np.linspace(-1.0, 1.0, 20)
    template_mask = np.zeros(20, dtype=bool)
    template_mask[5:15] = True
    estimate = np.zeros(20)
    estimate[template_mask] = original[template_mask]
    corrected = original - estimate
    masks = {
        "template": template_mask,
        "acquisition": np.ones(20, dtype=bool),
        "reference": ~template_mask,
    }

    result = diagnostic.channel_diagnostics(original, corrected, estimate, masks)

    assert result["subtraction_identity_relative_error"] == pytest.approx(0.0)
    assert result["estimate_to_original_template_rms"] == pytest.approx(1.0)
    assert result["corrected_to_original_template_rms"] == pytest.approx(0.0)


def test_spectral_diagnostics_report_band_preservation_ratios():
    sfreq = 200.0
    times = np.arange(2000) / sfreq
    original = np.sin(2.0 * np.pi * 10.0 * times)
    corrected = 0.5 * original
    estimate = original - corrected
    masks = {
        "template": np.ones(len(times), dtype=bool),
        "acquisition": np.arange(len(times)) < 1000,
        "reference": np.arange(len(times)) >= 1000,
    }

    rows, spectra = diagnostic.spectral_diagnostics(
        original,
        original,
        corrected,
        estimate,
        masks,
        sfreq,
    )
    alpha = next(row for row in rows if row["band"] == "alpha")

    assert alpha["corrected_to_reference_power"] == pytest.approx(0.25, rel=0.02)
    assert alpha["roundtrip_to_original_power"] == pytest.approx(1.0)
    assert "frequencies" in spectra


def test_matrix_self_weight_audit_supports_dense_payload():
    class Metadata:
        custom = {
            "artifact_template_matrices": [
                {
                    "channels": [
                        {
                            "channel_name": "Cz",
                            "averaging_matrix_A": {
                                "storage": "dense",
                                "shape": [2, 2],
                                "matrix": [[0.25, 0.75], [1.0, 0.0]],
                            },
                        }
                    ]
                }
            ]
        }

    class Context:
        metadata = Metadata()

    row = diagnostic.matrix_self_weight_diagnostics(Context())[0]
    assert row["self_weight_nonzero_fraction"] == pytest.approx(0.5)
    assert row["self_weight_mean"] == pytest.approx(0.125)
    assert not row["audit_truncated"]
