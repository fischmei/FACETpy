"""Fixed original eligibility and channel-time preservation denominators."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from tests.optimization import test_optimizer as fixtures

from facet.correction.flex.decisions import MatrixDecisions, TemplateSizePolicy
from facet.optimization import prepared, search
from facet.optimization.cache import EvaluationCache
from facet.optimization.metrics import BANDS, spectral_metrics, validate_clean_band_coverage

recording = fixtures.recording
settings = fixtures.settings


def af7_spectrum():
    frequencies = np.arange(481) * (5 / 24)
    psd = np.full(len(frequencies), 1e-12)
    theta = (frequencies >= 4) & (frequencies < 8)
    mask = theta.copy()
    clean_indices = np.flatnonzero(theta)[:3]
    mask[clean_indices] = False
    psd[clean_indices] = 7.91e-12 / (3 * (5 / 24))
    return frequencies, psd, mask


@pytest.mark.parametrize("chunk", [20, 10])
def test_af7_three_theta_bins_are_unavailable_not_zero(chunk):
    frequencies, psd, mask = af7_spectrum()
    diagnostics = validate_clean_band_coverage(frequencies, psd, mask, context=f"sub-01/ses-01/chunk-{chunk}/AF7")
    assert frequencies[1] - frequencies[0] == pytest.approx(0.2083333333)
    assert diagnostics["theta"]["coverage"] == pytest.approx(3 / 19)
    assert diagnostics["theta"]["power"] == pytest.approx(7.91e-12)
    assert not diagnostics["theta"]["valid"]
    original_mask = mask.copy()
    signal = np.sin(np.arange(960)) * 1e-6
    validity = [diagnostics[band]["valid"] for band in BANDS]
    first = spectral_metrics(signal, 200, frequencies, psd, mask, band_valid=validity)
    second = spectral_metrics(signal * 2, 200, frequencies, psd, mask, band_valid=validity)
    assert np.isnan(first["theta_preservation"])
    assert first["theta_preservation_weight"] == second["theta_preservation_weight"] == 0
    assert first["alpha_preservation_weight"] == second["alpha_preservation_weight"] == 1
    assert second["alpha_preservation"] == pytest.approx(4 * first["alpha_preservation"])
    np.testing.assert_array_equal(mask, original_mask)
    with pytest.raises(ValueError, match="non-finite samples"):
        spectral_metrics(signal * np.nan, 200, frequencies, psd, mask, band_valid=validity)
    with pytest.raises(ValueError, match="validity is inconsistent"):
        spectral_metrics(signal, 200, frequencies, psd, mask, band_valid=[True] * 3)


def test_two_level_aggregation_uses_fixed_channel_time_not_valid_chunk_means():
    def measurement(ratio, valid):
        return {
            "theta_preservation": ratio if valid else np.nan,
            "theta_preservation_weight": float(valid),
            "theta_valid_measurements": int(valid),
            "theta_invalid_measurements": int(not valid),
        }

    first = prepared.aggregate_metrics([measurement(1, True), measurement(0, False)])
    second = prepared.aggregate_metrics([measurement(3, True), measurement(5, True)])
    result = prepared.aggregate_metrics([first, second], [10, 20])
    assert result["theta_preservation"] == pytest.approx((10 + 20 * (3 + 5)) / 50)
    assert result["theta_preservation_weight"] == 50
    assert result["theta_valid_measurements"] == 3
    assert result["theta_invalid_measurements"] == 1
    with pytest.raises(ValueError, match="lost an eligible"):
        prepared.aggregate_metrics([measurement(np.nan, True), measurement(1, True)])


def test_preservation_objective_keeps_equal_band_weights():
    first, second = {}, {}
    for band in BANDS:
        first[f"{band}_log_deviation"] = 3.0
        first[f"{band}_log_deviation_weight"] = 1.0
        second[f"{band}_log_deviation"] = 0.0 if band != "theta" else np.nan
        second[f"{band}_log_deviation_weight"] = float(band != "theta")
    result = prepared.aggregate_metrics([first, second])
    # Theta: 3; alpha and beta: 1.5 each. Bands retain equal influence.
    assert result["nonpeak_eeg_log_deviation"] == 2.0


def test_isolated_failure_is_prepared_reported_and_reused(recording, settings, tmp_path, monkeypatch):
    path, _, triggers = recording
    dataset = search.DatasetSpec("sub-01_ses-01", path, triggers, None)
    real_spectrum = prepared.original_spectrum
    calls = []

    def occasional_bad_theta(*args):
        frequencies, psd, mask = real_spectrum(*args)
        calls.append(1)
        if len(calls) == 1:
            # Only the first EEG channel/chunk loses theta support.
            theta = np.flatnonzero((frequencies >= 4) & (frequencies < 8))
            mask[theta] = True
            mask[theta[:3]] = False
        return frequencies, psd, mask

    monkeypatch.setattr(prepared, "original_spectrum", occasional_bad_theta)
    settings = replace(settings, screening_channels=2, screening_sections=10)
    directory = prepared.prepare_dataset(dataset, settings, tmp_path)
    report = pd.read_csv(directory / "clean_band_coverage.csv")
    invalid = report[~report.valid]
    assert len(invalid) == 1 and invalid.iloc[0].band == "theta"
    assert invalid.iloc[0].coverage < 0.25
    assert "below 25%" in invalid.iloc[0].invalid_reason
    assert prepared.prepare_dataset(dataset, settings, tmp_path) == directory
    calls_before = len(calls)
    cache = EvaluationCache(tmp_path / "metrics.sqlite3", signature="coverage")
    try:
        first = prepared.evaluate_prepared(MatrixDecisions(), dataset, settings, directory, cache, "full")
        second = prepared.evaluate_prepared(
            replace(MatrixDecisions(), template_size=TemplateSizePolicy.minimum(3)),
            dataset,
            settings,
            directory,
            cache,
            "full",
        )
    finally:
        cache.close()
    assert len(calls) == calls_before
    for band in BANDS:
        assert np.isfinite(first[f"{band}_preservation"])
        assert first[f"{band}_preservation_weight"] == second[f"{band}_preservation_weight"]
        assert first[f"{band}_measurement_coverage"] >= 0.5
    assert first["theta_invalid_measurements"] == 1
    assert np.isfinite(first["nonpeak_eeg_log_deviation"])
    assert json.loads((directory / "preservation_coverage.json").read_text())["minimum_channel_time_coverage"] == 0.5


def test_overall_coverage_is_required_in_each_band_and_fidelity(tmp_path, settings):
    manifest = {"channels": ["AF7", "Cz"], "chunks": [{}, {}]}
    rows = [
        dict(
            dataset_id="sub-01",
            chunk=chunk,
            channel_index=channel,
            band=band,
            duration_seconds=10,
            valid=(channel == 1 or band != "theta"),
        )
        for chunk in range(2)
        for channel in range(2)
        for band in BANDS
    ]
    # Full dataset has exactly 50%, but screening picks only unavailable AF7.
    with pytest.raises(ValueError, match="screening/theta: 0.0%"):
        prepared.preservation_coverage_report(rows, manifest, settings, tmp_path)
    settings = replace(settings, screening_channels=2)
    prepared.preservation_coverage_report(rows, manifest, settings, tmp_path)
    rows[3]["valid"] = False  # Reduce theta to 25% of channel-time.
    with pytest.raises(ValueError, match="full/theta: 25.0%"):
        prepared.preservation_coverage_report(rows, manifest, settings, tmp_path)
    assert (tmp_path / "clean_band_coverage.csv").exists()
