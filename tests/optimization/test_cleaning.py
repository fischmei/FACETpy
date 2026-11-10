"""Fixed processor composition and provenance-based reference comparisons."""

import json
from dataclasses import FrozenInstanceError, replace

import mne
import numpy as np
import pytest
from scipy.io import savemat

from facet import AASCorrection, PCACorrection, QRSTriggerDetector, TriggerDetector
from facet.core import ProcessingContext
from facet.correction.flex.decisions import DirectionalQuota, MatrixDecisions
from facet.optimization import prepared, search
from facet.optimization.cache import EvaluationCache, file_identity
from facet.optimization.cleaning import DEFAULT_CLEANING, average_reference, clean_after_flex
from facet.optimization.metrics import objectives, reference_metrics
from facet.optimization.reference import alignment_path, compare_chunk, reference_plan


@pytest.fixture
def pair(tmp_path):
    rate = 200
    times = np.arange(5000) / rate
    rng = np.random.default_rng(2026)
    eeg = (
        np.array(
            [
                sum(np.sin(2 * np.pi * frequency * times + channel) for frequency in (6.7, 10.7, 17.3))
                + rng.normal(0, 0.2, len(times))
                for channel in range(3)
            ]
        )
        * 1e-6
    )
    eeg += 20e-6 * np.sin(2 * np.pi * 20 * times)
    ecg = (
        sum(
            (1 - 2 * ((times - beat) / 0.025) ** 2) * np.exp(-(((times - beat) / 0.025) ** 2))
            for beat in np.arange(0.8, times[-1], 0.85)
        )
        * 1e-3
    )
    raw = mne.io.RawArray(
        np.vstack([eeg, ecg]), mne.create_info(["Cz", "Pz", "Fz", "ECG"], rate, ["eeg"] * 3 + ["ecg"]), verbose=False
    )
    triggers = np.arange(100, 4801, 100)
    raw.set_annotations(mne.Annotations(triggers / rate, np.zeros(len(triggers)), ["R128"] * len(triggers)))
    source = tmp_path / "sub-01_task-checker_eeg.fif"
    reference_path = tmp_path / "sub-01_task-checker_preprocess_eeg.fif"
    raw.save(source, fmt="double", verbose=False)
    reference = raw.copy().pick(["Cz", "Pz", "Fz"])
    reference._data -= 20e-6 * np.sin(2 * np.pi * 20 * times)
    reference.save(reference_path, fmt="double", verbose=False)
    dataset = search.DatasetSpec(
        "sub-01",
        source,
        triggers,
        None,
        reference_path=reference_path,
        input_identity=file_identity(source),
        reference_identity=file_identity(reference_path),
    )
    settings = search.RunSettings(
        trigger_regex="^R128$",
        upsample_factor=1,
        chunk_padding_seconds=2,
        chunk_min_triggers=5,
        chunk_gap_seconds=None,
        preservation_ratio_bound=2,
        beta_preservation_minimum=0.2,
        beta_preservation_maximum=3,
        scanner_peak_minimum_hz=13,
        scanner_peak_maximum_hz=80,
        scanner_peak_prominence_db=6,
        scanner_peak_half_width_hz=0.25,
        max_motion_distance_low=0.05,
        max_motion_distance_high=10,
        chunk_seconds=30,
        evaluation_mode="combined",
        assume_reference_aligned=True,
    )
    return raw, reference, dataset, settings


def test_fixed_pipeline_order_and_immutability(pair, monkeypatch):
    raw, _, _, _ = pair
    context = TriggerDetector("^R128$").execute(ProcessingContext(raw))
    before = raw.get_data()
    calls = []
    for cls in (PCACorrection, QRSTriggerDetector, AASCorrection):
        execute = cls.execute

        def record(self, context, execute=execute):
            calls.append((self.name, dict(self._parameters)))
            return execute(self, context)

        monkeypatch.setattr(cls, "execute", record)
    result = clean_after_flex(context)
    assert [name for name, _ in calls] == ["pca_correction", "qrs_trigger_detector", "aas_correction"]
    assert result.get_raw().ch_names == ["Cz", "Pz", "Fz"]
    np.testing.assert_allclose(result.get_raw().get_data().mean(axis=0), 0, atol=1e-18)
    np.testing.assert_array_equal(raw.get_data(), before)
    assert result.metadata.custom["fixed_cleaning"] == DEFAULT_CLEANING.manifest()
    with pytest.raises(FrozenInstanceError):
        DEFAULT_CLEANING.sfreq = 250


def test_candidate_specific_cleaning_cache_and_modes(pair, tmp_path, monkeypatch):
    _, _, dataset, settings = pair
    directory = prepared.prepare_dataset(dataset, settings, tmp_path / "prepared")
    manifest = json.loads((directory / "prepared.json").read_text())
    assert manifest["channels"][-1] == "ECG"
    assert "ECG" not in manifest["eeg_channels"]
    cache = EvaluationCache(tmp_path / "cache.sqlite3", signature="test")
    calls = []
    clean = prepared.clean_after_flex

    def record(context):
        calls.append(DEFAULT_CLEANING.manifest())
        return clean(context)

    monkeypatch.setattr(prepared, "clean_after_flex", record)
    build = prepared.CachedBuilder.build_sparse

    def eeg_matrix_only(self, *args, **kwargs):
        assert not self.identity.endswith("/3"), "ECG must reuse scanner selection from EEG"
        return build(self, *args, **kwargs)

    monkeypatch.setattr(prepared.CachedBuilder, "build_sparse", eeg_matrix_only)
    baseline = MatrixDecisions()
    first = prepared.evaluate_prepared(baseline, dataset, settings, directory, cache, "full")
    assert calls and first["reference_metric_valid"]
    count = len(calls)
    assert prepared.evaluate_prepared(baseline, dataset, settings, directory, cache, "screening")["success"]
    assert len(calls) == count  # identical padded inputs/settings reused
    variant = replace(baseline, quota=DirectionalQuota.symmetric(20))
    prepared.evaluate_prepared(variant, dataset, settings, directory, cache, "full")
    assert len(calls) > count and all(value == calls[0] for value in calls)
    assert objectives(first, "combined") == (first["scanner_peak_residual"], first["reference_normalized_mse"])
    assert objectives(first, "reference") == (
        first["reference_normalized_mse"],
        first["reference_spectral_log_deviation"],
    )
    no_reference = replace(settings, evaluation_mode="non-reference")
    directory = prepared.prepare_dataset(dataset, no_reference, tmp_path / "prepared")
    monkeypatch.setattr(prepared, "clean_after_flex", lambda *_: pytest.fail("Non-reference ran downstream cleaning"))
    result = prepared.evaluate_prepared(baseline, dataset, no_reference, directory, cache, "screening")
    assert result["success"] and not result["reference_available"]
    assert "reference_normalized_mse" not in result
    cache.close()


def test_reference_alignment_resampling_missing_channels_and_bad_segments(pair, tmp_path):
    raw, reference, _, _ = pair
    reference = reference.copy().pick(["Pz", "Cz"]).resample(250, verbose=False)
    path = tmp_path / "reference.fif"
    plan = reference_plan(raw, reference, path, assume_aligned=True)
    assert plan["channels"] == ["Cz", "Pz"] and plan["missing_reference_channels"] == ["Fz"]
    candidate = average_reference(raw)
    candidate.set_annotations(mne.Annotations([5], [2], ["BAD_test"]))
    metrics, preview = compare_chunk(candidate, reference, plan, raw_start=0, core_start=0, core_stop=25)
    assert metrics["reference_valid_seconds"] == pytest.approx(23)
    assert preview["sfreq"] == 1000
    assert np.isfinite(metrics["reference_spectral_similarity"])
    # A compacted recording must provide a map; assume-aligned never waives it.
    compacted = reference.copy().crop(2, 20)
    with pytest.raises(ValueError, match="removed samples"):
        reference_plan(raw, compacted, path, assume_aligned=True)
    alignment_path(path).write_text(
        json.dumps(
            {
                "provenance": "Known acquisition interval",
                "intervals": [{"raw_start": 2, "reference_start": 0, "duration": 18}],
            }
        )
    )
    mapped = reference_plan(raw, compacted, path)
    assert mapped["coverage"] == pytest.approx(18 / 25)
    with pytest.raises(ValueError, match="timestamps missing"):
        reference_plan(raw, reference, tmp_path / "unmapped.fif")


def test_reference_matching_and_snr_definition(pair, tmp_path):
    _, reference, dataset, _ = pair
    assert prepared.match_reference(dataset, tmp_path, tmp_path, True) == dataset.path  # exact input name wins
    references = tmp_path / "references"
    references.mkdir()
    target = references / "sub-01_task-checker_preprocess_eeg.fif"
    reference.save(target, verbose=False)
    assert prepared.match_reference(dataset, references, tmp_path, True) == target
    result = reference_metrics(np.array([1.0, 3.0, 2.0]), np.array([2.0, 4.0, 3.0]))
    assert result["reference_snr_db"] == pytest.approx(10 * np.log10((4 + 16 + 9) / 3))


def test_pca_logged_failure_invalidates_trial(pair, monkeypatch):
    raw, _, _, _ = pair
    context = TriggerDetector("^R128$").execute(ProcessingContext(raw))
    monkeypatch.setattr(PCACorrection, "_calc_pca_residuals", lambda *_: (_ for _ in ()).throw(ValueError("test")))
    with pytest.raises(ValueError, match="Fixed cleaning failed"):
        clean_after_flex(context)


def test_missing_ecg_fails_before_cleaning(pair):
    raw, _, _, _ = pair
    with pytest.raises(ValueError, match="requires a good ECG"):
        clean_after_flex(ProcessingContext(raw.copy().drop_channels(["ECG"])))


def test_eeglab_mask_mapping_uses_published_provenance(pair, tmp_path):
    raw, reference, _, _ = pair
    mask = np.ones(raw.n_times, dtype=np.uint8)
    mask[400:800] = 0
    raw_path, ref_path = tmp_path / "raw.set", tmp_path / "clean.set"
    events = [{"bvmknum": i + 1, "type": "R128", "latency": latency} for i, latency in enumerate([1, 1201, 2401, 3601])]
    savemat(raw_path, {"event": events})
    savemat(
        ref_path,
        {
            "etc": {"clean_sample_mask": mask},
            "urevent": events,
            "event": [
                {**event, "urevent": i + 1, "latency": event["latency"] - (400 if event["latency"] > 800 else 0)}
                for i, event in enumerate(events)
            ],
        },
    )
    compacted = mne.io.RawArray(reference.get_data()[:, mask.astype(bool)], reference.info, verbose=False)
    plan = reference_plan(raw, compacted, ref_path, raw_path=raw_path)
    assert plan["intervals"] == [
        {"raw_start": 0, "reference_start": 0, "duration": 2},
        {"raw_start": 4, "reference_start": 2, "duration": 21},
    ]
    assert plan["provenance"]["matched_events"] == 4
    assert plan["coverage"] == pytest.approx(0.92)
