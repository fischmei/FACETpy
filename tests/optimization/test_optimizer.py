"""Numerical, scientific, persistence and fold isolation checks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import mne
import numpy as np
import pytest

from facet import Flex, Loader, TriggerDetector, UpSample
from facet.correction.flex.decisions import AveragingMatrixBuilder, MatrixDecisions, MatrixMetadata
from facet.optimization import search
from facet.optimization.cache import EvaluationCache, file_identity
from facet.optimization.metrics import original_spectrum, reference_metrics, spectral_metrics
from facet.optimization.prepared import (
    CachedBuilder,
    chunk_intervals,
    evaluate_prepared,
    prepare_dataset,
    subset_indices,
    validate_reference,
    worker_limit,
)


@pytest.fixture
def settings():
    return search.RunSettings(
        trigger_regex="^R128$",
        upsample_factor=2,
        chunk_padding_seconds=1,
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
        chunk_seconds=6,
        screening_channels=1,
        screening_sections=3,
    )


@pytest.fixture
def recording(tmp_path):
    sfreq = 200
    times = np.arange(4000) / sfreq
    rng = np.random.default_rng(42)
    physiological = (
        np.sin(2 * np.pi * 6.7 * times)
        + np.sin(2 * np.pi * 10.7 * times)
        + 0.2 * np.sin(2 * np.pi * 17.3 * times)
        + rng.normal(0, 0.2, len(times))
    ) * 1e-6
    data = np.vstack([physiological + 20e-6 * np.sin(2 * np.pi * 20 * times)] * 3)
    raw = mne.io.RawArray(data, mne.create_info(["Cz", "Pz", "Fz"], sfreq, "eeg"), verbose=False)
    triggers = np.arange(100, 3901, 100)
    raw.set_annotations(mne.Annotations(triggers / sfreq, np.zeros(len(triggers)), ["R128"] * len(triggers)))
    path = tmp_path / "sample_raw.fif"
    raw.save(path, overwrite=True, fmt="double", verbose=False)
    return path, raw, triggers


@pytest.mark.parametrize("seed", range(8))
def test_sparse_dense_all_conditional_anchors(seed, settings, tmp_path):
    optuna = pytest.importorskip("optuna")
    caps = search.SearchCapabilities(False, False, False, False)
    parameters = search.anchor_parameter_sets(caps) + [
        (str(i), p) for i, p in enumerate(search.balanced_seed_parameter_sets(caps, count=16))
    ]
    _, values = parameters[seed * 2]
    recipe = search.suggest_matrix_decisions(optuna.trial.FixedTrial(values), caps, settings)
    epochs = np.random.default_rng(seed).normal(size=(24, 64))
    builder = AveragingMatrixBuilder(recipe)
    metadata = MatrixMetadata(trigger_times=np.arange(24, dtype=float))
    dense = builder.build(epochs, metadata)
    sparse = builder.build_sparse(epochs, metadata)
    np.testing.assert_allclose(sparse.toarray(), dense, atol=1e-15)
    np.testing.assert_allclose(sparse @ epochs, dense @ epochs, atol=1e-14)
    cache = EvaluationCache(tmp_path / "cache.sqlite3", signature="test")
    cached_builder = CachedBuilder(recipe, cache, "epochs")
    np.testing.assert_allclose(cached_builder.build_sparse(epochs, metadata).toarray(), dense, atol=1e-15)
    cache.close()


def test_sparse_correction_preserves_alignment_scaling_and_gaps(recording, settings, tmp_path, monkeypatch):
    path, raw, triggers = recording
    context = TriggerDetector("^R128$").execute(Loader(str(path)).execute(None))
    # Exercise real interpolation gaps rather than only the no-gap branch.
    context.metadata.artifact_length = 80
    context.metadata.volume_gaps = True
    up = UpSample(2).execute(context)
    recipe = MatrixDecisions()
    flex = Flex(
        matrix_decisions=recipe,
        realign_after_averaging=True,
        interpolate_volume_gaps=True,
        apply_epoch_alpha_scaling=True,
        track_estimated_noise=False,
    )
    dense_result = flex.execute(up)
    from facet.correction.flex.decisions import AveragingMatrixBuilder

    monkeypatch.setattr(
        flex,
        "_calc_averaging_matrix",
        lambda epochs, **kwargs: AveragingMatrixBuilder(recipe).build_sparse(epochs, MatrixMetadata()),
    )
    # Metrics path invokes the common lifecycle directly, avoiding dense-only
    # report serialization and len(csr) compatibility helpers.
    epochs = np.load(
        prepare_dataset(
            search.DatasetSpec("sample", path, triggers, None),
            replace(settings, chunk_seconds=30, chunk_padding_seconds=2),
            tmp_path / "prepared",
        )
        / "0/0_epochs.npy",
        mmap_mode="r",
    )
    assert epochs.dtype == np.float64
    matrices, templates = {}, []
    for ch in range(3):
        from facet.helpers.utils import split_vector

        data = up.get_raw()._data[ch]
        epochs = split_vector(data - np.mean(data), up.get_triggers(), up.get_artifact_length())
        matrices[ch] = AveragingMatrixBuilder(recipe).build_sparse(epochs)
        templates.append(matrices[ch] @ epochs)
    copied = up.get_raw().copy()
    aligned, _ = flex._subtract_artifact_templates(
        raw=copied,
        averaging_matrices=matrices,
        artifacts_per_channel=templates,
        triggers=up.get_triggers(),
        artifact_offset=0,
        artifact_length=up.get_artifact_length(),
        sfreq=up.get_sfreq(),
        upsampling_factor=2,
    )
    np.testing.assert_array_equal(aligned, dense_result.get_triggers())
    np.testing.assert_allclose(copied._data, dense_result.get_raw()._data, atol=1e-17)


def test_preparation_is_reused_and_metrics_only(recording, settings, tmp_path, monkeypatch):
    path, raw, triggers = recording
    dataset = search.DatasetSpec("sample", path, triggers, None, input_identity=file_identity(path))
    directory = prepare_dataset(dataset, settings, tmp_path / "prepared")
    manifest = json.loads((directory / "prepared.json").read_text())
    assert len(manifest["chunks"]) >= 3
    assert isinstance(np.load(directory / "0/0_signal.npy", mmap_mode="r"), np.memmap)
    monkeypatch.setattr(Loader, "execute", lambda *args: pytest.fail("Prepared data reloaded"))
    assert prepare_dataset(dataset, settings, tmp_path / "prepared") == directory
    cache = EvaluationCache(tmp_path / "cache.sqlite3", signature="test")
    result = evaluate_prepared(MatrixDecisions(), dataset, settings, directory, cache, "screening")
    assert result["n_channels"] == 1
    assert result["n_chunks"] == 3
    assert result["success"]
    first_misses = cache.unit_misses
    evaluate_prepared(MatrixDecisions(), dataset, settings, directory, cache, "screening")
    assert cache.unit_misses == first_misses
    assert cache.unit_hits == 3
    expanded = evaluate_prepared(MatrixDecisions(), dataset, settings, directory, cache, "full")
    assert expanded["n_channels"] == 3
    assert cache.unit_hits >= 6
    assert not list((tmp_path / "prepared").rglob("*.edf"))
    cache.close()


def test_transactional_cache_resume_invalidation_and_fidelity(tmp_path):
    path = tmp_path / "cache.sqlite3"
    cache = EvaluationCache(path, signature="first")
    row = {"configuration_id": "a", "dataset_id": "d", "success": True, "value": 1}
    cache.add(row)
    cache.add({**row, "value": 99})
    assert cache.get("a", "d")["value"] == 1
    assert cache.get("a", "d", "screening") is None
    cache.close()
    cache = EvaluationCache(path, signature="first")
    assert cache.get("a", "d")["success"]
    cache.close()
    cache = EvaluationCache(path, signature="new_algorithm")
    assert cache.get("a", "d") is None
    cache.close()


def test_content_hash_invalidates_same_size_edits(tmp_path):
    path = tmp_path / "data.edf"
    path.write_bytes(b"1234")
    identity = file_identity(path)
    path.write_bytes(b"4321")
    assert file_identity(path) != identity


def test_mask_fixed_and_clean_coverage_guard(recording, settings):
    _, raw, triggers = recording
    original = raw._data[0]
    frequencies, psd, mask = original_spectrum(original, 200, triggers, settings)
    saved_mask = mask.copy()
    identity = spectral_metrics(original, 200, frequencies, psd, mask)
    zeroed = spectral_metrics(np.zeros_like(original), 200, frequencies, psd, mask)
    np.testing.assert_array_equal(mask, saved_mask)
    assert identity["nonpeak_eeg_log_deviation"] == pytest.approx(0)
    assert zeroed["nonpeak_eeg_log_deviation"] > 100
    contaminated = spectral_metrics(original, 200, frequencies, psd, np.ones_like(mask))
    assert not contaminated["clean_band_coverage_sufficient"]
    assert np.isnan(contaminated["nonpeak_eeg_log_deviation"])


@pytest.mark.parametrize("sfreq", [200, 5000])
def test_slow_volume_triggers_leave_clean_bands(sfreq, settings):
    # Regression: 2 s volume triggers previously masked every bin at 5 kHz.
    times = np.arange(64 * sfreq) / sfreq
    original = (
        sum(
            amplitude * np.sin(2 * np.pi * frequency * times)
            for frequency, amplitude in [(6.5, 2), (10.5, 2), (20, 20), (6.75, 1), (10.75, 1), (17.25, 1)]
        )
        * 1e-6
    )
    triggers = np.arange(0, len(times), 2 * sfreq)
    frequencies, psd, mask = original_spectrum(original, sfreq, triggers, settings)
    assert frequencies[1] - frequencies[0] <= 0.0625
    # Evidence-backed low-frequency harmonics and high-frequency peaks remain
    # excluded, while absent harmonics and adjacent physiology remain usable.
    for frequency in [6.5, 10.5, 20]:
        assert mask[np.argmin(abs(frequencies - frequency))]
    for frequency in [6, 10.75]:
        assert not mask[np.argmin(abs(frequencies - frequency))]
    metrics = spectral_metrics(original, sfreq, frequencies, psd, mask)
    assert metrics["clean_band_coverage_sufficient"]
    assert metrics["nonpeak_eeg_log_deviation"] == pytest.approx(0)
    for band in ["theta", "alpha", "nonpeak_beta"]:
        assert metrics[f"{band}_preservation"] == pytest.approx(1)


def test_bad_clean_coverage_stops_before_trials(recording, tmp_path, monkeypatch):
    from facet.optimization import prepared

    path, _, triggers = recording
    dataset = search.DatasetSpec("sample", path, triggers, None)
    monkeypatch.setattr(search, "preflight_datasets", lambda *args, **kwargs: {"sample": dataset})
    monkeypatch.setattr(search, "find_input_datasets", lambda *args, **kwargs: [path] * 3)
    monkeypatch.setattr(search, "unique_dataset_ids", lambda paths: {"sample": path})
    real_spectrum = original_spectrum

    def contaminated_spectrum(*args):
        frequencies, psd, mask = real_spectrum(*args)
        return frequencies, psd, np.ones_like(mask)

    monkeypatch.setattr(prepared, "original_spectrum", contaminated_spectrum)
    monkeypatch.setattr(search, "run_outer_fold", lambda **kwargs: pytest.fail("Trials started before validation"))
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="Clean-band coverage preflight failed") as error:
        search.main([str(tmp_path), str(output), "--trigger-regex", "^R128$", "--upsample-factor", "2"])
    message = str(error.value)
    for detail in ["sample", "chunk 0", "Cz", "theta: 0.0%", "alpha: 0.0%", "nonpeak_beta: 0.0%", "resolution="]:
        assert detail in message
    assert not list(output.rglob("prepared.json"))
    assert not list(output.rglob("*_signal.npy"))
    assert not list(output.rglob("screening_study.sqlite3"))


def test_reference_validation_and_missing_modes(recording):
    _, raw, _ = recording
    with pytest.raises(ValueError, match="timestamps"):
        validate_reference(raw, raw.copy())
    validate_reference(raw, raw.copy(), assume_aligned=True)
    with pytest.raises(ValueError, match="sampling"):
        validate_reference(raw, raw.copy().resample(100), assume_aligned=True)
    metrics = reference_metrics(raw._data[0], raw._data[0])
    assert metrics["reference_rms_residual"] == 0
    assert metrics["reference_normalized_mse"] == 0
    args = search.build_argument_parser().parse_args(["input", "output", "--evaluation-mode", "combined"])
    with pytest.raises(ValueError, match="requires --reference-directory"):
        search._validate_arguments(args)


def test_bounded_resources_and_reproducible_sections(recording, settings):
    _, raw, triggers = recording
    assert worker_limit(8, 1024, 400) == 2
    with pytest.raises(MemoryError):
        worker_limit(2, 256, 400)
    assert subset_indices(10, 3) == [0, 4, 9]
    intervals = chunk_intervals(triggers, raw.n_times, 200, settings)
    cores = [(chunk[2], chunk[3]) for chunk in intervals]
    assert all(a[1] == b[0] for a, b in zip(cores, cores[1:], strict=False))
    assert cores[0][0] == triggers[0]
    assert cores[-1][1] == raw.n_times


def test_recording_aggregation_does_not_treat_chunks_as_subjects():
    rows = [
        {"success": True, "scanner_peak_residual": x, "nonpeak_eeg_log_deviation": 1, "n_chunks": chunks}
        for x, chunks in [(1, 100), (3, 1), (5, 1)]
    ]
    aggregated = search.aggregate_dataset_rows(rows)
    assert aggregated["scanner_peak_residual"] == 3
    assert aggregated["n_datasets"] == 3


@pytest.mark.parametrize("mode", ["non-reference", "reference", "combined"])
def test_fold_isolation_and_resume(settings, tmp_path, monkeypatch, mode):
    pytest.importorskip("optuna")
    settings = replace(settings, evaluation_mode=mode)
    columns = {
        "non-reference": ("scanner_peak_residual", "nonpeak_eeg_log_deviation"),
        "reference": ("reference_normalized_mse", "reference_spectral_log_deviation"),
        "combined": ("scanner_peak_residual", "reference_normalized_mse"),
    }
    monkeypatch.setattr(search, "OBJECTIVE_COLUMNS", columns[mode])
    datasets = []
    for name in ["train_a", "train_b", "held_out"]:
        path = tmp_path / f"{name}.edf"
        path.write_bytes(b"fixture")
        datasets.append(search.DatasetSpec(name, path, np.arange(20), None))
    calls = []

    def evaluate(decisions, items, settings, cache, fidelity="full"):
        calls.append((fidelity, [item.dataset_id for item in items]))
        return [
            {
                "success": True,
                "scanner_peak_residual": 0.2,
                "nonpeak_eeg_log_deviation": 0.01,
                "reference_normalized_mse": 0.3,
                "reference_spectral_log_deviation": 0.2,
                "theta_preservation": 1,
                "alpha_preservation": 1,
                "nonpeak_beta_preservation": 1,
                "configuration_id": search.configuration_id(decisions),
                "recipe_json": json.dumps(decisions.to_dict()),
                "dataset_id": item.dataset_id,
            }
            for item in items
        ]

    monkeypatch.setattr(search, "evaluate_many", evaluate)
    heldout_calls = []

    def heldout(decisions, dataset, settings, cache):
        heldout_calls.append(dataset.dataset_id)
        return evaluate(decisions, [dataset], settings, cache)[0]

    monkeypatch.setattr(search, "evaluate_configuration", heldout)
    cache = EvaluationCache(tmp_path / "cache.sqlite3", signature="test")
    kwargs = dict(
        held_out=datasets[-1],
        training=datasets[:2],
        fold_directory=tmp_path / "fold",
        settings=settings,
        cache=cache,
        trials=2,
        promotion_count=3,
        screening_fraction=0.5,
        seed=42,
        timeout_seconds=None,
        rebuild_study=False,
    )
    search.run_outer_fold(**kwargs)
    assert heldout_calls == ["held_out"]
    assert all("held_out" not in ids for _, ids in calls[:-1])
    import optuna

    study = optuna.load_study(
        study_name="matrix_screening_held_out", storage=f"sqlite:///{tmp_path / 'fold/screening_study.sqlite3'}"
    )
    trial_count = len(study.trials)
    # Simulate a killed in-flight trial before the enlarged total budget.
    stale = study.ask()
    search.run_outer_fold(**{**kwargs, "trials": 3})
    fresh = optuna.load_study(
        study_name="matrix_screening_held_out", storage=f"sqlite:///{tmp_path / 'fold/screening_study.sqlite3'}"
    )
    assert fresh.trials[stale.number].state == optuna.trial.TrialState.FAIL
    assert sum(item.state == optuna.trial.TrialState.COMPLETE for item in fresh.trials) == 3
    assert len(fresh.trials) > trial_count
    selected = json.loads((tmp_path / "fold/selected_recipe.json").read_text())
    assert selected["training_datasets"] == ["train_a", "train_b"]
    cache.close()


def test_folder_combined_with_workers_final_model_and_resume(recording, tmp_path):
    import os
    import shutil
    import subprocess
    import sys

    path, raw, triggers = recording
    # Final-output comparison needs ECG and distinct EEG spatial signals;
    # identical channels become zero after average reference.
    raw = raw.copy()
    times = np.arange(raw.n_times) / raw.info["sfreq"]
    for channel in range(3):
        raw._data[channel] += 0.5e-6 * np.sin(2 * np.pi * (7.1 + channel * 0.4) * times)
    ecg = (
        sum(
            (1 - 2 * ((times - beat) / 0.025) ** 2) * np.exp(-(((times - beat) / 0.025) ** 2))
            for beat in np.arange(0.8, times[-1], 0.85)
        )
        * 1e-3
    )
    raw.add_channels([mne.io.RawArray(ecg[None], mne.create_info(["ECG"], 200, "ecg"), verbose=False)])
    raw.save(path, overwrite=True, fmt="double", verbose=False)
    inputs, references, output = (tmp_path / name for name in ("input", "references", "output"))
    inputs.mkdir()
    references.mkdir()
    for index in range(3):
        shutil.copyfile(path, inputs / f"sub{index}_raw.fif")
        reference = raw.copy()
        reference._data -= 20e-6 * np.sin(2 * np.pi * 20 * np.arange(raw.n_times) / 200)
        reference.save(references / f"sub{index}_raw.fif", fmt="double", verbose=False)
    script = Path(__file__).parents[2] / "examples/run_matrix_optimization.py"
    command = [
        sys.executable,
        str(script),
        str(inputs),
        str(output),
        "--reference-directory",
        str(references),
        "--assume-reference-aligned",
        "--evaluation-mode",
        "combined",
        "--trials",
        "1",
        "--promotion-count",
        "3",
        "--upsample-factor",
        "2",
        "--chunk-seconds",
        "6",
        "--chunk-min-triggers",
        "5",
        "--chunk-padding-seconds",
        "1",
        "--workers",
        "2",
        "--bootstrap-samples",
        "100",
        "--preservation-ratio-bound",
        "2",
        "--beta-preservation-maximum",
        "3",
    ]
    environment = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")}
    subprocess.run(command, env=environment, check=True, capture_output=True, timeout=90)
    recipe = json.loads((output / "recommended_recipe.json").read_text())
    assert len(recipe["training_datasets"]) == 3
    assert recipe["selection_pool"] == "feasible_pareto"
    assert (output / "combined/pareto.csv").exists()
    assert (output / "reference/pareto.csv").exists()
    for mode in ("non-reference", "reference", "combined"):
        assert (output / mode / "recommended_recipe.json").exists()
        assert (output / mode / "final_training_pareto.png").exists()
    assert (output / "combined/per_trial_results.csv").exists()
    assert recipe["fixed_preprocessing"]["parameters"]["pca_components"] == 0.95
    prepared = {path: path.stat().st_mtime_ns for path in output.rglob("*_signal.npy")}
    cache = EvaluationCache(
        output / "evaluation_cache.sqlite3",
        signature=json.loads((output / "run_manifest.json").read_text())["evaluation_signature"],
    )
    count = len(cache.rows)
    cache.close()
    subprocess.run(
        [*command, "--resume", "--workers", "1"], env=environment, check=True, capture_output=True, timeout=90
    )
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in prepared.items())
    assert json.loads((output / "recommended_recipe.json").read_text()) == recipe
    cache = EvaluationCache(
        output / "evaluation_cache.sqlite3",
        signature=json.loads((output / "run_manifest.json").read_text())["evaluation_signature"],
    )
    assert len(cache.rows) == count
    cache.close()


def test_motion_sparse_cache_matches_dense(settings, tmp_path):
    import optuna

    from facet import MotionEpochMetadata

    caps = search.SearchCapabilities(True, True, True, True)
    parameters = search.anchor_parameter_sets(caps)
    epochs = np.random.default_rng(10).normal(size=(30, 64))
    motion = MotionEpochMetadata(
        parameters=np.random.default_rng(1).normal(size=(30, 6)),
        stable=np.ones(30, dtype=bool),
        segment_ids=np.zeros(30, dtype=int),
    )
    metadata = MatrixMetadata(slices_per_volume=3, trigger_times=np.arange(30, dtype=float), motion=motion)
    cache = EvaluationCache(tmp_path / "cache.sqlite3", signature="motion")
    for _name, params in parameters:
        recipe = search.suggest_matrix_decisions(optuna.trial.FixedTrial(params), caps, settings)
        expected = AveragingMatrixBuilder(recipe).build(epochs, metadata)
        actual = CachedBuilder(recipe, cache, "dataset/chunk/channel").build_sparse(epochs, metadata)
        np.testing.assert_allclose(actual.toarray(), expected, atol=1e-15)
    cache.close()


def test_missing_recording_reference_is_explicit(recording, tmp_path):
    import shutil

    path, _, _ = recording
    inputs = tmp_path / "originals"
    references = inputs / "references"
    references.mkdir(parents=True)
    for index in range(3):
        shutil.copyfile(path, inputs / f"sub{index}_raw.fif")
    # A nested cleaned file must never be treated as a development recording.
    shutil.copyfile(path, references / "sub0_raw.fif")
    with pytest.raises(ValueError, match="Missing separately cleaned reference for sub1_raw"):
        search.main(
            [
                str(inputs),
                str(tmp_path / "results"),
                "--recursive",
                "--reference-directory",
                str(references),
                "--evaluation-mode",
                "combined",
            ]
        )
