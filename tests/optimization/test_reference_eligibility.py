"""Session-level technical exclusion before participant-grouped optimization."""

import json
from dataclasses import fields

import mne
import numpy as np
import pandas as pd
import pytest

from facet.optimization import eligibility, grouped, search


@pytest.fixture
def cohort(tmp_path):
    raw_root, ref_root, output = (tmp_path / name for name in ("raw", "references", "output"))
    for path in (raw_root, ref_root, output):
        path.mkdir()
    times = np.arange(2000) / 200
    data = np.vstack(
        [np.sin(2 * np.pi * 7 * times) * 1e-6, np.sin(2 * np.pi * 11 * times) * 1e-6, np.sin(2 * np.pi * times) * 1e-3]
    )
    original = mne.io.RawArray(data, mne.create_info(["Cz", "Pz", "ECG"], 200, ["eeg", "eeg", "ecg"]), verbose=False)
    triggers = np.arange(100, 1900, 100)
    original.set_annotations(mne.Annotations(triggers / 200, np.zeros(len(triggers)), ["R128"] * len(triggers)))
    paths, invalid = {}, set()
    for subject in range(6):
        for session in range(1, 2 if subject == 5 else 3):
            identifier = f"sub-{subject:02d}_ses-{session:02d}_task-rest_eeg"
            path = raw_root / (identifier + ".fif")
            raw = original.copy()
            reference = original.copy().pick(["Cz", "Pz"]).resample(250, verbose=False)
            if (subject, session) == (0, 2):
                reference.crop(0, 7.996)  # Deleted time without a verifiable map.
                invalid.add(identifier)
            if (subject, session) == (3, 2):
                raw.set_annotations(raw.annotations + mne.Annotations([5], [0], ["boundary"]))
                invalid.add(identifier)
            if subject == 5:
                reference.pick(["Cz"])
                invalid.add(identifier)
            raw.save(path, fmt="double", verbose=False)
            paths[identifier] = path
            if (subject, session) == (2, 1):
                invalid.add(identifier)  # Missing published reference.
            else:
                reference.save(ref_root / path.name, fmt="double", verbose=False)
    command = [
        str(raw_root),
        str(output),
        "--reference-directory",
        str(ref_root),
        "--evaluation-mode",
        "combined",
        "--assume-reference-aligned",
        "--cv-strategy",
        "grouped-kfold",
        "--include-pattern",
        "*task-rest_eeg.fif",
        "--upsample-factor",
        "1",
        "--chunk-seconds",
        "30",
        "--chunk-min-triggers",
        "5",
        "--trials",
        "1",
        "--promotion-count",
        "3",
        "--bootstrap-samples",
        "100",
    ]
    args = search.build_argument_parser().parse_args(command)
    values = {
        field.name: getattr(args, field.name) for field in fields(search.RunSettings) if hasattr(args, field.name)
    }
    return paths, invalid, search.RunSettings(**values), args, command


def test_strict_default_audits_every_session_then_fails(cohort):
    paths, invalid, settings, args, _ = cohort
    assert not args.exclude_invalid_references
    with pytest.raises(ValueError, match="strict mode"):
        eligibility.preflight_reference_sessions(paths, settings, args)
    report = pd.read_csv(args.output_directory / "reference_eligibility.csv")
    assert len(report) == 11
    assert set(report.loc[~report.eligible, "dataset_id"]) == invalid
    assert report.loc[~report.eligible, "exclusion_reason"].notna().all()
    assert not report.automatically_excluded.any()
    summary = json.loads((args.output_directory / "reference_eligibility.json").read_text())["summary"]
    assert summary["n_eligible_sessions"] == 7 and summary["n_eligible_participants"] == 5
    assert summary["n_invalid_sessions"] == 4 and summary["policy"] == "strict"


def test_exclusion_is_per_session_and_does_not_modify_inputs(cohort):
    paths, invalid, settings, args, _ = cohort
    args.exclude_invalid_references = True
    inputs = [*args.input_directory.glob("*.fif"), *args.reference_directory.glob("*.fif")]
    before = {path: path.read_bytes() for path in inputs}
    selected, report = eligibility.preflight_reference_sessions(paths, settings, args)
    assert set(selected) == set(paths) - invalid
    assert report["summary"]["n_excluded_sessions"] == 4
    assert set(report["summary"]["excluded_dataset_ids"]) == invalid
    assert "sub-00_ses-01_task-rest_eeg" in selected
    assert "sub-02_ses-02_task-rest_eeg" in selected
    assert not any(identifier.startswith("sub-05") for identifier in selected)
    assert before == {path: path.read_bytes() for path in inputs}
    assert set(inputs) == set(args.input_directory.glob("*.fif")) | set(args.reference_directory.glob("*.fif"))
    reasons = {row["dataset_id"]: row["exclusion_reason"] for row in report["sessions"]}
    assert "different duration" in reasons["sub-00_ses-02_task-rest_eeg"]
    assert "Missing separately cleaned reference" in reasons["sub-02_ses-01_task-rest_eeg"]
    assert "Internal boundary" in reasons["sub-03_ses-02_task-rest_eeg"]
    assert "two shared" in reasons["sub-05_ses-01_task-rest_eeg"]


def test_main_builds_folds_from_eligible_sessions_without_revalidating(cohort, monkeypatch):
    paths, invalid, _, args, command = cohort
    calls = []
    inspect = eligibility.inspect_reference_pair

    def record(dataset, settings):
        calls.append(dataset.dataset_id)
        return inspect(dataset, settings)

    monkeypatch.setattr(eligibility, "inspect_reference_pair", record)
    assert search.main(command + ["--exclude-invalid-references", "--preflight-only"]) == 0
    assert len(calls) == 9  # One missing reference and one invalid raw skip pair inspection.
    assignments = pd.read_csv(args.output_directory / "cv_fold_assignments.csv")
    assert set(assignments.dataset_id) == set(paths) - invalid
    test = assignments[assignments.role == "test"]
    assert len(test) == 7 and test.participant_id.nunique() == 5
    assert test.groupby("participant_id").outer_fold.nunique().eq(1).all()
    preflight = json.loads((args.output_directory / "preflight_manifest.json").read_text())
    assert preflight["n_recordings"] == 7 and preflight["n_participants"] == 5
    assert len(pd.read_csv(args.output_directory / "dataset_preflight.csv")) == 7
    assert not (args.output_directory / "prepared").exists()
    assert not (args.output_directory / "evaluation_cache.sqlite3").exists()


@pytest.mark.parametrize("folds", [["--outer-folds", "6"], ["--inner-folds", "5"]])
def test_insufficient_eligible_participants_fail_after_audit(cohort, folds):
    _, _, _, args, command = cohort
    with pytest.raises(ValueError, match="Eligible cohort cannot support requested CV.*7 sessions, 5 participants"):
        search.main(command + ["--exclude-invalid-references", "--preflight-only", *folds])
    assert len(pd.read_csv(args.output_directory / "reference_eligibility.csv")) == 11
    assert not (args.output_directory / "cv_fold_assignments.csv").exists()


def test_all_invalid_records_fail_with_a_complete_report(cohort):
    paths, _, settings, args, _ = cohort
    args.exclude_invalid_references = True
    paths = {key: value for key, value in paths.items() if key.startswith("sub-05")}
    with pytest.raises(ValueError, match="No eligible sessions remain"):
        eligibility.preflight_reference_sessions(paths, settings, args)
    summary = json.loads((args.output_directory / "reference_eligibility.json").read_text())["summary"]
    assert summary["n_eligible_sessions"] == 0 and summary["n_eligible_participants"] == 0


def test_resource_failures_are_not_used_to_filter_the_cohort(cohort, monkeypatch):
    paths, _, settings, args, _ = cohort
    args.exclude_invalid_references = True

    def allocation_failure(*_):
        raise MemoryError("insufficient process memory")

    monkeypatch.setattr(eligibility, "inspect_reference_pair", allocation_failure)
    with pytest.raises(MemoryError, match="insufficient process memory"):
        eligibility.preflight_reference_sessions(paths, settings, args)


def test_flag_rejects_non_reference_mode(tmp_path):
    args = search.build_argument_parser().parse_args([str(tmp_path), str(tmp_path), "--exclude-invalid-references"])
    with pytest.raises(ValueError, match="requires reference or combined"):
        search._validate_arguments(args)


def test_exclusion_requires_grouped_optimization(cohort):
    _, _, _, args, _ = cohort
    args.exclude_invalid_references = True
    args.cv_strategy = "leave-one-dataset-out"
    with pytest.raises(ValueError, match="requires participant-grouped CV"):
        search._validate_arguments(args)


def test_excluded_sessions_never_reach_processing_baselines_or_final_fit(cohort, monkeypatch):
    pytest.importorskip("optuna")
    monkeypatch.setattr(search, "OBJECTIVE_COLUMNS", search.OBJECTIVE_COLUMNS)
    paths, invalid, _, args, command = cohort
    prepared, evaluated, gates = [], [], []

    def prepare(dataset, settings, root):
        assert dataset.dataset_id not in invalid
        prepared.append(dataset.dataset_id)
        folder = root / dataset.dataset_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "prepared.json").write_text(json.dumps({"chunks": [{"job_mb": 1}]}))
        return folder

    def gate(training, *_):
        gates.extend(item.dataset_id for item in training)
        assert not set(gates) & invalid

    def evaluate(recipe, datasets, settings, cache, fidelity="full"):
        assert not {item.dataset_id for item in datasets} & invalid
        evaluated.extend(item.dataset_id for item in datasets)
        return [
            {
                "dataset_id": item.dataset_id,
                "configuration_id": search.configuration_id(recipe),
                "recipe_json": json.dumps(recipe.to_dict()),
                "success": True,
                "scanner_peak_residual": 0.2,
                "nonpeak_eeg_log_deviation": 0.05,
                "theta_preservation": 1.0,
                "alpha_preservation": 1.0,
                "nonpeak_beta_preservation": 1.0,
                "reference_normalized_mse": 0.3,
                "reference_spectral_log_deviation": 0.2,
                "reference_correlation": 0.8,
                "execution_time": 0.0,
                "fidelity": fidelity,
            }
            for item in datasets
        ]

    monkeypatch.setattr(search, "prepare_dataset", prepare)
    monkeypatch.setattr(search, "evaluate_many", evaluate)
    monkeypatch.setattr(grouped, "_validate_training_recording", gate)
    assert search.main(command + ["--exclude-invalid-references", "--outer-fold-index", "0"]) == 0
    allowed = set(paths) - invalid
    assert set(prepared) == allowed and set(evaluated).issubset(allowed)
    test = pd.read_csv(args.output_directory / "held_out_recording_results.csv")
    assert not set(test.dataset_id) & invalid
    assert (args.output_directory / "baseline_comparison.csv").exists()
    assert search.main(command + ["--exclude-invalid-references", "--final-only"]) == 0
    assert set(evaluated) == allowed and set(gates).issubset(allowed)
    final = json.loads((args.output_directory / "recommended_recipe.json").read_text())
    assert set(final["training_datasets"]) == allowed
    manifest = json.loads((args.output_directory / "run_manifest.json").read_text())
    assert set(manifest["datasets"]) == allowed
    assert manifest["reference_eligibility"]["n_eligible_participants"] == 5
