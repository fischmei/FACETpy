"""Participant partitions, validation-only selection and local synthetic runs."""

import json
from dataclasses import replace
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from facet.optimization import search
from facet.optimization.cache import EvaluationCache, digest
from facet.optimization.cv import (
    assignment_frame,
    balanced_folds,
    build_plan,
    participant_bootstrap,
    participant_rows,
    subject_id,
    validate_plan,
)
from facet.optimization.grouped import run_grouped
from facet.optimization.inner_cv import InnerValidation
from facet.optimization.prepared import match_reference


def datasets_at(root, participants=5, multiple=True):
    result = []
    for participant in range(participants):
        for session in range(2 if multiple and participant % 2 == 0 else 1):
            path = root / f"sub-{participant:02d}_ses-{session:02d}_task-rest_eeg.fif"
            path.write_bytes(b"synthetic metadata fixture")
            result.append(search.DatasetSpec(path.stem, path, np.arange(10), None, participant_id=subject_id(path)))
    return result


def run_settings():
    return search.RunSettings(
        trigger_regex="^R128$",
        upsample_factor=1,
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
    )


def test_balanced_subject_folds_and_all_sessions(tmp_path):
    datasets = datasets_at(tmp_path, 22)
    plan = build_plan(datasets, strategy="grouped-kfold")
    assert plan == build_plan(list(reversed(datasets)), strategy="grouped-kfold")
    assert plan != build_plan(datasets, strategy="grouped-kfold", seed=43)
    frame = assignment_frame(plan)
    test = frame[frame.role == "test"]
    assert len(test) == len(datasets)
    assert test.groupby("participant_id").outer_fold.nunique().eq(1).all()
    counts = test.groupby("outer_fold").participant_id.nunique()
    assert sorted(counts) == [4, 4, 4, 5, 5]
    for fold in plan["folds"]:
        seen = [item for inner in fold["inner"] for item in inner["validation"]]
        assert sorted(seen) == sorted(fold["train"])
        assert not set(seen) & set(fold["test"])
    plan["folds"][0]["train"].append(plan["folds"][0]["test"][0])
    with pytest.raises(ValueError, match="overlapping"):
        validate_plan(plan)


@pytest.mark.parametrize("path", ["rest.fif", "sub-01/sub-02_task-rest_eeg.set", "sub-01_sub-02_eeg.set"])
def test_subject_identity_requires_unambiguous_bids(path):
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        subject_id(Path(path))
    assert subject_id(Path("sub-01/ses-02/eeg/sub-01_ses-02_task-rest_eeg.set")) == "sub-01"


def test_small_counts_and_leave_subject_out(tmp_path):
    datasets = datasets_at(tmp_path)
    plan = build_plan(datasets, strategy="leave-one-subject-out")
    assert len(plan["folds"]) == 5
    assert len({item for fold in plan["folds"] for item in fold["test"]}) == len(datasets)
    with pytest.raises(ValueError, match="Need at least"):
        balanced_folds({"one": ["session"]}, 5, 42)


def test_session_means_and_equal_participant_weighting():
    rows = [
        {"participant_id": "one", "success": True, "scanner_peak_residual": value, "nonpeak_eeg_log_deviation": 1}
        for value in [1, 3, 5]
    ]
    rows += [{"participant_id": "two", "success": True, "scanner_peak_residual": 9, "nonpeak_eeg_log_deviation": 1}]
    participants = participant_rows(rows)
    assert participants[0]["scanner_peak_residual"] == 3
    assert search.aggregate_dataset_rows(participants)["scanner_peak_residual"] == 6
    assert search.aggregate_dataset_rows(rows)["scanner_peak_residual"] == 4
    first = participant_bootstrap(pd.DataFrame(participants), samples=100, seed=42)
    pd.testing.assert_frame_equal(first, participant_bootstrap(pd.DataFrame(participants), samples=100, seed=42))
    assert first.n_participants.eq(2).all()
    rows[0]["success"] = False
    assert not participant_rows(rows)[0]["success"]


def fake_metrics(recipe, items, settings, cache, fidelity="full"):
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
        for item in items
    ]


def test_inner_validation_rejects_outer_test_and_screens_each_subject(tmp_path, monkeypatch):
    datasets = datasets_at(tmp_path)
    plan = build_plan(datasets, strategy="grouped-kfold")
    fold = plan["folds"][0]
    training = [item for item in datasets if item.dataset_id in fold["train"]]
    calls = []

    def evaluate(recipe, items, settings, cache, fidelity="full"):
        calls.append({item.dataset_id for item in items})
        return fake_metrics(recipe, items, settings, cache, fidelity)

    monkeypatch.setattr(search, "evaluate_many", evaluate)
    evaluator = InnerValidation(training, fold["inner"], seed=42, screening_fraction=0.1, directory=tmp_path / "inner")
    result = evaluator.evaluate(search.MatrixDecisions(), training, run_settings(), None, "screening")
    assert len(result) == 4
    assert all(len(call) == 1 for call in calls)
    assert all(not call & set(fold["test"]) for call in calls)
    assert len(evaluator.last_scores) == 4
    with pytest.raises(ValueError, match="Outer-test"):
        evaluator.evaluate(search.MatrixDecisions(), datasets, run_settings(), None)


@pytest.mark.parametrize("mode", ["non-reference", "reference", "combined"])
def test_grouped_runner_isolation_resume_and_final(tmp_path, monkeypatch, mode):
    pytest.importorskip("optuna")
    datasets = datasets_at(tmp_path)
    plan = build_plan(datasets, strategy="grouped-kfold")
    settings = replace(run_settings(), cv_signature=plan, evaluation_mode=mode)
    columns = {
        "non-reference": ("scanner_peak_residual", "nonpeak_eeg_log_deviation"),
        "reference": ("reference_normalized_mse", "reference_spectral_log_deviation"),
        "combined": ("scanner_peak_residual", "reference_normalized_mse"),
    }
    monkeypatch.setattr(search, "OBJECTIVE_COLUMNS", columns[mode])
    args = search.build_argument_parser().parse_args(
        [
            str(tmp_path),
            str(tmp_path / "results"),
            "--cv-strategy",
            "grouped-kfold",
            "--trials",
            "1",
            "--promotion-count",
            "3",
            "--bootstrap-samples",
            "100",
        ]
    )
    cache = EvaluationCache(tmp_path / "cache.sqlite3", signature="test")
    current_outer = {item: fold["index"] for fold in plan["folds"] for item in fold["test"]}
    test_values = {"value": 10.0}
    from facet.optimization import grouped

    monkeypatch.setattr(grouped, "_validate_training_recording", lambda *args: None)
    phase = {"testing": False}
    original_test = grouped._outer_test

    def testing(*args, **kwargs):
        phase["testing"] = True
        try:
            return original_test(*args, **kwargs)
        finally:
            phase["testing"] = False

    monkeypatch.setattr(grouped, "_outer_test", testing)

    def evaluate(recipe, items, settings, cache, fidelity="full"):
        # During outer testing a recipe file must already be persisted; inner
        # evaluation never reads these values or the aggregate result files.
        rows = fake_metrics(recipe, items, settings, cache, fidelity)
        for row in rows:
            outer = current_outer[row["dataset_id"]]
            locked = args.output_directory / f"folds/outer_{outer:02d}/selected_recipe.json"
            if locked.exists():
                selected = json.loads(locked.read_text())
                assert row["dataset_id"] not in selected["training_datasets"]
            if phase["testing"]:
                assert locked.exists()
                for column in columns[mode]:
                    row[column] = test_values["value"]
        return rows

    monkeypatch.setattr(search, "evaluate_many", evaluate)
    run_grouped(datasets, settings, cache, plan, args)
    root = args.output_directory
    heldout = pd.read_csv(root / "held_out_participant_results.csv")
    assert heldout.participant_id.nunique() == 5 and len(heldout) == 5
    assert len(pd.read_csv(root / "held_out_recording_results.csv")) == len(datasets)
    assert len(pd.read_csv(root / "baseline_comparison.csv")) == 15
    assert len(pd.read_csv(root / "outer_fold_results.csv")) == 5
    assert json.loads((root / "nested_cv_status.json").read_text())["complete"]
    before = [(p, p.read_bytes()) for p in root.glob("folds/*/selected_recipe.json")]
    args.outer_fold_index = 0
    test_values["value"] = 1e9
    run_grouped(datasets, settings, cache, plan, args)
    assert all(path.read_bytes() == value for path, value in before)
    changed = pd.read_csv(root / "held_out_participant_results.csv")
    assert changed.loc[changed.outer_fold == 0, columns[mode][0]].eq(1e9).all()
    for fold in plan["folds"]:
        frame = pd.read_csv(root / f"folds/outer_{fold['index']:02d}/inner_cv_recording_results.csv")
        assert not set(frame.dataset_id) & set(fold["test"])
    args.final_only, args.outer_fold_index = True, None
    run_grouped(datasets, settings, cache, plan, args)
    final = json.loads((root / "recommended_recipe.json").read_text())
    assert len(final["training_participants"]) == 5
    assert final["inner_validation"]["inner"] == plan["final_inner"]
    incompatible = replace(settings, cv_signature={**plan, "seed": 100})
    with pytest.raises(ValueError, match="Incompatible CV"):
        run_grouped(datasets, incompatible, cache, plan, args)
    cache.close()


def test_synthetic_grouped_cli_end_to_end(tmp_path):
    pytest.importorskip("optuna")
    raw_root = tmp_path / "data"
    raw_root.mkdir()
    rate = 200
    times = np.arange(4000) / rate
    rng = np.random.default_rng(42)
    for participant in range(5):
        for session in range(2 if participant == 0 else 1):
            data = (
                np.array(
                    [
                        sum(np.sin(2 * np.pi * f * times) for f in (6.7, 10.7, 17.3)) + rng.normal(0, 0.2, len(times))
                        for _ in range(2)
                    ]
                )
                * 1e-6
            )
            data += 20e-6 * np.sin(2 * np.pi * 20 * times)
            raw = mne.io.RawArray(data, mne.create_info(["Cz", "Pz"], rate, "eeg"), verbose=False)
            triggers = np.arange(100, 3901, 100)
            raw.set_annotations(mne.Annotations(triggers / rate, np.zeros(len(triggers)), ["R128"] * len(triggers)))
            raw.save(raw_root / f"sub-{participant:02d}_ses-{session:02d}_task-rest_eeg.fif", verbose=False)
    root = tmp_path / "output"
    command = [
        str(raw_root),
        str(root),
        "--cv-strategy",
        "grouped-kfold",
        "--trials",
        "1",
        "--promotion-count",
        "3",
        "--bootstrap-samples",
        "100",
        "--upsample-factor",
        "1",
        "--chunk-seconds",
        "30",
        "--chunk-min-triggers",
        "5",
        "--preservation-ratio-bound",
        "2",
        "--beta-preservation-maximum",
        "3",
    ]
    assert search.main(command + ["--preflight-only"]) == 0
    assert (root / "dataset_preflight.csv").exists() and not (root / "evaluation_cache.sqlite3").exists()
    assert search.main(command) == 0
    assert len(pd.read_csv(root / "held_out_participant_results.csv")) == 5
    assert len(pd.read_csv(root / "held_out_recording_results.csv")) == 6
    assert (root / "participant_metrics.png").exists()
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["settings"]["optimizer_settings"]["cv_signature"]["strategy"] == "grouped-kfold"


def test_cv_cache_context_and_reference_entities(tmp_path):
    datasets = datasets_at(tmp_path)
    first = replace(run_settings(), cv_signature=build_plan(datasets, strategy="grouped-kfold"))
    second = replace(run_settings(), cv_signature=build_plan(datasets, strategy="grouped-kfold", seed=43))
    cache = EvaluationCache(tmp_path / "cache.sqlite3", signature=digest(first.evaluation_signature_payload()))
    cache.add({"dataset_id": "recording", "configuration_id": "candidate", "success": True})
    cache.close()
    other = EvaluationCache(tmp_path / "cache.sqlite3", signature=digest(second.evaluation_signature_payload()))
    assert other.get("candidate", "recording") is None
    other.close()
    assert first.evaluation_signature_payload() == replace(first, workers=2).evaluation_signature_payload()
    assert first.evaluation_signature_payload() != replace(first, chunk_seconds=60).evaluation_signature_payload()
    references = tmp_path / "references"
    references.mkdir()
    raw = tmp_path / "sub-01_ses-02_task-rest_run-1_eeg.set"
    raw.touch()
    dataset = search.DatasetSpec(raw.stem, raw, np.arange(10), None)
    for subject, session, run in ((1, 1, 1), (1, 2, 1), (1, 2, 2), (2, 2, 1)):
        (references / f"sub-{subject:02d}_ses-{session:02d}_task-rest_run-{run}_desc-clean_eeg.set").touch()
    match = match_reference(dataset, references, tmp_path, True)
    assert match.name == "sub-01_ses-02_task-rest_run-1_desc-clean_eeg.set"
    (references / "sub-01_ses-02_task-rest_run-1_desc-other_eeg.fif").touch()
    with pytest.raises(ValueError, match="Ambiguous reference"):
        match_reference(dataset, references, tmp_path, True)


def test_cli_flags_and_bounds_do_not_require_future_server_paths(tmp_path):
    parser = search.build_argument_parser()
    parsed = parser.parse_args(
        [
            "/data/users/cfischmei/NATVIEW_data",
            str(tmp_path),
            "--reference-directory",
            "/data/users/cfischmei/NATVIEW_cleaned",
            "--cv-strategy",
            "grouped-kfold",
            "--cv-group-by",
            "subject",
            "--outer-folds",
            "5",
            "--inner-folds",
            "4",
            "--cv-seed",
            "42",
            "--outer-fold-index",
            "0",
            "--include-pattern",
            "*task-rest_eeg.set",
            "--preflight-only",
        ]
    )
    assert parsed.outer_folds == 5 and parsed.inner_folds == 4 and parsed.outer_fold_index == 0
    local = parser.parse_args([str(tmp_path), str(tmp_path), "--outer-fold-index", "-1"])
    with pytest.raises(ValueError, match="zero-based"):
        search._validate_arguments(local)


def test_reference_gate_uses_only_current_outer_training(tmp_path, monkeypatch):
    from facet.optimization import validation
    from facet.optimization.grouped import _validate_training_recording

    datasets = datasets_at(tmp_path)
    plan = build_plan(datasets, strategy="grouped-kfold")
    settings = replace(run_settings(), evaluation_mode="combined")
    args = search.build_argument_parser().parse_args([str(tmp_path), str(tmp_path / "output")])
    seen = []
    monkeypatch.setattr(validation, "validate_single_recording", lambda dataset, *args: seen.append(dataset.dataset_id))
    for fold in plan["folds"]:
        training = [item for item in datasets if item.dataset_id in fold["train"]]
        _validate_training_recording(training, settings, None, args)
        assert seen[-1] in fold["train"] and seen[-1] not in fold["test"]


def test_grouped_output_cannot_reuse_legacy_run_manifest(tmp_path):
    from facet.optimization.grouped import check_compatibility

    datasets = datasets_at(tmp_path)
    plan = build_plan(datasets, strategy="grouped-kfold")
    settings = replace(run_settings(), cv_signature=plan)
    args = search.build_argument_parser().parse_args([str(tmp_path), str(tmp_path)])
    (tmp_path / "run_manifest.json").write_text(json.dumps({"settings": {"optimizer_settings": {}}}))
    with pytest.raises(ValueError, match="Incompatible CV strategy"):
        check_compatibility(datasets, settings, plan, args)
