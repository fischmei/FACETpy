"""Participant-grouped nested orchestration using the existing search engine."""

import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from facet import Loader
from facet.correction.flex.decisions import MatrixDecisions
from facet.optimization.cache import digest
from facet.optimization.cleaning import validate_cleaning_input
from facet.optimization.cv import assignment_frame, participant_bootstrap, participant_rows
from facet.optimization.inner_cv import InnerValidation
from facet.optimization.reference import reference_plan


def preflight_report(datasets, settings, plan, root, validated_rows=None):
    """Inspect acquisition/pairing metadata without correction, TPE or preparation."""
    from facet.optimization.search import _atomic_write_csv, write_json

    rows = []
    for dataset in datasets:
        row = {
            "dataset_id": dataset.dataset_id,
            "participant_id": dataset.participant_id,
            "raw": str(dataset.path),
            "reference": str(dataset.reference_path),
            "scanner_triggers": len(dataset.trigger_samples),
        }
        if settings.evaluation_mode != "non-reference" and validated_rows is not None:
            row.update(
                {
                    name: validated_rows[dataset.dataset_id][name]
                    for name in ("shared_channels", "reference_coverage", "raw_sfreq", "reference_sfreq")
                }
            )
        elif settings.evaluation_mode != "non-reference":
            raw = Loader(str(dataset.path), preload=False).execute(None).get_raw()
            reference = None
            try:
                validate_cleaning_input(raw)
                reference = Loader(str(dataset.reference_path), preload=False).execute(None).get_raw()
                evidence = reference_plan(
                    raw,
                    reference,
                    dataset.reference_path,
                    raw_path=dataset.path,
                    assume_aligned=settings.assume_reference_aligned,
                )
                write_json(root / f"{dataset.dataset_id}_reference_preflight.json", evidence)
                row.update(shared_channels=len(evidence["channels"]), reference_coverage=evidence["coverage"])
            finally:
                raw.close()
                if reference is not None:
                    reference.close()
        rows.append(row)
    _atomic_write_csv(pd.DataFrame(rows), root / "dataset_preflight.csv")
    write_json(
        root / "preflight_manifest.json",
        {
            "cv": plan,
            "n_recordings": len(datasets),
            "n_participants": len(plan["participants"]) if plan is not None else None,
            "settings": settings.signature_payload(),
            "datasets": [item.signature_payload() for item in datasets],
        },
    )
    if plan is not None:
        _atomic_write_csv(assignment_frame(plan), root / "cv_fold_assignments.csv")


def check_compatibility(datasets, settings, plan, args):
    """Check prior outer-report provenance before allocating prepared signals."""
    compatibility = {
        "plan": plan,
        "settings": settings.signature_payload(),
        "datasets": [item.signature_payload() for item in datasets],
        "optimization_seed": args.seed,
        "screening_fraction": args.screening_fraction,
        "promotion_count": args.promotion_count,
    }
    for name in ("workers", "memory_mb"):
        compatibility["settings"]["optimizer_settings"].pop(name, None)
    signature = digest(compatibility)
    prior_run = args.output_directory / "run_manifest.json"
    if prior_run.exists():
        previous = json.loads(prior_run.read_text())
        previous_plan = previous.get("settings", {}).get("optimizer_settings", {}).get("cv_signature")
        if previous_plan != plan:
            raise ValueError(
                "Incompatible CV strategy or assignments in the existing run manifest. Use a new output directory."
            )
    marker = args.output_directory / "cv_manifest.json"
    if marker.exists() and json.loads(marker.read_text())["signature"] != signature:
        raise ValueError(
            "Incompatible CV/preprocessing/data manifest. Use a new output directory; "
            "existing outer results cannot be mixed with different folds or settings."
        )
    return {"signature": signature, **compatibility}


def run_grouped(datasets, settings, cache, plan, args):
    from facet.optimization.search import _atomic_write_csv, write_json

    root = args.output_directory
    write_json(root / "cv_manifest.json", check_compatibility(datasets, settings, plan, args))
    _atomic_write_csv(assignment_frame(plan), root / "cv_fold_assignments.csv")
    by_id = {item.dataset_id: item for item in datasets}
    if args.final_only:
        _validate_training_recording(datasets, settings, cache, args)
        _fit_inner(
            datasets,
            plan["final_inner"],
            settings,
            cache,
            args,
            root / "final_model",
            args.seed,
            scope="Final recipe selected by participant-grouped internal validation on all development participants",
        )
        recipe = json.loads((root / "final_model/recommended_recipe.json").read_text())
        write_json(root / "recommended_recipe.json", recipe)
        return
    folds = plan["folds"] if args.outer_fold_index is None else [plan["folds"][args.outer_fold_index]]
    for fold in folds:
        directory = root / "folds" / f"outer_{fold['index']:02d}"
        training = [by_id[item] for item in fold["train"]]
        _validate_training_recording(training, settings, cache, args)
        selected = _fit_inner(
            training,
            fold["inner"],
            settings,
            cache,
            args,
            directory,
            args.seed + fold["index"],
            scope="Locked outer-fold recipe selected by inner validation only",
        )
        # Selection and the persisted recipe are complete before any test
        # candidate or baseline is evaluated. Test values are never fed back.
        _outer_test(
            [by_id[item] for item in fold["test"]], training, selected, settings, cache, directory, fold["index"]
        )
        _reports(root, settings, plan, args)


def _validate_training_recording(training, settings, cache, args):
    """Run/reuse the fixed pipeline gate using only this fold's training data."""
    if settings.evaluation_mode == "non-reference":
        return
    from facet.optimization.validation import validate_single_recording

    dataset = training[0]
    validate_single_recording(
        dataset, settings, cache, args.output_directory / "single_recording_validation" / dataset.dataset_id
    )


def _fit_inner(training, inner, settings, cache, args, directory, seed, *, scope):
    from facet.optimization.search import OBJECTIVE_COLUMNS, _atomic_write_csv, run_outer_fold, write_json

    evaluator = InnerValidation(
        training, inner, seed=seed, screening_fraction=args.screening_fraction, directory=directory
    )
    result = run_outer_fold(
        held_out=None,
        training=training,
        fold_directory=directory,
        settings=settings,
        cache=cache,
        trials=args.trials,
        promotion_count=args.promotion_count,
        screening_fraction=args.screening_fraction,
        seed=seed,
        timeout_seconds=args.timeout_seconds,
        rebuild_study=args.rebuild_studies,
        validation_evaluator=evaluator,
    )
    result.update(
        scope=scope,
        cv_strategy=args.cv_strategy,
        inner_validation=evaluator.signature,
        training_participants=sorted({item.participant_id for item in training}),
    )
    promoted = pd.read_csv(directory / "promoted_training_results.csv")
    selected = promoted[promoted.configuration_id == result["configuration_id"]].iloc[0]
    result["inner_validation_metrics"] = {name: float(selected[name]) for name in OBJECTIVE_COLUMNS}
    write_json(directory / "recommended_recipe.json", result)
    write_json(directory / "selected_recipe.json", result)
    for old, new in (("screening_trials.csv", "inner_cv_trials.csv"), ("training_pareto.csv", "inner_cv_pareto.csv")):
        _atomic_write_csv(pd.read_csv(directory / old), directory / new)
    return result


def _outer_test(test, training, selected, settings, cache, directory, fold_index):
    from facet.optimization.search import (
        SearchCapabilities,
        _atomic_write_csv,
        _fixed_trial_recipes,
        _load_optuna,
        anchor_parameter_sets,
        evaluate_many,
    )

    anchors = dict(
        _fixed_trial_recipes(
            _load_optuna(),
            anchor_parameter_sets(SearchCapabilities.from_datasets(training)),
            SearchCapabilities.from_datasets(training),
            settings,
        )
    )
    recipes = {
        "optimized": MatrixDecisions.from_dict(selected["recipe"]),
        "flex_default": MatrixDecisions(),
        "aas_per_target": anchors["aas_per_target"],
        "farm_per_target_k10": anchors["farm_per_target_k10"],
    }
    participant, recording = [], []
    by_id = {item.dataset_id: item.participant_id for item in test}
    for name, recipe in recipes.items():
        evaluated = evaluate_many(recipe, test, settings, cache, "full")
        if len(evaluated) != len(by_id) or {row["dataset_id"] for row in evaluated} != set(by_id):
            raise ValueError("Outer test returned missing, duplicate or unexpected recording results.")
        rows = [
            {**row, "participant_id": by_id[row["dataset_id"]], "outer_fold": fold_index, "configuration": name}
            for row in evaluated
        ]
        recording.extend(rows)
        participant.extend(participant_rows(rows))
    _atomic_write_csv(pd.DataFrame(recording), directory / "outer_recording_results.csv")
    _atomic_write_csv(pd.DataFrame(participant), directory / "outer_participant_results.csv")


def _reports(root, settings, plan, args):
    from facet.optimization.reports import stability_plot
    from facet.optimization.search import _atomic_write_csv, aggregate_dataset_rows, parameter_stability, write_json

    def collect(pattern):
        paths = sorted(root.glob(pattern))
        return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True) if paths else pd.DataFrame()

    participants = collect("folds/outer_*/outer_participant_results.csv")
    recordings = collect("folds/outer_*/outer_recording_results.csv")
    selected = participants[participants["configuration"] == "optimized"].copy()
    selected_recordings = recordings[recordings["configuration"] == "optimized"].copy()
    if selected["participant_id"].duplicated().any():
        raise ValueError("Duplicate held-out participants across outer results.")
    expected = {item: fold["index"] for fold in plan["folds"] for item in fold["test"]}
    participant_fold = {participant: expected[items[0]] for participant, items in plan["participants"].items()}
    if any(participant_fold[row.participant_id] != row.outer_fold for row in selected.itertuples()):
        raise ValueError("Held-out participant results disagree with the saved outer assignments.")
    _atomic_write_csv(selected, root / "held_out_participant_results.csv")
    _atomic_write_csv(selected_recordings, root / "held_out_recording_results.csv")
    _atomic_write_csv(selected_recordings, root / "held_out_results.csv")
    _atomic_write_csv(
        participant_bootstrap(selected, samples=args.bootstrap_samples, seed=args.cv_seed),
        root / "participant_bootstrap_summary.csv",
    )
    outer = [
        {"outer_fold": index, **aggregate_dataset_rows(group.to_dict("records"))}
        for index, group in selected.groupby("outer_fold")
    ]
    _atomic_write_csv(pd.DataFrame(outer), root / "outer_fold_results.csv")
    paired = []
    for baseline in ("flex_default", "aas_per_target", "farm_per_target_k10"):
        matched = selected.merge(
            participants[participants["configuration"] == baseline],
            on=["participant_id", "outer_fold"],
            suffixes=("_optimized", "_baseline"),
            validate="one_to_one",
        )
        for row in matched.to_dict("records"):
            result = {
                "participant_id": row["participant_id"],
                "outer_fold": row["outer_fold"],
                "baseline": baseline,
                "paired_success": bool(row["success_optimized"]) and bool(row["success_baseline"]),
            }
            for key in selected.select_dtypes(include="number"):
                if key in {"outer_fold", "n_sessions"}:
                    continue
                x, y = row.get(key + "_optimized"), row.get(key + "_baseline")
                if x is not None and y is not None:
                    result.update({f"{key}_optimized": x, f"{key}_baseline": y, f"{key}_delta": x - y})
            paired.append(result)
    _atomic_write_csv(pd.DataFrame(paired), root / "baseline_comparison.csv")
    paired_frame = pd.DataFrame(paired)
    intervals = []
    for baseline, group in paired_frame.groupby("baseline"):
        numeric = group[[column for column in group if column.endswith("_delta")]]
        summary = participant_bootstrap(numeric, samples=args.bootstrap_samples, seed=args.cv_seed)
        summary["baseline"] = baseline
        intervals.append(summary)
    _atomic_write_csv(pd.concat(intervals, ignore_index=True), root / "baseline_bootstrap_summary.csv")
    _baseline_plot(paired_frame, root / "baseline_comparison.png")
    for name in ("inner_cv_trials", "inner_cv_pareto"):
        frames = []
        for path in sorted(root.glob(f"folds/outer_*/{name}.csv")):
            frame = pd.read_csv(path)
            frame["outer_fold"] = int(path.parent.name.split("_")[-1])
            frames.append(frame)
        if frames:
            _atomic_write_csv(pd.concat(frames, ignore_index=True), root / f"{name}.csv")
    recipes = []
    for path in sorted(root.glob("folds/outer_*/selected_recipe.json")):
        recipe = json.loads(path.read_text())
        recipes.append({"recipe_json": json.dumps(recipe["recipe"])})
    stability = parameter_stability(pd.DataFrame(recipes))
    _atomic_write_csv(stability, root / "parameter_stability.csv")
    stability_plot(stability, root / "parameter_stability.png")
    write_json(
        root / "nested_cv_status.json",
        {
            "complete": len(selected) == len(plan["participants"]),
            "tested_participants": len(selected),
            "expected_participants": len(plan["participants"]),
            "completed_outer_folds": len(outer),
            "successful_participants": int(selected["success"].sum()),
            "interpretation": "Nested participant-held-out estimate; final-model selection is separate",
        },
    )
    _participant_plot(selected, root / "participant_metrics.png")


def _baseline_plot(frame, path):
    from facet.optimization.search import OBJECTIVE_COLUMNS

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, OBJECTIVE_COLUMNS, strict=True):
        for index, (name, group) in enumerate(frame.groupby("baseline")):
            axis.scatter(np.full(len(group), index), group[metric + "_delta"], label=name)
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.set(ylabel=f"Optimized minus baseline: {metric}", xticks=[])
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _participant_plot(frame, path):
    from facet.optimization.search import OBJECTIVE_COLUMNS

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for ax, column in zip(axes, OBJECTIVE_COLUMNS, strict=True):
        ax.scatter(frame["participant_id"], frame[column], c=frame["outer_fold"], cmap="tab10")
        ax.set(ylabel=column)
        ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
