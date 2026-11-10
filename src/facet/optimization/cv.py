"""Deterministic participant partitions and equal-participant summaries."""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def subject_id(path: Path) -> str:
    """Extract one consistent BIDS subject entity from filename and parents."""
    identifiers = set()
    for part in Path(path).parts:
        identifiers.update(re.findall(r"(?:^|_)sub-([A-Za-z0-9]+)(?=_|\.|$)", part))
    if len(identifiers) != 1:
        raise ValueError(
            f"Missing or ambiguous BIDS participant identifier in {path}: {sorted(identifiers)}. "
            "Grouped CV requires a consistent sub-<ID> entity; no recording-level fallback."
        )
    return "sub-" + identifiers.pop()


def participant_groups(datasets):
    groups = defaultdict(list)
    for dataset in datasets:
        if not dataset.participant_id:
            raise ValueError(f"Missing participant identifier for {dataset.dataset_id}.")
        groups[dataset.participant_id].append(dataset.dataset_id)
    return {key: sorted(value) for key, value in sorted(groups.items())}


def balanced_folds(groups, count, seed):
    """Balance participant counts first and session counts second, deterministically."""
    if count < 2 or len(groups) < count:
        raise ValueError(f"Need at least {count} participants for {count} folds; found {len(groups)}.")
    rng = np.random.default_rng(seed)
    participants = list(rng.permutation(sorted(groups)))
    participants.sort(key=lambda item: -len(groups[item]))
    folds, sessions = [[] for _ in range(count)], [0] * count
    for participant in participants:
        order = rng.permutation(count).tolist()
        index = min(order, key=lambda i: (len(folds[i]), sessions[i]))
        folds[index].append(str(participant))
        sessions[index] += len(groups[participant])
    return [sorted(fold) for fold in folds]


def build_plan(datasets, *, strategy, outer_folds=5, inner_folds=4, seed=42):
    """Make complete outer and inner membership maps without looking at scores."""
    groups = participant_groups(datasets)
    if strategy == "leave-one-subject-out":
        outer = [[participant] for participant in groups]
    elif strategy == "grouped-kfold":
        outer = balanced_folds(groups, outer_folds, seed)
    else:
        raise ValueError(f"Unsupported grouped CV strategy: {strategy}")
    plan = {
        "version": 1,
        "strategy": strategy,
        "group_by": "subject",
        "seed": seed,
        "inner_folds": inner_folds,
        "participants": groups,
        "folds": [],
    }
    for index, test in enumerate(outer):
        training = {key: value for key, value in groups.items() if key not in test}
        inner = balanced_folds(training, inner_folds, seed + index + 1)
        plan["folds"].append(
            {
                "index": index,
                "train": sorted(item for recordings in training.values() for item in recordings),
                "test": sorted(item for key in test for item in groups[key]),
                "inner": [
                    {
                        "train": sorted(
                            item for key, values in training.items() if key not in validation for item in values
                        ),
                        "validation": sorted(item for key in validation for item in groups[key]),
                    }
                    for validation in inner
                ],
            }
        )
    plan["final_inner"] = [
        {
            "train": sorted(item for key, values in groups.items() if key not in validation for item in values),
            "validation": sorted(item for key in validation for item in groups[key]),
        }
        for validation in balanced_folds(groups, inner_folds, seed)
    ]
    validate_plan(plan)
    return plan


def validate_plan(plan):
    """Reject subject/session leakage and incomplete validation partitions."""
    by_dataset = {item: participant for participant, items in plan["participants"].items() for item in items}
    all_ids = set(by_dataset)
    tested = []
    for fold in plan["folds"]:
        train, test = set(fold["train"]), set(fold["test"])
        if train & test or train | test != all_ids:
            raise ValueError("Outer fold has overlapping or missing recordings.")
        if {by_dataset[item] for item in train} & {by_dataset[item] for item in test}:
            raise ValueError("Participant leakage across outer train/test partitions.")
        tested.extend(test)
        validate_inner(fold["inner"], train, by_dataset)
    if len(tested) != len(all_ids) or set(tested) != all_ids:
        raise ValueError("Every recording and participant must be held out exactly once.")
    validate_inner(plan["final_inner"], all_ids, by_dataset)


def validate_inner(inner, training_ids, by_dataset):
    validated = []
    for fold in inner:
        train, validation = set(fold["train"]), set(fold["validation"])
        if not train or not validation or train & validation or train | validation != set(training_ids):
            raise ValueError("Invalid inner train/validation partition.")
        if {by_dataset[item] for item in train} & {by_dataset[item] for item in validation}:
            raise ValueError("Participant leakage across inner train/validation partitions.")
        validated.extend(validation)
    if len(validated) != len(training_ids) or set(validated) != set(training_ids):
        raise ValueError("Inner validation must cover each outer-training recording exactly once.")


def assignment_frame(plan):
    rows = []
    for fold in plan["folds"]:
        validation = {item: index for index, inner in enumerate(fold["inner"]) for item in inner["validation"]}
        for participant, recordings in plan["participants"].items():
            for item in recordings:
                rows.append(
                    {
                        "outer_fold": fold["index"],
                        "participant_id": participant,
                        "dataset_id": item,
                        "role": "test" if item in fold["test"] else "train",
                        "inner_validation_fold": validation.get(item, -1),
                    }
                )
    return pd.DataFrame(rows)


def participant_rows(rows):
    """Arithmetic session means, one row per participant; failures never disappear."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["participant_id"]].append(row)
    output = []
    for participant, recordings in sorted(groups.items()):
        result = {
            "participant_id": participant,
            "dataset_id": participant,
            "n_sessions": len(recordings),
            "success": all(bool(row.get("success")) for row in recordings),
            "failed_sessions": sum(not bool(row.get("success")) for row in recordings),
        }
        keys = {
            key
            for row in recordings
            for key, value in row.items()
            if isinstance(value, (int, float, np.number)) and key not in {"success", "outer_fold", "inner_fold"}
        }
        for key in sorted(keys):
            values = [float(row.get(key, np.nan)) for row in recordings]
            result[key] = float(np.mean(values)) if np.all(np.isfinite(values)) else float("nan")
        for key in ("outer_fold", "inner_fold", "configuration_id", "recipe_json", "fidelity", "configuration"):
            if key in recordings[0]:
                result[key] = recordings[0][key]
        output.append(result)
    return output


def participant_bootstrap(frame, *, samples, seed):
    """Bootstrap participants, retaining session dependence inside each row."""
    rng = np.random.default_rng(seed)
    rows = []
    columns = [
        name for name in frame.select_dtypes(include="number") if name not in {"outer_fold", "n_sessions", "success"}
    ]
    for column in columns:
        values = frame[column].to_numpy(float)
        if not len(values) or not np.all(np.isfinite(values)):
            continue
        estimates = [np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(samples)]
        rows.append(
            {
                "metric": column,
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "participant_std": float(np.std(values)),
                "ci_low": float(np.quantile(estimates, 0.025)),
                "ci_high": float(np.quantile(estimates, 0.975)),
                "n_participants": len(values),
            }
        )
    return pd.DataFrame(
        rows, columns=["metric", "median", "mean", "participant_std", "ci_low", "ci_high", "n_participants"]
    )
