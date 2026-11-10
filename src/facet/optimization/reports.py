"""Checkpoint/final reports. Plotting never occurs inside an evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from facet.optimization.metrics import objectives


def nondominated(values):
    values = np.asarray(values, dtype=float)
    return np.asarray(
        [
            np.all(np.isfinite(point)) and not np.any(np.all(values <= point, axis=1) & np.any(values < point, axis=1))
            for point in values
        ],
        dtype=bool,
    )


def scatter(frame, columns, path):
    if frame.empty or not set(columns).issubset(frame.columns):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(frame[columns[0]], frame[columns[1]], alpha=0.7)
    if "is_pareto" in frame:
        front = frame[frame["is_pareto"].astype(bool)]
        ax.scatter(front[columns[0]], front[columns[1]], facecolors="none", edgecolors="red", label="Pareto")
        ax.legend()
    ax.set(xlabel=columns[0], ylabel=columns[1])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def training_reports(frame, directory: Path):
    from facet.optimization.search import OBJECTIVE_COLUMNS

    scatter(frame, OBJECTIVE_COLUMNS, directory / "training_pareto.png")
    rows = []
    for _, row in frame.iterrows():
        recipe = json.loads(row["recipe_json"])
        rows.append(
            {
                "window_size": str(recipe["quota"].get("window_size", "global")),
                "kernel": recipe["weighting"]["kernel"],
                "residual": row.get("scanner_peak_residual", np.nan),
            }
        )
    pivot = pd.DataFrame(rows).pivot_table(index="window_size", columns="kernel", values="residual", aggfunc="median")
    pivot.to_csv(directory / "parameter_effects.csv")
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        picture = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set(
            xlabel="weighting kernel", ylabel="window size", title="Descriptive median residual (confounded parameters)"
        )
        fig.colorbar(picture, ax=ax)
        fig.tight_layout()
        fig.savefig(directory / "parameter_effects.png", dpi=130)
        plt.close(fig)


def mode_reports(frame, root):
    """Rank cached metrics separately without invoking correction.

    These descriptive rankings use all available evaluations and are never
    used for fold selection. Fold CSVs remain the authoritative validation.
    """
    if frame.empty:
        return
    frame[~frame["success"].astype(bool)].to_csv(root / "failed_evaluations.csv", index=False)
    for mode in ("non-reference", "reference", "combined"):
        directory = root / mode
        directory.mkdir(exist_ok=True)
        rows = []
        for row in frame.to_dict("records"):
            try:
                x, y = objectives(row, mode)
            except KeyError:
                continue
            if not row["success"] or not np.isfinite(x) or not np.isfinite(y):
                continue
            rows.append({**row, "objective_1": x, "objective_2": y})
        result = pd.DataFrame(rows)
        result.to_csv(directory / "per_dataset_results.csv", index=False)
        if result.empty:
            continue
        # Separate Pareto sets by fidelity and recording: raw recording
        # metrics are not comparable population-level training observations.
        result["is_pareto"] = False
        for _, group in result.groupby(["fidelity", "dataset_id"]):
            result.loc[group.index, "is_pareto"] = nondominated(group[["objective_1", "objective_2"]].values)
        result.to_csv(directory / "rankings.csv", index=False)
        result[result["is_pareto"]].to_csv(directory / "pareto.csv", index=False)
        scatter(
            result[result["fidelity"] == "full"], ("objective_1", "objective_2"), directory / "metric_comparison.png"
        )


def stability_plot(stability, path):
    if stability.empty:
        return
    grouped = stability.groupby("parameter")["fraction"].max().sort_values().tail(24)
    fig, ax = plt.subplots(figsize=(8, max(3, len(grouped) * 0.25)))
    ax.barh(grouped.index, grouped.values)
    ax.set(xlim=(0, 1), xlabel="Fraction of folds selecting the most common value")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def final_mode_reports(root, settings):
    """Compare modes using only final-model training evaluations.

    Alternate-mode recommendations describe the evaluated candidate pool;
    independent nested validation belongs only to the mode actually optimized.
    Held-out fold observations never enter this selection.
    """
    from facet.optimization.search import pareto_mask, select_ideal_recipe

    path = root / "final_model/promoted_training_results.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    mode_columns = {
        "non-reference": ("scanner_peak_residual", "nonpeak_eeg_log_deviation"),
        "reference": ("reference_normalized_mse", "reference_spectral_log_deviation"),
        "combined": ("scanner_peak_residual", "reference_normalized_mse"),
    }
    for mode, columns in mode_columns.items():
        if not set(columns).issubset(frame):
            continue
        usable = frame[frame["success"].astype(bool) & np.isfinite(frame[list(columns)]).all(axis=1)].copy()
        if usable.empty:
            continue
        directory = root / mode
        directory.mkdir(exist_ok=True)
        usable["is_pareto"] = pareto_mask(usable, columns)
        usable.to_csv(directory / "final_training_results.csv", index=False)
        scatter(usable, columns, directory / "final_training_pareto.png")
        try:
            selected = select_ideal_recipe(usable, settings, columns)
        except RuntimeError:
            continue
        payload = {
            "mode": mode,
            "recipe": json.loads(selected["recipe_json"]),
            "configuration_id": selected["configuration_id"],
            "objectives": {column: float(selected[column]) for column in columns},
            "selection_pool": selected["selection_pool"],
            "scope": "Final-model training candidate pool only; alternative-mode ranking is descriptive",
            "independently_validated_mode": settings.evaluation_mode,
        }
        (directory / "recommended_recipe.json").write_text(json.dumps(payload, indent=2))
    trials = []
    for path in sorted(root.glob("folds/*/screening_trials.csv")) + list(root.glob("final_model/screening_trials.csv")):
        trial = pd.read_csv(path)
        trial["study"] = str(path.parent.relative_to(root))
        trials.append(trial)
    if trials:
        pd.concat(trials, ignore_index=True).to_csv(
            root / settings.evaluation_mode / "per_trial_results.csv", index=False
        )
