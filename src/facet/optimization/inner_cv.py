"""Validation-only adapter for the existing TPE/screening/promotion engine."""

import math

import numpy as np
import pandas as pd

from facet.optimization.cv import participant_rows, validate_inner


class InnerValidation:
    """Evaluate a candidate on disjoint participant validation partitions.

    Flex is a per-recording transformation, with no cross-recording model to
    fit. Inner-training recordings are never used to score their paired
    validation fold. No downstream settings are learned. All model selection
    uses the pooled out-of-fold participant results, not outer-test results.
    """

    def __init__(self, training, inner, *, seed, screening_fraction, directory):
        self.datasets = {item.dataset_id: item for item in training}
        self.inner = inner
        self.seed = seed
        self.screening_fraction = screening_fraction
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        by_dataset = {item.dataset_id: item.participant_id for item in training}
        validate_inner(inner, set(self.datasets), by_dataset)
        self.signature = {
            "inner": inner,
            "seed": seed,
            "screening_fraction": screening_fraction,
            "aggregation": "session arithmetic means then equal-participant median/quantiles",
            "fit": "No cross-recording fitted model in Flex or fixed cleaning",
        }
        self.last_scores = []
        self._recordings = {}
        self._participants = {}
        self._scores = {}
        for filename, destination, fields in (
            ("inner_cv_recording_results.csv", self._recordings, ("configuration_id", "fidelity", "dataset_id")),
            (
                "inner_cv_participant_results.csv",
                self._participants,
                ("configuration_id", "fidelity", "participant_id"),
            ),
            ("inner_cv_fold_scores.csv", self._scores, ("configuration_id", "fidelity", "inner_fold")),
        ):
            path = self.directory / filename
            if path.exists():
                for row in pd.read_csv(path).to_dict("records"):
                    destination[tuple(row[field] for field in fields)] = row

    def evaluate(self, decisions, datasets, settings, cache, fidelity="full"):
        from facet.optimization.search import aggregate_dataset_rows, evaluate_many

        if not {item.dataset_id for item in datasets}.issubset(self.datasets):
            raise ValueError("Outer-test or unknown recordings passed to inner validation.")
        # Every participant is represented even during screening. Reduce
        # sessions within participants and EEG fidelity, never pool test data.
        participants, self.last_scores = [], []
        for index, fold in enumerate(self.inner):
            validation = [self.datasets[item] for item in fold["validation"]]
            if fidelity == "screening":
                validation = self._screening_sessions(validation, index)
            rows = evaluate_many(decisions, validation, settings, cache, fidelity)
            mapping = {item.dataset_id: item.participant_id for item in validation}
            if len(rows) != len(mapping) or {row["dataset_id"] for row in rows} != set(mapping):
                raise ValueError("Inner validation returned missing, duplicate or unexpected recording results.")
            rows = [
                {**row, "participant_id": mapping[row["dataset_id"]], "inner_fold": index, "fidelity": fidelity}
                for row in rows
            ]
            grouped = participant_rows(rows)
            participants.extend(grouped)
            self.last_scores.append(
                {
                    "inner_fold": index,
                    "validation_participants": sorted(set(mapping.values())),
                    **aggregate_dataset_rows(grouped),
                }
            )
            for row in rows:
                self._recordings[(row["configuration_id"], fidelity, row["dataset_id"])] = row
            for row in grouped:
                self._participants[(row["configuration_id"], fidelity, row["participant_id"])] = row
            self._scores[(rows[0]["configuration_id"], fidelity, index)] = {
                **self.last_scores[-1],
                "configuration_id": rows[0]["configuration_id"],
                "fidelity": fidelity,
            }
        # Diagnostic files are not selection inputs. Cache hits still rebuild
        # them on resume, so an interrupted CSV write cannot change ranking.
        from facet.optimization.search import _atomic_write_csv

        _atomic_write_csv(pd.DataFrame(self._recordings.values()), self.directory / "inner_cv_recording_results.csv")
        _atomic_write_csv(
            pd.DataFrame(self._participants.values()), self.directory / "inner_cv_participant_results.csv"
        )
        _atomic_write_csv(pd.DataFrame(self._scores.values()), self.directory / "inner_cv_fold_scores.csv")
        return participants

    def _screening_sessions(self, validation, fold_index):
        by_subject = {}
        for item in validation:
            by_subject.setdefault(item.participant_id, []).append(item)
        rng = np.random.default_rng(self.seed + fold_index)
        output = []
        for _, recordings in sorted(by_subject.items()):
            ordered = sorted(recordings, key=lambda item: item.dataset_id)
            count = max(1, math.ceil(len(ordered) * self.screening_fraction))
            indices = sorted(rng.choice(len(ordered), size=count, replace=False).tolist())
            output.extend(ordered[index] for index in indices)
        return output
