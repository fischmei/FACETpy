"""Immutable disk-backed acquisition chunks, prepared one channel at a time."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import mne
import numpy as np
import pandas as pd
from loguru import logger

from facet import Flex, Loader, MotionEpochMetadata, TriggerDetector
from facet.core import ProcessingContext, ProcessingMetadata
from facet.correction.flex.decisions import (
    AveragingMatrixBuilder,
    CandidateScoringMode,
    MatrixMetadata,
    pearson_scores,
    select_scored_candidates,
)
from facet.helpers.utils import split_vector
from facet.optimization.cache import digest
from facet.optimization.cleaning import (
    DEFAULT_CLEANING,
    clean_after_flex,
    good_eeg_names,
    validate_cleaning_input,
)
from facet.optimization.metrics import (
    BANDS,
    MIN_PRESERVATION_COVERAGE,
    original_spectrum,
    spectral_metrics,
    trigger_locked_metrics,
    validate_clean_band_coverage,
)
from facet.optimization.reference import compare_chunk, reference_plan

PREPARATION_VERSION = 4


def atomic_json(path, payload):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(
            payload, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)
        )
    )
    os.replace(temporary, path)


def worker_limit(requested, memory_mb, job_mb):
    """Reserve one job's working set per process; refuse unbounded allocation."""
    if min(requested, memory_mb, job_mb) <= 0:
        raise ValueError("Workers, memory and estimated working set must be positive.")
    if job_mb > memory_mb:
        raise MemoryError(f"One channel requires about {job_mb:.1f} MiB; budget is {memory_mb:.1f} MiB.")
    return min(requested, os.cpu_count() or 1, max(1, int(memory_mb // job_mb)))


def chunk_intervals(triggers, n_samples, sfreq, settings):
    """Deterministic nonoverlapping cores, padded for correction context.

    Split acquisition gaps before making bounded duration cores. Full fidelity
    covers every core, including a short trailing core; no trigger section is
    silently discarded because it is smaller than the preferred chunk size.
    """
    triggers = np.asarray(triggers, dtype=int)
    if len(triggers) < 2 or np.any(np.diff(triggers) <= 0):
        raise ValueError("At least two strictly increasing scanner triggers are required.")
    period = int(np.median(np.diff(triggers)))
    threshold = settings.chunk_gap_seconds or max(5.0, 5 * period / sfreq)
    groups = np.split(triggers, np.flatnonzero(np.diff(triggers) > threshold * sfreq) + 1)
    width = max(int(settings.chunk_seconds * sfreq), period * settings.chunk_min_triggers)
    padding = int(settings.chunk_padding_seconds * sfreq)
    chunks = []
    for group in groups:
        start, stop = max(0, int(group[0])), min(n_samples, int(group[-1]) + period)
        boundaries = list(range(start, stop, width)) + [stop]
        if len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < min(4 * sfreq, width):
            boundaries.pop(-2)
        for core_start, core_stop in zip(boundaries, boundaries[1:], strict=False):
            left, right = max(0, core_start - padding), min(n_samples, core_stop + padding)
            epoch_indices = np.flatnonzero((triggers >= left) & (triggers < right))
            if len(epoch_indices) < 2:
                raise ValueError("A bounded chunk has fewer than two triggers; increase --chunk-padding-seconds.")
            chunks.append((left, right, core_start, core_stop, epoch_indices))
    return chunks


def subset_indices(size, count):
    return np.unique(np.linspace(0, size - 1, min(size, count), dtype=int)).tolist()


def match_reference(dataset, reference_directory, input_directory, recursive):
    if reference_directory is None:
        return None
    relative = dataset.path.relative_to(input_directory)
    exact = reference_directory / relative
    if exact.exists():
        return exact
    iterator = reference_directory.rglob("*") if recursive else reference_directory.glob("*")
    from facet.io.loaders import SUPPORTED_EXTENSIONS

    def recording_key(path):
        # Processing suffixes and formats may differ; BIDS acquisition entities
        # must identify the same subject, session, task and run.
        entities = [part for part in path.name.split("_") if part.startswith(("sub-", "ses-", "task-", "run-", "acq-"))]
        return tuple(sorted(entities)) if any(p.startswith("sub-") for p in entities) else path.stem

    matches = [
        p
        for p in iterator
        if recording_key(p) == recording_key(dataset.path)
        and (p.suffix.lower() in SUPPORTED_EXTENSIONS or p.name.endswith(".fif.gz"))
    ]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous reference match for {dataset.dataset_id}: {matches}")
    return matches[0] if matches else None


def validate_reference(raw, reference, *, assume_aligned=False):
    if float(raw.info["sfreq"]) != float(reference.info["sfreq"]):
        raise ValueError("Reference sampling frequency differs; resample and align explicitly before optimization.")
    if raw.n_times != reference.n_times or raw.first_samp != reference.first_samp:
        raise ValueError("Reference sample interval differs; align explicitly before optimization.")
    names = [raw.ch_names[p] for p in mne.pick_types(raw.info, eeg=True, exclude="bads")]
    if not set(names).issubset(reference.ch_names):
        raise ValueError("Reference is missing required EEG channels.")
    for name in names:
        channel = reference.info["chs"][reference.ch_names.index(name)]
        source = raw.info["chs"][raw.ch_names.index(name)]
        if channel["unit"] != source["unit"] or channel["unit"] != 107:  # FIFF_UNIT_V
            raise ValueError(f"Unvalidated reference EEG units for {name}.")
    original_date, reference_date = raw.info["meas_date"], reference.info["meas_date"]
    if original_date is None or reference_date is None:
        if not assume_aligned:
            raise ValueError(
                "Reference timestamps are missing; use --assume-reference-aligned only after checking sample alignment."
            )
    elif original_date != reference_date:
        raise ValueError("Reference recording timestamp differs.")


def prepare_dataset(dataset, settings, root: Path):
    """Open one source lazily; write reusable float64 .npy arrays, never EEG exports."""
    directory = root / digest(
        {
            "dataset": dataset.signature_payload(),
            "settings": settings.evaluation_signature_payload(),
            "preparation_version": PREPARATION_VERSION,
        }
    )
    marker = directory / "prepared.json"
    if marker.exists():
        if settings.evaluation_mode != "non-reference" and not json.loads(marker.read_text())["reference_available"]:
            raise ValueError(f"Validated reference unavailable for {dataset.dataset_id}.")
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    raw = Loader(str(dataset.path), preload=False).execute(None).get_raw()
    reference = None
    try:
        plan = None
        full_cleaning = settings.evaluation_mode != "non-reference"
        if full_cleaning:
            validate_cleaning_input(raw)
        if dataset.reference_path is not None and full_cleaning:
            reference = Loader(str(dataset.reference_path), preload=False).execute(None).get_raw()
            try:
                plan = reference_plan(
                    raw,
                    reference,
                    dataset.reference_path,
                    assume_aligned=settings.assume_reference_aligned,
                    raw_path=dataset.path,
                )
            except ValueError:
                if settings.evaluation_mode != "non-reference":
                    raise
                logger.exception("Reference validation failed; non-reference correction remains available")
                reference.close()
                reference = None
        sfreq = float(raw.info["sfreq"])
        names = good_eeg_names(raw)
        eeg_names = list(names)
        if full_cleaning:
            ecg_picks = mne.pick_types(raw.info, ecg=True, exclude="bads")
            ecg_names = {raw.ch_names[p] for p in ecg_picks}
            names += [
                name
                for name in raw.ch_names
                if name not in names
                and name not in raw.info["bads"]
                and (name.upper() in {"ECG", "EKG"} or name in ecg_names)
            ]
        picks = [raw.ch_names.index(name) for name in names]
        if not picks:
            raise ValueError("No good EEG channels.")
        artifact = TriggerDetector(settings.trigger_regex)._compute_artifact_metadata(dataset.trigger_samples)
        intervals = chunk_intervals(dataset.trigger_samples, raw.n_times, sfreq, settings)
        manifest = {
            "sfreq": sfreq,
            "channels": [raw.ch_names[p] for p in picks],
            "eeg_channels": eeg_names,
            "artifact_length": artifact["artifact_length"] * settings.upsample_factor,
            "slices_per_volume": dataset.slices_per_volume,
            "chunks": [],
            "reference_available": reference is not None,
            "reference_plan": plan,
            "fixed_cleaning": DEFAULT_CLEANING.manifest() if full_cleaning else None,
            "preparation_version": PREPARATION_VERSION,
        }
        coverage_rows = []
        for index, (start, stop, core_start, core_stop, epoch_indices) in enumerate(intervals):
            if full_cleaning:
                # Integer-second padded origins share an exact grid with the
                # fixed target sampling rate and the reference resampling.
                start = int(np.floor(start / sfreq) * sfreq)
                epoch_indices = np.flatnonzero((dataset.trigger_samples >= start) & (dataset.trigger_samples < stop))
            n_epochs = len(epoch_indices)
            # Conservative allowance includes FFT resampling, epochs, template,
            # copy for subtraction, CSR worst case, and Python row lists.
            job_mb = estimated_job_mb(stop - start, settings.upsample_factor, n_epochs)
            if full_cleaning:
                job_mb += (stop - start) * len(raw.ch_names) * 8 * 12 / 1024**2
            worker_limit(1, settings.memory_mb, job_mb)
            triggers = (dataset.trigger_samples[epoch_indices] - start) * settings.upsample_factor
            chunk = {
                "start": start,
                "stop": stop,
                "core_start": core_start - start,
                "core_stop": core_stop - start,
                "triggers": triggers.tolist(),
                "epoch_indices": epoch_indices.tolist(),
                "job_mb": job_mb,
            }
            manifest["chunks"].append(chunk)
            folder = directory / str(index)
            folder.mkdir(exist_ok=True)
            for channel, pick in enumerate(picks):
                data = raw.get_data(picks=[pick], start=start, stop=stop)[0]
                original = data[core_start - start : core_stop - start]
                if raw.ch_names[pick] in eeg_names:
                    diagnostics = _save_original_metrics(
                        folder,
                        channel,
                        original,
                        sfreq,
                        dataset.trigger_samples[epoch_indices] - core_start,
                        artifact["artifact_length"],
                        settings,
                        f"dataset '{dataset.dataset_id}', chunk {index}, channel '{raw.ch_names[pick]}'",
                    )
                    for band, item in diagnostics.items():
                        coverage_rows.append(
                            dict(
                                dataset_id=dataset.dataset_id,
                                chunk=index,
                                channel=raw.ch_names[pick],
                                channel_index=channel,
                                band=band,
                                duration_seconds=(core_stop - core_start) / sfreq,
                                valid_measurements=int(item["valid"]),
                                invalid_measurements=int(not item["valid"]),
                                **{key: value for key, value in item.items() if key != "mask"},
                            )
                        )
                signal = mne.io.RawArray(
                    data[None, :], mne.create_info([raw.ch_names[pick]], sfreq, "eeg"), verbose=False
                )
                if settings.upsample_factor != 1:
                    signal.resample(
                        sfreq * settings.upsample_factor, npad="auto", window="boxcar", n_jobs=1, verbose=False
                    )
                upsampled = signal._data[0]
                np.save(folder / f"{channel}_signal.npy", upsampled)
                epochs = split_vector(upsampled - np.mean(upsampled), triggers, manifest["artifact_length"])
                if not np.all(np.isfinite(epochs)):
                    raise ValueError("Non-finite epoch data.")
                np.save(folder / f"{channel}_epochs.npy", epochs)
                del signal, upsampled, epochs, data
        manifest["preservation_coverage"] = preservation_coverage_report(coverage_rows, manifest, settings, directory)
        atomic_json(marker, manifest)  # Commit only a fully prepared dataset.
    finally:
        raw.close()
        if reference is not None:
            reference.close()
    return directory


def _save_original_metrics(folder, channel, original, sfreq, triggers, artifact_length, settings, label):
    frequencies, psd, mask = original_spectrum(original, sfreq, triggers, settings)
    diagnostics = validate_clean_band_coverage(
        frequencies,
        psd,
        mask,
        context=label,
    )
    original_locked = trigger_locked_metrics(original, triggers, artifact_length)
    np.savez(
        folder / f"{channel}_spectrum.npz",
        frequencies=frequencies,
        psd=psd,
        mask=mask,
        band_valid=np.array([diagnostics[band]["valid"] for band in BANDS]),
        original_rms=np.sqrt(np.mean(original**2)),
        original_trigger_locked_rms=original_locked["trigger_locked_rms"],
    )
    for item in diagnostics.values():
        item["welch_resolution_hz"] = float(frequencies[1] - frequencies[0])
    return diagnostics


def evaluation_indices(manifest, settings, fidelity):
    """One original-defined selection shared by coverage checks and every trial."""
    chunks = list(range(len(manifest["chunks"])))
    channels = list(range(len(manifest["channels"])))
    multiplier = {"screening": 1, "expanded": 2}.get(fidelity)
    if multiplier is not None:
        chunks = subset_indices(len(chunks), settings.screening_sections * multiplier)
        if settings.evaluation_mode == "non-reference":
            channels = subset_indices(len(channels), settings.screening_channels * multiplier)
    return chunks, channels


def preservation_coverage_report(rows, manifest, settings, directory):
    """Require majority channel-time support in every band at every fidelity."""
    from facet.optimization.search import _atomic_write_csv

    _atomic_write_csv(pd.DataFrame(rows), directory / "clean_band_coverage.csv")
    summary, failures = [], []
    for fidelity in ("screening", "expanded", "full"):
        chunks, channels = evaluation_indices(manifest, settings, fidelity)
        for band in BANDS:
            selected = [
                row
                for row in rows
                if row["chunk"] in chunks and row["channel_index"] in channels and row["band"] == band
            ]
            total = sum(row["duration_seconds"] for row in selected)
            valid = [row for row in selected if row["valid"]]
            fraction = sum(row["duration_seconds"] for row in valid) / total if total else 0.0
            summary.append(
                dict(
                    fidelity=fidelity,
                    band=band,
                    coverage=fraction,
                    valid_measurements=len(valid),
                    invalid_measurements=len(selected) - len(valid),
                    total_channel_seconds=total,
                )
            )
            if fraction < MIN_PRESERVATION_COVERAGE:
                failures.append(
                    f"{fidelity}/{band}: {fraction:.1%} valid channel-time ({len(valid)}/{len(selected)} measurements)"
                )
    payload = {"minimum_channel_time_coverage": MIN_PRESERVATION_COVERAGE, "summary": summary}
    atomic_json(directory / "preservation_coverage.json", payload)
    logger.info(
        "Original preservation coverage for {}: {} valid / {} unavailable measurements; reports in {}",
        rows[0]["dataset_id"] if rows else "unknown",
        sum(row["valid"] for row in rows),
        sum(not row["valid"] for row in rows),
        directory,
    )
    if failures:
        raise ValueError(
            f"Clean-band coverage preflight failed for dataset {rows[0]['dataset_id'] if rows else 'unknown'}: "
            + "; ".join(failures)
            + f". Require at least {MIN_PRESERVATION_COVERAGE:.0%} valid channel-time per band and fidelity; "
            f"see {directory / 'clean_band_coverage.csv'}. Individual measurements still require 25% clean bins, "
            "two bins and finite positive original power. No trials started."
        )
    return payload


def estimated_job_mb(samples, factor, epochs):
    # Includes interpreter/scientific stack reserve per worker, signal working
    # arrays and worst-case dense-selection CSR construction Python objects.
    return 256 + samples * factor * 8 * 16 / 1024**2 + epochs**2 * 96 / 1024**2


class CachedBuilder(AveragingMatrixBuilder):
    """Reuse signed candidate correlations across thresholds, k and kernels.

    Candidate scoring is bounded to 128 epochs per temporary array. No global
    pairwise correlation matrix is materialized. Cache rows are transactional.
    """

    def __init__(self, decisions, cache, identity):
        super().__init__(decisions)
        self.cache, self.identity = cache, identity

    def _cumulative_motion_distances(self, target_idx, candidate_indices, motion):
        identity = self.identity.rsplit("/", 1)[0]
        key = digest((identity, "cumulative_motion"))

        def compute():
            vectors = self._motion_vectors(motion)
            with np.errstate(over="ignore", invalid="ignore"):
                increments = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
            valid = np.isfinite(increments)
            distance = np.r_[0.0, np.cumsum(np.where(valid, increments, 0.0))]
            invalid = np.r_[0, np.cumsum(~valid, dtype=int)]
            finite = np.all(np.isfinite(vectors), axis=1)
            return np.concatenate((distance, invalid, finite))

        distance, invalid, finite = np.split(self.cache.calculation(key, compute), 3)
        starts, stops = np.minimum(target_idx, candidate_indices), np.maximum(target_idx, candidate_indices)
        valid = (invalid[stops] == invalid[starts]) & (finite[target_idx] > 0) & (finite[candidate_indices] > 0)
        return np.where(valid, distance[stops] - distance[starts], np.inf)

    def _motion_distances(self, target_idx, candidate_indices, motion):
        key = digest((self.identity.rsplit("/", 1)[0], "motion_distances", target_idx, candidate_indices.tolist()))
        return self.cache.calculation(
            key, lambda: AveragingMatrixBuilder._motion_distances(target_idx, candidate_indices, motion)
        )

    def _selected_candidates(self, *, epochs, target_idx, candidate_indices, metadata, motion):
        scoring = self.decisions.scoring
        if scoring.mode not in {CandidateScoringMode.SIGNED_PEARSON, CandidateScoringMode.ABSOLUTE_PEARSON}:
            return super()._selected_candidates(
                epochs=epochs,
                target_idx=target_idx,
                candidate_indices=candidate_indices,
                metadata=metadata,
                motion=motion,
            )
        key = digest((self.identity, target_idx, candidate_indices.tolist()))

        def compute():
            return np.concatenate(
                [
                    pearson_scores(epochs[target_idx], epochs[block])
                    for block in np.array_split(candidate_indices, max(1, (len(candidate_indices) + 127) // 128))
                ]
            )

        values = self.cache.calculation(key, compute)
        if scoring.mode == CandidateScoringMode.ABSOLUTE_PEARSON:
            values = np.where(np.isfinite(values), np.abs(values), values)
        return select_scored_candidates(
            candidate_indices,
            values,
            target_idx=target_idx,
            scoring=scoring,
            template_size=self.decisions.template_size,
        )


def evaluate_prepared(decisions, dataset, settings, directory, cache, fidelity, diagnostics_directory=None):
    manifest = json.loads((directory / "prepared.json").read_text())
    chunk_ids, channel_ids = evaluation_indices(manifest, settings, fidelity)
    full_cleaning = settings.evaluation_mode != "non-reference"
    sfreq, factor = manifest["sfreq"], settings.upsample_factor
    chunk_rows, weights = [], []
    flex = Flex(
        matrix_decisions=decisions,
        realign_after_averaging=True,
        interpolate_volume_gaps=True,
        apply_epoch_alpha_scaling=True,
        track_estimated_noise=False,
    )
    for index in chunk_ids:
        chunk = manifest["chunks"][index]
        folder = directory / str(index)
        triggers = np.asarray(chunk["triggers"], dtype=int)
        chunk_key = digest((directory.name, index, decisions.to_dict(), "full_cleaning"))
        if full_cleaning:
            completed = cache.get_unit(chunk_key)
            if completed is not None and diagnostics_directory is None:
                chunk_rows.append(completed["metrics"])
                weights.append(chunk["core_stop"] - chunk["core_start"])
                continue
            source = Loader(str(dataset.path), preload=False).execute(None).get_raw()
            try:
                corrected_raw = source.copy().crop(chunk["start"] / sfreq, (chunk["stop"] - 1) / sfreq).load_data()
            finally:
                source.close()
        motion = None
        if dataset.motion is not None:
            source = dataset.motion
            motion = MotionEpochMetadata(
                parameters=source.parameters,
                segment_ids=source.segment_ids,
                stable=source.stable,
                epoch_to_motion_index=source.epoch_to_motion_index[chunk["epoch_indices"]],
                rotation_scale=source.rotation_scale,
            )
        metadata = MatrixMetadata(
            trigger_times=triggers / (sfreq * factor), slices_per_volume=dataset.slices_per_volume, motion=motion
        )
        shared_matrix = None
        ecg_matrix = None
        aligned = None
        channel_rows = []
        for channel in channel_ids:
            unit_key = digest((directory.name, index, channel, decisions.to_dict()))
            unit = cache.get_unit(unit_key)
            if unit is not None and not full_cleaning:
                if aligned is None:
                    aligned = np.asarray(unit["aligned"], dtype=int)
                channel_rows.append(unit["metrics"])
                continue
            epochs = np.load(folder / f"{channel}_epochs.npy", mmap_mode="r")
            if full_cleaning and manifest["channels"][channel] not in manifest["eeg_channels"]:
                # Select scanner neighbors using EEG, not cardiac similarity:
                # scoring ECG itself can retain only the target and erase QRS.
                matrix = ecg_matrix
            elif shared_matrix is None:
                builder = CachedBuilder(decisions, cache, f"{directory.name}/{index}/{channel}")
                matrix = builder.build_sparse(epochs, metadata)
                if decisions.scoring.mode in {CandidateScoringMode.NONE, CandidateScoringMode.TEMPORAL_MOTION_COST}:
                    shared_matrix = matrix
            else:
                matrix = shared_matrix
            if ecg_matrix is None:
                ecg_matrix = matrix
            templates = matrix @ epochs
            # One writable channel copy; other recordings remain on disk.
            signal = np.array(np.load(folder / f"{channel}_signal.npy", mmap_mode="r"))
            raw = mne.io.RawArray(signal[None, :], mne.create_info(["EEG"], sfreq * factor, "eeg"), verbose=False)
            if aligned is None:
                aligned = flex._get_aligned_triggers(
                    raw, {0: matrix}, [templates], triggers, 0.0, manifest["artifact_length"], sfreq * factor, factor
                )
            flex._remove_artifacts(
                raw,
                {0: matrix},
                [templates],
                aligned,
                0,
                manifest["artifact_length"],
                track_estimated_noise=False,
                interpolate_untracked=True,
            )
            if factor != 1:
                raw.resample(sfreq, npad="auto", window="boxcar", n_jobs=1, verbose=False)
            if full_cleaning:
                # Scanner-correct ECG as well as EEG before QRS detection.
                # ECG is excluded from PCA, cardiac AAS and EEG metrics.
                pick = corrected_raw.ch_names.index(manifest["channels"][channel])
                corrected_raw._data[pick] = raw._data[0]
                if manifest["channels"][channel] not in manifest["eeg_channels"]:
                    del raw, signal, epochs, templates
                    continue
            corrected = raw._data[0, chunk["core_start"] : chunk["core_stop"]]
            with np.load(folder / f"{channel}_spectrum.npz") as spectrum:
                metrics = spectral_metrics(
                    corrected,
                    sfreq,
                    spectrum["frequencies"],
                    spectrum["psd"],
                    spectrum["mask"],
                    band_valid=spectrum["band_valid"],
                )
            locked = trigger_locked_metrics(
                corrected, triggers // factor - chunk["core_start"], manifest["artifact_length"] // factor
            )
            metrics.update(locked)
            with np.load(folder / f"{channel}_spectrum.npz") as spectrum:
                original_locked = float(spectrum["original_trigger_locked_rms"])
                metrics["original_signal_rms"] = float(spectrum["original_rms"])
                metrics["trigger_locked_power_ratio"] = (
                    (locked["trigger_locked_rms"] / original_locked) ** 2 if original_locked > 0 else float("nan")
                )
            cache.add_unit(unit_key, metrics, aligned)
            channel_rows.append(metrics)
            del raw, signal, epochs, templates
        metrics = aggregate_metrics(channel_rows)
        if full_cleaning:
            metadata = ProcessingMetadata(
                triggers=aligned // factor, artifact_length=manifest["artifact_length"] // factor, upsampling_factor=1
            )
            cleaned = clean_after_flex(ProcessingContext(corrected_raw, metadata=metadata)).get_raw()
            reference = Loader(str(dataset.reference_path), preload=False).execute(None).get_raw()
            try:
                comparison, preview = compare_chunk(
                    cleaned,
                    reference,
                    manifest["reference_plan"],
                    raw_start=chunk["start"] / sfreq,
                    core_start=(chunk["start"] + chunk["core_start"]) / sfreq,
                    core_stop=(chunk["start"] + chunk["core_stop"]) / sfreq,
                )
            finally:
                reference.close()
            if diagnostics_directory is not None and index == chunk_ids[0]:
                from facet.optimization.validation import comparison_plots

                comparison_plots(
                    dataset, corrected_raw, preview, diagnostics_directory, chunk_start=chunk["start"] / sfreq
                )
            metrics.update(comparison)
            cache.add_unit(chunk_key, metrics, aligned)
            del corrected_raw, cleaned
        chunk_rows.append(metrics)
        weights.append(chunk["core_stop"] - chunk["core_start"])
    result = aggregate_metrics(chunk_rows, np.asarray(weights) / sfreq)
    channel_seconds = (
        sum(weights) / sfreq * sum(manifest["channels"][channel] in manifest["eeg_channels"] for channel in channel_ids)
    )
    for band in BANDS:
        result[f"{band}_measurement_coverage"] = result[f"{band}_preservation_weight"] / channel_seconds
    if full_cleaning:
        result["reference_valid_seconds"] = sum(row["reference_valid_seconds"] for row in chunk_rows)
        result["reference_evaluated_seconds"] = sum(weights) / sfreq
        result["reference_evaluation_coverage"] = (
            result["reference_valid_seconds"] / result["reference_evaluated_seconds"]
        )
    result.update(
        n_chunks=len(chunk_ids),
        successful_chunks=len(chunk_ids),
        n_channels=sum(manifest["channels"][channel] in manifest["eeg_channels"] for channel in channel_ids),
        reference_available=manifest["reference_available"],
    )
    # Preparation fixes eligibility. Candidate-dependent loss is fatal rather
    # than silently shrinking a preservation denominator.
    required = ["nonpeak_eeg_log_deviation", *(f"{band}_preservation" for band in BANDS)]
    if any(not np.isfinite(result[key]) for key in required):
        raise ValueError("No finite preservation summary on the fixed original-defined measurement set.")
    result["success"] = True
    return result


def aggregate_metrics(rows, weights=None):
    """Preserve fixed eligible channel-time denominators through both levels."""
    result = {}
    weights = np.ones(len(rows)) if weights is None else np.asarray(weights, dtype=float)
    for key in sorted({key for row in rows for key in row}):
        if key.endswith("_weight"):
            continue
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
        if key.endswith(("_valid_measurements", "_invalid_measurements")):
            result[key] = float(np.sum(values))
            continue
        if any(f"{key}_weight" in row for row in rows):
            fixed = np.asarray([row[f"{key}_weight"] for row in rows]) * weights
            eligible = fixed > 0
            if not np.all(np.isfinite(values[eligible])):
                raise ValueError(f"Candidate lost an eligible preservation measurement: {key}.")
            result[f"{key}_weight"] = float(fixed.sum())
            result[key] = (
                float(np.average(values[eligible], weights=fixed[eligible])) if eligible.any() else float("nan")
            )
            continue
        result[key] = float(np.average(values, weights=weights)) if np.all(np.isfinite(values)) else float("nan")
    if any(f"{band}_log_deviation_weight" in result for band in BANDS):
        deviations = [result[f"{band}_log_deviation"] for band in BANDS]
        result["nonpeak_eeg_log_deviation"] = float(np.mean(deviations))
    return result
