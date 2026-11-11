"""Technical reference eligibility, fixed before any CV or candidate scoring."""

from dataclasses import replace
from math import lcm

import numpy as np
import pandas as pd
from loguru import logger

from facet import Loader
from facet.core import ProcessorError
from facet.optimization.cache import file_identity
from facet.optimization.cleaning import DEFAULT_CLEANING, validate_cleaning_input
from facet.optimization.cv import subject_id
from facet.optimization.prepared import match_reference
from facet.optimization.reference import alignment_path, reference_plan


def pair_recording(dataset, args):
    """Attach an unambiguous separate reference and content fingerprints."""
    reference = match_reference(dataset, args.reference_directory, args.input_directory, args.recursive)
    if reference is None and args.evaluation_mode != "non-reference":
        raise ValueError(f"Missing separately cleaned reference for {dataset.dataset_id}.")
    if reference is not None and reference.resolve() == dataset.path.resolve():
        raise ValueError(f"Reference and original must be separate recordings: {dataset.dataset_id}.")
    return replace(
        dataset,
        reference_path=reference,
        input_identity=file_identity(dataset.path),
        reference_identity=file_identity(reference) if reference is not None else None,
        alignment_identity=file_identity(alignment_path(reference))
        if reference is not None and alignment_path(reference).exists()
        else None,
    )


def inspect_reference_pair(dataset, settings):
    """Use existing fixed-cleaning prerequisites and provenance validation.

    No correction, EEG agreement metric, or parameter-dependent threshold is
    evaluated here. Input recordings are opened read-only and closed per pair.
    """
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
        # These are the same sampling-rate requirements used by compare_chunk;
        # fail here rather than only after a candidate has been processed.
        rates = DEFAULT_CLEANING.sfreq, float(reference.info["sfreq"])
        if any(not np.isfinite(rate) or rate <= 0 or rate != round(rate) for rate in rates):
            raise ValueError("Reference comparison requires positive integer sampling frequencies.")
        if lcm(*(int(rate) for rate in rates)) > 2000:
            raise ValueError("Sample-exact common comparison rate exceeds 2000 Hz.")
        details = {
            "raw_sfreq": float(raw.info["sfreq"]),
            "reference_sfreq": rates[1],
            "shared_channels": len(evidence["channels"]),
            "reference_coverage": evidence["coverage"],
        }
        return evidence, details
    finally:
        raw.close()
        if reference is not None:
            reference.close()


def preflight_reference_sessions(paths, settings, args):
    """Audit every session, then apply strict failure or explicit exclusion."""
    from facet.optimization.search import _atomic_write_csv, preflight_datasets, write_json

    rows, eligible, evidence_by_id = [], {}, {}
    # Expected reader/provenance errors are per-session failures. Environment,
    # allocation and programming assertions remain fatal, never cohort filters.
    invalid_input = (ValueError, OSError, RuntimeError, ProcessorError, TypeError, KeyError, IndexError, OverflowError)
    for identifier, path in paths.items():
        try:
            participant = subject_id(path)
        except ValueError:
            participant = None  # Grouped CLI checks identity separately and fails rather than falling back.
        row = {
            "dataset_id": identifier,
            "participant_id": participant,
            "session_id": next((part for part in path.stem.split("_") if part.startswith("ses-")), ""),
            "raw": str(path),
            "reference": "",
            "validation_status": "invalid",
            "eligible": False,
            "validation_stage": "raw_preflight",
            "error_type": "",
            "exclusion_reason": "",
        }
        try:
            dataset = preflight_datasets(
                {identifier: path},
                trigger_regex=args.trigger_regex,
                motion_directory=args.motion_directory,
                rotation_scale=args.rotation_scale,
            )[identifier]
            row["scanner_triggers"] = len(dataset.trigger_samples)
            row["validation_stage"] = "reference_pairing"
            dataset = pair_recording(dataset, args)
            row["reference"] = str(dataset.reference_path)
            row["validation_stage"] = "reference_compatibility"
            evidence, details = inspect_reference_pair(dataset, settings)
        except invalid_input as exc:
            row.update(error_type=type(exc).__name__, exclusion_reason=str(exc))
            logger.warning("Reference eligibility failed for {} ({}): {}", identifier, row["validation_stage"], exc)
        else:
            row.update(details, validation_status="valid", eligible=True, validation_stage="complete")
            eligible[identifier] = dataset
            evidence_by_id[identifier] = evidence
        row["automatically_excluded"] = bool(args.exclude_invalid_references and not row["eligible"])
        rows.append(row)
    valid = [row for row in rows if row["eligible"]]
    invalid = [row for row in rows if not row["eligible"]]
    summary = {
        "policy": "exclude_invalid_references" if args.exclude_invalid_references else "strict",
        "n_discovered_sessions": len(rows),
        "n_eligible_sessions": len(valid),
        "n_invalid_sessions": len(invalid),
        "n_excluded_sessions": len(invalid) if args.exclude_invalid_references else 0,
        "n_eligible_participants": len({row["participant_id"] for row in valid})
        if all(row["participant_id"] for row in valid)
        else None,
        "eligible_dataset_ids": list(eligible),
        "excluded_dataset_ids": [row["dataset_id"] for row in invalid] if args.exclude_invalid_references else [],
    }
    report = {"summary": summary, "sessions": rows}
    # Output failures are fatal, not a reason to exclude a valid input session.
    csv_path = args.output_directory / "reference_eligibility.csv"
    _atomic_write_csv(pd.DataFrame(rows), csv_path)
    write_json(args.output_directory / "reference_eligibility.json", report)
    for identifier, evidence in evidence_by_id.items():
        write_json(args.output_directory / f"{identifier}_reference_preflight.json", evidence)
    logger.info(
        "Reference eligibility: {}/{} sessions, {} identified participants",
        len(valid),
        len(rows),
        summary["n_eligible_participants"],
    )
    if invalid and not args.exclude_invalid_references:
        reasons = "\n".join(f"{row['dataset_id']}: {row['error_type']}: {row['exclusion_reason']}" for row in invalid)
        raise ValueError(
            f"Reference compatibility preflight failed in strict mode for {len(invalid)}/{len(rows)} sessions. "
            f"See {csv_path}; opt in with --exclude-invalid-references to retain valid sessions.\n{reasons}"
        )
    if not eligible:
        raise ValueError(f"No eligible sessions remain after reference compatibility preflight; see {csv_path}.")
    return eligible, report
