"""Processing report writers for the FACETpy CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from facet import Pipeline

from ._cli_pipeline import (
    ADD_ON_MODE_DESCRIPTIONS,
    CORRECTION_MATRIX_DESCRIPTIONS,
    CORRECTION_MODE_DESCRIPTIONS,
    PROCESS_PATTERN_DESCRIPTIONS,
    _selected_add_on_modes,
)


def _processing_failure_record(input_path: Path, target_dir: Path, exc: Exception) -> dict[str, str]:
    """Build a serializable failure record for one input file."""
    return {
        "input_path": str(input_path),
        "output_dir": str(target_dir),
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }


def _write_processing_error(target_dir: Path, record: dict[str, str]) -> Path:
    """Write a per-recording processing error marker."""
    marker_path = target_dir / "processing_error.json"
    marker_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return marker_path


def _write_batch_failures(output_root: Path, failures: list[dict[str, str]]) -> Path | None:
    """Write a batch-level failure manifest when any inputs failed."""
    if not failures:
        return None

    manifest_path = output_root / "processing_failures.json"
    manifest_path.write_text(json.dumps({"failures": failures}, indent=2), encoding="utf-8")
    return manifest_path


def _processor_report(processor, index: int) -> dict:
    """Return a JSON-friendly processor description for process reports."""
    return {
        "index": index,
        "name": processor.name,
        "type": processor.__class__.__name__,
        "description": getattr(processor, "description", ""),
        "parameters": _json_safe(getattr(processor, "_parameters", {})),
    }


def _chunk_report(chunk, result) -> dict:
    """Return a compact result description for one processed chunk."""
    return {
        "index": int(chunk.index),
        "total": int(chunk.total),
        "start_sample": int(chunk.start_sample),
        "stop_sample": int(chunk.stop_sample),
        "duration_seconds": float(chunk.duration_seconds),
        "output_path": str(chunk.output_path),
        "success": bool(result.success),
        "error": None if result.error is None else str(result.error),
    }


def _chunked_results(chunked_result) -> list:
    """Return chunk results when the result object is iterable."""
    try:
        return list(chunked_result)
    except TypeError:
        return []


def _collect_artifact_template_reports(chunked_result) -> list[dict]:
    """Collect AAS-style template matrix reports from successful chunk contexts."""
    chunks = list(getattr(chunked_result, "chunks", []))
    reports: list[dict] = []

    for chunk, result in zip(chunks, _chunked_results(chunked_result), strict=False):
        context = getattr(result, "context", None)
        if context is None:
            continue

        custom = getattr(context.metadata, "custom", {})
        for report in custom.get("artifact_template_matrices", []):
            report_payload = _json_safe(report)
            report_payload.setdefault(
                "chunk",
                {
                    "index": int(chunk.index),
                    "total": int(chunk.total),
                    "start_sample": int(chunk.start_sample),
                    "stop_sample": int(chunk.stop_sample),
                    "output_path": str(chunk.output_path),
                },
            )
            reports.append(report_payload)

    return reports


def _write_artifact_template_matrix_report(target_dir: Path, input_path: Path, chunked_result) -> Path:
    """Write the per-recording artifact template matrix report."""
    reports = _collect_artifact_template_reports(chunked_result)
    report_path = target_dir / "artifact_template_matrices.json"
    payload = {
        "source_path": str(input_path),
        "description": (
            "AAS-style corrections build artifact templates with N = A @ D. "
            "D is the epoch data matrix, A is the averaging matrix, and N is the estimated artifact template matrix."
        ),
        "report_count": len(reports),
        "reports": reports,
    }
    if not reports:
        payload["note"] = "No AAS-style artifact template matrix report was produced for this run."

    report_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return report_path


def _write_pipeline_description(
    target_dir: Path,
    input_path: Path,
    args: argparse.Namespace,
    pipeline: Pipeline,
    chunked_result,
    matrix_report_path: Path,
) -> Path:
    """Write a per-recording JSON description of the process pipeline."""
    chunks = list(getattr(chunked_result, "chunks", []))
    results = _chunked_results(chunked_result)
    add_on_modes, selected_by_pattern = _selected_add_on_modes(args)
    payload = {
        "source_path": str(input_path),
        "output_dir": str(target_dir),
        "pattern": args.pattern,
        "pattern_description": PROCESS_PATTERN_DESCRIPTIONS.get(args.pattern),
        "correction_mode": args.correction_mode,
        "correction_mode_description": CORRECTION_MODE_DESCRIPTIONS.get(args.correction_mode),
        "correction_matrix_description": CORRECTION_MATRIX_DESCRIPTIONS.get(args.correction_mode),
        "add_on_modes": add_on_modes,
        "add_on_modes_selected_by_pattern": selected_by_pattern,
        "add_on_mode_descriptions": {mode: ADD_ON_MODE_DESCRIPTIONS[mode] for mode in add_on_modes},
        "channel_sequential": bool(args.channel_sequential),
        "process_options": {
            key: _json_safe(value)
            for key, value in vars(args).items()
            if key not in {"func", "input", "input_list", "input_dir"}
        },
        "processors": [
            _processor_report(processor, index)
            for index, processor in enumerate(pipeline.processors, 1)
        ],
        "result": {
            "success": bool(chunked_result.was_successful()),
            "execution_time_seconds": float(getattr(chunked_result, "execution_time", 0.0)),
            "chunks_manifest": (
                str(chunked_result.manifest_path)
                if getattr(chunked_result, "manifest_path", None) is not None
                else None
            ),
            "artifact_template_matrix_report": str(matrix_report_path),
            "chunks": [
                _chunk_report(chunk, result)
                for chunk, result in zip(chunks, results, strict=False)
            ],
        },
    }

    description_path = target_dir / "pipeline_description.json"
    description_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return description_path


def _json_safe(value):
    """Convert nested metadata values into JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value
