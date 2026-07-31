"""Diagnostic report processors for trigger/acquisition sanity checks."""

from __future__ import annotations

import numpy as np
from loguru import logger

from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor


@register_processor
class AnalyzeDataReport(Processor):
    """Generate a FACETpy-style data analysis summary.

    This mirrors the purpose of MATLAB FACET ``AnalyzeData`` while keeping
    output in FACETpy's structured metadata + concise logger format.

    The generated report is stored in
    ``metadata.custom["analyze_data_report"]``.
    """

    name = "analyze_data_report"
    description = "Create structured recording and trigger summary report"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = True

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw()
        sfreq = float(raw.info["sfreq"])
        n_samples = int(raw.n_times)
        n_channels = int(len(raw.ch_names))
        duration_s = n_samples / sfreq if sfreq > 0 else 0.0

        report = {
            "samples": n_samples,
            "sampling_rate_hz": sfreq,
            "duration_s": duration_s,
            "channels": n_channels,
            "channel_names": raw.ch_names.copy(),
        }

        if context.has_triggers():
            triggers = np.asarray(context.get_triggers(), dtype=int)
            report["num_triggers"] = int(len(triggers))
            report["first_trigger_sample"] = int(triggers[0])
            report["last_trigger_sample"] = int(triggers[-1])

            if len(triggers) > 1:
                diffs = np.diff(triggers)
                report["trigger_distance"] = {
                    "min": int(np.min(diffs)),
                    "max": int(np.max(diffs)),
                    "mean": float(np.mean(diffs)),
                    "std": float(np.std(diffs)),
                    "median": float(np.median(diffs)),
                }
                vals, counts = np.unique(diffs, return_counts=True)
                hist = sorted(
                    [{"distance_samples": int(v), "count": int(c)} for v, c in zip(vals, counts, strict=False)],
                    key=lambda x: x["count"],
                    reverse=True,
                )
                report["trigger_distance_histogram"] = hist[:10]

        logger.info(
            "AnalyzeDataReport: samples={}, sfreq={:.3f}Hz, duration={:.2f}s, channels={}",
            n_samples,
            sfreq,
            duration_s,
            n_channels,
        )
        if context.has_triggers():
            logger.info("AnalyzeDataReport: triggers={}", report["num_triggers"])

        metadata = context.metadata.copy()
        metadata.custom[self.name] = report
        return context.with_metadata(
            metadata,
            copy_estimated_noise=False,
        )


@register_processor
class CheckDataReport(Processor):
    """Run FACETpy-style data checks and emit a structured report.

    This mirrors MATLAB FACET ``CheckData`` intent and stores results in
    ``metadata.custom["check_data_report"]``.

    Parameters
    ----------
    require_triggers : bool, optional
        If ``True``, missing triggers are treated as an error (default: True).
    strict : bool, optional
        If ``True``, raise ``ProcessorValidationError`` when checks fail
        (default: True).
    """

    name = "check_data_report"
    description = "Run trigger/data sanity checks and store report"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = True

    def __init__(self, require_triggers: bool = True, strict: bool = True) -> None:
        self.require_triggers = require_triggers
        self.strict = strict
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw()
        errors: list[str] = []
        warnings: list[str] = []

        if raw.n_times < 2:
            errors.append("Raw has fewer than 2 samples.")
        if raw.info["sfreq"] <= 0:
            errors.append("Sampling frequency must be positive.")

        if self.require_triggers:
            if not context.has_triggers():
                errors.append("No triggers available.")
            else:
                self._check_triggers(context, errors, warnings)
        elif context.has_triggers():
            self._check_triggers(context, errors, warnings)

        report = {
            "strict": self.strict,
            "require_triggers": self.require_triggers,
            "num_errors": len(errors),
            "num_warnings": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }

        if errors:
            logger.error("CheckDataReport failed with {} error(s): {}", len(errors), "; ".join(errors))
            if self.strict:
                raise ProcessorValidationError("CheckDataReport failed: " + "; ".join(errors))
        if warnings:
            logger.warning("CheckDataReport found {} warning(s): {}", len(warnings), "; ".join(warnings))
        if not errors and not warnings:
            logger.info("CheckDataReport: all checks passed")

        metadata = context.metadata.copy()
        metadata.custom[self.name] = report
        return context.with_metadata(
            metadata,
            copy_estimated_noise=False,
        )

    def _check_triggers(self, context: ProcessingContext, errors: list[str], warnings: list[str]) -> None:
        triggers = np.asarray(context.get_triggers(), dtype=int)
        if len(triggers) == 0:
            errors.append("Trigger array is empty.")
            return

        if np.any(triggers < 0):
            errors.append("Trigger array contains negative sample indices.")
        if np.any(triggers >= context.get_raw().n_times):
            errors.append("Trigger array contains indices outside raw range.")
        if np.any(np.diff(triggers) <= 0):
            errors.append("Trigger array is not strictly increasing.")

        if context.get_artifact_length() is None or context.get_artifact_length() <= 0:
            warnings.append("artifact_length is missing or non-positive.")

        if len(triggers) > 1:
            diffs = np.diff(triggers)
            if np.std(diffs) > 0.02 * max(np.mean(diffs), 1.0):
                warnings.append("High trigger-distance variability detected (>2% of mean).")
