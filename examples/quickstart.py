"""Quickstart - minimal fMRI artifact correction pipeline.

The fewest steps needed to correct a bundled recording in trigger-section
chunks and export numbered EDF files. Chunking keeps large recordings from
being loaded into memory all at once.
"""

from __future__ import annotations

from pathlib import Path

from facet import (
    AASCorrection,
    DownSample,
    Pipeline,
    TriggerDetector,
    UpSample,
)

INPUT_FILE = Path("./examples/datasets/NiazyFMRI.edf")
OUTPUT_DIR = Path("./output/quickstart_chunks")

pipeline = Pipeline(
    [
        TriggerDetector(regex=r"\b1\b"),
        UpSample(factor=10),
        AASCorrection(window_size=30),
        DownSample(factor=10),
    ],
    name="Quickstart",
)

# Trigger-section chunking writes one padded output window per detected
# scan/trigger section, not one output file per individual trigger.
result = pipeline.run_chunked(
    input_path=str(INPUT_FILE),
    output_dir=str(OUTPUT_DIR),
    output_extension=".edf",
    trigger_section_padding_seconds=60.0,
    trigger_section_min_triggers=31,
    channel_sequential=True,
)

result.print_summary()
