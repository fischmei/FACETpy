"""
Quickstart — minimal fMRI artifact correction pipeline.

The fewest steps needed to correct a recording in trigger-section chunks andexport numbered EDF files. Chunking keeps large recordings from being loaded
into memory all at once.
Run this first to verify your installation.
"""

from pathlib import Path

from facet import (
    AASCorrection,
    DownSample,
    DropChannelsMatching,
    Pipeline,
    TriggerDetector,
    UpSample,
)

# The SET source preserves the Status trigger channel used for AAS.
INPUT_FILE = "/home/cfischmei/projects/facetpy_data/rawdata/EEGfMRI20220801_20220801_163809.mff"
#INPUT_FILE  = "./examples/datasets/NiazyFMRI.set"
OUTPUT_DIR  = Path("./output/quickstart_chunks")
EGI_E1_TO_E128 = r"^E(?:[1-9]|[1-9]\d|1[01]\d|12[0-8])$"

pipeline = Pipeline([
    # Remove EGI numbered EEG channels on a copy, while keeping trigger and
    # auxiliary channels such as TREV, ECG, or respiration.
    DropChannelsMatching(regex=EGI_E1_TO_E128),
    TriggerDetector(regex=r"\b1\b"),
    UpSample(factor=10),
    AASCorrection(window_size=30),
    DownSample(factor=10),
], name="Quickstart")

# Fixed 3 cuts, kept for later testing:
# result = pipeline.run_chunked(
#     input_path=INPUT_FILE,
#     output_dir=str(OUTPUT_DIR),
#     output_extension=".edf",
#     min_chunks=3,
#     max_chunks=3,
#     chunk_by_trigger_sections=False,
#     channel_sequential=True,
# )

# Trigger-section chunks: one output per scan block, padded by 10 seconds
# before the first trigger and after the last trigger.
result = pipeline.run_chunked(
    input_path=INPUT_FILE,
    output_dir=str(OUTPUT_DIR),
    output_extension=".edf",
    trigger_section_padding_seconds=60.0,
    trigger_section_min_triggers=16,
    channel_sequential=True,
)

result.print_summary()
