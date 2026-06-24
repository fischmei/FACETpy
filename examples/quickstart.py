"""
Quickstart — minimal fMRI artifact correction pipeline.

The fewest steps needed to load, correct, and export an EDF recording.
Run this first to verify your installation.
"""

from facet import (
    AASCorrection,
    DownSample,
    EDFExporter,
    Loader,
    Pipeline,
    TriggerDetector,
    UpSample,
)

# The SET source preserves the Status trigger channel used for AAS.
#INPUT_FILE  = "/home/cfischmei/projects/facetpy_data/rawdata/EEGfMRI20220801_20220801_163809.mff"
INPUT_FILE  = "./examples/datasets/NiazyFMRI.set"
OUTPUT_FILE = "./output/corrected_quickstart2.edf"

pipeline = Pipeline([
    Loader(path=INPUT_FILE, preload=True),
    TriggerDetector(regex=r"\b1\b"),
    UpSample(factor=10),
    AASCorrection(window_size=30),
    DownSample(factor=10),
    EDFExporter(path=OUTPUT_FILE, overwrite=True),
], name="Quickstart")

result = pipeline.run()
result.print_summary()
