"""
Memory-efficient pipeline using channel-sequential execution.

``pipeline.run(channel_sequential=True)`` groups consecutive processors that
declare ``channel_wise = True`` (or ``run_once = True``) into a single
per-channel pass.  For each channel the full sequence executes before moving
to the next::

    ch 1 → HighPassFilter → UpSample → TriggerAligner → AAS → DownSample → store
    ch 2 → HighPassFilter → UpSample → (skip aligner) → AAS → DownSample → store
    …

Because TriggerAligner has ``run_once = True`` it executes only for the first
channel (updating trigger metadata) and is silently skipped for the rest.

The output array is pre-allocated at the final sfreq so the 10× upsampled
data never exists for all channels simultaneously.

Concepts used
-------------
channel_wise = True  Processor can run on a single-channel context.
run_once     = True  In channel_sequential mode: run for the first channel
                     only, skip for subsequent channels.

These flags have no relation to multiprocessing.
"""

from pathlib import Path

from facet import (
    AASCorrection,
    DownSample,
    EDFExporter,
    HighPassFilter,
    Loader,
    LowPassFilter,
    MetricsReport,
    Pipeline,
    RMSCalculator,
    TriggerAligner,
    TriggerDetector,
    UpSample,
)

INPUT_FILE = "./examples/datasets/NiazyFMRI.edf"
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = str(OUTPUT_DIR / "corrected_memory_efficient.edf")
TRIGGER_REGEX = r"\b1\b"
UPSAMPLE = 10

pipeline = Pipeline(
    [
        # --- serial pre-steps (channel_wise=False) ---------------------------
        Loader(path=INPUT_FILE, preload=True, artifact_to_trigger_offset=-0.005),
        TriggerDetector(regex=TRIGGER_REGEX),
        # --- channel_sequential batch (channel_wise=True) --------------------
        # All six steps below run as one per-channel pass.
        # TriggerAligner has run_once=True: executes on ch 1, skipped for ch 2…N.
        HighPassFilter(freq=1.0),
        UpSample(factor=UPSAMPLE),
        TriggerAligner(ref_trigger_index=0),  # run_once=True — fires once, then skips
        AASCorrection(window_size=30, correlation_threshold=0.975),
        DownSample(factor=UPSAMPLE),
        LowPassFilter(freq=70),
        # --- serial post-steps (channel_wise=False) --------------------------
        RMSCalculator(),
        MetricsReport(),
        EDFExporter(path=OUTPUT_FILE, overwrite=True),
    ],
    name="Memory-efficient AAS",
)

result = pipeline.run(channel_sequential=True)
result.print_summary()
