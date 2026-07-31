Processors Guide
================

Processors are the building blocks of FACETpy. This guide covers all available processors
and how to use them effectively.

What is a Processor?
--------------------

A processor is a single processing step that:

- Takes a ``ProcessingContext`` as input
- Performs one specific operation
- Returns a new ``ProcessingContext`` as output
- Can validate prerequisites before processing
- Tracks its execution in the context history

.. code-block:: python

   from facet.preprocessing import HighPassFilter

   # Create processor
   hpf = HighPassFilter(freq=1.0)

   # Execute on context
   output_context = hpf.execute(input_context)

Processor Lifecycle
-------------------

When you call ``processor.execute(context)``:

1. **Validation** - Checks prerequisites (triggers, raw data, etc.)
2. **Processing** - Executes main logic
3. **History** - Adds entry to processing history
4. **Return** - Returns new context

.. code-block:: python

   # Automatic validation
   try:
       result = processor.execute(context)
   except ProcessorValidationError as e:
       print(f"Validation failed: {e}")

Live Progress Bars
------------------

The modern console can render a dedicated progress bar for any long-running
processor. Import the ``processor_progress`` helper and advance it as your work
completes. When the console runs in classic mode the helper becomes a no-op, so
it is safe to call regardless of user preferences.

.. code-block:: python

   from facet.console import processor_progress
   from facet.core import Processor

   class EpochAverager(Processor):
       name = "Average epochs"

       def execute(self, context):
           epochs = list(context.get_epochs())
           with processor_progress(total=len(epochs), message="Averaging") as prog:
               for epoch in epochs:
                   self._process_epoch(epoch)
                   prog.advance(message=f"Processed {prog.current:.0f} epochs")
           return context

``processor_progress`` also supports ``update(current=..., total=...)`` and
``complete()`` so you can report arbitrary metrics (samples, files, estimated
seconds, etc.). Combine these updates with your own loguru messages for a rich
view of what the processor is doing.

.. note::

   Processors with ``channel_wise = True`` (e.g. ``Filter`` or
   ``AASCorrection``) automatically emit channel-wise progress when the
   pipeline runs them in parallel or channel_sequential mode. For
   memory-sensitive workloads, ``channel_sequential=True`` is usually the
   preferred execution mode. For other workloads consider tracking epochs,
   files, or optimization iterations; anything you surface through
   ``processor_progress`` appears in the live console without affecting legacy
   logging.

Available Processors
--------------------

I/O Processors
~~~~~~~~~~~~~~

Loading Data
^^^^^^^^^^^^

**Loader** - Load EEG data with automatic format detection

.. code-block:: python

   from facet.io import Loader

   loader = Loader(
       path="data.edf",
       preload=True,  # Load into memory
   )
   context = loader.execute(None)

**BIDSLoader** - Load BIDS format data

.. code-block:: python

   from facet.io import BIDSLoader

   loader = BIDSLoader(
       root="/path/to/bids",
       subject="01",
       session="01",
       task="rest",
       run="01",
   )
   context = loader.execute(None)

Use ``run`` when your BIDS dataset has multiple recordings for the same
subject/session/task combination.

Exporting Data
^^^^^^^^^^^^^^

**EDFExporter** - Export to EDF format

.. code-block:: python

   from facet.io import EDFExporter

   exporter = EDFExporter(
       path="output.edf",
       overwrite=True
   )
   exporter.execute(context)

**BIDSExporter** - Export to BIDS format

.. code-block:: python

   from facet.io import BIDSExporter

   exporter = BIDSExporter(
       root="/path/to/bids",
       subject="01",
       session="01",
       task="rest"
   )
   exporter.execute(context)

Preprocessing Processors
~~~~~~~~~~~~~~~~~~~~~~~~~

Filtering
^^^^^^^^^

**HighPassFilter** - Remove slow drifts

.. code-block:: python

   from facet.preprocessing import HighPassFilter

   hpf = HighPassFilter(freq=0.5)  # 0.5 Hz cutoff
   context = hpf.execute(context)

**LowPassFilter** - Remove high-frequency noise

.. code-block:: python

   from facet.preprocessing import LowPassFilter

   lpf = LowPassFilter(freq=100.0)  # 100 Hz cutoff

**BandPassFilter** - Keep specific frequency range

.. code-block:: python

   from facet.preprocessing import BandPassFilter

   bpf = BandPassFilter(l_freq=0.5, h_freq=100.0)

**NotchFilter** - Remove specific frequencies (e.g., line noise)

.. code-block:: python

   from facet.preprocessing import NotchFilter

   notch = NotchFilter(freqs=[50, 100, 150])  # 50 Hz and harmonics

Resampling
^^^^^^^^^^

**UpSample** - Increase sampling rate

.. code-block:: python

   from facet.preprocessing import UpSample

   upsampler = UpSample(factor=10)  # 10x upsampling
   context = upsampler.execute(context)

**DownSample** - Decrease sampling rate

.. code-block:: python

   from facet.preprocessing import DownSample

   downsampler = DownSample(factor=10)  # 10x downsampling

**Resample** - Change to specific sampling rate

.. code-block:: python

   from facet.preprocessing import Resample

   resampler = Resample(sfreq=1000.0)  # Resample to 1000 Hz

Trigger Detection
^^^^^^^^^^^^^^^^^

**TriggerDetector** - Detect fMRI triggers using regex

.. code-block:: python

   from facet.preprocessing import TriggerDetector

   detector = TriggerDetector(
       regex=r"\b1\b",  # Pattern to match
   )
   context = detector.execute(context)

   # Check detected triggers
   triggers = context.get_triggers()
   print(f"Found {len(triggers)} triggers")

**QRSTriggerDetector** - Detect R-peaks for cardiac (BCG) artifact correction

.. code-block:: python

   from facet.preprocessing import QRSTriggerDetector

   detector = QRSTriggerDetector(
       save_to_annotations=False  # Optionally persist peaks as MNE annotations
   )

**MissingTriggerDetector** - Detect and interpolate missing triggers

.. code-block:: python

   from facet.preprocessing import MissingTriggerDetector

   detector = MissingTriggerDetector(
       expected_tr=2.0,  # Expected TR in seconds
       tolerance=0.1  # 10% tolerance
   )

**TriggerExplorer** / **InteractiveTriggerExplorer** - Inspect and QA trigger
detection results interactively

.. code-block:: python

   from facet.preprocessing import TriggerExplorer

   explorer = TriggerExplorer()
   context = explorer.execute(context)

Alignment
^^^^^^^^^

**TriggerAligner** - Align triggers using cross-correlation

.. code-block:: python

   from facet.preprocessing import TriggerAligner

   aligner = TriggerAligner(
       ref_trigger_index=0,  # Reference trigger
       search_window=50  # Maximum lag search window in samples
   )
   context = aligner.execute(context)

**SubsampleAligner** - Subsample-precision alignment

.. code-block:: python

   from facet.preprocessing import SubsampleAligner

   aligner = SubsampleAligner(
       ref_trigger_index=0,
       search_window=20
   )

**SliceAligner** - Align artifacts slice-by-slice

.. code-block:: python

   from facet.preprocessing import SliceAligner

   aligner = SliceAligner()
   context = aligner.execute(context)

Data Transforms
^^^^^^^^^^^^^^^

**CutAcquisitionWindow** / **PasteAcquisitionWindow** - Remove and restore the
acquisition window around fMRI triggers for cleaner downstream processing.

.. code-block:: python

   from facet.preprocessing import CutAcquisitionWindow, PasteAcquisitionWindow

   cutter = CutAcquisitionWindow()
   paster = PasteAcquisitionWindow()

**Crop** - Crop the raw recording to a time range

.. code-block:: python

   from facet.preprocessing import Crop

   crop = Crop(tmin=10.0, tmax=300.0)  # Keep 10 s – 300 s

**MagicErasor** - Interactively erase selected signal segments with multiple modes

.. code-block:: python

   from facet.preprocessing import MagicErasor

   # Opens an interactive editor and stays open until "Done" is clicked.
   # Multiple edits can be applied in one session.
   erasor = MagicErasor(
       picks="eeg",          # channels to edit
       default_mode="zero",  # zero | mean | interpolate | generated_eeg
       random_seed=42,       # optional, used for generated_eeg mode
   )
   context = erasor.execute(context)

MagicErasor precision controls:

- ``View Center (s)``: move the visible time window.
- ``Window (s)``: zoom in/out horizontally for precise segment selection.
- ``Y Zoom``: zoom amplitude vertically (default ``0.5``, max ``3.0``).
- ``Apply Edit`` applies the current selection without closing the window.
- ``Done`` confirms all edits and returns the updated context.
- ``Cancel`` closes without applying the session changes.

**PickChannels** / **DropChannels** - Select or remove channels

.. code-block:: python

   from facet.preprocessing import PickChannels, DropChannels

   picker = PickChannels(picks=["Fp1", "Fp2", "F3", "F4"])
   dropper = DropChannels(channels=["ECG", "EOG"])

**ChannelStandardizer** - Convert EEG channels to predefined standards

.. code-block:: python

   from facet.preprocessing import ChannelStandardizer

   # Convert to classic 10-20 subset and keep auxiliary channels (stim/ecg)
   standardizer = ChannelStandardizer("10-20", keep_auxiliary=True)
   context = standardizer.execute(context)

   # Or use an explicit custom subset
   custom = ChannelStandardizer(["Cz", "Pz", "Oz"], on_missing="ignore", keep_auxiliary=False)
   context = custom.execute(context)

**PrintMetric** - Print a context metadata value to the console

.. code-block:: python

   from facet.preprocessing import PrintMetric

   printer = PrintMetric(key="triggers")
   printer.execute(context)

Correction Processors
~~~~~~~~~~~~~~~~~~~~~

**Shared Flex template engine**

``Flex`` inherits directly from ``Processor`` and provides the common
trigger-locked template workflow. For every processed channel, it extracts an
epoch matrix ``D`` (epochs by samples), obtains an averaging matrix ``A`` from
the selected strategy, calculates the template matrix ``N = A @ D``, and
subtracts row ``n`` of ``N`` from target epoch ``n``. Trigger realignment,
optional template scaling and gap interpolation, estimated-noise accumulation,
and matrix reporting are also shared by this engine.

The existing named template corrections are direct ``Flex`` subclasses kept
for compatibility. ``AASCorrection`` retains its block running-average rule,
``FARMCorrection`` retains its absolute-correlation ranking,
``CorrespondingSliceCorrection`` and ``VolumeTriggerCorrection`` retain their
volume-aware index rules, ``SliceTriggerCorrection`` retains its odd/even
selection, and ``MoosmannCorrection`` retains its motion-informed weights.
They share the surrounding correction engine, but each still constructs a
different ``A``.

``ANCCorrection``, ``PCACorrection``, and ``VolumeArtifactCorrection`` remain
independent processors because they do not construct templates as ``A @ D``.
They respectively apply adaptive filtering to an existing noise estimate,
reconstruct PCA residuals, and blend artifacts around volume transitions.

**AASCorrection** - Averaged Artifact Subtraction

The compatibility strategy for the established AAS averaging matrix:

.. code-block:: python

   from facet.correction import AASCorrection

   aas = AASCorrection(
       window_size=30,  # Sliding window size
       correlation_threshold=0.975,  # Correlation threshold
       realign_after_averaging=True,  # Realign to template
   )
   context = aas.execute(context)

**Flex** - Flexible correlation-based artifact subtraction

``Flex`` creates a separate artifact template for every target epoch and EEG
channel. Its candidate window is exact and future-first: it takes the next
``window_size`` non-target epochs when they exist. Near the end of the
recording, it fills only the missing positions with the nearest preceding
epochs. The target epoch is never included. If the recording contains fewer
than ``window_size + 1`` epochs, all available non-target epochs are used.

For example, with eight epochs numbered 0 through 7 and ``window_size=4``:

- target 2 uses candidates ``[3, 4, 5, 6]``;
- target 6 uses ``[3, 4, 5, 7]`` (one future epoch and three backfilled epochs);
- target 7 uses ``[3, 4, 5, 6]`` (four backfilled epochs).

Candidates with a signed Pearson correlation greater than or equal to
``threshold`` are accepted. All candidates that pass remain selected, even
when their number exceeds ``min_accepted``. If too few pass, the remaining
candidates are added from highest to lowest signed correlation until
``min_accepted`` is reached (or every available candidate has been used).
Ties prefer the epoch nearest to the target, then the lower epoch index.

``N_distribution="equal"`` gives each of the ``k`` selected epochs weight
``1 / k``. ``N_distribution="normal"`` instead applies a temporal Gaussian:
the unnormalized weight for selected epoch ``j`` and target ``n`` is
``exp(-0.5 * (abs(j - n) / sigma)**2)``, where ``sigma`` is the larger of 1
and one third of the effective candidate-window size. The weights are then
normalized to sum to one. Thus, ``"normal"`` favors selected epochs closer in
time; it does not favor past epochs over future epochs.

Together, these parameters describe a continuum within Flex's native
future-first strategy rather than a collection of named presets. A high
threshold with a small ``min_accepted`` produces a selective template, while
setting ``min_accepted=window_size`` forces every available candidate into the
template. Intermediate threshold, minimum-count, window, and weighting choices
span the behavior between those endpoints. They do not exactly recreate the
legacy AAS, FARM, slice/volume, or motion-informed matrices; use the named
compatibility subclass when one of those specific rules is required.

.. code-block:: python

   from facet.correction import Flex

   flex = Flex(
       window_size=30,          # Consider at most 30 non-target epochs.
       threshold=0.975,         # Automatically accept signed r >= 0.975.
       min_accepted=5,          # Backfill by correlation until at least 5 are used.
       N_distribution="normal", # Give temporally closer selections more weight.
   )
   context = flex.execute(context)

**ANCCorrection** - Adaptive Noise Cancellation

Removes residual artifacts:

.. code-block:: python

   from facet.correction import ANCCorrection

   anc = ANCCorrection(
       filter_order=5,  # Filter order
       hp_freq=1.0,  # Highpass frequency
       use_c_extension=True  # Use C implementation if available
   )
   context = anc.execute(context)

**PCACorrection** - PCA-based artifact removal

.. code-block:: python

   from facet.correction import PCACorrection

   pca = PCACorrection(
       n_components=0.95,  # Keep 95% variance
       hp_freq=1.0  # Highpass before PCA
   )
   context = pca.execute(context)

Evaluation Processors
~~~~~~~~~~~~~~~~~~~~~

**ReferenceIntervalSelector** - Select clean reference data for metrics

.. code-block:: python

   from facet.evaluation import ReferenceIntervalSelector

   ref_selector = ReferenceIntervalSelector(channel="Fp1")
   context = ref_selector.execute(context)

**SignalIntervalSelector** - Select the evaluated signal interval for metrics

.. code-block:: python

   from facet.evaluation import SignalIntervalSelector

   sig_selector = SignalIntervalSelector(channel="Fp1")
   context = sig_selector.execute(context)

   # Access selected interval
   eval_interval = context.metadata.custom['evaluation_interval']

**SNRCalculator** - Calculate Signal-to-Noise Ratio

.. code-block:: python

   from facet.evaluation import SNRCalculator

   snr_calc = SNRCalculator()
   context = snr_calc.execute(context)

   # Access results
   snr = context.metadata.custom['metrics']['snr']
   snr_per_channel = context.metadata.custom['metrics']['snr_per_channel']

**RMSCalculator** - Calculate RMS ratio

.. code-block:: python

   from facet.evaluation import RMSCalculator

   rms_calc = RMSCalculator()
   context = rms_calc.execute(context)

   rms_ratio = context.metadata.custom['metrics']['rms_ratio']

**MedianArtifactCalculator** - Calculate median artifact amplitude

.. code-block:: python

   from facet.evaluation import MedianArtifactCalculator

   median_calc = MedianArtifactCalculator()
   context = median_calc.execute(context)

   median_artifact = context.metadata.custom['metrics']['median_artifact']

**RMSResidualCalculator** - Calculate RMS residual ratio

.. code-block:: python

   from facet.evaluation import RMSResidualCalculator

   rms_res = RMSResidualCalculator()
   context = rms_res.execute(context)

   rms_residual = context.metadata.custom['metrics']['rms_residual']

**FFTAllenCalculator** - FFT-based quality metric (Allen 2000)

.. code-block:: python

   from facet.evaluation import FFTAllenCalculator

   allen = FFTAllenCalculator()
   context = allen.execute(context)

**FFTNiazyCalculator** - FFT-based quality metric (Niazy 2005)

.. code-block:: python

   from facet.evaluation import FFTNiazyCalculator

   niazy = FFTNiazyCalculator()
   context = niazy.execute(context)

**MetricsReport** - Print metrics report

.. code-block:: python

   from facet.evaluation import MetricsReport

   report = MetricsReport()
   report.execute(context)

Composite Processors
~~~~~~~~~~~~~~~~~~~~

**SequenceProcessor** - Execute processors in sequence

.. code-block:: python

   from facet.core import SequenceProcessor

   # Group multiple processors
   correction_sequence = SequenceProcessor([
       AASCorrection(window_size=30),
       ANCCorrection(filter_order=5),
       PCACorrection(n_components=0.95)
   ])

   context = correction_sequence.execute(context)

**ConditionalProcessor** - Execute conditionally

.. code-block:: python

   from facet.core import ConditionalProcessor

   # Only run if condition is met
   conditional_anc = ConditionalProcessor(
       condition=lambda ctx: ctx.metadata.custom.get('needs_anc', False),
       processor=ANCCorrection(filter_order=5),
       else_processor=None  # Skip if False
   )

**SwitchProcessor** - Switch between processors

.. code-block:: python

   from facet.core import SwitchProcessor

   # Select processor based on context
   adaptive_correction = SwitchProcessor(
       selector=lambda ctx: "high_artifact" if ctx.metadata.artifact_length > 100 else "low_artifact",
       cases={
           "high_artifact": AASCorrection(window_size=50),
           "low_artifact": AASCorrection(window_size=20)
       },
       default=AASCorrection(window_size=30)
   )

Processor Requirements
----------------------

Each processor may require certain data to be present in the context:

Common Requirements
~~~~~~~~~~~~~~~~~~~

- ``requires_raw`` - Needs MNE Raw data
- ``requires_triggers`` - Needs trigger positions

Checking Requirements
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check processor requirements
   print(f"Requires raw: {processor.requires_raw}")
   print(f"Requires triggers: {processor.requires_triggers}")

   # Check context
   if context.has_triggers():
       print("Triggers available")
   else:
       print("No triggers - detector needed")

Processor Properties
--------------------

Each processor has properties that describe its behavior:

.. code-block:: python

   processor = AASCorrection(window_size=30)

   # Metadata
   print(processor.name)  # "aas_correction"
   print(processor.description)  # "Averaged Artifact Subtraction"
   print(processor.version)  # "1.0.0"

   # Behavior flags
   print(processor.requires_raw)  # True
   print(processor.requires_triggers)  # True
   print(processor.modifies_raw)  # True
   print(processor.parallel_safe)  # True

Using Processors
----------------

Direct Execution
~~~~~~~~~~~~~~~~

Execute processor directly:

.. code-block:: python

   processor = TriggerDetector(regex=r"\b1\b")
   output_context = processor.execute(input_context)

In a Pipeline
~~~~~~~~~~~~~

Add to pipeline:

.. code-block:: python

   pipeline = Pipeline([
       Loader(path="data.edf"),
       TriggerDetector(regex=r"\b1\b"),
       AASCorrection(window_size=30)
   ])

Callable Interface
~~~~~~~~~~~~~~~~~~

Processors are callable:

.. code-block:: python

   processor = HighPassFilter(freq=1.0)

   # These are equivalent
   context = processor.execute(input_context)
   context = processor(input_context)

Chaining Processors
~~~~~~~~~~~~~~~~~~~

Chain processor calls with the pipe operator:

.. code-block:: python

   from facet import load, TriggerDetector, UpSample, TriggerAligner, AASCorrection

   context = (
       load("data.edf", preload=True)
       | TriggerDetector(regex=r"\b1\b")
       | UpSample(factor=10)
       | TriggerAligner(ref_trigger_index=0)
       | AASCorrection(window_size=30)
   )

Processor Discovery
-------------------

Using the Registry
~~~~~~~~~~~~~~~~~~

Get processor by name:

.. code-block:: python

   from facet.core import get_processor, list_processors

   # Get processor class
   ProcessorClass = get_processor("aas_correction")
   processor = ProcessorClass(window_size=30)

   # List all processors
   all_processors = list_processors()
   for name, proc_class in all_processors.items():
       print(f"{name}: {proc_class.__name__}")

   # List by module category
   correction_processors = list_processors(category="correction")
   preprocessing_processors = list_processors(category="preprocessing")

Best Practices
--------------

1. **Validate Context**

   Check that context has required data:

   .. code-block:: python

      if not context.has_triggers():
          raise ValueError("Triggers required for this processor")

2. **Use Appropriate Parameters**

   Choose parameters based on your data:

   .. code-block:: python

      # High artifact amplitude
      AASCorrection(window_size=50, correlation_threshold=0.98)

      # Low artifact amplitude
      AASCorrection(window_size=20, correlation_threshold=0.95)

3. **Check Results**

   Verify processor output:

   .. code-block:: python

      context = processor.execute(context)

      # Check raw data modified
      if processor.modifies_raw:
          assert context.get_raw() is not None

4. **Handle Errors**

   Catch validation errors:

   .. code-block:: python

      from facet.core import ProcessorValidationError

      try:
          context = processor.execute(context)
      except ProcessorValidationError as e:
          print(f"Prerequisites not met: {e}")

5. **Log Processing**

   Track what's being done:

   .. code-block:: python

      from loguru import logger

      logger.info(f"Applying {processor.name}")
      context = processor.execute(context)
      logger.info(f"Completed {processor.name}")

Processor Comparison
--------------------

Correction Algorithms
~~~~~~~~~~~~~~~~~~~~~

+----------------+------------------+----------------+----------------+
| Algorithm      | Speed            | Effectiveness  | Use Case       |
+================+==================+================+================+
| AAS            | Fast             | High           | Primary        |
+----------------+------------------+----------------+----------------+
| ANC            | Medium           | Medium-High    | Residual       |
+----------------+------------------+----------------+----------------+
| PCA            | Slow             | Medium         | Alternative    |
+----------------+------------------+----------------+----------------+

Typical workflow:
1. AAS (primary correction)
2. ANC (residual artifacts)
3. PCA (optional, if needed)

Filter Types
~~~~~~~~~~~~

+----------------+------------------+---------------------------+
| Filter         | Purpose          | When to Use               |
+================+==================+===========================+
| HighPass       | Remove drift     | After correction          |
+----------------+------------------+---------------------------+
| LowPass        | Remove HF noise  | Before analysis           |
+----------------+------------------+---------------------------+
| BandPass       | Frequency range  | Specific analysis         |
+----------------+------------------+---------------------------+
| Notch          | Line noise       | 50/60 Hz and harmonics    |
+----------------+------------------+---------------------------+

Performance Tips
----------------

1. **Processor Order**

   Optimal order:

   - Load data
   - Detect triggers
   - Upsample
   - Align
   - Correct (AAS → ANC → PCA)
   - Downsample
   - Filter
   - Evaluate
   - Export

2. **Parallelization**

   Enable for multi-channel data:

   .. code-block:: python

      pipeline.run(parallel=True, n_jobs=-1)

3. **Memory Management**

   Use preload wisely:

   .. code-block:: python

      # Small files - preload
      Loader(path="small.edf", preload=True)

      # Large files - don't preload
      Loader(path="large.edf", preload=False)

Next Steps
----------

- Learn how to create :doc:`custom_processors`
- Understand :doc:`parallel_processing` for performance
- Explore :doc:`pipelines` for workflow composition
- Check :doc:`../api/preprocessing` for detailed API reference
