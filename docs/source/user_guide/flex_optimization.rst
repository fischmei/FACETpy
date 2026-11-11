Efficient folder-based Flex optimization
========================================

Run on at least three independent development recordings. Install the optional
Optuna dependency, then use the single supported entry point:

.. code-block:: bash

   uv sync --extra optimization
   uv run python examples/run_matrix_optimization.py \
       /path/to/eeg_folder /path/to/results --recursive \
       --trials 100 --promotion-count 15 --workers 2 \
       --memory-mb 2048 --trigger-regex '^R128$' \
       --evaluation-mode non-reference

``uv run facet-optimize`` accepts the same arguments. The economical defaults
are 24 screening trials per fold (and final model), eight promotions, one
worker, 2 GiB estimated working memory, two screening channels, and three
screening sections. ``--trials`` is a total budget, including preset seeds and
equivalent pruned trials, not an additional budget when resuming. TPE begins
after at least ten startup trials. Presets are also retained as baselines;
therefore the promotion count has a floor equal to the available preset count.

For references, add:

.. code-block:: bash

   --reference-directory /path/to/separately_cleaned \
   --evaluation-mode combined

By default, reference and combined modes require a matching, validated reference
for every recording. All sessions are audited in ``reference_eligibility.csv``
before strict failure. Explicit ``--exclude-invalid-references`` instead retains
valid sessions and rebuilds CV on that cohort; see
:doc:`participant_grouped_optimization`. There is no silent fallback.
References match relative input paths,
then unique BIDS acquisition entities (subject, session, task, run, acquisition)
across supported formats; preprocessing suffixes may differ. Channel names must
match exactly and EEG must be in SI volts after loading. Missing channels are
reported and excluded; at least two shared good EEG channels are required.
Both recordings are explicitly average-referenced over the shared good channels
for comparison. Missing timestamps require ``--assume-reference-aligned`` after
independent verification, unless explicit sample provenance is available.
A reference is a published comparison output, not neural ground truth.
Non-reference mode never requires downstream cleaning or reference metrics.

Fixed downstream cleaning and two checkpoints
----------------------------------------------

NATVIEW supplies the broad stage order, not FACETpy's algorithms or settings.
``facet.optimization.cleaning.clean_after_flex(context)`` also runs independently
of Optuna. It expects a multichannel Flex-corrected context with MRI triggers.
``CleaningSettings`` is frozen; the optimizer uses one ``DEFAULT_CLEANING``
throughout every trial and fidelity. ``fixed_preprocessing.json`` records it.

.. list-table:: Stage correspondence
   :header-rows: 1
   :widths: 25 75

   * - General stage
     - FACETpy implementation and fixed choice
   * - Scanner-template subtraction
     - Existing Flex lifecycle, with only matrix decisions searched. ECG also
       receives scanner subtraction using the first good EEG channel's matrix,
       avoiding selection based on cardiac similarity; it is excluded from EEG metrics.
   * - Residual PCA
     - ``PCACorrection`` defaults: 0.95 explained-variance components, no PCA
       high-pass, no retained noise array. This is FACETpy PCA, not FASTR OBS.
   * - ANC
     - Disabled throughout: no validated retained ANC noise reference. Its
       processor exists, but enabling it is not required for stage comparability.
   * - Cardiac correction
     - ``BandPassFilter(1, 40, picks="ecg")`` stabilizes the existing detector
       against ECG drift/onset transients. ``QRSTriggerDetector`` then supplies
       events from the MRI acquisition interval, excluding pre-scanner padding;
       indices are mapped back to the unchanged EEG timeline. It supplies
       cardiac triggers to ``AASCorrection`` with its window 10 and correlation
       0.975 defaults. Cardiac realignment is disabled to preserve event timing.
       This is heartbeat AAS, not a second scanner correction or median PAS.
   * - Resampling and EEG filtering
     - ``Resample(200)`` then ``BandPassFilter(1, 70)``. The passband follows the
       FACETpy complete-pipeline example; 200 Hz supports it with anti-alias margin.
   * - Bad channels and segments
     - Existing bad-channel marks are honored; no interpolation. Existing BAD
       annotations exclude samples from comparison without compacting time.
       FACETpy's interactive MagicErasor is not used in automated trials.
   * - Average reference
     - Existing ``RawTransform`` wrapping MNE average rereferencing.
   * - ASR / ICA rejection
     - No existing FACETpy automated processors found; omitted and recorded.

Checkpoint 1 uses the unchanged Flex-only scanner residual, suppression,
clean-band coverage and theta/alpha/beta preservation metrics. Contamination
masks are calculated once from uncorrected EEG. Checkpoint 2 compares the fixed
pipeline output with the reference. QRS events, PCA fits and heartbeat templates
are recomputed for each different Flex output; the settings never change.
Processing runs on padded chunks and scores their cores; filtering/PCA/BCG at
chunk boundaries need not equal whole-record processing. Preserve chunk settings
when deploying a selected recipe.

Reference time correspondence
-----------------------------

NATVIEW SET files may retain ``EEG.etc.clean_sample_mask``. The comparison reads
that published mask and verifies original event marker IDs and ``urevent``
timestamps against the raw file. It reports matched-event count and maximum
timestamp error. Compacted event latencies are checked within one reference
sample, allowing fractional events adjacent to rejection boundaries; the binary
sample map itself is exact. This is metadata recovery, not a fitted EEG correlation lag.

The published mask can describe an intermediate rejection stage. When its
retained count exceeds final EEG length, FACETpy requires final boundary cuts
and retained-event ``urevent`` origins to establish each continuous segment.
Short unanchored segments require uniquely consistent boundary durations and
mask constraints. Boundary durations are not blindly summed: a later boundary
may count deletion from an already compacted intermediate recording. The final
map must be an ordered subset of the original mask, contain exactly the final
EEG sample count, and agree independently with original marker IDs/timestamps.
Missing or ambiguous segment origins, incompatible clocks, or contradictory
durations fail with a provenance diagnostic. No EEG correlation alignment,
length-only truncation, interpolation, shift or stretch repairs a failed map.
The cleaning stage responsible cannot be inferred from sample counts alone.

To inspect a server recording without loading EEG signals or creating files:

.. code-block:: bash

   uv run python -m facet.optimization.eeglab_diagnostic \
       /path/to/raw/sub-02_ses-01_task-rest_eeg.set \
       /path/to/cleaned/sub-02_ses-01_task-rest_eeg.set

The CLI reads explicitly selected MAT metadata variables (including ``etc``
for the sample mask), excluding ``data`` and signal/ICA activation arrays.
It prints mask counts, boundaries, per-segment urevent origin ranges, original
event clock checks, mapping examples, history and verification/blocking evidence
to stdout. ``--max-events 0`` prints every mapping; ``--json`` prints the complete
structured report. Exit status 2 means provenance is blocked/unsupported;
nested SET structures are not loaded because they may contain signal arrays.
Successful reconstruction on synthetic or other recordings does not establish
the timeline of an uninspected recording: review that recording's diagnostic
before using its reference metrics.

Otherwise, removed samples require ``<reference filename>.alignment.json``:

.. code-block:: json

   {
     "provenance": "Verified acquisition timestamps and rejection log",
     "intervals": [
       {"raw_start": 0, "reference_start": 0, "duration": 20},
       {"raw_start": 25, "reference_start": 20, "duration": 30}
     ]
   }

Times are seconds from the first stored sample in each file. Ordered, disjoint
intervals must lie inside both files. Equal-duration continuous files can use
matching absolute starts or an explicit user assertion. Compacted references
are never silently treated as aligned, even with ``--assume-reference-aligned``.
For sample-exact comparison, both signals are explicitly resampled to the least
common integer sampling rate, capped at 2000 Hz (200/250 Hz becomes 1000 Hz).
This interpolation is fixed by metadata, not optimized against the signal.
Grid-intersection boundaries are rounded inward deterministically; usable
seconds and channel/timeline coverage are reported. No fitted scale, polarity,
lag, channel permutation or missing-channel interpolation is used. PSDs are
computed separately on contiguous valid intervals of at least two seconds.

Single-recording gate and NATVIEW commands
-------------------------------------------

Run one recording before optimization:

.. code-block:: bash

   uv run facet-optimize data/natview/raw output/natview_validation \
       --reference-directory data/natview/cleaned --evaluation-mode combined \
       --recursive --include-pattern 'sub-01_ses-01_task-checker_eeg.set' \
       --validate-only --upsample-factor 1 --chunk-seconds 30 \
       --chunk-min-triggers 10 --chunk-padding-seconds 5 --memory-mb 6000

This checks the full acquisition with a baseline Flex matrix and screens a
second matrix using identical downstream settings. It writes metrics, alignment
evidence, resource measurements, a four-stage time/PSD plot and preview arrays
under ``single_recording_validation/``. Finite reference metrics are required.
This is an execution gate, not a score threshold or a scientific quality claim;
its reference values never select a recipe. All reference runs perform the gate
before starting TPE. Failure stops the run. ``--validate-only`` does not load Optuna.

The recorded Subject 01 checkerboard validation passed over seven acquisition
chunks: 60 shared channels, 491 original event matches, and 186.952 valid seconds
out of 205.8 evaluated seconds. One fractional event at a rejection boundary
differs by one reference sample (4 ms); original event times agree to numerical
precision. The baseline mean channel RMS difference was 16.79 microvolts and
mean correlation 0.270. This is an unoptimized compatibility check. The two
candidate checks took 61.5 seconds after preparation, with 2.4 GiB peak process
RSS on the local run. See the :download:`validation report
<../../benchmarks/natview_fixed_pipeline_validation.json>` and
:download:`checkpoint metrics <../../benchmarks/natview_fixed_pipeline_validation.csv>`.

After reviewing that validation, run the combined search:

.. code-block:: bash

   uv run facet-optimize data/natview/raw output/natview_combined \
       --reference-directory data/natview/cleaned --evaluation-mode combined \
       --recursive --include-pattern '*task-checker_eeg.set' \
       --upsample-factor 1 --chunk-seconds 30 --chunk-min-triggers 10 \
       --chunk-padding-seconds 5 --workers 1 --memory-mb 6000 --trials 24

The filename filter selects Scanner ON SET files, excluding Scanner OFF/Outside
tasks and alternative headers whose binary companions may be absent. A factor
of one avoids expensive oversampling on a 16 GB laptop; it is fixed, not searched.
Use another output directory when changing fixed settings. This command is an
example, not evidence that a scientifically useful recommendation has been found.

Architecture and fixed correction lifecycle
-------------------------------------------

``facet.optimization.search`` retains the previous runner's conditional
``MatrixDecisions`` space, presets, constrained multi-objective TPE, and nested
leave-one-recording-out selection. ``cache``, ``prepared``, ``metrics``, and
``reports`` separate persistence, signal preparation, evaluation, and reporting.
The example file is a compatibility wrapper for diagnostic example imports.
The public Flex processor and the other processors retain their defaults and
behavior. The new ``AveragingMatrixBuilder.build_sparse`` method preserves the
same candidate, selection, and normalized weighting rules as the dense builder.

Only averaging-matrix choices vary. The lifecycle always upsamples, constructs
``N = A @ D``, realigns against the first selected good EEG channel, scales each
epoch template, subtracts, interpolates gaps, and downsamples. These operations
call the existing Flex implementation. No ANC or separately selected AAS/FARM
algorithm is introduced. Future lifecycle search can extend the fixed settings
and preparation signatures without changing the conditional matrix search.

Discovery uses FACETpy's supported readers, including compressed FIF and MFF
directories. Scanner triggers must be strictly increasing and within the
recording. Motion sidecars use the previous runner's ``.npz``, ``.npy`` or text
schema: ``parameters``, ``segment_ids``, ``stable``, and an explicit
``epoch_to_motion_index`` when epoch/volume mapping is ambiguous. Missing motion
disables corresponding search branches in that training fold. No held-out
motion capabilities influence training search choices.

Preparation and bounded resources
---------------------------------

Each source requests lazy loading where the format supports it. MNE loads
embedded SET arrays in full even with ``preload=False``; allow for that reader
overhead when choosing a memory budget. Triggers are detected once per session in preflight.
Fixed acquisition sections are subdivided into ``--chunk-seconds`` cores (120 s
by default), with ``--chunk-padding-seconds`` of context on either side. Very
short tails merge into the preceding core to improve spectral resolution. All
scanner-on cores are covered at full fidelity. Pre-scanner/post-scanner signal
is not part of optimization coverage. Chunk-local global candidate pools are
scoped to the fixed padded chunk, not the entire original recording; changing
chunk duration changes the algorithm's candidate universe and invalidates caches.
Use longer chunks when long-range temporal or motion comparisons are required.
``--chunk-min-triggers`` sets the preferred minimum core width, rather than
silently dropping short acquisition sections.

Float64 upsampled signals, zero-mean epoch matrices, original Welch spectra,
contamination masks, original RMS/phase-locked statistics, and validated reference
intervals are prepared per channel and stored as NumPy arrays. Arrays are mapped
read-only. Correction needs one writable channel, its epochs/templates, sparse
weights and bounded scoring temporaries. Other recordings remain on disk.
Screening and full evaluation use identical prepared epochs and context. In
non-reference mode, selected channels/cores change. Reference/combined modes
always clean all EEG/ECG channels so the reference convention is fixed; only
selected cores change. Candidate-specific completed chunk metrics are cached
transactionally, including the full fixed pipeline. Different matrix decisions
never reuse PCA, QRS or BCG outputs. Incomplete downstream chunks are recomputed.

Processes evaluate independent recordings; trials remain serial. A conservative
per-job estimate includes interpreter reserve, resampling working arrays and
worst-case selected sparse weights. Concurrency decreases automatically to fit
``--memory-mb``; if one job exceeds it, the run stops before allocating that job.
Reduce chunk duration/upsampling or increase the budget in that case. The budget
is an estimate, not an operating-system RSS limit. Scientific-library imports,
file-backed page cache, MNE readers and platform allocators can affect actual RSS.
BLAS thread pools are limited to one in workers. No nested process pools run.

Sparse CSR weights avoid dense quadratic matrices for local selections. Signed
correlation rows are cached across thresholds, template sizes and kernels;
absolute scoring reuses signed values. Correlations use at most 128 candidate
epochs in a temporary array. Temporal and motion decisions share weights across
channels when truly independent of signal. Cumulative motion paths and motion
distances are cached. Global select-all rules can still have quadratic nonzero
weights; the resource estimate includes this case. Candidate-distance/scoring
caches may grow substantially for many global pools; disk capacity remains a
practical limit.

Cache correctness and resume
----------------------------

``evaluation_cache.sqlite3`` uses WAL and transactional, uniquely keyed rows.
Only completed results are committed; failed configurations retain tracebacks.
Correction units and candidate calculations are reusable across fidelities and
folds. Dataset/configuration/fidelity results retain all available metrics,
independent of the chosen ranking mode. CSV is exported at checkpoints/finally;
it is not rewritten after each evaluation. No corrected EEG is exported in trials.

Keys include streaming hashes of input and reference contents (including
BrainVision/EEGLAB companions or MFF contents), trigger/motion contents, fixed
preprocessing/chunking, spectral definitions, dependency versions and algorithm
source hashes. Hashing adds one sequential file read per session to prevent
unsafe reuse after a same-size edit. Older unused cache signatures coexist in
SQLite; old prepared directories can occupy disk until manually retired.
The optional migration of an explicitly compatible old CSV is a one-time action.

Re-run the same command, optionally adding ``--resume``. Each fold's Optuna SQLite
study and atomic sampler state checkpoint preserve completed trials and TPE random
state. A stale running trial is marked failed and requeued. Completed correction
units survive interrupted dataset evaluation. Exact replay across a hard process
kill between an Optuna commit and sampler checkpoint is not guaranteed; ordinary
checkpointed resume is deterministic. Use only sampler checkpoint files created
by a trusted run. Changed search signatures require ``--rebuild-studies`` or a new
output directory. ``--rebuild-cache`` resets aggregate evaluation rows for that
signature; independently valid prepared arrays and correction units remain reusable.

Fidelity, independent validation and final model
------------------------------------------------

1. Screening uses a seeded training-recording subset, evenly spaced EEG channels,
   and early/middle/late cores. Set ``--screening-fraction``,
   ``--screening-channels`` and ``--screening-sections`` to control it.
2. Expanded screening uses every training recording, twice as many channels and
   twice as many cores, capped at the available counts. Nondominated candidates
   and preset baselines advance.
3. Full training evaluates all good EEG channels and all scanner-on cores.
   Preservation constraints are applied, then a normalized distance to the ideal
   point selects a feasible Pareto recipe. There is no infeasible fallback.
4. Only the selected recipe is evaluated on the held-out recording. Chunks are
   duration weighted within a recording; recordings receive equal aggregation
   weight. Held-out results never determine their fold's recipe or promotion.

After all folds, an independent final study fits/selects a deployable recipe on
all development recordings. This is separate from the held-out validation table;
its training results are not an independent generalization estimate.

Metrics and interpretation
--------------------------

A contamination mask is constructed only from original EEG and trigger timing.
It includes prominent original spectral peaks (13--80 Hz by default). Below
that range, only prominent peaks within two Welch bins of a trigger-rate
fundamental or harmonic are included; timing alone does not exclude a harmonic.
Welch windows span at least 16 seconds and 16 trigger periods when the chunk
permits, giving comparable frequency resolution across sampling rates. Mask
half-widths use the smaller of the configured width and a quarter of the harmonic
spacing, with a two-bin minimum for Hann leakage. Neighboring prominent peaks
provide evidence for sidebands. The mask is fixed across corrections. This
heuristic cannot perfectly distinguish physiological peaks from artifact.

Welch power is summed as frequency-bin mass; masked integration never bridges
excluded gaps. Scanner residual power and suppression in dB measure power in the
fixed contaminated bins. Theta, alpha and beta preservation use only clean bins,
with log deviations, power changes and coverage reported. A band needs at least
two clean bins and 25% clean coverage; otherwise preservation is unavailable.
Preparation checks every channel and chunk before any optimization trials. If
coverage or finite positive original band power is insufficient, it stops with
the dataset, channel, chunk, affected bands and frequency resolution. Inspect
original spectra and trigger timing; longer chunks can help resolve short
recordings, and scanner-peak settings should be justified by the original EEG.
Unavailable coverage is never replaced by a plausible score. After upgrading
the spectral definition, use a new output directory or ``--rebuild-studies``
when rerunning an existing output directory; prepared data and metric cache
entries are invalidated automatically.

Phase-locked and non-phase-locked RMS diagnostics use complete nonoverlapping trigger epochs;
phase locking does not uniquely identify scanner artifact or neural activity.
Spectral preservation is a proxy, not direct proof of physiological preservation.

The two minimized objectives are:

* non-reference: scanner residual/original power and clean-band mean absolute log deviation;
* reference: normalized reference MSE and mean absolute log PSD ratio (1--40 Hz);
* combined: scanner residual/original power and normalized reference MSE.

All modes enforce training theta/alpha and beta preservation quantile bounds.
Defaults are theta/alpha 0.8--1.25 and beta 0.5--1.5. No all-zero EEG recipe can
be selected merely for suppressing artifact power. If no feasible recipe exists,
inspect coverage and diagnostics before changing scientifically justified bounds.
Reference SNR (dB), RMS residual, median absolute residual, normalized MSE and
correlation are distinct from FACETpy's scanner-off reference metrics. They use
the final cleaned output; non-reference metrics use the earlier Flex checkpoint.
Reference-relative SNR is ``10 log10(mean(reference²) / mean((candidate-reference)²))``;
it measures agreement, not neural SNR. Normalized MSE divides that error power
by reference power. Spectral similarity is PSD cosine similarity (larger is
better); spectral log deviation and absolute log bandpower ratios are smaller
when closer. PSD diagnostics use 1--40 Hz to avoid cutoff-edge comparisons.
Separate 12 Hz and 24 Hz narrow-band reference deviations are included for
checkerboard assessment; they are descriptive, not optimization objectives or
proof of physiological preservation. No unrelated metrics are summed.

Reports and validation limits
-----------------------------

``run_manifest.json`` records settings, software and input provenance.
``folds/<recording>/`` contains screening trials/Pareto results, expanded training,
full training/Pareto CSVs, selected recipe and held-out metrics. ``final_model/``
contains the separate final study and recommendation. ``recommended_recipe.json``
is the deployable recipe plus fixed lifecycle, upsampling and acquisition/chunk
settings. Use those same chunk settings in deployment to preserve candidate pools.
``per_dataset_metrics.csv``, per-mode rankings/Pareto CSVs, failed evaluation
tracebacks, resource statistics, held-out metric comparisons, parameter-effect
heatmaps, stability plots and recording-bootstrap intervals are generated at
checkpoints or completion. Per-mode rankings across the cache are descriptive;
the fold-specific tables alone define independent validation. Parameter-effect
heatmaps are confounded summaries, not causal parameter effects. With only a few
recordings, bootstrap intervals are exploratory; folds share training data.
Each mode also receives a final-training Pareto plot and a recommendation from
the evaluated final-model training candidate pool, subject to the same
preservation constraints. Alternative-mode recommendations are descriptive;
independent held-out validation applies only to the mode actually optimized.
The optimized mode includes a combined per-trial CSV with study identifiers.

The old public grid-search APIs and their tested examples remain for compatibility.
Unused experimental code is preserved under ``old/`` with original paths listed
in ``old/README.md``. See :doc:`flex_optimization_benchmark` for measured synthetic
costs and limitations, and ``tests/optimization/test_optimizer.py`` for numerical,
cache, reference, resource and fold-isolation checks. Real scanner data still need
scientific assessment; synthetic numerical equivalence is not clinical validation.

Participant-grouped experiments
---------------------------------

Use the grouped workflow when participants contribute multiple recordings:

.. toctree::
   :maxdepth: 1

   participant_grouped_optimization
