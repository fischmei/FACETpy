Flex optimization synthetic benchmark
=====================================

Reproduce with the optimization extra installed:

.. code-block:: bash

   uv run python scripts/benchmark_flex_optimization.py /tmp/flex-benchmark

Measured on this macOS ARM64 environment using Python 3.12.16, FACETpy 2.0.2,
NumPy 2.1.3, SciPy 1.17.1 and MNE 1.10.2. Each method runs in a fresh process
with one BLAS thread. These are single synthetic measurements, not statistical
speedup estimates or evidence of performance on clinical recordings.
Machine-readable results are preserved in ``docs/benchmarks/flex_synthetic.json``.

Matrix microbenchmark
---------------------

The same seeded 800-by-256 epoch array and default matrix recipe were used.
The dense and sparse template checksums agreed; automated tests additionally
compare every weight and corrected sample within floating-point tolerance.

.. list-table:: Matrix construction and multiplication
   :header-rows: 1

   * - Method
     - Wall time (s)
     - Matrix bytes
     - Process peak RSS (MiB)
   * - Original dense
     - 0.0501
     - 5,120,000
     - 195.89
   * - Sparse CSR
     - 0.0495
     - 51,204
     - 194.30

Sparse storage was approximately 100 times smaller for this local candidate
recipe. The construction times were effectively similar; no meaningful matrix
construction speed improvement is claimed from this small measurement.

Repeated lifecycle benchmark
----------------------------

A 40-second, three-channel, 200 Hz synthetic recording with regular triggers,
a periodic scanner artifact and non-phase-locked EEG was evaluated with five
preset recipes repeated three times, representing cache opportunities across
folds. The original route runs the existing dense Flex lifecycle, reloads,
redetects triggers, upsamples, computes original spectra and exports corrected
EDF for every evaluation. The new route includes initial preparation, SQLite
and NumPy cache writes, sparse correction and all metrics; repeats reuse the
metric cache. Imports, Optuna studies and plots are excluded from timing.
This isolates evaluation cost; it is not a timed historical full optimizer run.

.. list-table:: Repeated evaluation
   :header-rows: 1

   * - Method
     - Wall time (s)
     - Process peak RSS (MiB)
     - Output/cache snapshot bytes
   * - Repeated original lifecycle
     - 0.4728
     - 193.42
     - 758,400
   * - Prepared metrics and cache
     - 0.2803
     - 192.50
     - 5,183,392

The measured prepared path took about 59% of the original route's wall time.
Original PSD calculations decreased from 45 to 3, channel upsamplings from 45
to 3, and corrected channel evaluations from 45 to 15. Ten complete dataset
metric-cache hits avoided repeat corrections. Triggers were supplied by the
common synthetic fixture for the prepared route; normal preflight adds one
trigger detection per recording/session, compared with 15 repeated detections
in this original-route benchmark.

Both operating-system block-input and block-output counters returned zero.
They therefore provide no useful physical-disk traffic measurement on this
platform. Snapshot bytes count the generated files, including active SQLite WAL
for the prepared route; they are not cumulative write bytes. Cache preparation
used more disk space than the tiny repeated EDF outputs. This is a real upfront
cost, not a claimed I/O reduction for short recordings. Longer recordings,
repeated studies and many trials should be benchmarked separately before
extrapolating. Peak RSS includes the interpreter and scientific-library stack,
so the small RSS difference is not evidence of a large process-memory reduction.

Remaining costs and limitations
-------------------------------

Correlation-based selection still examines each target's candidate epochs.
Chunk-local global pools can require quadratic work and selected sparse storage;
correlation rows can consume substantial SQLite space. Upsampled float64 arrays
and epoch matrices deliberately trade reusable disk space for repeated CPU/RAM
allocations. Hashing inputs adds an initial sequential read. Downsampling,
alignment, scaling, subtraction and corrected PSD calculation remain necessary
for each previously unseen correction unit. Conservative worker estimates are
not hard operating-system memory limits. Real EEG, large global pools, very
high sampling rates and long acquisitions have not been benchmarked here.
