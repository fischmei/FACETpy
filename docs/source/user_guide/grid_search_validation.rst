Grid Search and K-Fold Validation
=================================

FACETpy's Pareto grid-search runner evaluates the public ``Flex`` parameters
with this processing order:

.. code-block:: text

   TriggerDetector -> UpSample -> Flex -> DownSample -> evaluation metrics

The example runner in ``examples/run_kfold.py`` performs leave-one-dataset-out
validation. For ``N`` input recordings it:

#. evaluates every valid parameter combination once on every dataset, using
   trigger-section chunking;
#. caches the resulting scalar metrics, not corrected EEG chunks;
#. creates ``N`` folds, excluding one dataset from each training table;
#. calculates a training-only Pareto set and selects the point closest to the
   normalized ideal objective values;
#. evaluates that one selected configuration on the held-out dataset.

This layout avoids validation leakage: a held-out dataset does not influence
the configuration selected for its fold. The global grid cache avoids repeating
the same expensive correction run separately for every fold.

Each completed parameter combination is atomically checkpointed in
``global_grid/dataset_cache/<dataset>.csv``. If a run is interrupted midway
through a dataset, start it again with the same arguments and without
``--rebuild-cache``; the runner validates the partial cache and evaluates only
the missing combinations. Transient NFS or other network-mount errors,
including Linux ``EREMOTEIO`` (errno 121), are retried with bounded exponential
backoff. Atomic replacement ensures that a failed write cannot leave a
half-written CSV.

Running the Example
-------------------

.. code-block:: bash

   uv run python examples/run_kfold.py \
       --input-dir /path/to/eeg/files \
       --output-dir /path/to/results \
       --workers 1

Use ``--rebuild-cache`` after changing datasets, grid values, the upsampling
factor, or chunk settings. The manifest protects against accidentally reusing
an incompatible cache. More than one worker runs independent configurations
or folds concurrently and can use substantially more memory.

Combination Progress and Grid
-----------------------------

Before processing begins, ``global_grid/global_combinations.csv`` records every
valid combination with a stable, one-based ``combination_number``. Progress
logs use the same number and include the full parameter dictionary:

.. code-block:: text

   Dataset 'subject_01': running combination 7/36: {...}

Use the combination number to connect a log entry or heatmap row back to exact
parameter values. Invalid combinations where ``min_accepted > window_size``
are omitted consistently from both the CSV and execution.

Understanding the Pareto Reports
--------------------------------

The default objectives maximize SNR and minimize residual RMS and Niazy FFT
distortion. A configuration is Pareto-optimal when no comparable configuration
is at least as good in every objective and strictly better in one.

Several Pareto points are normal: multi-objective selection produces a set of
trade-offs rather than one universal winner. Each fold also has its own
training-only set. With three objectives, a two-dimensional projection can
make a starred point look dominated because its advantage is on the hidden
third axis.

The reports make those distinctions explicit:

``*_pareto_2d.png``
   A readable two-objective projection. Stars are members of the complete
   multi-objective Pareto set and the diamond is the selected ideal-point
   compromise.

``*_pareto_3d.png``
   The first three objectives without the hidden-axis ambiguity.

``*_pareto_matrix.png``
   Every pairwise objective projection plus diagonal distributions. This is
   usually the most useful figure when the Pareto set is large.

``*_pareto.csv``
   Raw aggregate values, ``pareto_eligible``, ``is_pareto``,
   ``ideal_point_distance``, and ``is_selected_compromise`` for reproducible
   inspection.

Configurations with fewer successful runs are not directly comparable to
complete configurations. They remain in the result and heatmap outputs, but
are excluded from Pareto eligibility whenever a higher common success rate is
available.

Other Visualizations
--------------------

``*_combination_heatmap.png``
   Shows every configuration against every Pareto objective. Each objective is
   independently normalized so 1 is the best comparable observed value.
   Stars mark Pareto configurations, the diamond marks the selected compromise,
   and a cross marks incomplete or Pareto-ineligible rows. Raw values remain in
   the CSV files because normalized values should only be used visually.

``*_parameter_effects.png``
   Shows the marginal mean and standard deviation of the configured
   parameter-effect metric for each parameter value.

``parameter_stability.png``
   Shows how often each parameter value was selected across held-out folds.
   Concentrated bars indicate stable selection; a diffuse pattern suggests
   dataset sensitivity or several nearly equivalent configurations.

``held_out_metrics_heatmap.png``
   Shows per-metric z-scores across held-out datasets. Standardization makes
   fold-to-fold variation visible without comparing incompatible metric units.

``held_out_runtime.png``
   Compares held-out execution time and, when available, the successful chunk
   fraction for each dataset.

Every visualization has a corresponding CSV containing the unnormalized
values needed for statistical analysis.
