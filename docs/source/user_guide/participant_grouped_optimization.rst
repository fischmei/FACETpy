Participant-grouped nested Flex optimization
============================================

``--cv-strategy grouped-kfold`` uses five outer participant folds and four
inner participant folds by default. ``--cv-strategy leave-one-subject-out``
tests one participant at a time, retaining grouped inner validation.
``--cv-strategy leave-one-dataset-out`` remains the backward-compatible default;
that recording-level option does not guarantee participant isolation.

Grouping and model selection
----------------------------

Grouped modes require one consistent BIDS ``sub-<ID>`` entity in the filename
and parent directories. Missing/contradictory identifiers fail; no recording-level
fallback occurs. Every session/run from one participant stays together.
Assignments balance participant counts first, session counts second, with seeded
tie breaking. Every participant is tested exactly once. Each outer training set
must have at least ``--inner-folds`` participants. ``--outer-fold-index`` is
zero-based and selects a fold without changing the saved assignments.

For each outer training set, each candidate is scored separately on the
validation participants in every inner split. Sessions are averaged within
participant; the existing median objectives and preservation quantiles operate
on equally weighted participant rows. Participants are pooled once each, not
weighted by session count or by equally weighting unequal-size fold means.
Failed sessions make their participant unsuccessful. Missing/non-finite values
are not replaced with good scores. Per-fold validation metrics are saved.

Flex constructs its matrix independently per recording. No cross-participant
model is fitted on inner training recordings and no downstream setting is
learned. Inner training/validation membership is explicit and validated; no
artificial fitting stage is introduced. For this stateless transformation,
pooling full-fidelity out-of-fold participant scores is numerically equivalent
to scoring each outer-training participant once. All recipe selection remains
inside outer training participants; only the outer test estimates independent
performance. Inner scores must not be presented as independent test performance.

The existing multi-objective TPE space, anchors, promotion, preservation
constraints and normalized ideal-point Pareto selection are reused. Both
checkpoints, all evaluation modes and fixed cleaning settings are unchanged.
Screening represents every training participant and selects a deterministic
``--screening-fraction`` of sessions within each participant. EEG channel/core
fidelity follows existing rules; reference modes still clean all channels.
Expanded/full promotion uses all validation sessions. No outer-test scores or
reference comparisons rank trials, promote candidates or select recipes.
The single-recording execution gate uses an outer-training recording.

Optional technical session exclusion
------------------------------------

Reference/combined preflight checks every discovered raw/reference pair before
constructing CV folds. Strict mode remains the default: any invalid session
stops the run after the complete audit has been written. Add
``--exclude-invalid-references`` to retain only technically valid sessions:

.. code-block:: bash

   uv run facet-optimize "$RAW" "$OUT" "${COMMON[@]}" \
       --cv-strategy grouped-kfold --outer-folds 5 \
       --exclude-invalid-references --preflight-only

Use the flag on subsequent pilot, full-CV and final-only commands too; exclusions
are never silently enabled by a previous report. This flag is available only
in reference and combined modes. ``reference_eligibility.csv`` lists every
session, participant/session IDs, raw/reference paths, validation status/stage,
exception type and exact exclusion reason. ``reference_eligibility.json`` records
the policy, complete audit, and final eligible session/participant counts;
the run manifest also records the summary. Existing ``dataset_preflight.csv``
continues to describe the retained cohort.

Optimization with this flag requires ``grouped-kfold`` or
``leave-one-subject-out`` to keep each participant's sessions together;
single-recording validation and preflight-only audits need no fitting strategy.

Validation reuses scanner-trigger checks, fixed-cleaning prerequisites, channel
and voltage checks, provenance-based time alignment and supported comparison
sampling rates. It computes no Flex-dependent agreement score. Only failed
sessions are excluded: participants keep all their other valid sessions. All
folds are rebuilt using the eligible recordings, and requested outer/inner fold
counts must still be supported. Participants with no valid sessions cannot
enter the experiment. Audit files are written even if too few participants remain.

Excluded recordings never enter preparation, trial evaluation, baselines, outer
testing or final fitting. Input EEG/reference files are never edited, moved or
deleted. Environment/allocation failures remain fatal rather than defining the
cohort. Candidate-dependent correction failures remain evaluation failures;
they do not trigger additional automatic session exclusions. Performance
estimates therefore apply to the documented technically eligible cohort.
If eligibility changes, use a new output directory: existing fold/study
compatibility checks reject mixing different cohorts.

Baselines and final recipe
----------------------------

After persisting the selected outer recipe, the same outer test participants
are evaluated with default ``MatrixDecisions()``, existing ``aas_per_target``
and ``farm_per_target_k10`` anchors, and the optimized recipe. All use the same
fixed downstream cleaning. Inspired anchors do not claim exact historical AAS
or FARM reproduction. Paired optimized-minus-baseline participant differences
are reported for both checkpoints. Positive deltas favor the optimized recipe
for correlation/SNR; negative deltas favor it for errors/residuals. Test baseline
values are descriptive and never tune the optimized configuration.

Grouped CV does not automatically fit a final model. ``--final-only`` performs
separate internal grouped selection on all participants and writes
``final_model/recommended_recipe.json`` plus the root ``recommended_recipe.json``.
Outer recipes stay under ``folds/outer_XX/``. Final validation performance is not
an unseen-participant estimate; report outer CV performance for that purpose.

Reproducibility and outputs
-----------------------------

``cv_manifest.json`` and ``cv_fold_assignments.csv`` save the complete outer/inner
membership. Strategy, grouping, seed, assignments, data/source fingerprints and
evaluation/preprocessing settings enter compatibility checks. Fidelities remain
separate. Different CV seeds/settings require a new output directory; incompatible
outer reports cannot be mixed. Worker/memory limits may change on resume and
trial budgets may increase. Keep promotion count and screening fraction fixed.
The existing SQLite metric/calculation caches and TPE sampler persistence are
reused. Identical per-recording results may be reused across folds without adding
that recording to another validation partition.

Individual outer folds may run sequentially into one output directory. Use
separate directories for concurrent fold jobs: root reports are not designed
for concurrent writers. ``nested_cv_status.json`` labels partial experiments.
Participant bootstrap resamples participants with session dependence retained;
intervals are exploratory and conditional on observed fits, not refitted studies.

Outputs include:

* ``dataset_preflight.csv`` / ``preflight_manifest.json``: pairing and grouping
  metadata and separate reference provenance reports, using ``--preflight-only``.
* ``cv_fold_assignments.csv``: recording outer role and inner validation fold;
  test rows have inner index -1. Other inner folds train on that recording.
* Per-outer-fold ``inner_cv_recording_results.csv``,
  ``inner_cv_participant_results.csv`` and ``inner_cv_fold_scores.csv``.
* ``inner_cv_trials.csv`` and ``inner_cv_pareto.csv`` per fold and combined.
  Original screening/promoted-training filenames remain compatible; in grouped
  mode their values are inner validation scores.
* ``outer_fold_results.csv``, ``held_out_participant_results.csv`` and
  ``held_out_recording_results.csv`` for independent test performance.
* ``baseline_comparison.csv``, ``baseline_bootstrap_summary.csv`` and paired plots.
* ``participant_bootstrap_summary.csv``, participant/fold plots,
  ``parameter_stability.csv`` and stability plots.
* Existing per-recording metric caches, failure diagnostics and
  ``resource_statistics.json``. Cached rows alone are not CV performance estimates.

Ubuntu commands
---------------

These paths are future server inputs; local development/tests do not access them.
From the repository in Bash, set up common arguments once:

.. code-block:: bash

   uv sync --extra optimization
   RAW=/data/users/cfischmei/NATVIEW_data
   OUT=output/natview_nested
   COMMON=(
     --reference-directory /data/users/cfischmei/NATVIEW_cleaned
     --evaluation-mode combined --recursive
     --include-pattern '*task-rest_eeg.set'
     --cv-group-by subject --inner-folds 4 --cv-seed 42 --seed 42
     --upsample-factor 1 --chunk-seconds 30 --chunk-min-triggers 10
     --chunk-padding-seconds 5 --workers 2 --memory-mb 6000
   )

1. Validate discovery, pairing, grouping and reference metadata without EEG
   correction, prepared signal caches or Optuna:

   .. code-block:: bash

      uv run facet-optimize "$RAW" "$OUT" "${COMMON[@]}" \
        --cv-strategy grouped-kfold --outer-folds 5 --preflight-only

2. Pilot outer fold 0 with a small trial budget (anchors remain baselines even
   when the trial budget is smaller than the anchor set):

   .. code-block:: bash

      uv run facet-optimize "$RAW" "$OUT" "${COMMON[@]}" \
        --cv-strategy grouped-kfold --outer-folds 5 --outer-fold-index 0 --trials 3

3. Resume the pilot and run all five outer folds:

   .. code-block:: bash

      uv run facet-optimize "$RAW" "$OUT" "${COMMON[@]}" \
        --cv-strategy grouped-kfold --outer-folds 5 --trials 100 --resume

4. Optional nested leave-one-participant-out evaluation, in a different output:

   .. code-block:: bash

      uv run facet-optimize "$RAW" output/natview_loso "${COMMON[@]}" \
        --cv-strategy leave-one-subject-out --trials 100

5. Separate final recipe selection using all participants; outer estimates stay
   unchanged:

   .. code-block:: bash

      uv run facet-optimize "$RAW" "$OUT" "${COMMON[@]}" \
        --cv-strategy grouped-kfold --outer-folds 5 --final-only --trials 100

Adjust resource limits to the server. Trial counts are totals per study,
including queued anchor seeds. Budgets above startup are needed for TPE proposals.
No participant/session counts are hardcoded: inspect the preflight manifest.
Filename filtering excludes scanner-off/outside tasks and alternate EEG formats.
Pairing uses subject, session, task, run and acquisition entities, and rejects
ambiguous references. Published sample provenance handles deleted intervals as
described in :doc:`flex_optimization`.

Local checks and GitHub preparation
-------------------------------------

Tests use synthetic recordings/metadata, including a complete five-fold CLI run,
all three evaluation modes, participant/session balance, held-out-score
perturbation, individual-fold resume, final fitting and incompatible-context
rejection. No server connection or dataset download is part of testing.

.. code-block:: bash

   uv run pytest tests/optimization
   uv run ruff check src/facet/optimization tests/optimization
   LC_ALL=C uv run sphinx-build -b html docs/source docs/build

Inspect staged and unstaged work, including earlier local optimizer/cleaning
changes. Data under ``data/``, generated ``output/``/``results/``, prepared
caches and study databases are ignored. Never force-add EEG or generated arrays.

.. code-block:: bash

   git status --short
   git diff
   git diff --cached
   git add .gitignore README.md pyproject.toml examples/run_matrix_optimization.py
   git add src/facet/optimization tests/optimization
   git add docs/source/user_guide/flex_optimization.rst
   git add docs/source/user_guide/participant_grouped_optimization.rst
   git diff --cached --stat
   # Review earlier staged changes and required local Flex dependencies too.
   git commit -m "optimization, cv: add nested participant-grouped validation"
   git push origin HEAD

Nothing is committed or pushed automatically.
