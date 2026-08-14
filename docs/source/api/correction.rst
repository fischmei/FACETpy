Correction API
==============

Correction processors for removing fMRI artifacts from EEG data.

.. currentmodule:: facet.correction

Correction Architecture
-----------------------

``Flex`` inherits directly from :class:`facet.core.Processor` and owns the
shared trigger-locked template engine. For each processed channel, the engine
extracts the epoch data matrix ``D``, constructs the averaging matrix ``A``
from an explicit :class:`MatrixDecisions` recipe, computes the
artifact-template matrix ``N = A @ D``, and subtracts each row of ``N`` from
its target epoch.

The CLI now instantiates ``Flex`` for every template-subtraction correction
mode. Named recipes reproduce the former AAS, FARM, volume, slice,
corresponding-slice, and Moosmann choices as closely as the composable decision
space permits. Complete decisions and the closest legacy resemblance are
written to the pipeline JSON and displayed at the top of the HTML report.

Public legacy names remain as Flex-backed compatibility adapters:

.. code-block:: text

   Processor
   +-- Flex
       +-- AASCorrection
       +-- FARMCorrection
       +-- CorrespondingSliceCorrection
       +-- VolumeTriggerCorrection
       +-- SliceTriggerCorrection
       +-- MoosmannCorrection

They no longer contain independent matrix implementations. For new
experiments, ``Flex(matrix_decisions=...)`` constructs ``A`` from independent
directional-quota, sampling, motion-eligibility, target-inclusion, scoring,
template-size, distance, and weighting decisions. There is no named-algorithm
branch in that builder. The unmodified historical source snapshots are kept
under ``src/facet/correction/archived_algos`` for reference and are not loaded
by the CLI.

``ANCCorrection``, ``PCACorrection``, and ``VolumeArtifactCorrection`` remain
independent ``Processor`` subclasses. They use adaptive filtering, PCA
reconstruction, and volume-transition blending rather than the ``A @ D``
template workflow.

AAS Correction
--------------

AASCorrection
~~~~~~~~~~~~~

.. autoclass:: AASCorrection
   :members:
   :undoc-members:
   :show-inheritance:

AveragedArtifactSubtraction (alias)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AveragedArtifactSubtraction
   :members:
   :undoc-members:
   :show-inheritance:

Flexible Correction
-------------------

Flex
~~~~

.. autoclass:: Flex
   :members:
   :undoc-members:
   :show-inheritance:

FlexCorrection (alias)
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FlexCorrection
   :members:
   :undoc-members:
   :show-inheritance:

Composable Averaging-Matrix Decisions
-------------------------------------

The recipe is evaluated separately for every target epoch. A finite
``DirectionalQuota`` requests ``past`` and ``future`` candidates *after*
sampling and motion eligibility. If one direction is short, the remaining
quota is completed from the other direction. A global quota ignores
``window_size`` and returns every eligible candidate on the active sampling
lattice.

The subsequent stages apply target inclusion or exclusion, candidate scoring,
template-size selection, and normalized weighting. Signed and absolute Pearson
modes rank higher scores first and use ``threshold`` (``ct``) as their
acceptance gate. ``minimum_k`` keeps every threshold match and supplements the
best rejected candidates only when fewer than ``k`` pass. ``maximum_k`` caps
the passing set without supplementation, and ``exactly_k`` both caps and
supplements as needed. Invalid correlations cannot pass or supplement.

``temporal_motion_cost`` instead ranks lower costs first. Its independently
scaled temporal and cumulative-motion terms are added before ranking. It
supports ``maximum_k`` and ``exactly_k`` because an ungated cost list has no
meaningful minimum. ``none`` is paired only with ``select_all`` and calculates
neither a score nor an acceptance gate. Limited selections break ties by
smaller target distance and then candidate index.

For example, this recipe requests ten alternating future candidates. Because
the directional quota is counted after sampling, these are offsets
``+1, +3, ..., +19`` when available:

.. code-block:: python

   from facet.correction import (
       CandidateScoringPolicy,
       DirectionalQuota,
       Flex,
       MatrixDecisions,
       SamplingPolicy,
       TargetPolicy,
       TemplateSizePolicy,
       WeightingPolicy,
   )

   decisions = MatrixDecisions(
       quota=DirectionalQuota.future_only(10),
       sampling=SamplingPolicy.alternating(),
       target_policy=TargetPolicy.EXCLUDE,
       scoring=CandidateScoringPolicy.absolute_pearson(threshold=0.975),
       template_size=TemplateSizePolicy.exactly(k=5),
       weighting=WeightingPolicy.student_t(
           scale=3.0,
           degrees_of_freedom=4.0,
       ),
   )
   correction = Flex(matrix_decisions=decisions)

Acquisition-dependent decisions fail validation when their metadata is
missing. Same-slice sampling reads ``ProcessingMetadata.slices_per_volume``.
Motion eligibility, temporal-motion scoring, and motion-distance weighting read
``ProcessingMetadata.custom["artifact_epoch_motion"]`` as either a
``MotionEpochMetadata`` instance or an equivalent mapping. Volume-level motion
rows require an explicit artifact-epoch mapping; use
``MotionEpochMetadata.from_volume_parameters`` for regular slice sequences.

AveragingMatrixBuilder
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AveragingMatrixBuilder
   :members:

MatrixDecisions
~~~~~~~~~~~~~~~

.. autoclass:: MatrixDecisions
   :members:

DirectionalQuota
~~~~~~~~~~~~~~~~

.. autoclass:: DirectionalQuota
   :members:

SamplingPolicy
~~~~~~~~~~~~~~

.. autoclass:: SamplingPolicy
   :members:

MotionEligibility
~~~~~~~~~~~~~~~~~

.. autoclass:: MotionEligibility
   :members:

CandidateScoringPolicy
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CandidateScoringPolicy
   :members:

TemplateSizePolicy
~~~~~~~~~~~~~~~~~~

.. autoclass:: TemplateSizePolicy
   :members:

CorrelationPolicy (compatibility)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CorrelationPolicy
   :members:

WeightingPolicy
~~~~~~~~~~~~~~~

.. autoclass:: WeightingPolicy
   :members:

MatrixMetadata and MotionEpochMetadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MatrixMetadata
   :members:

.. autoclass:: MotionEpochMetadata
   :members:

FARM Correction
---------------

FARMCorrection
~~~~~~~~~~~~~~

.. autoclass:: FARMCorrection
   :members:
   :undoc-members:
   :show-inheritance:

Volume Artifact Correction
--------------------------

VolumeArtifactCorrection
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VolumeArtifactCorrection
   :members:
   :undoc-members:
   :show-inheritance:

RemoveVolumeArtifactCorrection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RemoveVolumeArtifactCorrection
   :members:
   :undoc-members:
   :show-inheritance:

Flex Compatibility Strategies
-----------------------------

CorrespondingSliceCorrection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CorrespondingSliceCorrection
   :members:
   :undoc-members:
   :show-inheritance:

VolumeTriggerCorrection
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VolumeTriggerCorrection
   :members:
   :undoc-members:
   :show-inheritance:

SliceTriggerCorrection
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SliceTriggerCorrection
   :members:
   :undoc-members:
   :show-inheritance:

MoosmannCorrection
~~~~~~~~~~~~~~~~~~

.. autoclass:: MoosmannCorrection
   :members:
   :undoc-members:
   :show-inheritance:

ANC Correction
--------------

ANCCorrection
~~~~~~~~~~~~~

.. autoclass:: ANCCorrection
   :members:
   :undoc-members:
   :show-inheritance:

AdaptiveNoiseCancellation (alias)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdaptiveNoiseCancellation
   :members:
   :undoc-members:
   :show-inheritance:

PCA Correction
--------------

PCACorrection
~~~~~~~~~~~~~

.. autoclass:: PCACorrection
   :members:
   :undoc-members:
   :show-inheritance:

Legacy Weighting Aliases
------------------------

AvgArtWghtCorrespondingSliceCorrection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AvgArtWghtCorrespondingSliceCorrection
   :members:
   :undoc-members:
   :show-inheritance:

AvgArtWghtVolumeTriggerCorrection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AvgArtWghtVolumeTriggerCorrection
   :members:
   :undoc-members:
   :show-inheritance:

AvgArtWghtSliceTriggerCorrection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AvgArtWghtSliceTriggerCorrection
   :members:
   :undoc-members:
   :show-inheritance:

AvgArtWghtMoosmannCorrection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AvgArtWghtMoosmannCorrection
   :members:
   :undoc-members:
   :show-inheritance:
