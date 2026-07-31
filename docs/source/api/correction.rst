Correction API
==============

Correction processors for removing fMRI artifacts from EEG data.

.. currentmodule:: facet.correction

Correction Architecture
-----------------------

``Flex`` inherits directly from :class:`facet.core.Processor` and owns the
shared trigger-locked template engine. For each processed channel, the engine
extracts the epoch data matrix ``D``, delegates construction of the averaging
matrix ``A`` to the active strategy, computes the artifact-template matrix
``N = A @ D``, and subtracts each row of ``N`` from its target epoch.

The established template algorithms are direct ``Flex`` subclasses. They
reuse extraction, template calculation, trigger realignment, subtraction,
estimated-noise accumulation, and matrix reporting while preserving their own
averaging-matrix rules and public constructors:

.. code-block:: text

   Processor
   +-- Flex
       +-- AASCorrection
       +-- FARMCorrection
       +-- CorrespondingSliceCorrection
       +-- VolumeTriggerCorrection
       +-- SliceTriggerCorrection
       +-- MoosmannCorrection

These subclasses are compatibility strategies, not parameter aliases. The
four native ``Flex`` controls do not exactly reproduce AAS's block running
average, FARM's absolute-correlation ranking, the slice/volume indexing rules,
or Moosmann's motion-informed weights. Named Flex strategies or presets can
eventually provide a migration path without changing those matrices.

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
