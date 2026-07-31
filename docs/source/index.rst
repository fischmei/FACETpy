.. FACETpy documentation master file

.. image:: _static/logo.png
   :align: center
   :width: 300px

Welcome to FACETpy's Documentation!
====================================

**FACETpy** (fMRI Artifact Correction and Evaluation Toolbox for Python) is a comprehensive,
modular toolkit for correcting fMRI-induced artifacts in EEG data.

FACETpy provides a modular architecture with:

* 🧩 **Modular Design** - Composable processors for flexible workflows
* 🔌 **Plugin System** - Easy extensibility via decorators
* ⚡ **Parallel Processing** - Built-in multiprocessing support
* 🔗 **MNE Integration** - First-class MNE-Python compatibility
* 📝 **Type Hints** - Full IDE support and type safety
* 🎯 **Beginner Friendly** - Clear API and comprehensive docs

Quick Start
-----------

Install FACETpy from PyPI:

.. code-block:: bash

   pip install facetpy

Repository: `GitHub - H0mire/facetpy <https://github.com/H0mire/facetpy>`_

For the **full experience** (contribution workflow and local example datasets/scripts),
use the bootstrap install script (WSL/macOS/Linux):

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/H0mire/facetpy/main/scripts/bootstrap.sh | sh
   cd facetpy

Run a complete correction pipeline:

.. code-block:: python

   from facet import create_standard_pipeline

   pipeline = create_standard_pipeline(
       input_path="data.edf",
       output_path="corrected.edf",
       trigger_regex=r"\b1\b",
       evaluate=True,
   )

   result = pipeline.run()
   print(f"SNR: {result.metrics['snr']:.2f}")

Documentation Overview
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   project_overview
   getting_started/installation
   getting_started/quickstart
   getting_started/tutorial
   getting_started/examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/use_cases
   user_guide/architecture
   user_guide/configuration
   user_guide/pipelines
   user_guide/helper_cookbook
   user_guide/grid_search_validation
   user_guide/mne_integration
   user_guide/processors
   user_guide/parallel_processing
   user_guide/custom_processors

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/top_level
   api/core
   api/io
   api/preprocessing
   api/correction
   api/evaluation

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/ai_agents_processors
   development/changelog
   development/roadmap

Key Features
------------

Modular Processors
~~~~~~~~~~~~~~~~~~

Every processing step is a self-contained processor that can be used independently:

.. code-block:: python

   from facet.preprocessing import TriggerDetector, UpSample
   from facet.correction import AASCorrection

   detector = TriggerDetector(regex=r"\b1\b")
   upsampler = UpSample(factor=10)
   aas = AASCorrection(window_size=30)

Pipeline-Based Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~

Compose processors into declarative pipelines:

.. code-block:: python

   from facet import Pipeline, Loader, EDFExporter, TriggerDetector, UpSample, AASCorrection, DownSample

   pipeline = Pipeline([
       Loader(path="data.edf"),
       TriggerDetector(regex=r"\b1\b"),
       UpSample(factor=10),
       AASCorrection(window_size=30),
       DownSample(factor=10),
       EDFExporter(path="corrected.edf")
   ])

   result = pipeline.run(parallel=True)

Plugin System
~~~~~~~~~~~~~

Create custom processors with simple decorators:

.. code-block:: python

   from facet.core import Processor, register_processor

   @register_processor
   class MyCustomProcessor(Processor):
       name = "my_processor"

       def process(self, context):
           # Your custom logic
           return context

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Automatic parallelization for compatible processors:

.. code-block:: python

   # Automatically uses all CPU cores
   result = pipeline.run(parallel=True, n_jobs=-1)

Available Processors
--------------------

**I/O**
   * Loader, BIDSLoader
   * EDFExporter, BIDSExporter

**Preprocessing**
   * Filtering: HighPassFilter, LowPassFilter, BandPassFilter, NotchFilter
   * Resampling: UpSample, DownSample, Resample
   * Triggers: TriggerDetector, QRSTriggerDetector, MissingTriggerDetector
   * Alignment: TriggerAligner, SliceAligner, SubsampleAligner
   * Transforms: CutAcquisitionWindow, PasteAcquisitionWindow, Crop, PickChannels, DropChannels

**Correction**
   * AASCorrection - Averaged Artifact Subtraction
   * ANCCorrection - Adaptive Noise Cancellation (requires compiled C extension)
   * PCACorrection - PCA-based artifact removal

**Evaluation**
   * SNRCalculator, RMSCalculator, RMSResidualCalculator
   * MedianArtifactCalculator
   * FFTAllenCalculator, FFTNiazyCalculator
   * MetricsReport, RawPlotter

Citation
--------

If you use FACETpy in your research, please cite:

.. code-block:: bibtex

   @software{facetpy2025,
     title = {FACETpy: fMRI Artifact Correction and Evaluation Toolbox for Python},
     author = {Mueller, Janik Michael},
     year = {2025},
     version = {2.0.0}
   }

License
-------

FACETpy is released under the **GPLv3** license. See the ``LICENSE`` file for details.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
