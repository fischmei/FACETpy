Installation
============

Requirements
------------

* Python 3.11, 3.12, or 3.13
* MNE-Python 1.10.2
* NumPy 2.1.3
* SciPy >= 1.15.3
* scikit-learn >= 1.4.2
* `uv <https://docs.astral.sh/uv/>`_ (contributors only)

Install from PyPI (Recommended)
-------------------------------

For normal usage, install FACETpy from PyPI. This is sufficient for running
your own scripts and the repository examples.

.. code-block:: bash

   pip install facetpy

The package name on PyPI is ``facetpy``; import it in Python as ``facet``.

Full Setup from Source (Contributing, Example Datasets/Scripts, Early Access to features)
-----------------------------------------------------------------------------------------

FACETpy is managed with uv.

Unix (macOS/Linux)
~~~~~~~~~~~~~~~~~~

For contributors on Unix-like systems, the quickest setup is the bootstrap URL:

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/H0mire/facetpy/main/scripts/bootstrap.sh | sh
   cd facetpy

You can also run the installer script in an existing clone:

.. code-block:: bash

   ./scripts/install.sh

Other platforms (including Windows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For contributors outside Unix-like shells, use uv directly:

.. code-block:: bash

   git clone https://github.com/H0mire/facetpy.git
   cd facetpy
   uv sync --locked

The Unix ``./scripts/install.sh`` script:

* checks for Python 3.11/3.12/3.13
* checks whether uv is installed
* prompts to install uv if missing
* runs ``uv sync --locked``

The bootstrap script:

* clones FACETpy into ``./facetpy``
* runs ``./scripts/install.sh`` inside that clone

Manual uv setup (contributors):

.. code-block:: text

   uv sync --locked

Run contributor commands inside the uv environment (Unix and Windows):

.. code-block:: text

   uv run pytest

Optional extras (Unix and Windows):

.. code-block:: text

   uv sync --extra deeplearning     # TensorFlow-based extras
   uv sync --extra notebooks        # Jupyter notebook support
   uv sync --extra gui              # PyQt6 GUI components
   uv sync --extra gui-all          # all supported Qt bindings
   uv sync --extra docs             # Sphinx + MyST documentation toolchain
   uv sync --all-extras             # all optional dependencies

Development Installation
------------------------

The standard ``uv sync`` includes development dependencies such as
``pytest`` and ``ruff``. Documentation tooling (``sphinx``, ``myst-parser``)
is installed via ``--extra docs``.

Optional Components
-------------------

C Extension for ANC
~~~~~~~~~~~~~~~~~~~

Strong recommendation: compile the FastRANC C extension once after installing.
ANC runs significantly faster with it.

Without uv:

.. code-block:: bash

   python -m facet.build

With uv:

.. code-block:: bash

   uv run build-fastranc

This will create:

* ``libfastranc.so`` (Linux)
* ``libfastranc.dylib`` (macOS)
* ``fastranc.dll`` (Windows)

If the C extension is not compiled, ``ANCCorrection`` falls back to a slower
Python implementation and the rest of the toolbox still works.

Verify Installation
-------------------

To verify your installation:

.. code-block:: python

   import facet
   print(facet.__version__)

This should print the version number without errors.

Troubleshooting
---------------

MNE-Python Issues
~~~~~~~~~~~~~~~~~

If you encounter issues with MNE-Python installation:

.. code-block:: bash

   pip install --upgrade mne

See `MNE installation guide <https://mne.tools/stable/install/index.html>`_ for platform-specific instructions.

Import Errors
~~~~~~~~~~~~~

If you get import errors, reinstall dependencies:

.. code-block:: bash

   pip install --upgrade --force-reinstall facetpy

For contributor/source environments, run:

.. code-block:: bash

   uv sync --locked
