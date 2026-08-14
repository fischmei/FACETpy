<p align="center">
  <a href="https://facetpy.readthedocs.io/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/H0mire/facetpy/main/docs/source/_static/logo_dark_theme.png">
      <img src="https://raw.githubusercontent.com/H0mire/facetpy/main/docs/source/_static/logo_light_theme.png" alt="FACETpy logo" width="300">
    </picture>
  </a>
</p>

<h3 align="center">FACETpy - EEG-Data Correction Framework</h3>

<p align="center">
  A Python toolbox for correcting EEG artifacts using Averaged Artifact Subtraction (AAS) and other advanced methods. Built on MNE-Python.
  <br>
  <a href="https://facetpy.readthedocs.io/"><strong>Explore FACETpy docs »</strong></a>
  <br>
  <br>
  <a href="https://github.com/H0mire/facetpy/issues/new?assignees=&labels=bug">Report bug</a>
  ·
  <a href="https://github.com/H0mire/facetpy/issues/new?assignees=&labels=feature">Request feature</a>
  ·
  <a href="https://facetpy.readthedocs.io/">Documentation</a>
</p>

[![Documentation Status](https://readthedocs.org/projects/facetpy/badge/?version=latest)](https://facetpy.readthedocs.io/en/latest/?badge=latest)

Built on [MNE-Python](https://mne.tools), FACETpy provides a modular pipeline architecture that lets researchers process, evaluate, and compare correction results with minimal code.

CLI reference PDF: [Download the FACETpy CLI Overview](docs/source/_static/FACETpy_CLI_Overview.pdf).

**Key features**

- Load EEG from EDF, GDF, and BIDS formats
- Artifact correction: AAS, PCA, Adaptive Noise Cancellation (ANC)
- Full evaluation suite: SNR, RMS, Median Artifact, FFT-based metrics
- Batch processing across subjects/sessions with `Pipeline.map()`
- Command-line processing with `facetpy-run` for single files, folders, MFF inputs, BIDS export, viewing, and analysis
- Generate synthetic EEG for algorithm testing
- Rich progress display in the terminal


## Supported artifacts

| Artifact | Origin | Correction methods |
|---|---|---|
| **Gradient artifact (GA)** | MRI scanner gradient switching during simultaneous EEG-fMRI | AAS, PCA |
| **Ballistocardiogram (BCG)** | Cardiac-induced electrode movements in the MRI magnetic field | AAS (after BCG detection), PCA |
| **Cardioballistic / pulse artifact** | Pulsatile skin/vessel movement under electrodes | BCG detector + AAS |
| **Motion artifact** | Head movement, cable pull, electrode shift | Preprocessing filters, PCA |
| **Power-line noise (50/60 Hz)** | Electrical mains interference | Notch filter (via MNE preprocessing) |
| **Muscle (EMG) artifact** | Jaw, neck muscle activity contaminating high-frequency EEG | High-frequency filtering, PCA |
| **Amplifier saturation / drift** | Slow DC drift or clipping from long recordings | Baseline correction, filtering |

For a detailed description of use cases (EEG-fMRI research, batch studies, benchmarking, clinical pipelines, and more) see the [Use Cases & Supported Artifacts](https://facetpy.readthedocs.io/en/latest/user_guide/use_cases.html) page in the documentation.


## Quick start

Quick installation from PyPI (requires Python 3.11/3.12/3.13):

```bash
pip install facetpy
```

Strongly recommended for fast ANC performance:

```bash
python -m facet.build
```

For the Full Setup + Early Access to features  (Repository, Examples, Contributing):  
Unix (WSL/macOS/Linux) - bootstrap shortcut:

```bash
curl -fsSL https://raw.githubusercontent.com/H0mire/facetpy/main/scripts/bootstrap.sh | sh
cd facetpy
```

## Preview

<p align="center">
  <img src="https://raw.githubusercontent.com/H0mire/facetpy/main/docs/source/_static/run_example.png" alt="FACETpy example run" width="700" />
</p>


```python
from facet import (
    Pipeline, Loader, EDFExporter,
    TriggerDetector, UpSample, DownSample, AASCorrection,
)

pipeline = Pipeline([
    Loader(path="data.edf", preload=True),
    TriggerDetector(regex=r"\b1\b"),
    UpSample(factor=10),
    AASCorrection(window_size=30),
    DownSample(factor=10),
    EDFExporter(path="corrected.edf", overwrite=True),
], name="Quickstart")

result = pipeline.run()
result.print_summary()   # Done in 4.2s  snr=18.3  rms_ratio=0.14
```


## Installation

Requires **Python 3.11, 3.12, or 3.13**.
For normal usage, `uv` is not required.

### Normal usage (recommended): install from PyPI

```bash
pip install facetpy
```

The package name on PyPI is `facetpy`; import it in Python as `facet`.

### Contributing setup (source + uv)

uv is required for contribution workflows (tests, linting, docs).

Unix (macOS/Linux) - bootstrap shortcut (installs uv):

```bash
curl -fsSL https://raw.githubusercontent.com/H0mire/facetpy/main/scripts/bootstrap.sh | sh
cd facetpy
```

Unix (macOS/Linux) - existing clone:

```bash
./scripts/install.sh
```

Other platforms (including Windows) with uv installed:

```bash
git clone https://github.com/H0mire/facetpy.git
cd facetpy
uv sync --locked
```

The Unix `./scripts/install.sh` script:
- checks for Python 3.11/3.12/3.13
- checks whether uv is installed
- asks whether uv should be installed if missing
- runs `uv sync --locked`

The bootstrap script:
- clones FACETpy into `./facetpy`
- runs `./scripts/install.sh` inside that clone

Manual uv setup (contributors):

```bash
# 1 — verify Python
python --version

# 2 — install uv (pick one)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pip install --user uv

# 3 — install repository dependencies
uv sync --locked
```

Optional contributor extras:
```text
uv sync --extra deeplearning     # TensorFlow-based models
uv sync --extra notebooks        # Jupyter notebook support
uv sync --extra gui              # PyQt6 GUI components
uv sync --extra gui-all          # all supported Qt bindings
uv sync --extra docs             # Sphinx documentation toolchain
uv sync --all-extras             # everything above
```

Run contributor commands with `uv run ...` (for example, `uv run pytest`).


### Build the C extension (strongly recommended for ANC)

The fast Adaptive Noise Cancellation (ANC) path is significantly faster with
the compiled FastRANC C extension. Build it once after installing.

Without uv:

```bash
python -m facet.build
```

With uv:

```bash
uv run build-fastranc
```

If the extension is not compiled, ANC uses a slower Python fallback and the
rest of the toolbox still works.


## Command line interface

Installing FACETpy exposes two console commands:

```bash
facetpy-run --help
facetpy-to-bids --help
```

`facetpy-run` is the main command. It mirrors the pipeline concepts used in
the Python API while adding batch input discovery, chunked processing, BIDS
conversion, viewing, and analysis utilities. For an illustrated reference,
download the [FACETpy CLI Overview PDF](docs/source/_static/FACETpy_CLI_Overview.pdf).

### CLI subcommands

| Command | Purpose |
|---|---|
| `facetpy-run process` | Run correction over one file, a list, or an input folder |
| `facetpy-run to-bids` | Convert corrected output files into a BIDS dataset |
| `facetpy-run modes` | List correction modes and add-on modes |
| `facetpy-run patterns` | List reusable pipeline patterns |
| `facetpy-run viewer` / `facetpy-run view` | Open or save raw EEG plots with `RawPlotter` |
| `facetpy-run analysis` | Run data checks and optional quality metrics |

Use these discovery commands first:

```bash
facetpy-run --help
facetpy-run process --help
facetpy-run modes
facetpy-run patterns
facetpy-run analysis --list-metrics
```

### Processing EEG files

`process` accepts single files, repeated files, path lists, or folders. It also
understands MFF directories. The default processing flow is scanner/gradient
correction; add `--mode bcg` when you also want QRS-triggered BCG cleanup.

```bash
# One input
facetpy-run process \
  --input raw.edf \
  --output-dir output/corrected \
  --overwrite

# Many explicit inputs
facetpy-run process \
  --input sub-01.mff \
  --input sub-02.mff \
  --output-dir output/corrected \
  --on-error continue

# Text file with one path per line
facetpy-run process \
  --input-list inputs.txt \
  --output-dir output/corrected \
  --on-error continue

# Folder scan, including nested folders
facetpy-run process \
  --input-dir /path/to/rawdata \
  --recursive \
  --extensions .mff .edf .set \
  --output-dir output/corrected \
  --on-error continue \
  --overwrite
```

The output directory is always a folder because chunked processing can create
one or more corrected files per input. For batch runs, each source input gets
its own subfolder unless `--flat-output` is used. Every recording also gets a
dedicated `*_facetpy.log`, a self-describing `quality_metrics.json`, the
pipeline and template-matrix JSON files, and a static HTML quality-control
report. Recursive and list-based runs never mix log records from different
EEGs.

### Patterns, correction modes, and add-on modes

Patterns define the whole pipeline shape:

| Pattern | Use when |
|---|---|
| `--pattern standard` | **Default.** Full docs-style scanner workflow with pattern-selected PCA and ANC; add `--mode bcg` for cardiac cleanup |
| `--pattern quickstart` | You explicitly want the smaller scanner workflow without the standard PCA and ANC stages |
| `--pattern bcg` | You want the specialized BCG-only path |

Correction modes choose the main template-subtraction strategy:

```bash
--correction-mode aas
--correction-mode farm
--correction-mode flex
--correction-mode volume-trigger
--correction-mode slice-trigger
--correction-mode corresponding-slice
--correction-mode moosmann
```

Flex exposes the correlation-based choices directly: `--window-size` sets the
non-target candidate window, `--flex-threshold` selects correlated epochs,
`--flex-min-accepted` supplements that selection when too few pass, and
`--flex-distribution equal|normal` controls their averaging weights.

All CLI template-subtraction modes instantiate `Flex`; the selected name maps
to a complete decision recipe for candidate quota, sampling, motion,
target inclusion, scoring, template size, and weighting. For each channel,
Flex extracts `D`, builds `A` from those decisions, computes `N = A @ D`, and
subtracts the corresponding row of `N` at each trigger. The report records the
whole recipe and the legacy algorithm it most closely resembles.

The old hard-coded AAS, FARM, and weighted source snapshots live in
`src/facet/correction/archived_algos` and are not imported by the CLI. Public
legacy class names remain available as thin Flex-backed compatibility
adapters. `ANCCorrection`, `PCACorrection`, and `VolumeArtifactCorrection`
remain independent add-on processors because they use adaptive filtering,
PCA reconstruction, and volume-transition correction rather than constructing
`A`.

Add-on modes are layered around the selected correction mode and can be passed
more than once:

```bash
--mode volume-artifact   # before template subtraction
--mode pca               # after template subtraction
--mode anc               # after downsampling
--mode bcg               # QRS-triggered BCG cleanup after scanner correction
```

BCG can also be run alone with `--pattern bcg`. Use that specialized path when
you only want QRS trigger detection and BCG artifact subtraction, without the
scanner/gradient correction stages.

Examples:

```bash
# Full standard pattern with FARM, PCA, and ANC
facetpy-run process \
  --input raw.edf \
  --output-dir output/farm_standard \
  --pattern standard \
  --correction-mode farm \
  --mode pca \
  --mode anc \
  --on-error continue \
  --overwrite

# Scanner correction plus QRS-triggered BCG cleanup
facetpy-run process \
  --input raw.edf \
  --output-dir output/scanner_bcg \
  --pattern quickstart \
  --mode bcg \
  --overwrite

# Flexible correlation selection with temporal-normal weights
facetpy-run process \
  --input raw.edf \
  --output-dir output/flex \
  --correction-mode flex \
  --window-size 30 \
  --flex-threshold 0.95 \
  --flex-min-accepted 8 \
  --flex-distribution normal \
  --overwrite

# Motion-informed Moosmann weighting
facetpy-run process \
  --input raw.edf \
  --output-dir output/moosmann \
  --pattern standard \
  --correction-mode moosmann \
  --motion-rp-file motion.txt \
  --on-error continue \
  --overwrite

# QRS-triggered BCG-only correction
facetpy-run process \
  --input raw.edf \
  --output-dir output/bcg \
  --pattern bcg \
  --bcg-window-size 20 \
  --overwrite
```

### Memory, chunking, and large files

`facetpy-run process` uses chunked execution for large recordings. By default,
it tries trigger-section chunks so each correction chunk contains detected
triggers. Use fixed-length chunks when you want memory-estimated divisions
instead:

```bash
facetpy-run process \
  --input raw.mff \
  --output-dir output/fixed_chunks \
  --fixed-length-chunks \
  --min-chunks 3 \
  --max-chunks 12
```

Useful memory and chunk controls:

```bash
--channel-sequential / --no-channel-sequential
--max-memory-mb 4096
--memory-fraction 0.5
--disable-chunking
--force-full-run
--trigger-section-padding-seconds 10
--trigger-section-min-triggers 16
--trigger-section-gap-seconds 30
--trigger-section-max-sections 2
```

`--channel-sequential` is enabled by default and is the safest option for
upsampling-heavy pipelines because high-memory correction stages run one
channel at a time.

Trigger-section planning never reduces Flex's required trigger count. When the
normal estimate is smaller than that context, FACETpy enlarges only to the
smallest valid trigger-aligned window that fits safely in currently available
RAM. Computational windows may overlap for correction context, while each
output contains only its non-overlapping core. `--disable-chunking` requests
one checked full-run window; `--force-full-run` bypasses that safety refusal
and should be used only when the memory estimate is known to be conservative.
The older `--memory-budget-mb` spelling remains an alias for
`--max-memory-mb`.

### Process outputs and reports

Each successful `process` output folder includes corrected chunk files, a
self-contained HTML quality-control report, and JSON metadata:

| File | Contents |
|---|---|
| `<source>_cleaning_report.html` | Offline before/during/after report with temporal spectra, Welch PSD, amplitude distributions, embedded Flex matrices, graph-Laplacian energy, clean topomap GIF, coherence matrices/network, and the complete pipeline record |
| `chunks_manifest.json` | Source path, chunk windows, output files, runtime, success/error per chunk |
| `pipeline_description.json` | Pattern, correction mode, BCG mode status, add-on modes, processor list, parameters, HTML report path, and corrected output paths |
| `artifact_template_matrices.json` | Flex-engine `N = A @ D` reports: epoch matrix `D`, averaging matrix `A`, artifact-template matrix `N` preview, and ordered diagnostic-plot records |

The HTML file embeds every plot and animation as a data URI, so it can be
opened or archived without a separate report-assets folder. Its comparisons
use exact non-overlapping exported chunk cores, match EEG channels by name, and
never treat overlap context as independent data. To keep reporting bounded for
large studies, signal analytics use up to eight stratified chunks, up to three
temporal strata per selected core, at most 120 seconds of paired data, a 1 kHz
report-rate cap, and 16 million samples per phase; the report records every
analyzed window and any report-rate change. The complete chunk manifest is
also embedded, including core/overlap boundaries, chunking mode, and memory
budget. In a multi-input `--flat-output` run, source-specific hashed HTML and
JSON names prevent equal basenames from overwriting one another.

Temporal frequency plots and graph frequency are intentionally separate:
Welch PSD is shown in Hz and µV²/Hz, while the electrode-graph Laplacian panel
uses graph eigenvalue λ and empirical graph-mode energy. Scalp maps prefer
recorded electrode positions, then recognized MNE templates. If the channel
names do not support defensible positions, spatial sections show an explicit
unavailable notice instead of inventing a head layout. Coherence is presented
as an 8–13 Hz alpha-band, rank-thresholded exploratory sensor-space summary
because reference choice and volume conduction can affect it. The thresholded
edges and communities are not statistical-significance or causal-connectivity
claims.

If a recording fails and `--on-error continue` is used, FACETpy writes
`processing_error.json` in that recording's output folder and
`processing_failures.json` at the batch root.

### Viewer and analysis

Use `viewer` to inspect recordings before or after correction:

```bash
facetpy-run viewer \
  --input output/corrected/sub-01_chunk_001_of_001.edf \
  --viewer-mode mne \
  --show \
  --n-channels 30 \
  --scalings auto

facetpy-run viewer \
  --input raw.edf \
  --viewer-mode matplotlib \
  --channel Cz \
  --start 30 \
  --duration 10 \
  --output output/cz_preview.png
```

Use `analysis` for data checks and metrics:

```bash
# List metric names
facetpy-run analysis --list-metrics

# Default metric suite
facetpy-run analysis \
  --input raw.edf \
  --detect-events \
  --metrics \
  --output-json output/mff_analysis_metrics.json

# Select specific metrics
facetpy-run analysis \
  --input raw.edf \
  --metric rms \
  --metric legacy-snr \
  --metric report \
  --output-json output/selected_metrics.json
```

By default, inapplicable metrics are skipped and recorded in the output JSON.
Use `--no-skip-inapplicable-metrics` when you want those validation failures to
stop the command.

### Convert corrected outputs to BIDS

Corrected chunks can be converted to BIDS with either `facetpy-run to-bids` or
the convenience command `facetpy-to-bids`:

```bash
facetpy-run to-bids \
  --input-dir output/corrected \
  --recursive \
  --extensions .edf \
  --output-dir output/bids \
  --task facetcorrected \
  --detect-events \
  --overwrite \
  --on-error continue
```

BIDS labels can be provided explicitly with `--subject`, `--session`, and
`--task`; otherwise, the CLI derives subject/run values from the corrected file
names.

To convert an existing BIDS dataset in the other direction, use the dedicated
command:

```bash
facetpy-convert-bids \
  --bids-root /path/to/bids_dataset \
  --output-dir /path/to/edf_output
```

The command recursively discovers supported recordings inside `eeg/`
directories in EEGLAB (`.set`), BrainVision (`.vhdr`), EDF, BDF, and FIF
format. Sidecars are matched from the recording stem: `_events.tsv` supplies
EDF+ annotations, `_channels.tsv` supplies channel types and bad-channel
status, and `_eeg.json` supplies portable recording metadata such as line
frequency. Unsupported or incomplete recordings are reported and skipped
without stopping the remaining conversions.

### Setup notes for CLI users

- `facetpy-run` is available after `pip install facetpy` or after activating a
  source checkout environment created with `uv sync`.
- From a source checkout, use `.venv/bin/facetpy-run ...` or `uv run facetpy-run ...`
  if the command is not on your shell `PATH`.
- Build FastRANC once with `python -m facet.build` for faster ANC correction.
- Interactive MNE viewing may require a GUI backend. If a headless machine
  cannot show windows, use `--viewer-mode matplotlib --output plot.png`.
- For recursive MFF processing, pass `--input-dir`, `--recursive`, and
  `--extensions .mff`.


## Running the examples

All examples are in the `examples/` folder and use the bundled
`NiazyFMRI.edf` dataset.

To run repository examples, clone the repository and install FACETpy in your
active Python environment (no uv required):

```bash
git clone https://github.com/H0mire/facetpy.git
cd facetpy
pip install facetpy
```

Then run from the project root:

```bash
# Recommended order for new users:
python examples/quickstart.py          # minimal pipeline
python examples/evaluation.py          # metrics & comparison
python examples/advanced_workflows.py  # conditional, parallel, factory
python examples/batch_processing.py    # multiple files at once
python examples/inline_steps.py        # custom steps & pipe operator
python examples/complete_pipeline_example.py  # full clinical pipeline
python examples/eeg_generation_visualization_example.py  # synthetic EEG
```


## Testing

```bash
# Run the full test suite
uv run pytest

# Only fast unit tests (skip slow integration tests)
uv run pytest -m "not slow"

# A single test file
uv run pytest tests/test_core_pipeline.py -v

# With coverage report
uv run pytest --cov=facet --cov-report=html
```

Open the coverage report:

```bash
python -m webbrowser htmlcov/index.html
```


## Documentation

```bash
# Install docs dependencies
uv sync --extra docs

# Build HTML docs
uv run sphinx-build -b html docs/source docs/build

```

Open docs locally:

```bash
python -m webbrowser docs/build/index.html
```

Full online documentation: https://facetpy.readthedocs.io/

For comprehensive build instructions, theme configuration, and contribution guidelines see [`docs/README.md`](docs/README.md).
For PyPI release steps, see [`RELEASING.md`](RELEASING.md).


## Contributing

Contributing uses the source/uv workflow. See installation guide above.  
Use `./scripts/install.sh` on Unix, otherwise run `uv sync --locked`.
Follow [`docs/source/development/contributing.rst`](docs/source/development/contributing.rst) for the full setup and checks.


## Project structure

```
src/facet/
├── cli.py          facetpy-run command line entry point
├── _cli_*.py       CLI parser, pipeline builders, BIDS helpers, report writers
├── core/           Pipeline, Processor, ProcessingContext, BatchResult
├── io/             Loader, BIDSLoader, EDFExporter, BIDSExporter
├── preprocessing/  Filters, Resample, TriggerDetector, Alignment, Transforms
├── correction/     Flex template strategies, PCA, ANC, volume transitions
├── evaluation/     SNRCalculator, RMSCalculator, MetricsReport, RawPlotter
├── misc/           EEGGenerator (synthetic data)
└── pipelines.py    create_standard_pipeline() factory

examples/
├── quickstart.py                         minimal pipeline
├── evaluation.py                         metrics & comparison
├── advanced_workflows.py                 conditional, parallel, factory
├── batch_processing.py                   multiple files
├── inline_steps.py                       custom steps & pipe operator
├── complete_pipeline_example.py          full clinical pipeline
└── eeg_generation_visualization_example.py  synthetic EEG
```


## VS Code Tasks

Tasks are defined in `.vscode/tasks.json` and can be run via **Ctrl+Shift+P** → **Tasks: Run Task**.

| Task | Shortcut | Description |
|---|---|---|
| **Test: Run All** | default test task | Full test suite with coverage report |
| **Test: Run Current File** | | Run pytest on the file open in the editor |
| **Test: Unit Only** | | Only tests marked `@pytest.mark.unit` |
| **Test: Integration Only** | | Only tests marked `@pytest.mark.integration` |
| **Test: Skip Slow** | | All tests except those marked `@pytest.mark.slow` |
| **Test: Show Coverage Report** | | Open `htmlcov/index.html` in the browser |
| **Lint: Check (Ruff)** | | Check `src/` and `tests/` for lint errors |
| **Lint: Fix (Ruff)** | | Auto-fix lint errors in place |
| **Format: Check (Ruff)** | | Verify formatting without changing files |
| **Format: Apply (Ruff)** | | Apply ruff formatting to `src/` and `tests/` |
| **Build: FastRANC C Extension** | | Compile the FastRANC C extension |
| **Build: Install Dependencies** | | `uv sync` |
| **Build: Install All Extras** | | `uv sync --all-extras` |
| **Build: Update Dependencies** | | `uv lock --upgrade` |
| **Docs: Build HTML** | | Build Sphinx documentation |
| **Docs: Open in Browser** | | Open the built docs in the browser |
| **Docs: Build & Open** | | Build docs and open immediately |
| **Run: Current Python File** | | Execute the file open in the editor |
| **Review: Uncommitted Changes (Codex)** | | Codex AI review of all local changes |
| **Review: Against Branch (Codex)** | | Codex AI review against a selected base branch (prompts for branch) |
| **QA: Full Check (Lint + Format + Test)** | | Lint + format check + full test suite in sequence |

## License

GPLv3 — see `LICENSE` for details.

Author: Janik Michael Mueller
