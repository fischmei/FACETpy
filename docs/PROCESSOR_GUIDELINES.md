# FACETpy Processor Guidelines

> Living document for writing clean, consistent, and performant processors.
> Use this as a checklist when creating new processors **and** when refactoring existing ones.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Processor Anatomy](#2-processor-anatomy)
3. [Context Handling Rules](#3-context-handling-rules)
4. [Validation](#4-validation)
5. [Method Structure & Size](#5-method-structure--size)
6. [File Organization](#6-file-organization)
7. [Logging Guidelines](#7-logging-guidelines)
8. [Console & Progress Reporting](#8-console--progress-reporting)
9. [Error Handling](#9-error-handling)
10. [Performance Considerations](#10-performance-considerations)
11. [Import Conventions](#11-import-conventions)
12. [Registration, Versioning & Lifecycle](#12-registration-versioning--lifecycle)
13. [Testing Contract](#13-testing-contract)
14. [Processor Checklist](#14-processor-checklist)
15. [Anti-Patterns (What to Avoid)](#15-anti-patterns-what-to-avoid)
16. [Reference Template](#16-reference-template)

---

## 1. Design Principles

Every processor should follow these core principles:

| Principle | What it means in practice |
|---|---|
| **Single Responsibility** | One processor does one thing. If the docstring needs "and", split it. |
| **Immutability** | Never mutate the incoming context. Always return a new one via `with_raw()` / `with_metadata()`. |
| **Transparency** | A reader should understand what a processor does from its class name, `name` attribute, and the first 5 lines of `process()`. |
| **Fail Fast** | Validate prerequisites in `validate()`, not halfway through `process()`. |
| **No Surprises** | No hidden side effects. No writing files unless that is the processor's purpose. No mutating `self` during `process()`. |

---

## 2. Processor Anatomy

Every processor follows a consistent structure with clearly separated sections.

### 2.1 Class-Level Attributes

```python
class MyProcessor(Processor):
    """One-line summary.

    Extended description of what this processor does, when to use it,
    and any algorithmic background (with references if applicable).
    """

    name: str = "my_processor"
    description: str = "Human-readable description for pipeline.describe()"
    version: str = "1.0.0"

    requires_triggers: bool = False
    requires_raw: bool = True
    modifies_raw: bool = True
    parallel_safe: bool = True
```

Rules:
- `name` must be a unique, lowercase, snake_case identifier.
- `description` is a short sentence shown in `pipeline.describe()` output.
- Set the flags honestly — they drive automatic validation and parallel execution decisions.

### 2.2 `__init__` — Configuration Only

```python
def __init__(self, cutoff: float, order: int = 4):
    self.cutoff = cutoff
    self.order = order
    super().__init__()
```

Rules:
- Store parameters as plain attributes (no leading underscore for public config).
- Call `super().__init__()` **last** so `_get_parameters()` captures all attributes.
- Never do heavy computation, I/O, or validation here.
- Never store mutable shared state that `process()` would modify at runtime.
- Type-hint every parameter.

### 2.3 `validate` — Pre-Flight Checks

```python
def validate(self, context: ProcessingContext) -> None:
    super().validate(context)
    if self.cutoff <= 0:
        raise ProcessorValidationError(f"cutoff must be positive, got {self.cutoff}")
    if context.get_sfreq() < 2 * self.cutoff:
        raise ProcessorValidationError(
            f"Nyquist violation: sfreq={context.get_sfreq()} < 2 * cutoff={self.cutoff}"
        )
```

Rules:
- Always call `super().validate(context)` first to get the base `requires_raw` / `requires_triggers` checks.
- Validate **parameter sanity** (ranges, types, combinations).
- Validate **context preconditions** (sampling rate, channel count, metadata fields).
- Only raise `ProcessorValidationError`.
- Never read or transform large data arrays here — keep it O(1).

### 2.4 `process` — The Core Logic

```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    raw = context.get_raw().copy()

    logger.info("Applying {}-order filter at {} Hz", self.order, self.cutoff)

    raw.filter(l_freq=self.cutoff, h_freq=None, picks="eeg")

    return context.with_raw(raw)
```

Rules:
- First line: obtain what you need from the context.
- Last line: return a new context.
- In between: the algorithm, broken into private helpers when it exceeds ~30 lines.
- Never return `None` — always return a `ProcessingContext`.

### 2.5 Private Helpers — `_do_something()`

Extract any non-trivial sub-computation into private methods:

```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    raw = context.get_raw().copy()
    data = raw.get_data()

    artifact_template = self._compute_template(data, context.get_triggers())
    corrected = self._subtract_template(data, artifact_template)

    raw._data[:] = corrected
    return context.with_raw(raw)

def _compute_template(self, data: np.ndarray, triggers: np.ndarray) -> np.ndarray:
    ...

def _subtract_template(self, data: np.ndarray, template: np.ndarray) -> np.ndarray:
    ...
```

Rules:
- Helpers are pure functions when possible (take data in, return data out).
- Helpers should not access or mutate `context` directly — pass the needed arrays as arguments.
- Name them with a leading underscore.
- Each helper should fit on one screen (~40 lines).

### 2.6 `_get_parameters` — Override Only When Needed

The base class auto-captures public instance attributes. Override only if you need to exclude large objects or format values:

```python
def _get_parameters(self) -> Dict[str, Any]:
    return {"cutoff": self.cutoff, "order": self.order}
```

**Gotcha:** The auto-capture grabs every public instance attribute. If your processor stores a large array, a callable, or a non-serializable object as a public attribute, it will end up in the history dict. Either rename it with a leading underscore (`self._large_matrix`) or override `_get_parameters` to exclude it.

### 2.7 Type Hints

All public method signatures must be fully annotated. Private helpers should be annotated at minimum on the return type.

```python
def __init__(self, cutoff: float, order: int = 4) -> None:
    ...

def validate(self, context: ProcessingContext) -> None:
    ...

def process(self, context: ProcessingContext) -> ProcessingContext:
    ...

def _compute_template(self, data: np.ndarray, triggers: np.ndarray) -> np.ndarray:
    ...
```

Rules:
- Use `Optional[X]` (or `X | None` on Python 3.11+) for parameters that accept `None`.
- Use `np.ndarray` for NumPy arrays — don't use `Any`.
- Use `mne.io.Raw` for MNE Raw objects (not `BaseRaw` unless the processor truly accepts both).
- Import types from `typing` at the module top: `from typing import Any, Dict, List, Optional`.
- Don't over-annotate local variables — let the types flow from the annotated functions.

### 2.8 Docstring Format

The project uses **NumPy-style docstrings** everywhere. This is non-negotiable — Sphinx and the documentation build depend on it.

**Class docstring:**

```python
class AASCorrection(Processor):
    """Remove fMRI gradient artifacts using Averaged Artifact Subtraction.

    Computes a weighted average of neighboring artifact epochs and subtracts
    the template from each epoch. Weights are derived from Pearson correlation
    between epochs.

    References
    ----------
    Allen et al., 2000. "A method for removing imaging artifact from continuous
    EEG recorded during functional MRI." NeuroImage, 12(2), 230-239.

    Parameters
    ----------
    window_size : int
        Number of neighboring epochs to average (default: 30).
    correlation_threshold : float
        Minimum Pearson r to include an epoch in the average (default: 0.975).
    """
```

**Method docstring (for non-trivial helpers):**

```python
def _find_correlated_epochs(
    self, data: np.ndarray, target_idx: int, n_neighbors: int
) -> np.ndarray:
    """Find the most correlated neighboring epochs for template averaging.

    Parameters
    ----------
    data : np.ndarray
        EEG data array, shape (n_channels, n_samples).
    target_idx : int
        Index of the target epoch.
    n_neighbors : int
        Maximum number of neighbors to consider.

    Returns
    -------
    np.ndarray
        Indices of selected epochs, sorted by correlation (descending).
    """
```

Rules:
- Class docstring: one-line summary, blank line, extended description, optional `References`, then `Parameters`.
- Method docstring: one-line summary, optional extended description, `Parameters`, `Returns` (and `Raises` if applicable).
- Don't repeat information already in the type hints — the docstring adds *meaning*, not just types.
- `process()` and `validate()` typically don't need their own docstrings (they inherit from the base class). Only add one if the behavior is non-obvious.

---

## 3. Context Handling Rules

### 3.1 Never Mutate the Input Context

```python
# WRONG — mutates the context that was passed in
def process(self, context):
    context._metadata.triggers = new_triggers
    return context

# RIGHT — returns a new context
def process(self, context):
    return context.with_triggers(new_triggers)
```

### 3.2 Always Copy Raw Before Modifying

```python
raw = context.get_raw().copy()  # always copy first
raw.filter(...)
return context.with_raw(raw)
```

### 3.3 Use `with_*` Methods for All Context Changes

| Change | Method |
|---|---|
| Replace raw data | `context.with_raw(new_raw)` |
| Replace metadata | `context.with_metadata(new_metadata)` |
| Replace triggers | `context.with_triggers(new_triggers)` |

Never assign to private attributes:
```python
# WRONG
new_ctx._metadata = new_metadata
new_ctx._metadata.triggers = triggers

# RIGHT
new_ctx = context.with_metadata(new_metadata)
new_ctx = context.with_triggers(triggers)
```

### 3.4 Metadata Updates

When you need to update a single metadata field without replacing the whole object:

```python
new_metadata = context.metadata.copy()
new_metadata.artifact_length = computed_length
return context.with_metadata(new_metadata)
```

For custom data:

```python
new_metadata = context.metadata.copy()
new_metadata.custom["my_key"] = my_value
return context.with_metadata(new_metadata)
```

### 3.5 Raw Data Access

Prefer MNE's public API over direct `_data` access:

```python
# Preferred — uses public API
data = raw.get_data(picks="eeg")
raw = mne.io.RawArray(corrected_data, raw.info)

# Acceptable when performance matters — document why
# Direct _data access avoids a full array copy on large datasets
raw._data[ch_idx, start:stop] = corrected_segment
```

When direct `_data` access is truly needed for performance, add a brief inline comment explaining why.

### 3.6 Estimated Noise Propagation

The context carries an optional noise estimate (`context.get_estimated_noise()`) that correction processors build and downstream processors (filters, resamplers) must propagate. This is a cross-cutting concern that every processor modifying raw data should consider.

**Who creates noise estimates:**
- Correction processors (AAS, ANC, PCA) compute the estimated artifact and store it via `context.set_estimated_noise()` or `context.accumulate_noise()`.

**Who must propagate noise estimates:**
- Any processor that transforms the raw signal (filtering, resampling) must apply the same transformation to the noise estimate, if present.

**Standard pattern for noise propagation:**

```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    raw = context.get_raw().copy()

    raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)
    new_ctx = context.with_raw(raw)

    if context.has_estimated_noise():
        noise = context.get_estimated_noise().copy()
        noise_raw = mne.io.RawArray(noise, raw.info)
        noise_raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)
        new_ctx.set_estimated_noise(noise_raw.get_data())

    return new_ctx
```

**Rules:**
- Always guard with `context.has_estimated_noise()` — most pipelines won't have noise estimates until after a correction step.
- Apply the **same transformation** to noise as to raw (same filter params, same resampling factor).
- Use `logger.debug` when skipping noise propagation:
  ```python
  logger.debug("No noise estimate present — skipping noise propagation")
  ```
- Never silently drop the noise estimate. If your processor can't meaningfully transform it, pass it through unchanged.

### 3.7 MNE Verbosity Suppression

MNE operations (loading files, filtering, creating `RawArray`) produce verbose console output that interferes with the modern console display. Wrap chatty MNE calls with `suppress_stdout()`:

```python
from facet.logging_config import suppress_stdout

def process(self, context: ProcessingContext) -> ProcessingContext:
    with suppress_stdout():
        raw = mne.io.read_raw_edf(self.path, preload=True)
    ...
```

**When to use `suppress_stdout()`:**
- Loading files (`read_raw_edf`, `read_raw_gdf`, `read_raw_bids`)
- Creating `RawArray` objects (e.g., during serialization/deserialization)
- Any MNE call that prints progress bars or info messages to stdout

**When NOT needed:**
- `raw.filter()`, `raw.resample()` — these respect MNE's `verbose` parameter; pass `verbose=False` instead.
- `raw.get_data()`, `raw.copy()` — these are silent.

---

## 4. Validation

### 4.1 What to Validate

| Category | Examples |
|---|---|
| Parameter ranges | `cutoff > 0`, `0 < factor <= 100`, `window_size >= 1` |
| Parameter types | Not usually needed if type hints are used, but for callables or complex inputs |
| Context preconditions | Sampling rate sufficient, triggers present, channels exist |
| Data shape | Channel count > 0, enough samples for the window |
| Logical constraints | Target sfreq != current sfreq, start < stop |

### 4.2 What NOT to Validate

- Don't validate things the base class already checks (`requires_raw`, `requires_triggers`).
- Don't read large data arrays in `validate()`.
- Don't catch exceptions from MNE here — let them propagate from `process()`.

### 4.3 Validation for Loaders

Loaders create the context, so they have no incoming context to validate. Override `validate()` to check file paths and configuration:

```python
def validate(self, context: ProcessingContext) -> None:
    # Don't call super() — there's no incoming context to check
    if not Path(self.path).exists():
        raise ProcessorValidationError(f"File not found: {self.path}")
```

---

## 5. Method Structure & Size

### 5.1 Target Lengths

| Method | Target | Hard Limit |
|---|---|---|
| `__init__` | < 15 lines | 25 lines |
| `validate` | < 15 lines | 25 lines |
| `process` | < 40 lines | 60 lines |
| Private helper | < 40 lines | 60 lines |
| Total class | < 150 lines | 250 lines |

If a processor exceeds the hard limit, consider splitting it into a `SequenceProcessor` or extracting a utility module.

### 5.2 Canonical Flow Inside `process()`

Every `process()` method follows the same six-phase structure. The phases always appear in this order, separated by blank lines. Not every processor needs all phases, but the ones that are present must follow this sequence.

```
Phase 1 — EXTRACT          Get data and metadata from context
Phase 2 — LOG ENTRY        One logger.info summarizing what will happen
Phase 3 — COMPUTE          Core algorithm, delegated to helpers, with progress
Phase 4 — NOISE            Propagate or accumulate noise estimates (if applicable)
Phase 5 — BUILD RESULT     Construct new context via with_raw / with_metadata
Phase 6 — RETURN           Return the new context (never None)
```

**Simple processor** (all 6 phases fit in ~15 lines):

```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    # --- EXTRACT ---
    raw = context.get_raw().copy()

    # --- LOG ---
    logger.info("Applying highpass filter at {} Hz", self.cutoff)

    # --- COMPUTE ---
    raw.filter(l_freq=self.cutoff, h_freq=None, picks="eeg", verbose=False)

    # --- NOISE ---
    if context.has_estimated_noise():
        noise = context.get_estimated_noise().copy()
        noise_raw = mne.io.RawArray(noise, raw.info)
        noise_raw.filter(l_freq=self.cutoff, h_freq=None, verbose=False)
        new_ctx = context.with_raw(raw)
        new_ctx.set_estimated_noise(noise_raw.get_data())
        return new_ctx

    # --- RETURN ---
    return context.with_raw(raw)
```

**Complex processor** (phases are still visible, heavy work is in helpers):

```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    # --- EXTRACT ---
    raw = context.get_raw().copy()
    triggers = context.get_triggers()
    artifact_length = context.get_artifact_length()
    sfreq = context.get_sfreq()
    eeg_channels = mne.pick_types(raw.info, eeg=True, exclude="bads")

    # --- LOG ---
    logger.info(
        "Applying AAS correction: {} channels, {} triggers, window={}",
        len(eeg_channels), len(triggers), self.window_size,
    )

    # --- COMPUTE (phase 1: templates) ---
    epochs = self._extract_epochs(raw, triggers, artifact_length, eeg_channels)
    templates = self._compute_templates(epochs, eeg_channels, raw.ch_names)

    # --- COMPUTE (phase 2: subtraction) ---
    self._subtract_artifacts(raw, templates, triggers, artifact_length, eeg_channels)

    # --- NOISE ---
    new_ctx = context.with_raw(raw)
    new_ctx.accumulate_noise(self._estimated_artifacts)

    # --- BUILD RESULT ---
    new_metadata = context.metadata.copy()
    new_metadata.custom["aas_n_artifacts"] = len(triggers)

    # --- RETURN ---
    return new_ctx.with_metadata(new_metadata)
```

**Key rules:**
- Phases are separated by **blank lines** — a reader can visually scan the structure.
- The EXTRACT block is a flat sequence of assignments. No logic, no conditionals.
- The LOG line is a single `logger.info` call. Never multiple info lines at the top.
- The COMPUTE block is where all algorithmic work happens, delegated to `_helpers`.
- NOISE and BUILD RESULT are distinct — don't intermix metadata updates with computation.
- The RETURN is always the last line, always returns a `ProcessingContext`.

### 5.3 When to Extract Helpers

Extract a private method when:
- A block has a clear name ("compute the averaging matrix", "align triggers").
- A block is reused or could be tested independently.
- The `process()` method exceeds 40 lines.
- A deeply nested block (3+ levels of indentation) can be flattened.

### 5.4 Readability & Cognitive Complexity

Code is read far more often than it is written. Optimize for the reader.

#### Blank Lines as Paragraph Breaks

Use blank lines to separate logical groups — just like paragraphs in prose. Each group should do one thing:

```python
# GOOD — grouped by concern, easy to scan
triggers = context.get_triggers()
artifact_length = context.get_artifact_length()
sfreq = context.get_sfreq()

eeg_channels = mne.pick_types(raw.info, eeg=True, exclude="bads")
n_channels = len(eeg_channels)

templates = self._compute_templates(epochs, eeg_channels)

self._subtract_artifacts(raw, templates, triggers)

return context.with_raw(raw)
```

```python
# BAD — wall of code with no visual structure
triggers = context.get_triggers()
artifact_length = context.get_artifact_length()
sfreq = context.get_sfreq()
eeg_channels = mne.pick_types(raw.info, eeg=True, exclude="bads")
n_channels = len(eeg_channels)
templates = self._compute_templates(epochs, eeg_channels)
self._subtract_artifacts(raw, templates, triggers)
return context.with_raw(raw)
```

#### Early Returns for Guard Clauses

Reduce nesting by handling edge cases first and returning early:

```python
# GOOD — flat structure, main logic at top level
def _compute_template(self, epochs: np.ndarray, target_idx: int) -> np.ndarray:
    if len(epochs) == 0:
        return np.zeros_like(epochs[0])

    if len(epochs) < self.window_size:
        neighbors = epochs
    else:
        neighbors = self._find_neighbors(epochs, target_idx)

    return np.mean(neighbors, axis=0)
```

```python
# BAD — deep nesting, main logic buried
def _compute_template(self, epochs, target_idx):
    if len(epochs) > 0:
        if len(epochs) >= self.window_size:
            neighbors = self._find_neighbors(epochs, target_idx)
        else:
            neighbors = epochs
        return np.mean(neighbors, axis=0)
    else:
        return np.zeros_like(epochs[0])
```

#### Maximum Nesting Depth

Hard limit: **3 levels** of indentation inside any method (class + def + one level of control flow). If you need more, extract a helper:

```python
# BAD — 4 levels deep
def process(self, context):
    for ch_idx in eeg_channels:                  # level 1
        for epoch_idx in range(n_epochs):         # level 2
            if correlation > threshold:            # level 3
                for sample in range(length):       # level 4 — too deep
                    ...

# GOOD — extracted inner loop
def process(self, context):
    for ch_idx in eeg_channels:
        self._correct_channel(ch_idx, epochs, ...)

def _correct_channel(self, ch_idx, epochs, ...):
    for epoch_idx in range(n_epochs):
        if correlation > threshold:
            self._subtract_epoch(...)
```

#### Naming for Readability

Names should make the code read like a narrative:

```python
# GOOD — reads like prose
eeg_channels = mne.pick_types(raw.info, eeg=True)
artifact_template = self._compute_weighted_average(epochs, weights)
corrected_data = raw_data - artifact_template

# BAD — cryptic abbreviations
chs = mne.pick_types(raw.info, eeg=True)
tmpl = self._calc_avg(ep, w)
d2 = d1 - tmpl
```

Variable naming conventions:

| Context | Convention | Example |
|---|---|---|
| Channel index | `ch_idx` | `for ch_idx in eeg_channels:` |
| Channel name | `ch_name` | `ch_name = raw.ch_names[ch_idx]` |
| Loop counter with total | `idx` + describe | `for idx, ch_idx in enumerate(eeg_channels):` |
| Data arrays | descriptive noun | `artifact_template`, `corrected_data`, `epoch_matrix` |
| Counts | `n_` prefix | `n_channels`, `n_epochs`, `n_samples` |
| Boolean flags | `is_`/`has_` prefix | `is_parallel`, `has_noise_estimate` |

#### Avoid Inline Complexity

Don't pack complex expressions into a single line:

```python
# BAD — too much happening on one line
averaging_matrices[ch_idx] = self._calc_averaging_matrix(
    np.squeeze(epochs.get_data(copy=False)[:, idx, :]),
    window_size=self.window_size,
    rel_window_offset=self.rel_window_position,
    correlation_threshold=self.correlation_threshold
)

# GOOD — intermediate variable makes intent clear
channel_epochs = np.squeeze(epochs.get_data(copy=False)[:, idx, :])

averaging_matrices[ch_idx] = self._calc_averaging_matrix(
    channel_epochs,
    window_size=self.window_size,
    rel_window_offset=self.rel_window_position,
    correlation_threshold=self.correlation_threshold,
)
```

#### Comments: When and How

Comments explain **why**, not **what**. If you need a comment to explain what a line does, the code should be rewritten to be self-explanatory:

```python
# BAD — narrates the obvious
# Get the triggers from the context
triggers = context.get_triggers()
# Calculate the number of epochs
n_epochs = len(triggers)

# GOOD — explains a non-obvious decision
# Use Pearson correlation rather than Euclidean distance because
# it is invariant to amplitude differences between epochs
weights = self._correlate_epochs(channel_epochs, target_idx)
```

Section-level comments (marking the canonical phases) are acceptable in `process()` because they serve as visual landmarks:

```python
# --- EXTRACT ---
...

# --- COMPUTE ---
...
```

#### Cognitive Complexity Budget

Think of each method as having a complexity budget. Each of these costs one point:

| Element | Cost |
|---|---|
| `if` / `elif` / `else` | +1 each |
| `for` / `while` loop | +1 each |
| Nesting (each additional level) | +1 per level |
| `try` / `except` | +1 |
| Boolean operator (`and` / `or`) in condition | +1 each |
| `break` / `continue` | +1 each |

**Target: ≤ 10 per method. Hard limit: 15.** If a method exceeds 15, it must be split.

---

## 6. File Organization

Not every processor needs its own file, and not everything should be packed into one. The right granularity depends on complexity and cohesion.

### 6.1 Decision Matrix

| Situation | Approach | Example |
|---|---|---|
| **Complex standalone processor** (> 150 lines, multiple helpers) | **Own file** | `flex.py` → `Flex` template engine |
| **Family of related processors** sharing a base class | **One file** for the family | `filtering.py` → `Filter`, `HighPassFilter`, `LowPassFilter`, `BandPassFilter` |
| **Simple thin processors** that are conceptually a group | **One file** per group | `transforms.py` → `Crop`, `PickChannels`, `DropChannels` |
| **Utility logic shared across processors** in the same module | **Module-level `_utils.py`** file | `preprocessing/_utils.py` → `extract_epoch_with_padding()` |
| **Utility logic shared across modules** | **`helpers/` package** | `helpers/crosscorr.py` → `crosscorrelation()` |
| **Mixin providing shared behavior** to processors in one module | **Same file** as the processors, or a `_mixins.py` if large | `evaluation/metrics.py` → `ReferenceDataMixin` |

### 6.2 File Size Limits

| File type | Target | Hard limit | Action when exceeded |
|---|---|---|---|
| Single-processor file | < 250 lines | 400 lines | Extract helpers to `_utils.py` |
| Multi-processor file | < 400 lines | 600 lines | Split into separate files per processor or sub-group |
| Utility module | < 150 lines | 300 lines | Split by domain |

### 6.3 What Stays in the Processor File

Keep together when it's **cohesive** — when the helper only makes sense in the context of that processor:

```
correction/
  flex.py                ← shared D → A → N template lifecycle + default strategy
  aas.py                 ← legacy AAS matrix strategy only
  farm.py                ← FARM matrix strategy only
```

The shared epoch extraction, subtraction, realignment, and reporting lifecycle
belongs to ``Flex``. Matrix builders such as ``_calc_averaging_matrix`` stay on
the concrete strategy class because their selection rules differ, while each
strategy inherits the common Flex lifecycle.

### 6.4 When to Extract to a Utility Module

Move logic out when any of these apply:

1. **Two or more processors** import the same function (DRY).
2. **The function is a pure computation** — it takes arrays in and returns arrays out, with no knowledge of `Processor` or `ProcessingContext`.
3. **The processor file exceeds the hard limit** and the helper is self-contained.
4. **The helper is independently testable** and would benefit from its own focused test file.

Extraction targets:

| Scope of reuse | Target location | Naming |
|---|---|---|
| Within one module (e.g., `preprocessing`) | `preprocessing/_utils.py` | Leading underscore = private to the package |
| Across modules | `helpers/<descriptive_name>.py` | Public, documented |

Example — shared epoch extraction:

```
preprocessing/
  __init__.py
  alignment.py           ← imports from _utils
  trigger_detection.py   ← imports from _utils
  _utils.py              ← extract_epoch_with_padding(), get_pre_post_samples()
```

```python
# In alignment.py
from ._utils import extract_epoch_with_padding
```

### 6.5 When to Split a Multi-Processor File

Split when:
- The file exceeds 600 lines.
- Processors in the file don't share a base class or meaningful logic.
- A reader looking for one processor has to scroll past 500 lines of unrelated code.

Example — `evaluation/metrics.py` (992 lines, 8 processors) should become:

```
evaluation/
  __init__.py
  _mixins.py             ← ReferenceDataMixin
  snr.py                 ← SNRCalculator, LegacySNRCalculator
  rms.py                 ← RMSCalculator, RMSResidualCalculator
  spectral.py            ← FFTAllenCalculator, FFTNiazyCalculator
  artifacts.py           ← MedianArtifactCalculator
  report.py              ← MetricsReport
  visualization.py       ← RawPlotter (already separate)
```

### 6.6 Module `__init__.py` as the Public API

Regardless of internal file splits, the `__init__.py` is the public surface. Users import from the package, not from individual files:

```python
# Users write:
from facet.evaluation import SNRCalculator, RMSCalculator

# NOT:
from facet.evaluation.snr import SNRCalculator
```

When splitting files, update `__init__.py` to re-export everything. Internal file structure is an implementation detail:

```python
# evaluation/__init__.py
from .snr import SNRCalculator, LegacySNRCalculator
from .rms import RMSCalculator, RMSResidualCalculator
from .spectral import FFTAllenCalculator, FFTNiazyCalculator
from .artifacts import MedianArtifactCalculator
from .report import MetricsReport
from .visualization import RawPlotter
```

### 6.7 Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Processor file (single) | Lowercase, descriptive noun | `aas.py`, `anc.py`, `pca.py` |
| Processor file (group) | Lowercase, plural noun or domain | `filtering.py`, `transforms.py` |
| Private utility module | Leading underscore | `_utils.py`, `_mixins.py` |
| Shared helper module | Descriptive name in `helpers/` | `helpers/crosscorr.py` |

---

## 7. Logging Guidelines

### 7.1 Log Levels

| Level | When to use | Example |
|---|---|---|
| `logger.debug` | Internal algorithm details only useful during development or debugging. | `logger.debug("Epoch {} correlation: {:.4f}", i, corr)` |
| `logger.info` | Key milestones visible during normal operation. One or two per processor. | `logger.info("Detected {} triggers in {} s of data", n, duration)` |
| `logger.warning` | Recoverable issues that the user should be aware of. | `logger.warning("Channel {} excluded: zero variance", ch_name)` |
| `logger.error` | Unrecoverable issues (before raising an exception). | `logger.error("No triggers found — cannot proceed with correction")` |

### 7.2 Logging Rules

1. **One `logger.info` at the start** of `process()` summarizing what will happen:
   ```python
   logger.info("Applying AAS correction with window_size={}", self.window_size)
   ```

2. **One `logger.info` at the end** with key results (optional, only if meaningful):
   ```python
   logger.info("Correction complete: {} artifacts subtracted across {} channels", n_artifacts, n_channels)
   ```

3. **Use `logger.debug` for loop internals** — never `logger.info` inside a per-channel or per-epoch loop:
   ```python
   # WRONG — floods the log
   for ch in channels:
       logger.info("Processing channel {}", ch)

   # RIGHT
   for ch in channels:
       logger.debug("Processing channel {}", ch)
   ```

4. **Use loguru's lazy formatting** — never f-strings:
   ```python
   # WRONG — evaluates even if debug is disabled
   logger.debug(f"Matrix shape: {matrix.shape}")

   # RIGHT — lazy evaluation
   logger.debug("Matrix shape: {}", matrix.shape)
   ```

5. **Don't log in `__init__` or `validate()`** unless something unusual happens (a warning).

6. **Don't log in private helpers** unless they contain non-trivial iteration. Let `process()` own the narrative.

### 7.3 Structured Logging Context

For processors that iterate over channels, use loguru's `bind()` for context:

```python
with logger.contextualize(processor=self.name, step="correction"):
    for ch_idx, ch_name in enumerate(channels):
        logger.debug("Processing channel {}/{}", ch_idx + 1, len(channels))
```

---

## 8. Console & Progress Reporting

FACETpy ships a console system (`facet.console`) that provides live progress bars and metric displays during pipeline execution. The pipeline handles the high-level lifecycle (step started/completed), but **processors are responsible for reporting their own sub-step progress** when they perform long-running, iterative work.

### 8.1 Architecture Overview

```mermaid
flowchart TD
    A["Pipeline"] --> B["console.start_pipeline()"]
    B --> C["for each processor"]
    C --> D["console.step_started(i, name)"]
    D --> E["set_current_step_index(i)"]
    E --> F["processor.execute(context)"]
    F --> G["processor_progress(...)"]
    G --> H["set_current_step_index(None)"]
    H --> I["console.step_completed(i, name, duration)"]
    I --> J["console.pipeline_complete()"]
```

The pipeline sets a thread-local step index before calling your processor. This is what allows `processor_progress()` to route updates to the correct step in the console display. You don't need to manage this yourself — just use the APIs below.

### 8.2 When to Use Progress Reporting

| Situation | Use progress? | Reasoning |
|---|---|---|
| Iterating over channels (correction, evaluation) | **Yes** | Gives per-channel feedback on long operations |
| Single MNE call (filter, resample) | **No** | MNE is fast and has its own verbose output |
| Loading/exporting a file | **No** | Usually I/O-bound and fast enough |
| Iterating over epochs for template computation | **Yes** | Can be slow with many triggers |

**Rule of thumb**: if the loop body takes > 0.5 seconds per iteration and there are > 5 iterations, add a progress bar.

### 8.3 Using `processor_progress()`

Import and use it as a context manager inside `process()`:

```python
from facet.console import processor_progress

def process(self, context: ProcessingContext) -> ProcessingContext:
    raw = context.get_raw().copy()
    eeg_channels = mne.pick_types(raw.info, eeg=True)

    logger.info("Applying correction across {} channels", len(eeg_channels))

    with processor_progress(total=len(eeg_channels), message="Correcting") as progress:
        for idx, ch_idx in enumerate(eeg_channels):
            ch_name = raw.ch_names[ch_idx]

            corrected = self._correct_channel(raw._data[ch_idx], ...)
            raw._data[ch_idx] = corrected

            progress.advance(1, message=f"{ch_name} ({idx + 1}/{len(eeg_channels)})")

    return context.with_raw(raw)
```

**API:**

| Method | Purpose |
|---|---|
| `processor_progress(total, message)` | Create a progress context manager. `total` is the number of steps; `message` is the initial label. |
| `progress.advance(n, message=...)` | Move forward by `n` steps and optionally update the displayed message. |
| `progress.update(completed, message=...)` | Set absolute progress (instead of advancing). |
| `progress.complete()` | Mark as done (called automatically when the `with` block exits). |

### 8.4 Progress Message Conventions

Keep messages short and informative:

```python
# Good — channel name and count
progress.advance(1, message=f"{ch_name} ({idx + 1}/{total})")

# Good — phase name
progress.advance(1, message=f"Computing template for epoch {i}")

# Bad — too verbose
progress.advance(1, message=f"Now processing channel {ch_name} with cutoff {self.cutoff} Hz at index {idx}")

# Bad — no useful info
progress.advance(1)
```

### 8.5 Multiple Progress Phases

If a processor has two distinct phases (e.g., AAS computes averaging matrices first, then subtracts artifacts), use **two separate** `processor_progress()` blocks:

```python
def process(self, context):
    raw = context.get_raw().copy()
    channels = mne.pick_types(raw.info, eeg=True)

    with processor_progress(total=len(channels), message="Computing templates") as progress:
        for idx, ch_idx in enumerate(channels):
            self._compute_template(ch_idx, ...)
            progress.advance(1, message=f"Template {idx + 1}/{len(channels)}")

    with processor_progress(total=len(channels), message="Subtracting artifacts") as progress:
        for idx, ch_idx in enumerate(channels):
            self._subtract(ch_idx, ...)
            progress.advance(1, message=f"Corrected {idx + 1}/{len(channels)}")

    return context.with_raw(raw)
```

### 8.6 Reporting Metrics with `report_metric()`

Evaluation processors should report computed metrics so they appear in the console's metrics panel:

```python
from facet.console import report_metric

def process(self, context: ProcessingContext) -> ProcessingContext:
    snr_value = self._compute_snr(context)

    report_metric("snr", float(snr_value), unit="dB", fmt=".2f")

    new_metadata = context.metadata.copy()
    new_metadata.custom.setdefault("metrics", {})["snr"] = float(snr_value)
    return context.with_metadata(new_metadata)
```

**API:**

```python
report_metric(name: str, value: float, unit: str = "", fmt: str = ".4g")
```

- `name`: Metric key (e.g., `"snr"`, `"rms_ratio"`, `"rms_residual"`).
- `value`: Numeric value.
- `unit`: Display unit (e.g., `"dB"`, `"µV"`).
- `fmt`: Format specifier for display.

This reports to both the live console panel **and** to loguru (so it also appears in log files).

### 8.7 What Processors Should NOT Do

- **Don't call `console.step_started()` / `console.step_completed()`** — the pipeline manages the step lifecycle.
- **Don't call `set_current_step_index()`** — the pipeline sets this before invoking your processor.
- **Don't call `get_console()` directly** — use `processor_progress()` and `report_metric()` which handle the console routing internally.
- **Don't use `print()`** — use `logger.*` or `report_metric()`. In modern mode, stdout is captured and routed to the log panel, but raw prints look messy there.
- **Don't create Rich progress bars directly** — always use `processor_progress()` so it integrates with the console layout.

### 8.8 Console Modes

The system supports two modes, controlled by `FACET_CONSOLE_MODE`:

| Mode | Behavior |
|---|---|
| `modern` (default) | Rich-based live display with panels for steps, progress, metrics, and logs |
| `classic` | Plain loguru output, no live display |

Your processor code doesn't need to care about the mode — `processor_progress()` and `report_metric()` work in both. In classic mode, progress updates are simply logged; in modern mode, they update the live display.

---

## 9. Error Handling

### 9.1 Exception Types

| Exception | When to raise |
|---|---|
| `ProcessorValidationError` | In `validate()` — preconditions not met. |
| `ProcessorError` | In `process()` — algorithmic failure that the pipeline should catch. |
| Standard exceptions | Let `ValueError`, `TypeError`, etc. from NumPy/MNE propagate naturally. |

### 9.2 Rules

- **Don't catch and silence exceptions** unless you have a clear recovery strategy.
- **Don't return `None`** from `process()` to signal failure — raise an exception instead.
- **Include diagnostic information** in error messages (array shapes, parameter values):
  ```python
  raise ProcessorError(
      f"Artifact length ({artifact_len}) exceeds data length ({n_samples})"
  )
  ```
- **Don't use early returns** for error conditions in `process()` — raise instead:
  ```python
  # WRONG — silent failure
  if len(triggers) == 0:
      logger.warning("No triggers, returning unchanged")
      return context

  # RIGHT — fail fast
  # (This should have been caught in validate() anyway)
  raise ProcessorError("No triggers available for correction")
  ```

### 9.3 Graceful Degradation

Some processors legitimately need optional behavior (e.g., noise filtering is skipped when no noise estimate exists). Use explicit guard clauses with debug logging:

```python
if context.has_estimated_noise():
    noise = self._filter_noise(context.get_estimated_noise(), ...)
    new_ctx.set_estimated_noise(noise)
else:
    logger.debug("No noise estimate present — skipping noise filtering")
```

---

## 10. Performance Considerations

### 10.1 Memory

- **Copy raw only once** at the start of `process()`. Don't create intermediate copies.
- **Avoid full-array copies** when modifying a subset of channels/samples — use in-place operations on the copy.
- **Don't store large arrays** as instance attributes. Pass them through method arguments.
- **Use `context.cache_set()`** for expensive intermediate results that downstream processors might need.

### 10.2 Computation

- **Vectorize with NumPy** — avoid Python loops over samples or time points.
- **Use MNE built-ins** when available (filtering, resampling, epoch extraction).
- **Profile before optimizing** — don't add complexity for speculative performance gains.
- **Mark `parallel_safe = True`** only when the processor operates independently per channel and does not depend on cross-channel state.

### 10.3 I/O

- **Lazy-load data** when possible (use `preload=False` in MNE loaders for large files).
- **Don't read/write files** in processors that aren't I/O processors.

---

## 11. Import Conventions

### 11.1 All Imports at the Top

```python
import numpy as np
from scipy.signal import firls, butter
from loguru import logger
from ..core import Processor, ProcessingContext, ProcessorValidationError
```

### 11.2 When Lazy Imports Are Acceptable

Only for **optional heavyweight dependencies** that not all users have installed:

```python
def process(self, context):
    try:
        import tensorflow as tf
    except ImportError:
        raise ProcessorError(
            "TensorFlow is required for DeepCorrection. "
            "Install with: pip install facetpy[deeplearning]"
        )
```

Standard library, NumPy, SciPy, and MNE are **never** lazy-imported — they are always available.

### 11.3 No Inline Imports

```python
# WRONG — import inside a loop or deep in a method
def _compute(self, data):
    import numpy as np  # already a core dependency
    ...

# RIGHT — at module top
import numpy as np
```

---

## 12. Registration, Versioning & Lifecycle

### 12.1 Processor Registration

The project includes a global registry (`facet.core.registry`) that maps processor names to classes. Registration enables config-driven pipeline construction and serialization.

**Register every public processor** in its module's `__init__.py` or at the bottom of the processor file:

```python
from ..core.registry import register_processor

register_processor("aas_correction", AASCorrection)
```

Rules:
- The registry key must match the processor's `name` attribute.
- Don't register internal/experimental processors — only stable public ones.
- Registration happens at import time, not at instantiation.

### 12.2 Aliases

When a processor has both a short name and a descriptive name, provide an alias at the bottom of the file:

```python
AveragedArtifactSubtraction = AASCorrection
```

Rules:
- The alias is the long, spelled-out name; the class itself uses the short name.
- Export both from `__init__.py`.
- Register only the canonical short name, not the alias.

### 12.3 Versioning

Every processor has a `version` class attribute. Bump it when the processor's **output changes** for the same input:

| Change | Bump? | Example |
|---|---|---|
| Bug fix that changes output | Yes → patch (`1.0.0` → `1.0.1`) | Fixed off-by-one in epoch boundary |
| New parameter with default preserving old behavior | No | Added `method` param with default `"fir"` |
| New parameter that changes default behavior | Yes → minor (`1.0.0` → `1.1.0`) | Changed default window from 30 to 25 |
| Algorithm rewrite | Yes → major (`1.0.0` → `2.0.0`) | Switched from simple average to weighted average |

The version is recorded in the processing history, enabling reproducibility.

### 12.4 Deprecation

When renaming or removing a processor or parameter, provide a smooth transition:

**Deprecating a parameter:**

```python
import warnings

def __init__(self, window_size: int = 30, window: int | None = None):
    if window is not None:
        warnings.warn(
            "The 'window' parameter is deprecated, use 'window_size' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        window_size = window
    self.window_size = window_size
    super().__init__()
```

**Deprecating a processor:**

```python
class OldProcessor(NewProcessor):
    """Deprecated — use NewProcessor instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "OldProcessor is deprecated, use NewProcessor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
```

Rules:
- Use `warnings.warn` with `DeprecationWarning`, not `logger.warning` — deprecation warnings are part of Python's standard mechanism and can be filtered by users.
- Keep deprecated code for at least one minor version before removing.
- Document the deprecation in the class docstring.

### 12.5 `parallel_safe`, `channel_wise`, and `run_once` in Practice

Three class-level flags control how a processor participates in the two
execution modes.  They are **independent concepts**:

| Flag | Default | Meaning |
|---|---|---|
| `parallel_safe` | `True` | Processor can run in separate worker processes without data races |
| `channel_wise` | `False` | Processor can operate on a single-channel subset context (enables both `parallel=True` and `channel_sequential=True`) |
| `run_once` | `False` | In `channel_sequential` mode: execute only for the first channel; skip silently for the rest |

A processor is **parallel-safe** only when all of these hold:

1. **No cross-channel dependencies** — channel A's computation doesn't read channel B's data.
2. **No shared mutable state** — no writes to `self` during `process()`.
3. **No global side effects** — no file writes, shared caches, or global variables.
4. **Deterministic output** — same input always produces same output.

Common cases:

| Processor type | `parallel_safe` | `channel_wise` | `run_once` | Why |
|---|---|---|---|---|
| Per-channel filter | Yes | Yes | No | Each channel filtered independently |
| Per-channel correction (AAS, PCA) | Yes | Yes | No | Artifact template computed per channel |
| Trigger alignment | Yes | Yes | **Yes** | Only needs one reference channel; metadata result reused |
| Trigger detection | No | No | No | Reads annotations/stim channels globally |
| Resampling | Yes | Yes | No | Operates independently per channel |
| I/O (load, export) | No | No | No | File operations are inherently serial |
| Evaluation metrics | No | No | No | Typically aggregates across channels |

Minimal example of a custom channel-wise processor:

```python
class MyChannelProcessor(Processor):
    name         = "my_channel_processor"
    requires_raw = True
    modifies_raw = True
    parallel_safe = True   # no cross-channel dependencies
    channel_wise  = True   # can run on a single-channel context

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        # each invocation receives exactly one channel
        return context.with_raw(raw)
```

See `examples/channelwise_execution.py` for a complete runnable demonstration including flag inspection, backend comparison, and a benchmark helper.

---

## 13. Testing Contract

Every processor must have corresponding tests. The test structure mirrors the source structure.

### 13.1 Required Test Cases

| Test | What it verifies |
|---|---|
| `test_<processor>_basic` | Happy path with valid input produces expected output shape and type. |
| `test_<processor>_validation` | Invalid parameters or missing prerequisites raise `ProcessorValidationError`. |
| `test_<processor>_immutability` | Input context is unchanged after `process()`. |
| `test_<processor>_history` | History entry is added after execution via `execute()`. |
| `test_<processor>_edge_cases` | Empty data, single channel, single sample, etc. |

### 13.2 Test Structure

```python
class TestMyProcessor:
    def test_basic(self, sample_context):
        proc = MyProcessor(cutoff=10.0)
        result = proc.execute(sample_context)
        assert result.get_raw().info["sfreq"] == sample_context.get_sfreq()

    def test_validation_bad_cutoff(self, sample_context):
        proc = MyProcessor(cutoff=-1.0)
        with pytest.raises(ProcessorValidationError):
            proc.execute(sample_context)

    def test_immutability(self, sample_context):
        proc = MyProcessor(cutoff=10.0)
        original_data = sample_context.get_raw().get_data().copy()
        proc.execute(sample_context)
        np.testing.assert_array_equal(sample_context.get_raw().get_data(), original_data)
```

### 13.3 Markers

- `@pytest.mark.unit` — fast, no I/O, no heavy computation.
- `@pytest.mark.integration` — requires file I/O or full pipeline execution.
- `@pytest.mark.slow` — takes > 5 seconds.

---

## 14. Processor Checklist

Use this checklist before submitting a new or refactored processor:

### Structure
- [ ] Class inherits from `Processor` (or a valid subclass).
- [ ] `name` is unique, lowercase, snake_case.
- [ ] `description` is a concise, human-readable sentence.
- [ ] All four flags (`requires_triggers`, `requires_raw`, `modifies_raw`, `parallel_safe`) are explicitly set.
- [ ] `__init__` stores config, calls `super().__init__()` last, does nothing else.
- [ ] `validate()` calls `super().validate(context)` first (unless it's a loader).
- [ ] `process()` follows the 6-phase canonical flow: Extract → Log → Compute → Noise → Build → Return.
- [ ] `process()` always returns a `ProcessingContext` (never `None`).

### Context & Data Integrity
- [ ] Input context is never mutated.
- [ ] `raw.copy()` is called before any modification.
- [ ] New context is returned via `with_raw()` / `with_metadata()` / `with_triggers()`.
- [ ] No direct assignment to `context._metadata` or other private attributes.
- [ ] `_data` access is documented when used for performance.
- [ ] Noise estimate is propagated when transforming raw data (filter, resample).
- [ ] Chatty MNE calls are wrapped in `suppress_stdout()` or use `verbose=False`.

### Readability & Code Quality
- [ ] `process()` is ≤ 60 lines; complex logic is in private helpers.
- [ ] Cognitive complexity is ≤ 15 per method (target ≤ 10).
- [ ] No nested indentation deeper than 3 levels in any method.
- [ ] Blank lines separate logical groups (paragraph-style spacing).
- [ ] Variables have descriptive names (`artifact_template`, not `tmpl`).
- [ ] Complex expressions are broken into named intermediate variables.
- [ ] Guard clauses use early returns to keep main logic flat.
- [ ] Comments explain **why**, not **what**.
- [ ] No magic numbers — constants are named or parameterized.
- [ ] No code duplication with other processors — shared logic is extracted to utilities.
- [ ] All imports are at the module top (except optional heavyweight deps).
- [ ] No unused imports.

### File Organization
- [ ] Single-processor file is < 400 lines; multi-processor file is < 600 lines.
- [ ] Shared helpers used by 2+ processors live in `_utils.py` or `helpers/`.
- [ ] Related processor families (shared base class) are in one file.
- [ ] Complex standalone processors have their own file.
- [ ] `__init__.py` re-exports all public classes — users never import from internal files.

### Logging & Progress
- [ ] One `logger.info` at the start of `process()` with key parameters.
- [ ] `logger.debug` for internal details; never `logger.info` in loops.
- [ ] `logger.warning` for recoverable issues.
- [ ] Loguru lazy formatting (no f-strings inside `logger.*` calls).
- [ ] Long channel/epoch loops use `processor_progress()` for progress reporting.
- [ ] Evaluation processors use `report_metric()` for computed metrics.
- [ ] No direct `print()` calls — use `logger.*` or `report_metric()`.
- [ ] No direct use of `get_console()`, `set_current_step_index()`, or Rich progress bars.

### Error Handling
- [ ] Parameter validation happens in `validate()`, not `process()`.
- [ ] Errors include diagnostic info (shapes, values, parameter names).
- [ ] No silent failures (no early returns for error conditions without logging).

### Testing
- [ ] Basic happy-path test exists.
- [ ] Validation failure test exists.
- [ ] Immutability test exists.
- [ ] Edge case tests exist (empty data, single channel, etc.).
- [ ] Tests use fixtures from `conftest.py`.

### Type Hints & Documentation
- [ ] All public method signatures are fully type-annotated.
- [ ] Private helpers have at least return type annotations.
- [ ] Class docstring uses NumPy-style format with `Parameters` section.
- [ ] Class docstring explains what, why, and when to use.
- [ ] `__init__` parameters are documented (type + purpose).
- [ ] Non-obvious algorithmic choices have inline comments.

### Registration & Lifecycle
- [ ] Processor is registered via `register_processor()` if it's a stable public processor.
- [ ] `version` is bumped when output changes for the same input.
- [ ] Deprecated parameters use `warnings.warn(DeprecationWarning)`.
- [ ] `parallel_safe` is set correctly per §12.5 criteria.

---

## 15. Anti-Patterns (What to Avoid)

### 15.1 Mutating the Input Context

```python
# BAD
context._metadata.triggers = new_triggers
return context
```

### 15.2 Returning None

```python
# BAD — base class converts this to the input context, hiding the fact
# that your processor didn't do anything
def process(self, context):
    if not self.should_run:
        return None  # silent no-op
```

### 15.3 God Methods

```python
# BAD — 170-line process() with nested loops, conditionals, and inline math
def process(self, context):
    # ... 170 lines of interleaved computation, logging, and error handling
```

### 15.4 Mutating `self` During Processing

```python
# BAD — makes the processor non-reentrant and non-thread-safe
def process(self, context):
    self.sfreq = context.get_sfreq() * self.factor  # mutates self!
    ...
```

### 15.5 Lazy Imports for Core Dependencies

```python
# BAD — numpy and scipy are always available
def _compute(self, data):
    import numpy as np
    from scipy.signal import butter
```

### 15.6 Inline Imports in Methods

```python
# BAD
def process(self, context):
    import random  # standard library, put at top
```

### 15.7 Empty Validate

```python
# BAD — if you have nothing to validate, don't override at all;
# the base class already handles requires_raw and requires_triggers
def validate(self, context):
    pass
```

### 15.8 Swallowing Exceptions

```python
# BAD — hides bugs
try:
    result = self._compute(data)
except Exception:
    logger.warning("Computation failed, returning unchanged")
    return context
```

### 15.9 Inconsistent Parameter Naming

```python
# BAD — one says "freq", another says "frequency", another says "cutoff"
class HighPass(Processor):
    def __init__(self, freq): ...

class LowPass(Processor):
    def __init__(self, frequency): ...

class BandPass(Processor):
    def __init__(self, low_cutoff, high_cutoff): ...
```

Pick one convention and use it across related processors.

### 15.10 Duplicated Code Across Processors

If three loaders share the same bad-channel handling logic, extract it into a shared utility function or a base class method. Don't copy-paste.

---

## 16. Reference Template

Use this template when creating a new processor:

```python
"""
<Module docstring — one line describing the module's purpose.>
"""

import numpy as np
import mne
from loguru import logger

from ..core import Processor, ProcessingContext, ProcessorValidationError
from ..console import processor_progress


class MyProcessor(Processor):
    """Apply <concise description of what this does>.

    <Optional: Extended description, algorithmic background, references.>

    Parameters
    ----------
    param_a : float
        Description of param_a.
    param_b : int, optional
        Description of param_b (default: 10).
    """

    name = "my_processor"
    description = "Concise sentence for pipeline.describe()"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(self, param_a: float, param_b: int = 10):
        self.param_a = param_a
        self.param_b = param_b
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if self.param_a <= 0:
            raise ProcessorValidationError(
                f"param_a must be positive, got {self.param_a}"
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        sfreq = context.get_sfreq()
        eeg_channels = mne.pick_types(raw.info, eeg=True)

        logger.info("Running my_processor with param_a={}, param_b={}", self.param_a, self.param_b)

        data = raw.get_data()

        with processor_progress(total=len(eeg_channels), message="Processing") as progress:
            for idx, ch_idx in enumerate(eeg_channels):
                ch_name = raw.ch_names[ch_idx]
                data[ch_idx] = self._process_channel(data[ch_idx], sfreq)
                progress.advance(1, message=f"{ch_name} ({idx + 1}/{len(eeg_channels)})")

        raw._data[:] = data
        return context.with_raw(raw)

    def _process_channel(self, channel_data: np.ndarray, sfreq: float) -> np.ndarray:
        """Pure computation on a single channel — no context access."""
        ...
        return channel_data
```

For evaluation processors, also use `report_metric()`:

```python
from ..console import report_metric

class MyMetricCalculator(Processor):
    name = "my_metric"
    modifies_raw = False

    def process(self, context: ProcessingContext) -> ProcessingContext:
        value = self._compute(context)

        report_metric("my_metric", float(value), unit="dB", fmt=".2f")

        new_metadata = context.metadata.copy()
        new_metadata.custom.setdefault("metrics", {})["my_metric"] = float(value)
        return context.with_metadata(new_metadata)
```

---

## Appendix: Known Issues in Current Processors

The following issues were identified during the audit that triggered this guideline document. They should be addressed during refactoring:

| Issue | Affected Processors | Guideline Section |
|---|---|---|
| Direct `context._metadata` mutation | Resample, AASCorrection, TriggerAligner, ANCCorrection | §3.3 |
| Direct `raw._data` write without copy first | Filter (noise), SubsampleAligner | §3.2 |
| Inconsistent noise estimate propagation | Filter, NotchFilter, Resample | §3.6 |
| Missing `suppress_stdout()` on MNE calls | BIDSLoader | §3.7 |
| Empty `validate()` overrides | BIDSLoader | §4.3, §15.7 |
| `process()` exceeds 100 lines | ANCCorrection, MissingTriggerDetector, SubsampleAligner, TriggerDetector | §5.1 |
| Lazy imports for core deps (scipy, random) | ANCCorrection, TriggerAligner, QRSTriggerDetector | §11.1 |
| Missing type annotations | Various | §2.7 |
| Inconsistent docstring format | Various | §2.8 |
| No processor registration | Various | §12.1 |
| `self` mutation in `process()` | UpSample, DownSample | §15.4 |
| Duplicated bad-channel logic | BIDSLoader | §6.4, §15.10 |
| f-strings in logger calls | Various | §7.2 |
| Silent early returns on error | ANCCorrection | §9.2 |
| Inconsistent parameter naming | HighPassFilter vs LowPassFilter vs BandPassFilter | §15.9 |
| Unused imports | trigger_detection.py (pearsonr) | §11.3 |
| Dummy parent parameters (`sfreq=1.0`) | UpSample, DownSample | §15.4 |
| Monolithic multi-processor file (992 lines) | evaluation/metrics.py | §6.5 |
