"""Tests for memory-aware channel-parallel execution."""

import mne
import numpy as np
import pytest

from facet.core import ParallelExecutor, ProcessingContext, Processor

pytestmark = pytest.mark.unit


class _ScaleChannels(Processor):
    """Simple channel-wise processor with deterministic signal/noise output."""

    name = "scale_channels"
    parallel_safe = True
    channel_wise = True

    def __init__(self, factor: float = 2.0):
        self.factor = factor
        super().__init__()

    def _get_parameters(self):
        return {"factor": self.factor}

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        raw._data *= self.factor
        noise = None
        if context.has_estimated_noise():
            noise = context.get_estimated_noise() * self.factor
        return context.with_raw(
            raw,
            estimated_noise=noise,
            copy_estimated_noise=False,
        )


@pytest.fixture
def parallel_context():
    """Return a small context with annotations and estimated noise."""
    data = np.arange(4 * 40, dtype=float).reshape(4, 40) * 1e-6
    info = mne.create_info(
        ["EEG001", "EEG002", "EEG003", "EEG004"],
        sfreq=20.0,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, first_samp=100, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.25],
            duration=[0.1],
            description=["marker"],
            orig_time=raw.info["meas_date"],
        )
    )
    context = ProcessingContext(raw=raw)
    context.set_estimated_noise(np.arange(data.size, dtype=float).reshape(data.shape))
    return context


def test_channel_subset_avoids_full_raw_copy(monkeypatch, parallel_context):
    """Selected-channel construction must not copy then discard the full Raw."""
    raw = parallel_context.get_raw()

    def fail_full_copy():
        raise AssertionError("full Raw.copy() should not be called")

    monkeypatch.setattr(raw, "copy", fail_full_copy)

    subset = ParallelExecutor(n_jobs=2, backend="serial")._create_channel_subset_context(
        parallel_context,
        [2, 0],
    )

    assert subset.get_raw().ch_names == ["EEG003", "EEG001"]
    assert subset.get_raw().first_samp == raw.first_samp
    np.testing.assert_array_equal(
        subset.get_data(),
        raw._data[[2, 0]],
    )
    np.testing.assert_array_equal(
        subset.get_estimated_noise(),
        parallel_context.get_estimated_noise()[[2, 0]],
    )
    assert subset.get_raw().annotations.description.tolist() == ["marker"]
    np.testing.assert_allclose(
        subset.get_raw().annotations.onset,
        raw.annotations.onset,
    )


def test_channel_parallel_merge_preserves_serial_output(parallel_context):
    """Splitting and merging channels must not alter numerical output."""
    processor = _ScaleChannels(factor=3.0)
    expected = processor.execute(parallel_context)

    actual = ParallelExecutor(n_jobs=2, backend="serial").execute(
        processor,
        parallel_context,
    )

    np.testing.assert_array_equal(actual.get_data(), expected.get_data())
    np.testing.assert_array_equal(
        actual.get_estimated_noise(),
        expected.get_estimated_noise(),
    )
    assert actual.get_raw().ch_names == expected.get_raw().ch_names
    assert actual.get_sfreq() == expected.get_sfreq()


def test_multiprocessing_preserves_serial_output(monkeypatch, parallel_context):
    """Spawned workers should preserve the same deterministic channel output."""
    processor = _ScaleChannels(factor=3.0)
    expected = processor.execute(parallel_context)
    executor = ParallelExecutor(n_jobs=2, backend="multiprocessing")
    monkeypatch.setattr(
        executor,
        "_available_memory_bytes",
        lambda: 16 * 1024**3,
    )

    actual = executor.execute(processor, parallel_context)

    np.testing.assert_array_equal(actual.get_data(), expected.get_data())
    np.testing.assert_array_equal(
        actual.get_estimated_noise(),
        expected.get_estimated_noise(),
    )


def test_multiprocessing_worker_count_is_capped_by_available_memory(
    monkeypatch,
    parallel_context,
):
    """Automatic safety should reduce jobs when process overhead exceeds RAM."""
    executor = ParallelExecutor(n_jobs=4, backend="multiprocessing")
    monkeypatch.setattr(
        executor,
        "_available_memory_bytes",
        lambda: 200 * 1024**2,
    )

    worker_count = executor._resolve_worker_count(
        processor=_ScaleChannels(),
        context=parallel_context,
        task_count=4,
    )

    # Half of 200 MiB is reserved for workers. One 64 MiB worker fits, while
    # two workers exceed the allowance before their signal arrays are counted.
    assert worker_count == 1


def test_multiprocessing_inputs_are_generated_lazily(
    monkeypatch,
    parallel_context,
):
    """Worker payloads should be produced as the pool consumes them."""
    executor = ParallelExecutor(n_jobs=2, backend="multiprocessing")
    created = []
    original_create = executor._create_channel_subset_context

    def record_create(context, channel_indices):
        created.append(channel_indices)
        return original_create(context, channel_indices)

    monkeypatch.setattr(executor, "_create_channel_subset_context", record_create)

    class _InlinePool:
        def __init__(self, processes):
            assert processes == 2

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def imap(self, worker, contexts, chunksize):
            assert chunksize == 1
            assert created == []
            assert not isinstance(contexts, list)
            for context_data in contexts:
                yield worker(context_data)

    monkeypatch.setattr("facet.core.parallel.mp.Pool", _InlinePool)

    results = executor._execute_multiprocessing(
        processor=_ScaleChannels(),
        context=parallel_context,
        channel_chunks=[[0, 1], [2, 3]],
        worker_count=2,
    )

    assert created == [[0, 1], [2, 3]]
    assert len(results) == 2
