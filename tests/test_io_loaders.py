"""Tests for EEG data loaders."""

import mne
import numpy as np
import pytest

from facet.core import ProcessorValidationError
from facet.io.loaders import _EXTENSION_READERS, BIDSLoader, Loader, _apply_sample_window

pytestmark = pytest.mark.unit


@pytest.fixture
def raw_factory():
    """Return a factory that creates synthetic Raw objects."""

    def _factory(n_times: int = 500, sfreq: float = 100.0) -> mne.io.RawArray:
        rng = np.random.RandomState(42)
        ch_names = ["EEG001", "EEG002"]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        data = rng.standard_normal((len(ch_names), n_times)) * 1e-6
        return mne.io.RawArray(data, info, verbose=False)

    return _factory


def test_apply_sample_window_crops_data(raw_factory):
    raw = raw_factory()
    total_samples = raw.n_times

    cropped_raw, start, stop = _apply_sample_window(raw, 10, 60)

    assert cropped_raw.n_times == 50
    assert start == 10
    assert stop == 60
    assert total_samples == 500


def test_apply_sample_window_handles_open_end(raw_factory):
    raw = raw_factory()
    total_samples = raw.n_times

    cropped_raw, start, stop = _apply_sample_window(raw, 100, None)

    assert cropped_raw.n_times == total_samples - 100
    assert start == 100
    assert stop == total_samples


def test_apply_sample_window_validates_bounds(raw_factory):
    raw = raw_factory()

    with pytest.raises(ValueError):
        _apply_sample_window(raw, -1, 10)

    raw = raw_factory()
    with pytest.raises(ValueError):
        _apply_sample_window(raw, 10, 5)

    raw = raw_factory()
    with pytest.raises(ValueError):
        _apply_sample_window(raw, 0, 10_000)


def test_auto_loader_applies_sample_window(monkeypatch, raw_factory):
    read_preload_values = []

    def fake_read_raw_edf(path, *args, **kwargs):
        read_preload_values.append(kwargs.get("preload"))
        return raw_factory()

    monkeypatch.setitem(_EXTENSION_READERS, ".edf", (fake_read_raw_edf, "EDF"))

    loader = Loader(path="./examples/datasets/NiazyFMRI.edf", start_sample=50, stop_sample=150)
    context = loader.execute(None)

    raw = context.get_raw()
    metadata = context.metadata

    assert raw.n_times == 100
    assert metadata.acq_start_sample == 50
    assert metadata.acq_end_sample == 150
    assert read_preload_values == [False]


def test_auto_loader_invalid_window_raises(monkeypatch, raw_factory):
    def fake_read_raw_edf(path, *args, **kwargs):
        return raw_factory()

    monkeypatch.setitem(_EXTENSION_READERS, ".edf", (fake_read_raw_edf, "EDF"))

    loader = Loader(path="./examples/datasets/NiazyFMRI.edf", start_sample=600, stop_sample=550)

    with pytest.raises(ProcessorValidationError):
        loader.execute(None)


def test_bids_loader_passes_run_to_bids_path(monkeypatch, raw_factory, tmp_path):
    captured: dict[str, object] = {}

    def fake_read_raw_bids(bids_path, *args, **kwargs):
        captured["bids_path"] = bids_path
        return raw_factory()

    monkeypatch.setattr("facet.io.loaders.read_raw_bids", fake_read_raw_bids)

    loader = BIDSLoader(
        root=str(tmp_path),
        subject="01",
        session="01",
        task="rest",
        run="02",
    )

    context = loader.execute(None)

    assert context.get_raw().n_times == 500
    bids_path = captured["bids_path"]
    assert bids_path.run == "02"
