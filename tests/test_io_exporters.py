"""Tests for EEG data exporters."""

from pathlib import Path

import mne
import pytest

from facet.core import ProcessingContext, ProcessorValidationError
from facet.io import exporters as exporters_module
from facet.io.exporters import (
    SUPPORTED_EXPORT_EXTENSIONS,
    BDFExporter,
    BIDSExporter,
    BrainVisionExporter,
    EDFExporter,
    EEGLABExporter,
    Exporter,
    FIFExporter,
)
from facet.io.loaders import SUPPORTED_EXTENSIONS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("exporter_class", "extension", "expected_format"),
    [
        (BDFExporter, ".bdf", "bdf"),
        (BrainVisionExporter, ".vhdr", "brainvision"),
        (EEGLABExporter, ".set", "eeglab"),
    ],
)
def test_raw_exporters_copy_metadata_but_not_signal_data(
    monkeypatch,
    sample_context,
    temp_dir,
    exporter_class,
    extension,
    expected_format,
):
    source_raw = sample_context.get_raw()
    captured = {}

    def fail_copy(self):
        raise AssertionError("Exporter must not copy the complete Raw object")

    def fake_export(self, fname, fmt="auto", *args, **kwargs):
        captured["raw"] = self
        captured["fname"] = fname
        captured["fmt"] = fmt

    monkeypatch.setattr(type(source_raw), "copy", fail_copy)
    monkeypatch.setattr(type(source_raw), "export", fake_export)

    out_path = temp_dir / f"out{extension}"
    result = exporter_class(path=str(out_path)).execute(sample_context)

    assert result is sample_context
    assert captured["raw"] is not source_raw
    assert captured["raw"]._data is source_raw._data
    assert captured["fname"] == str(out_path)
    assert captured["fmt"] == expected_format


def test_edf_exporter_sanitizes_copied_metadata_only(
    monkeypatch,
    sample_context,
    temp_dir,
):
    source_raw = sample_context.get_raw()
    with source_raw.info._unlock():
        source_raw.info["device_info"] = {"type": "Test Device"}

    captured = {}

    def fail_copy(self):
        raise AssertionError("Exporter must not copy the complete Raw object")

    def fake_export(self, fname, fmt="auto", *args, **kwargs):
        captured["raw"] = self
        captured["device_type"] = self.info["device_info"]["type"]

    monkeypatch.setattr(type(source_raw), "copy", fail_copy)
    monkeypatch.setattr(type(source_raw), "export", fake_export)

    result = EDFExporter(path=str(temp_dir / "out.edf")).execute(sample_context)

    assert result is sample_context
    assert captured["raw"] is not source_raw
    assert captured["raw"]._data is source_raw._data
    assert captured["device_type"] == "Test_Device"
    assert source_raw.info["device_info"]["type"] == "Test Device"


def test_fif_exporter_saves_context_raw_without_copy(
    monkeypatch,
    sample_context,
    temp_dir,
):
    source_raw = sample_context.get_raw()
    captured = {}

    def fail_copy(self):
        raise AssertionError("Exporter must not copy the complete Raw object")

    def fake_save(self, fname, *args, **kwargs):
        captured["raw"] = self
        captured["fname"] = fname

    monkeypatch.setattr(type(source_raw), "copy", fail_copy)
    monkeypatch.setattr(type(source_raw), "save", fake_save)

    out_path = temp_dir / "out.fif"
    result = FIFExporter(path=str(out_path)).execute(sample_context)

    assert result is sample_context
    assert captured["raw"] is source_raw
    assert captured["fname"] == str(out_path)


def test_exporting_lazy_raw_does_not_preload_input_context(
    monkeypatch,
    sample_edf_file,
    temp_dir,
):
    source_raw = mne.io.read_raw_edf(
        sample_edf_file,
        preload=False,
        verbose="ERROR",
    )
    context = ProcessingContext(raw=source_raw, raw_original=source_raw)

    def fake_export(self, fname, fmt="auto", *args, **kwargs):
        # BDF and EEGLAB writers preload their input internally.  Simulate that
        # behavior and verify it affects only the lightweight export shell.
        self.load_data()
        assert self.preload is True

    monkeypatch.setattr(type(source_raw), "export", fake_export)

    BDFExporter(path=str(temp_dir / "out.bdf")).execute(context)

    assert source_raw.preload is False
    assert not hasattr(source_raw, "_data")


def test_bids_exporter_copies_only_selected_channel_data(
    monkeypatch,
    sample_context,
    temp_dir,
):
    source_raw = sample_context.get_raw()
    source_raw.set_channel_types({source_raw.ch_names[-1]: "stim"})
    source_channels = source_raw.ch_names.copy()
    source_data = source_raw._data.copy()
    captured = {}

    def fail_copy(self):
        raise AssertionError("Exporter must not copy the complete Raw object")

    def fake_write_raw_bids(*, raw, bids_path, **kwargs):
        captured["raw"] = raw
        captured["bids_path"] = bids_path

    monkeypatch.setattr(type(source_raw), "copy", fail_copy)
    monkeypatch.setattr(exporters_module, "write_raw_bids", fake_write_raw_bids)

    result = BIDSExporter(
        root=str(temp_dir / "bids"),
        subject="01",
        task="test",
    ).execute(sample_context)

    assert result is sample_context
    assert captured["raw"] is not source_raw
    assert captured["raw"].ch_names == source_channels[:-1]
    assert source_raw.ch_names == source_channels
    assert source_raw._data.shape == source_data.shape
    assert (source_raw._data == source_data).all()


def test_top_level_export_convenience_function(monkeypatch, sample_context, temp_dir):
    import facet

    captured = {}

    def fake_execute(self, context):
        captured["context"] = context
        captured["path"] = self.path
        captured["overwrite"] = self.overwrite
        return context

    monkeypatch.setattr(Exporter, "execute", fake_execute)

    out_path = temp_dir / "convenience.set"
    result = facet.export(sample_context, str(out_path), overwrite=False)

    assert result is sample_context
    assert captured["context"] is sample_context
    assert captured["path"] == str(out_path)
    assert captured["overwrite"] is False


def test_supported_export_extensions_cover_loader_extensions():
    for ext in SUPPORTED_EXTENSIONS:
        assert ext in SUPPORTED_EXPORT_EXTENSIONS


def test_exporter_routes_to_edf_exporter(monkeypatch, sample_context, temp_dir):
    captured = {}

    def fake_process(self, context):
        captured["path"] = self.path
        captured["overwrite"] = self.overwrite
        return context

    monkeypatch.setattr(EDFExporter, "process", fake_process)

    out_path = temp_dir / "out.edf"
    result = Exporter(path=str(out_path), overwrite=False).execute(sample_context)

    assert result is sample_context
    assert captured["path"] == str(out_path)
    assert captured["overwrite"] is False


def test_exporter_routes_fif_gz_to_fif_exporter(monkeypatch, sample_context, temp_dir):
    captured = {}

    def fake_process(self, context):
        captured["path"] = self.path
        captured["overwrite"] = self.overwrite
        return context

    monkeypatch.setattr(FIFExporter, "process", fake_process)

    out_path = temp_dir / "out.fif.gz"
    result = Exporter(path=str(out_path), overwrite=True).execute(sample_context)

    assert result is sample_context
    assert captured["path"] == str(out_path)
    assert captured["overwrite"] is True


def test_eeglab_exporter_uses_set_format(monkeypatch, sample_context, temp_dir):
    captured = {}
    raw_type = type(sample_context.get_raw())

    def fake_export(self, fname, fmt="auto", *args, **kwargs):
        captured["fname"] = fname
        captured["fmt"] = fmt
        captured["overwrite"] = kwargs.get("overwrite")

    monkeypatch.setattr(raw_type, "export", fake_export)

    out_path = temp_dir / "out.set"
    result = EEGLABExporter(path=str(out_path), overwrite=False).execute(sample_context)

    assert result is sample_context
    assert captured["fname"] == str(out_path)
    assert captured["fmt"] == "eeglab"
    assert captured["overwrite"] is False


def test_exporter_rejects_gdf_with_clear_error(sample_context, temp_dir):
    out_path = temp_dir / "out.gdf"
    with pytest.raises(ProcessorValidationError, match="GDF export is not supported"):
        Exporter(path=str(out_path)).execute(sample_context)


def test_exporter_rejects_unknown_extension(sample_context, temp_dir):
    out_path = Path(temp_dir) / "out.unknown"
    with pytest.raises(ProcessorValidationError, match="Unsupported export extension"):
        Exporter(path=str(out_path)).execute(sample_context)
