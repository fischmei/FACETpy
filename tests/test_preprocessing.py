"""
Tests for preprocessing processors.
"""

import mne
import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata, ProcessorValidationError
from facet.preprocessing import (
    AnalyzeDataReport,
    BandPassFilter,
    ChannelStandardizer,
    CheckDataReport,
    CutAcquisitionWindow,
    DownSample,
    DropChannelsMatching,
    HighPassFilter,
    LowPassFilter,
    MagicErasor,
    MATLABPreFilter,
    MissingTriggerCompleter,
    SliceAligner,
    SliceTriggerGenerator,
    SubsampleAligner,
    TriggerAligner,
    TriggerDetector,
    UpSample,
)


@pytest.mark.unit
class TestTriggerDetector:
    """Tests for TriggerDetector processor."""

    def test_trigger_detection_from_annotations(self, sample_edf_file):
        """Test detecting triggers from annotations."""
        from facet.io import Loader

        # Load file with triggers
        loader = Loader(path=str(sample_edf_file), preload=True)
        context = loader.execute(None)

        # Detect triggers
        detector = TriggerDetector(regex=r"\b1\b")
        result = detector.execute(context)

        # Check triggers were found
        assert result.has_triggers()
        triggers = result.get_triggers()
        assert len(triggers) > 0

    def test_trigger_detection_sets_artifact_length(self, sample_edf_file):
        """Test that trigger detection calculates artifact length."""
        from facet.io import Loader

        loader = Loader(path=str(sample_edf_file), preload=True)
        context = loader.execute(None)

        detector = TriggerDetector(regex=r"\b1\b")
        result = detector.execute(context)

        # Check artifact length was calculated
        assert result.get_artifact_length() is not None
        assert result.get_artifact_length() > 0

    def test_no_triggers_found(self, sample_raw):
        """Test behavior when no triggers are found."""
        context = ProcessingContext(raw=sample_raw)

        # Try to detect with pattern that won't match
        detector = TriggerDetector(regex=r"NONEXISTENT")
        result = detector.execute(context)

        # Should return unchanged context
        assert not result.has_triggers()

    def test_trigger_detection_from_eeg_status_channel(self):
        """Detect binary triggers when a Status channel is not typed as stim."""
        sfreq = 250.0
        n_samples = 1000
        trigger_positions = np.array([100, 250, 400, 550, 700], dtype=int)
        data = np.random.randn(3, n_samples) * 1e-6
        status = np.zeros(n_samples)
        status[trigger_positions] = 1
        data = np.vstack([data, status])

        info = mne.create_info(
            ch_names=["EEG001", "EEG002", "EEG003", "Status"],
            sfreq=sfreq,
            ch_types=["eeg", "eeg", "eeg", "eeg"],
        )
        raw = mne.io.RawArray(data, info, verbose=False)
        context = ProcessingContext(raw=raw)

        result = TriggerDetector(regex=r"\b1\b").execute(context)

        assert result.has_triggers()
        np.testing.assert_array_equal(result.get_triggers(), trigger_positions)

    def test_trigger_detection_accepts_weighted_positive_channel(self):
        """Detect trigger channels whose pulse value is positive but not one."""
        sfreq = 250.0
        n_samples = 1000
        trigger_positions = np.array([100, 250, 400, 550, 700], dtype=int)
        data = np.random.randn(2, n_samples) * 1e-6
        trigger_channel = np.zeros(n_samples)
        trigger_channel[trigger_positions[:2]] = 1
        trigger_channel[trigger_positions[2:]] = 8
        data = np.vstack([data, trigger_channel])

        info = mne.create_info(
            ch_names=["EEG001", "EEG002", "Aux"],
            sfreq=sfreq,
            ch_types=["eeg", "eeg", "eeg"],
        )
        raw = mne.io.RawArray(data, info, verbose=False)
        context = ProcessingContext(raw=raw)

        result = TriggerDetector(regex=r"\b1\b").execute(context)

        assert result.has_triggers()
        np.testing.assert_array_equal(result.get_triggers(), trigger_positions)

    def test_discrete_channel_still_respects_nonmatching_regex(self):
        """Do not use positive fallback when the regex is not trigger-like."""
        sfreq = 250.0
        n_samples = 1000
        data = np.random.randn(2, n_samples) * 1e-6
        trigger_channel = np.zeros(n_samples)
        trigger_channel[[100, 250, 400]] = 8
        data = np.vstack([data, trigger_channel])

        info = mne.create_info(
            ch_names=["EEG001", "EEG002", "Status"],
            sfreq=sfreq,
            ch_types=["eeg", "eeg", "eeg"],
        )
        raw = mne.io.RawArray(data, info, verbose=False)
        context = ProcessingContext(raw=raw)

        result = TriggerDetector(regex=r"NONEXISTENT").execute(context)

        assert not result.has_triggers()


@pytest.mark.unit
class TestResampling:
    """Tests for resampling processors."""

    def test_upsample(self, sample_context):
        """Test upsampling."""
        original_sfreq = sample_context.get_raw().info["sfreq"]

        upsampler = UpSample(factor=2)
        result = upsampler.execute(sample_context)

        new_sfreq = result.get_raw().info["sfreq"]

        # Check frequency doubled
        assert new_sfreq == original_sfreq * 2

    def test_downsample(self, sample_context):
        """Test downsampling."""
        # First upsample
        upsampler = UpSample(factor=4)
        context_upsampled = upsampler.execute(sample_context)

        original_sfreq = context_upsampled.get_raw().info["sfreq"]

        # Then downsample
        downsampler = DownSample(factor=2)
        result = downsampler.execute(context_upsampled)

        new_sfreq = result.get_raw().info["sfreq"]

        # Check frequency halved
        assert new_sfreq == original_sfreq / 2

    def test_resampling_updates_triggers(self, sample_context):
        """Test that resampling updates trigger positions."""
        original_triggers = sample_context.get_triggers().copy()
        original_sfreq = sample_context.get_raw().info["sfreq"]

        # Upsample by 2
        upsampler = UpSample(factor=2)
        result = upsampler.execute(sample_context)

        new_triggers = result.get_triggers()
        new_sfreq = result.get_raw().info["sfreq"]

        # Triggers should be scaled
        scaling_factor = new_sfreq / original_sfreq
        expected_triggers = (original_triggers * scaling_factor).astype(int)

        np.testing.assert_array_equal(new_triggers, expected_triggers)

    def test_upsample_scales_sample_metadata(self, sample_context):
        """Upsampling should scale artifact/window metadata in sample units."""
        metadata = sample_context.metadata.copy()
        metadata.artifact_length = 51
        metadata.pre_trigger_samples = 21
        metadata.post_trigger_samples = 30
        metadata.acq_start_sample = 101
        metadata.acq_end_sample = 901
        metadata.custom["acquisition"] = {
            "pre_trigger_samples": 21,
            "post_trigger_samples": 30,
            "acq_start_sample": 101,
            "acq_end_sample": 901,
        }
        context = sample_context.with_metadata(metadata)

        result = UpSample(factor=2).execute(context)

        assert result.metadata.artifact_length == 102
        assert result.metadata.pre_trigger_samples == 42
        assert result.metadata.post_trigger_samples == 60
        assert result.metadata.acq_start_sample == 202
        assert result.metadata.acq_end_sample == 1802
        assert result.metadata.custom["acquisition"]["pre_trigger_samples"] == 42
        assert result.metadata.custom["acquisition"]["post_trigger_samples"] == 60
        assert result.metadata.custom["acquisition"]["acq_start_sample"] == 202
        assert result.metadata.custom["acquisition"]["acq_end_sample"] == 1802

    def test_downsample_scales_sample_metadata(self, sample_context):
        """Downsampling should scale artifact/window metadata in sample units."""
        metadata = sample_context.metadata.copy()
        metadata.artifact_length = 51
        metadata.pre_trigger_samples = 21
        metadata.post_trigger_samples = 30
        metadata.acq_start_sample = 101
        metadata.acq_end_sample = 901
        metadata.custom["acquisition"] = {
            "pre_trigger_samples": 21,
            "post_trigger_samples": 30,
            "acq_start_sample": 101,
            "acq_end_sample": 901,
        }
        context = sample_context.with_metadata(metadata)

        result = DownSample(factor=2).execute(context)

        # Half-up rounding: 51/2 -> 25.5 -> 26, 21/2 -> 10.5 -> 11.
        assert result.metadata.artifact_length == 26
        assert result.metadata.pre_trigger_samples == 11
        assert result.metadata.post_trigger_samples == 15
        assert result.metadata.acq_start_sample == 51
        assert result.metadata.acq_end_sample == 451
        assert result.metadata.custom["acquisition"]["pre_trigger_samples"] == 11
        assert result.metadata.custom["acquisition"]["post_trigger_samples"] == 15
        assert result.metadata.custom["acquisition"]["acq_start_sample"] == 51
        assert result.metadata.custom["acquisition"]["acq_end_sample"] == 451

    def test_upsample_downsample_roundtrip(self, sample_context):
        """Test upsampling then downsampling returns to original."""
        original_sfreq = sample_context.get_raw().info["sfreq"]
        original_shape = sample_context.get_raw()._data.shape

        # Upsample then downsample
        upsampler = UpSample(factor=4)
        downsampler = DownSample(factor=4)

        context_up = upsampler.execute(sample_context)
        context_down = downsampler.execute(context_up)

        # Check we're back to original frequency
        final_sfreq = context_down.get_raw().info["sfreq"]
        assert final_sfreq == original_sfreq

        # Shape should be similar (may differ slightly due to resampling)
        final_shape = context_down.get_raw()._data.shape
        assert final_shape[0] == original_shape[0]  # Same channels
        assert abs(final_shape[1] - original_shape[1]) < 10  # Similar samples


@pytest.mark.unit
class TestTriggerAligner:
    """Tests for TriggerAligner processor."""

    def test_trigger_alignment(self, sample_context):
        """Test trigger alignment."""
        aligner = TriggerAligner(ref_trigger_index=0)
        result = aligner.execute(sample_context)

        # Check triggers were aligned
        assert result.has_triggers()

        # Triggers may have changed slightly
        original_triggers = sample_context.get_triggers()
        aligned_triggers = result.get_triggers()

        assert len(aligned_triggers) == len(original_triggers)

    def test_alignment_requires_triggers(self, sample_raw):
        """Test that alignment requires triggers."""
        context = ProcessingContext(raw=sample_raw)

        aligner = TriggerAligner(ref_trigger_index=0)

        with pytest.raises(ProcessorValidationError):
            aligner.execute(context)


@pytest.mark.unit
class TestAcquisitionAlignment:
    """Tests for acquisition window and slice/subsample alignment."""

    def _build_shifted_context(self, shift_samples=3):
        """Create a minimal context with a known shift between artifacts."""
        sfreq = 200
        artifact_length = 40
        n_samples = 400
        triggers = np.array([100, 200])

        data = np.zeros((1, n_samples), dtype=float)
        template = np.sin(np.linspace(0, np.pi, artifact_length)) * 1e-6

        # Reference artifact aligned with trigger
        data[0, triggers[0] : triggers[0] + artifact_length] += template
        # Second artifact shifted in time
        shifted_start = triggers[1] + shift_samples
        data[0, shifted_start : shifted_start + artifact_length] += template

        info = mne.create_info(ch_names=["EEG001"], sfreq=sfreq, ch_types=["eeg"])
        raw = mne.io.RawArray(data, info, verbose=False)

        metadata = ProcessingMetadata()
        metadata.triggers = triggers
        metadata.artifact_length = artifact_length
        metadata.artifact_to_trigger_offset = 0.0
        metadata.upsampling_factor = 1

        return ProcessingContext(raw=raw, raw_original=raw.copy(), metadata=metadata)

    def test_cut_acquisition_sets_metadata(self):
        """CutAcquisitionWindow should derive acquisition metadata."""
        context = self._build_shifted_context()
        cutter = CutAcquisitionWindow()
        result = cutter.execute(context)

        assert result.metadata.acq_start_sample is not None
        assert result.metadata.acq_end_sample is not None
        assert result.metadata.pre_trigger_samples is not None
        assert result.metadata.post_trigger_samples is not None

    def test_slice_aligner_corrects_shift(self):
        """SliceAligner should align shifted triggers on upsampled data."""
        context = self._build_shifted_context(shift_samples=3)
        context = CutAcquisitionWindow().execute(context)

        aligner = SliceAligner(ref_trigger_index=0, search_window=6)
        result = aligner.execute(context)

        aligned = result.get_triggers()
        assert aligned[0] == context.get_triggers()[0]
        assert aligned[1] - context.get_triggers()[1] == 3

    def test_subsample_aligner_records_shifts(self):
        """SubsampleAligner should adjust triggers and record shift metadata."""
        context = self._build_shifted_context(shift_samples=2)
        context = CutAcquisitionWindow().execute(context)

        aligner = SubsampleAligner(ref_trigger_index=0, search_window=5)
        result = aligner.execute(context)

        aligned = result.get_triggers()
        expected = context.get_triggers()[1] + 2
        assert abs(aligned[1] - expected) <= 3

        alignment_meta = result.metadata.custom.get("subsample_alignment")
        assert alignment_meta is not None
        recorded_shift = alignment_meta["shifts"][1]
        assert abs(recorded_shift - 2) <= 3

    def test_alignment_requires_artifact_length(self, sample_context):
        """Test that alignment requires artifact length."""
        # Remove artifact length
        metadata = sample_context.metadata.copy()
        metadata.artifact_length = None
        context = sample_context.with_metadata(metadata)

        aligner = TriggerAligner(ref_trigger_index=0)

        with pytest.raises(ProcessorValidationError):
            aligner.execute(context)


@pytest.mark.unit
class TestFiltering:
    """Tests for filtering processors."""

    def test_highpass_filter(self, sample_context):
        """Test highpass filtering."""
        original_data = sample_context.get_raw()._data.copy()

        hpf = HighPassFilter(freq=1.0)
        result = hpf.execute(sample_context)

        filtered_data = result.get_raw()._data

        # Data should have changed
        assert not np.array_equal(original_data, filtered_data)

        # Check it's still reasonable
        assert not np.any(np.isnan(filtered_data))
        assert not np.any(np.isinf(filtered_data))

    def test_lowpass_filter(self, sample_context):
        """Test lowpass filtering."""
        original_data = sample_context.get_raw()._data.copy()

        lpf = LowPassFilter(freq=40.0)
        result = lpf.execute(sample_context)

        filtered_data = result.get_raw()._data

        # Data should have changed
        assert not np.array_equal(original_data, filtered_data)

    def test_bandpass_filter(self, sample_context):
        """Test bandpass filtering."""
        original_data = sample_context.get_raw()._data.copy()

        bpf = BandPassFilter(l_freq=1.0, h_freq=40.0)
        result = bpf.execute(sample_context)

        filtered_data = result.get_raw()._data

        # Data should have changed
        assert not np.array_equal(original_data, filtered_data)

    def test_filter_also_filters_noise(self, sample_context_with_noise):
        """Test that filtering also filters estimated noise."""
        original_noise = sample_context_with_noise.get_estimated_noise().copy()

        hpf = HighPassFilter(freq=1.0)
        result = hpf.execute(sample_context_with_noise)

        filtered_noise = result.get_estimated_noise()

        # Noise should have been filtered too
        assert not np.array_equal(original_noise, filtered_noise)

    def test_filter_chain(self, sample_context):
        """Test chaining multiple filters."""
        from facet.core import Pipeline

        pipeline = Pipeline([HighPassFilter(freq=0.5), LowPassFilter(freq=50.0)])

        result = pipeline.run(initial_context=sample_context)

        assert result.success is True
        filtered_data = result.context.get_raw()._data

        # Should be finite
        assert np.all(np.isfinite(filtered_data))


@pytest.mark.unit
class TestMATLABPreFilter:
    """Tests for MATLABPreFilter processor."""

    def test_prefilter_gaussian_hp_changes_signal(self, sample_context):
        original = sample_context.get_raw()._data.copy()
        processor = MATLABPreFilter(gauss_hp_frequency=1.0)
        result = processor.execute(sample_context)
        filtered = result.get_raw()._data

        assert filtered.shape == original.shape
        assert not np.array_equal(filtered, original)

    def test_prefilter_custom_transfer_function(self, sample_context):
        original = sample_context.get_raw()._data.copy()
        processor = MATLABPreFilter(
            transfer_frequencies=np.array([0.0, 0.2, 0.25, 1.0]),
            transfer_amplitudes=np.array([0.0, 0.0, 1.0, 1.0]),
            gauss_hp_frequency=None,
        )
        result = processor.execute(sample_context)
        filtered = result.get_raw()._data
        assert filtered.shape == original.shape
        assert not np.array_equal(filtered, original)

    def test_prefilter_propagates_noise(self, sample_context_with_noise):
        original_noise = sample_context_with_noise.get_estimated_noise().copy()
        processor = MATLABPreFilter(gauss_hp_frequency=1.0)
        result = processor.execute(sample_context_with_noise)
        assert result.get_estimated_noise() is not None
        assert not np.array_equal(result.get_estimated_noise(), original_noise)


@pytest.mark.unit
class TestMATLABTriggerParity:
    """Tests for deterministic missing-trigger completion and slice generation."""

    def test_missing_trigger_completer(self, sample_raw):
        missing = np.array([100, 140, 180, 260, 300, 340, 380], dtype=int)  # missing one trigger

        metadata = ProcessingMetadata(
            triggers=missing,
            artifact_length=40,
            volume_gaps=True,
            slices_per_volume=4,
        )
        context = ProcessingContext(raw=sample_raw, raw_original=sample_raw.copy(), metadata=metadata)

        processor = MissingTriggerCompleter(volumes=2, slices=4, add_annotations=False, strict=False)
        result = processor.execute(context)

        assert result.has_triggers()
        assert len(result.get_triggers()) == 8
        assert 220 in result.get_triggers()

    def test_slice_trigger_generator(self, sample_raw):
        metadata = ProcessingMetadata(triggers=np.array([100, 500], dtype=int), artifact_length=400, volume_gaps=True)
        context = ProcessingContext(raw=sample_raw, raw_original=sample_raw.copy(), metadata=metadata)

        processor = SliceTriggerGenerator(slices=4, duration_samples=50, relative_position=0.0)
        result = processor.execute(context)

        expected = np.array([100, 150, 200, 250, 500, 550, 600, 650], dtype=int)
        np.testing.assert_array_equal(result.get_triggers(), expected)
        assert result.metadata.slices_per_volume == 4


@pytest.mark.unit
class TestDiagnosticReports:
    """Tests for AnalyzeDataReport and CheckDataReport processors."""

    def test_analyze_data_report(self, sample_context):
        result = AnalyzeDataReport().execute(sample_context)
        report = result.metadata.custom.get("analyze_data_report")
        assert report is not None
        assert report["samples"] == sample_context.get_raw().n_times
        assert report["channels"] == sample_context.get_n_channels()

    def test_check_data_report_passes(self, sample_context):
        result = CheckDataReport(require_triggers=True, strict=True).execute(sample_context)
        report = result.metadata.custom.get("check_data_report")
        assert report is not None
        assert report["num_errors"] == 0

    def test_check_data_report_non_strict(self, sample_raw):
        context = ProcessingContext(raw=sample_raw)
        result = CheckDataReport(require_triggers=True, strict=False).execute(context)
        report = result.metadata.custom.get("check_data_report")
        assert report is not None
        assert report["num_errors"] >= 1


@pytest.mark.unit
class TestCrop:
    """Tests for the Crop processor."""

    def test_crop_tmin_tmax(self, sample_context):
        """Crop with tmin and tmax shortens the recording."""
        from facet.preprocessing import Crop

        raw = sample_context.get_raw()
        original_duration = raw.times[-1]

        tmin = 1.0
        tmax = original_duration - 1.0
        result = sample_context | Crop(tmin=tmin, tmax=tmax)

        new_duration = result.get_raw().times[-1]
        assert new_duration < original_duration

    def test_crop_tmin_only(self, sample_context):
        """Crop with only tmin trims the start."""
        from facet.preprocessing import Crop

        raw = sample_context.get_raw()
        original_n_times = raw.n_times

        result = sample_context | Crop(tmin=1.0)
        assert result.get_raw().n_times < original_n_times

    def test_crop_no_args_is_passthrough(self, sample_context, monkeypatch):
        """Crop with no arguments keeps recording unchanged when selection is cancelled."""
        from facet.preprocessing import Crop

        monkeypatch.setattr(Crop, "_show_interactive_crop_selector", lambda self, raw: None)
        original_n_times = sample_context.get_raw().n_times
        result = sample_context | Crop()
        assert result.get_raw().n_times == original_n_times

    def test_crop_no_args_uses_interactive_selection(self, sample_context, monkeypatch):
        """Crop with no arguments uses interval returned by interactive selector."""
        from facet.preprocessing import Crop

        monkeypatch.setattr(Crop, "_show_interactive_crop_selector", lambda self, raw: (1.0, 2.0))

        result = sample_context | Crop()
        sfreq = sample_context.get_raw().info["sfreq"]

        assert result.get_raw().n_times < sample_context.get_raw().n_times
        assert np.isclose(result.get_raw().times[-1], 1.0, atol=1.0 / sfreq)

    def test_crop_rejects_invalid_bounds(self, sample_context):
        """Crop raises validation error when tmax <= tmin."""
        from facet.preprocessing import Crop

        with pytest.raises(ProcessorValidationError):
            sample_context | Crop(tmin=2.0, tmax=1.0)

    def test_crop_preserves_metadata(self, sample_context):
        """Crop preserves triggers and other metadata."""
        from facet.preprocessing import Crop

        result = sample_context | Crop(tmin=0.0)
        assert result.has_triggers()


@pytest.mark.unit
class TestRawTransform:
    """Tests for the RawTransform processor."""

    def test_transform_is_applied(self, sample_context):
        """RawTransform calls the supplied function."""
        from facet.preprocessing import RawTransform

        called = []

        def my_transform(raw):
            called.append(True)
            return raw.copy()

        result = sample_context | RawTransform("my_transform", my_transform)

        assert called == [True]
        assert isinstance(result.get_raw(), type(sample_context.get_raw()))

    def test_transform_modifies_raw(self, sample_context):
        """RawTransform can return a modified Raw object."""
        import numpy as np

        from facet.preprocessing import RawTransform

        original_max = np.abs(sample_context.get_raw()._data).max()

        def scale_up(raw):
            r = raw.copy()
            r._data *= 100
            return r

        result = sample_context | RawTransform("scale_up", scale_up)
        new_max = np.abs(result.get_raw()._data).max()
        assert new_max > original_max

    def test_transform_name_used_in_history(self, sample_context):
        """The name passed to RawTransform appears in processor repr."""
        from facet.preprocessing import RawTransform

        proc = RawTransform("my_custom_step", lambda raw: raw.copy())
        # The name attribute is set on the instance
        assert proc.name == "my_custom_step"


@pytest.mark.unit
class TestDropChannelsMatching:
    """Tests for regex-based channel removal."""

    def test_drop_egi_e1_to_e128_on_copy(self):
        """E1-E128 should be removed while non-matching channels are preserved."""
        ch_names = ["E1", "E2", "E128", "E129", "TREV", "ECG"]
        ch_types = ["eeg", "eeg", "eeg", "eeg", "stim", "ecg"]
        data = np.zeros((len(ch_names), 100))
        raw = mne.io.RawArray(
            data,
            mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types=ch_types),
            verbose=False,
        )
        context = ProcessingContext(raw=raw)
        regex = r"^E(?:[1-9]|[1-9]\d|1[01]\d|12[0-8])$"

        result = context | DropChannelsMatching(regex=regex)

        assert context.get_raw().ch_names == ch_names
        assert result.get_raw() is not context.get_raw()
        assert result.get_raw().ch_names == ["E129", "TREV", "ECG"]


@pytest.mark.unit
class TestChannelStandardizer:
    """Tests for ChannelStandardizer."""

    @staticmethod
    def _build_context() -> ProcessingContext:
        """Create a context with EEG + auxiliary channels."""
        ch_names = [
            "Fp1",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "T3",
            "C3",
            "Cz",
            "C4",
            "T4",
            "T5",
            "P3",
            "Pz",
            "P4",
            "T6",
            "O1",
            "O2",
            "ECG",
            "STI 014",
        ]
        ch_types = ["eeg"] * 19 + ["ecg", "stim"]
        data = np.random.randn(len(ch_names), 500) * 1e-6
        info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        return ProcessingContext(raw=raw, raw_original=raw.copy(), metadata=ProcessingMetadata())

    def test_standard_1020_alias_renaming_and_auxiliary(self):
        """Legacy alias channels should be mapped and renamed to target labels."""
        context = self._build_context()
        result = context | ChannelStandardizer("10-20", on_missing="raise", keep_auxiliary=True, rename_aliases=True)

        expected_eeg = [
            "Fp1",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "T7",
            "C3",
            "Cz",
            "C4",
            "T8",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "O1",
            "O2",
        ]
        assert result.get_raw().ch_names[: len(expected_eeg)] == expected_eeg
        assert result.get_raw().ch_names[-2:] == ["ECG", "STI 014"]

    def test_custom_standard_subset(self):
        """Custom ordered subsets should be respected exactly."""
        context = self._build_context()
        result = context | ChannelStandardizer(["Cz", "Pz", "O1"], on_missing="raise", keep_auxiliary=False)
        assert result.get_raw().ch_names == ["Cz", "Pz", "O1"]

    def test_unknown_standard_raises(self, sample_context):
        """Invalid standard names should fail validation."""
        with pytest.raises(ProcessorValidationError):
            sample_context | ChannelStandardizer("does_not_exist")

    def test_missing_channels_raise_when_configured(self):
        """on_missing='raise' should fail when requested channels are absent."""
        context = self._build_context()
        with pytest.raises(ProcessorValidationError):
            context | ChannelStandardizer(["Fz", "CPz"], on_missing="raise", keep_auxiliary=False)

    def test_missing_channels_ignored(self):
        """on_missing='ignore' should keep only available channels."""
        context = self._build_context()
        result = context | ChannelStandardizer(["Fz", "CPz"], on_missing="ignore", keep_auxiliary=False)
        assert result.get_raw().ch_names == ["Fz"]

    def test_noise_propagation_tracks_selected_channels(self):
        """Estimated noise should be subset in the same channel order."""
        context = self._build_context()
        original_noise = np.random.randn(*context.get_raw()._data.shape) * 1e-7
        context.set_estimated_noise(original_noise)

        result = context | ChannelStandardizer(["Fp1", "Cz", "O2"], on_missing="raise", keep_auxiliary=False)
        noise = result.get_estimated_noise()

        assert noise is not None
        assert noise.shape == (3, context.get_raw().n_times)

        source_names = context.get_raw().ch_names
        expected_indices = [source_names.index(ch_name) for ch_name in ["Fp1", "Cz", "O2"]]
        np.testing.assert_allclose(noise, original_noise[expected_indices, :])


@pytest.mark.unit
class TestMagicErasor:
    """Tests for the MagicErasor processor."""

    def test_magic_erasor_zero_mode(self, sample_context, monkeypatch):
        """Zero mode should overwrite the selected interval with zeros."""
        start_sample, end_sample = 10, 30

        def _fake_editor(self, data, sfreq, target_picks, preview_channel, channel_names):
            self._apply_edit(
                data=data,
                target_picks=target_picks,
                start_sample=start_sample,
                end_sample=end_sample,
                mode="zero",
                sfreq=sfreq,
                edit_index=0,
            )
            return [
                {
                    "mode": "zero",
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start_time": start_sample / sfreq,
                    "end_time": end_sample / sfreq,
                }
            ]

        monkeypatch.setattr(MagicErasor, "_show_interactive_editor", _fake_editor)

        original = sample_context.get_raw().get_data().copy()
        result = sample_context | MagicErasor()
        edited = result.get_raw().get_data()

        assert np.allclose(edited[:, start_sample:end_sample], 0.0)
        assert not np.allclose(original[:, start_sample:end_sample], 0.0)

    def test_magic_erasor_multiple_edits_recorded(self, sample_context, monkeypatch):
        """Multiple in-session edits should be reflected in metadata."""
        first = (20, 40)
        second = (60, 90)

        def _fake_editor(self, data, sfreq, target_picks, preview_channel, channel_names):
            self._apply_edit(data, target_picks, first[0], first[1], "mean", sfreq, 0)
            self._apply_edit(data, target_picks, second[0], second[1], "interpolate", sfreq, 1)
            return [
                {
                    "mode": "mean",
                    "start_sample": first[0],
                    "end_sample": first[1],
                    "start_time": first[0] / sfreq,
                    "end_time": first[1] / sfreq,
                },
                {
                    "mode": "interpolate",
                    "start_sample": second[0],
                    "end_sample": second[1],
                    "start_time": second[0] / sfreq,
                    "end_time": second[1] / sfreq,
                },
            ]

        monkeypatch.setattr(MagicErasor, "_show_interactive_editor", _fake_editor)

        result = sample_context | MagicErasor(default_mode="mean")
        edit_meta = result.metadata.custom.get("magic_erasor")

        assert edit_meta is not None
        assert edit_meta["n_edits"] == 2
        assert edit_meta["edits"][0]["mode"] == "mean"
        assert edit_meta["edits"][1]["mode"] == "interpolate"

    def test_magic_erasor_generated_eeg_mode(self, sample_context, monkeypatch):
        """Generated EEG mode should replace data with synthetic content."""
        start_sample, end_sample = 40, 80

        def _fake_editor(self, data, sfreq, target_picks, preview_channel, channel_names):
            self._apply_edit(data, target_picks, start_sample, end_sample, "generated_eeg", sfreq, 0)
            return [
                {
                    "mode": "generated_eeg",
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start_time": start_sample / sfreq,
                    "end_time": end_sample / sfreq,
                }
            ]

        monkeypatch.setattr(MagicErasor, "_show_interactive_editor", _fake_editor)

        original = sample_context.get_raw().get_data().copy()
        result = sample_context | MagicErasor(random_seed=42)
        edited = result.get_raw().get_data()

        assert not np.allclose(
            edited[:, start_sample:end_sample],
            original[:, start_sample:end_sample],
        )

    def test_magic_erasor_generated_eeg_matches_environment_stats(self, sample_context, monkeypatch):
        """Generated EEG mode should match surrounding mean and amplitude."""
        start_sample, end_sample = 300, 420
        window = end_sample - start_sample

        raw = sample_context.get_raw().copy()
        sfreq = float(raw.info["sfreq"])
        time_axis = np.arange(raw.n_times) / sfreq

        for ch_idx in range(raw._data.shape[0]):
            baseline = (ch_idx + 1) * 8e-6
            amplitude = (ch_idx + 1) * 2e-6
            raw._data[ch_idx] = baseline + amplitude * np.sin(2 * np.pi * (1.5 + 0.3 * ch_idx) * time_axis)
            raw._data[ch_idx, start_sample:end_sample] = baseline + (40.0 * amplitude) * np.cos(
                2 * np.pi * 6.0 * time_axis[start_sample:end_sample]
            )

        context = sample_context.with_raw(raw)

        def _fake_editor(self, data, sfreq, target_picks, preview_channel, channel_names):
            self._apply_edit(data, target_picks, start_sample, end_sample, "generated_eeg", sfreq, 0)
            return [
                {
                    "mode": "generated_eeg",
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start_time": start_sample / sfreq,
                    "end_time": end_sample / sfreq,
                }
            ]

        monkeypatch.setattr(MagicErasor, "_show_interactive_editor", _fake_editor)

        result = context | MagicErasor(random_seed=7)
        edited = result.get_raw().get_data()

        assert not np.allclose(
            edited[:, start_sample:end_sample],
            raw._data[:, start_sample:end_sample],
        )

        for ch_idx in range(edited.shape[0]):
            environment = np.concatenate(
                [
                    raw._data[ch_idx, start_sample - window : start_sample],
                    raw._data[ch_idx, end_sample : end_sample + window],
                ]
            )
            segment = edited[ch_idx, start_sample:end_sample]
            np.testing.assert_allclose(float(np.mean(segment)), float(np.mean(environment)), rtol=0.0, atol=1e-10)
            np.testing.assert_allclose(float(np.std(segment)), float(np.std(environment)), rtol=5e-3, atol=1e-12)

    def test_magic_erasor_cancel_keeps_context(self, sample_context, monkeypatch):
        """Cancelled editor should keep the context unchanged."""

        def _fake_editor(self, data, sfreq, target_picks, preview_channel, channel_names):
            return None

        monkeypatch.setattr(MagicErasor, "_show_interactive_editor", _fake_editor)

        result = sample_context | MagicErasor()
        np.testing.assert_allclose(result.get_raw().get_data(), sample_context.get_raw().get_data())

    def test_magic_erasor_invalid_default_mode(self, sample_context):
        """Invalid default mode should fail validation."""
        with pytest.raises(ProcessorValidationError):
            sample_context | MagicErasor(default_mode="does_not_exist")

    def test_magic_erasor_propagates_noise(self, sample_context_with_noise, monkeypatch):
        """Noise estimate should be edited consistently for compatible modes."""
        start_sample, end_sample = 5, 25

        def _fake_editor(self, data, sfreq, target_picks, preview_channel, channel_names):
            self._apply_edit(data, target_picks, start_sample, end_sample, "zero", sfreq, 0)
            return [
                {
                    "mode": "zero",
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start_time": start_sample / sfreq,
                    "end_time": end_sample / sfreq,
                }
            ]

        monkeypatch.setattr(MagicErasor, "_show_interactive_editor", _fake_editor)

        result = sample_context_with_noise | MagicErasor()
        noise = result.get_estimated_noise()
        assert noise is not None
        assert np.allclose(noise[:, start_sample:end_sample], 0.0)


@pytest.mark.integration
class TestPreprocessingIntegration:
    """Integration tests for preprocessing workflow."""

    def test_full_preprocessing_workflow(self, sample_edf_file):
        """Test complete preprocessing workflow."""
        from facet.core import Pipeline
        from facet.io import Loader

        pipeline = Pipeline(
            [
                Loader(path=str(sample_edf_file), preload=True),
                TriggerDetector(regex=r"\b1\b"),
                UpSample(factor=2),
                TriggerAligner(ref_trigger_index=0),
                HighPassFilter(freq=0.5),
                DownSample(factor=2),
            ]
        )

        result = pipeline.run()

        assert result.success is True
        assert result.context.has_triggers()
        assert result.context.get_raw() is not None
