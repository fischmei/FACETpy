"""
Tests for correction processors.
"""

import mne
import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata, ProcessorValidationError
from facet.correction import (
    AASCorrection,
    ANCCorrection,
    CorrespondingSliceCorrection,
    FARMCorrection,
    MoosmannCorrection,
    PCACorrection,
    SliceTriggerCorrection,
    VolumeArtifactCorrection,
    VolumeTriggerCorrection,
)


@pytest.mark.unit
class TestAASCorrection:
    """Tests for AASCorrection processor."""

    def test_aas_initialization(self):
        """Test AAS processor initialization."""
        aas = AASCorrection(window_size=30, correlation_threshold=0.975)

        assert aas.window_size == 30
        assert aas.correlation_threshold == 0.975
        assert aas.apply_epoch_alpha_scaling is False

    def test_aas_requires_triggers(self, sample_raw):
        """Test that AAS requires triggers."""
        context = ProcessingContext(raw=sample_raw)

        aas = AASCorrection(window_size=5)

        with pytest.raises(ProcessorValidationError):
            aas.execute(context)

    def test_aas_requires_artifact_length(self, sample_context):
        """Test that AAS requires artifact length."""
        # Remove artifact length
        metadata = sample_context.metadata.copy()
        metadata.artifact_length = None
        context = sample_context.with_metadata(metadata)

        aas = AASCorrection(window_size=5)

        with pytest.raises(ProcessorValidationError):
            aas.execute(context)

    def test_aas_execution(self, sample_context):
        """Test AAS execution."""
        original_data = sample_context.get_raw()._data.copy()

        aas = AASCorrection(window_size=5)
        result = aas.execute(sample_context)

        # Check result is valid
        assert result.get_raw() is not None
        corrected_data = result.get_raw()._data

        # Data should have changed
        assert not np.array_equal(original_data, corrected_data)

        # Check estimated noise was created
        assert result.has_estimated_noise()
        noise = result.get_estimated_noise()
        assert noise.shape == corrected_data.shape

    def test_aas_can_skip_noise_without_changing_correction(self, sample_context):
        """AAS exposes Flex's memory-saving diagnostic-noise switch."""
        tracked = AASCorrection(
            window_size=5,
            track_estimated_noise=True,
        ).execute(sample_context)
        untracked = AASCorrection(
            window_size=5,
            track_estimated_noise=False,
        ).execute(sample_context)

        np.testing.assert_array_equal(
            untracked.get_raw()._data,
            tracked.get_raw()._data,
        )
        assert not untracked.has_estimated_noise()

    def test_aas_saves_matrix_plot_for_chunk_runs(self, sample_context, tmp_path):
        """AAS should emit a matrix-plot image when a chunk output path is available."""
        output_path = tmp_path / "chunk_output.edf"
        metadata = sample_context.metadata.copy()
        metadata.custom["chunk"] = {"output_path": str(output_path)}
        context = sample_context.with_metadata(metadata)

        aas = AASCorrection(window_size=5)
        aas.execute(context)

        generated_plot = output_path.with_suffix(".aas_matrices.png")
        assert generated_plot.exists()

    def test_aas_reduces_artifact_amplitude(self, sample_raw_with_artifacts, sample_triggers):
        """Test that AAS actually reduces artifact amplitude."""
        from facet.core import ProcessingMetadata

        # Create context with artifacts
        metadata = ProcessingMetadata()
        metadata.triggers = sample_triggers
        metadata.artifact_length = 50
        context = ProcessingContext(
            raw=sample_raw_with_artifacts, raw_original=sample_raw_with_artifacts.copy(), metadata=metadata
        )

        # Calculate RMS before
        rms_before = np.sqrt(np.mean(sample_raw_with_artifacts._data**2))

        # Apply AAS
        aas = AASCorrection(window_size=5)
        result = aas.execute(context)

        # Calculate RMS after
        rms_after = np.sqrt(np.mean(result.get_raw()._data ** 2))

        # RMS should be reduced (artifacts removed)
        assert rms_after < rms_before

    def test_aas_with_small_window(self, sample_context):
        """Test AAS with small window size."""
        aas = AASCorrection(window_size=3)
        result = aas.execute(sample_context)

        assert result.get_raw() is not None

    def test_aas_with_realignment(self, sample_context):
        """Test AAS with trigger realignment."""
        aas = AASCorrection(window_size=5, realign_after_averaging=True)
        result = aas.execute(sample_context)

        # Triggers may have been adjusted
        assert result.has_triggers()

    def test_aas_interpolate_volume_gaps(self, sample_context):
        """Interpolating volume gaps should fill estimated-noise gaps."""
        base = AASCorrection(window_size=5, realign_after_averaging=False, interpolate_volume_gaps=False).execute(
            sample_context
        )
        interp = AASCorrection(window_size=5, realign_after_averaging=False, interpolate_volume_gaps=True).execute(
            sample_context
        )

        triggers = sample_context.get_triggers()
        artifact_length = sample_context.get_artifact_length()
        gap_start = triggers[0] + artifact_length
        gap_end = triggers[1]

        noise_base = base.get_estimated_noise()[:, gap_start:gap_end]
        noise_interp = interp.get_estimated_noise()[:, gap_start:gap_end]

        assert np.allclose(noise_base, 0.0)
        assert np.any(np.abs(noise_interp) > 0.0)

    def test_aas_epoch_alpha_scaling_reduces_amplitude_residual(self):
        """MATLAB-like per-epoch alpha scaling should reduce residuals for drifting amplitudes."""
        sfreq = 100.0
        artifact_length = 20
        n_epochs = 12
        spacing = 30
        triggers = (np.arange(n_epochs) * spacing + 20).astype(int)
        n_samples = int(triggers[-1] + artifact_length + 20)

        template = np.sin(np.linspace(0, np.pi, artifact_length))
        scales = np.linspace(0.5, 1.8, n_epochs)

        data = np.zeros((1, n_samples), dtype=float)
        for idx, trg in enumerate(triggers):
            data[0, trg : trg + artifact_length] += scales[idx] * template

        raw = mne.io.RawArray(data, mne.create_info(["EEG1"], sfreq=sfreq, ch_types=["eeg"]), verbose=False)
        metadata = ProcessingMetadata()
        metadata.triggers = triggers
        metadata.artifact_length = artifact_length
        metadata.artifact_to_trigger_offset = 0.0
        context = ProcessingContext(raw=raw, raw_original=raw.copy(), metadata=metadata)

        base = AASCorrection(
            window_size=n_epochs,
            correlation_threshold=0.8,
            realign_after_averaging=False,
            apply_epoch_alpha_scaling=False,
        ).execute(context)
        scaled = AASCorrection(
            window_size=n_epochs,
            correlation_threshold=0.8,
            realign_after_averaging=False,
            apply_epoch_alpha_scaling=True,
        ).execute(context)

        def _artifact_rms(ctx: ProcessingContext) -> float:
            ch = ctx.get_raw()._data[0]
            epochs = [ch[t : t + artifact_length] for t in triggers]
            stacked = np.concatenate(epochs)
            return float(np.sqrt(np.mean(stacked**2)))

        rms_base = _artifact_rms(base)
        rms_scaled = _artifact_rms(scaled)
        assert rms_scaled < rms_base


@pytest.mark.unit
class TestANCCorrection:
    """Tests for ANCCorrection processor."""

    def test_anc_initialization(self):
        """Test ANC processor initialization."""
        anc = ANCCorrection(filter_order=5, hp_freq=1.0)

        assert anc.filter_order == 5
        assert anc.hp_freq == 1.0

    def test_anc_requires_estimated_noise(self, sample_context):
        """Test that ANC requires estimated noise."""
        anc = ANCCorrection(filter_order=5)

        with pytest.raises(ProcessorValidationError):
            anc.execute(sample_context)

    def test_anc_execution(self, sample_context_with_noise):
        """Test ANC execution."""
        original_data = sample_context_with_noise.get_raw()._data.copy()

        anc = ANCCorrection(filter_order=5, use_c_extension=False)
        result = anc.execute(sample_context_with_noise)

        corrected_data = result.get_raw()._data

        # Data may have changed (depends on noise)
        assert corrected_data.shape == original_data.shape
        assert not np.any(np.isnan(corrected_data))
        assert not np.any(np.isinf(corrected_data))

    def test_anc_python_fallback(self, sample_context_with_noise):
        """Test ANC Python fallback when C extension unavailable."""
        anc = ANCCorrection(filter_order=3, use_c_extension=False)
        result = anc.execute(sample_context_with_noise)

        # Should complete successfully
        assert result.get_raw() is not None

    def test_anc_acquisition_window_uses_trigger_min_max(self, sample_context_with_noise):
        """Acquisition window should be robust to unsorted trigger arrays."""
        metadata = sample_context_with_noise.metadata.copy()
        metadata.triggers = np.array([300, 120, 280, 200], dtype=int)
        metadata.artifact_length = 40
        context = sample_context_with_noise.with_metadata(metadata)

        anc = ANCCorrection(filter_order=5, use_c_extension=False)
        start, end = anc._get_acquisition_window(context)

        assert start == 80
        assert end == 340

    @pytest.mark.requires_c_extension
    def test_anc_c_extension(self, sample_context_with_noise):
        """Test ANC with C extension."""
        anc = ANCCorrection(filter_order=5, use_c_extension=True)

        # May skip if C extension not available
        try:
            result = anc.execute(sample_context_with_noise)
            assert result.get_raw() is not None
        except ImportError:
            pytest.skip("C extension not available")


@pytest.mark.unit
class TestPCACorrection:
    """Tests for PCACorrection processor."""

    def test_pca_initialization(self):
        """Test PCA processor initialization."""
        pca = PCACorrection(n_components=0.95)

        assert pca.n_components == 0.95

    def test_pca_requires_triggers(self, sample_raw):
        """Test that PCA requires triggers."""
        context = ProcessingContext(raw=sample_raw)

        pca = PCACorrection(n_components=0.95)

        with pytest.raises(ProcessorValidationError):
            pca.execute(context)

    def test_pca_execution(self, sample_context):
        """Test PCA execution."""
        original_data = sample_context.get_raw()._data.copy()

        pca = PCACorrection(n_components=2)  # Use integer for predictable behavior
        result = pca.execute(sample_context)

        corrected_data = result.get_raw()._data

        # Data should have changed
        assert not np.array_equal(original_data, corrected_data)

        # Check estimated noise was updated
        assert result.has_estimated_noise()

    def test_pca_with_variance_threshold(self, sample_context):
        """Test PCA with variance threshold."""
        pca = PCACorrection(n_components=0.95)  # Keep 95% variance
        result = pca.execute(sample_context)

        assert result.get_raw() is not None

    def test_pca_can_skip_noise_without_changing_correction(self, sample_context):
        """PCA residual storage is optional and does not affect subtraction."""
        tracked = PCACorrection(
            n_components=2,
            track_estimated_noise=True,
        ).execute(sample_context)
        untracked = PCACorrection(
            n_components=2,
            track_estimated_noise=False,
        ).execute(sample_context)

        np.testing.assert_array_equal(
            untracked.get_raw()._data,
            tracked.get_raw()._data,
        )
        assert not untracked.has_estimated_noise()

    def test_pca_with_zero_components_skips(self, sample_context):
        """Test that PCA with n_components=0 skips processing."""
        original_data = sample_context.get_raw()._data.copy()

        pca = PCACorrection(n_components=0)
        result = pca.execute(sample_context)

        # Data should be unchanged
        np.testing.assert_array_equal(result.get_raw()._data, original_data)

    def test_pca_with_auto_components(self, sample_context):
        """Test PCA with MATLAB-style auto component selection."""
        pca = PCACorrection(n_components="auto")
        result = pca.execute(sample_context)
        assert result.get_raw() is not None
        assert result.has_estimated_noise()

    def test_pca_acquisition_window_uses_trigger_min_max(self, sample_context):
        """Acquisition window should be robust to unsorted trigger arrays."""
        metadata = sample_context.metadata.copy()
        metadata.triggers = np.array([300, 120, 280, 200], dtype=int)
        metadata.artifact_length = 40
        context = sample_context.with_metadata(metadata)

        pca = PCACorrection(n_components=0.95)
        start, end = pca._get_acquisition_window(context)

        assert start == 80
        assert end == 340


@pytest.mark.unit
class TestFARMCorrection:
    """Tests for FARMCorrection processor."""

    def test_farm_initialization(self):
        """Test FARM processor initialization."""
        farm = FARMCorrection(window_size=20, correlation_threshold=0.9, search_half_window=60)
        assert farm.window_size == 20
        assert farm.correlation_threshold == 0.9
        assert farm.search_half_window == 60
        assert farm.apply_epoch_alpha_scaling is False

    def test_farm_execution(self, sample_context):
        """Test FARM execution."""
        original_data = sample_context.get_raw()._data.copy()

        farm = FARMCorrection(window_size=5, realign_after_averaging=False)
        result = farm.execute(sample_context)

        corrected_data = result.get_raw()._data
        assert corrected_data.shape == original_data.shape
        assert not np.array_equal(original_data, corrected_data)
        assert result.has_estimated_noise()

    def test_farm_matrix_rows_are_nonzero(self):
        """FARM should keep valid averaging rows even for low-correlation epochs."""
        rng = np.random.RandomState(0)
        epochs = rng.randn(12, 50)
        farm = FARMCorrection(window_size=4, correlation_threshold=0.9999, search_half_window=6)
        matrix = farm._calc_averaging_matrix(
            epochs=epochs,
            window_size=4,
            rel_window_offset=0.0,
            correlation_threshold=0.9999,
        )
        row_sums = matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-12)

    def test_farm_search_half_window_does_not_scale_with_window_size(self):
        """FARM should not expand the candidate search span via a window-size multiplier."""
        farm = FARMCorrection(window_size=5, search_half_window=None, search_half_window_factor=9.0)

        assert farm._resolve_search_half_window(5) == 5


@pytest.mark.unit
class TestVolumeArtifactCorrection:
    """Tests for VolumeArtifactCorrection processor."""

    def _build_volume_gap_context(self) -> ProcessingContext:
        sfreq = 250.0
        n_channels = 3
        n_samples = 2500
        data = np.zeros((n_channels, n_samples), dtype=float)

        triggers = np.array([100, 140, 180, 220, 260, 300, 340, 380, 480, 520, 560, 600, 640, 680, 720], dtype=int)
        artifact_length = 40
        pre_samples = 5
        post_samples = artifact_length - pre_samples - 1

        base_epoch = np.sin(np.linspace(0, 2 * np.pi, artifact_length, endpoint=False)) * 5e-6
        for ch in range(n_channels):
            for trig in triggers:
                start = trig - pre_samples
                stop = start + artifact_length
                if start < 0 or stop > n_samples:
                    continue
                data[ch, start:stop] += base_epoch

            # Add stronger volume-transition artifact on both slices around the gap.
            pre_trig = triggers[7]
            post_trig = triggers[8]
            pre_start = pre_trig - pre_samples
            post_start = post_trig - pre_samples
            transition = np.linspace(0.0, 20e-6, artifact_length)
            data[ch, pre_start : pre_start + artifact_length] += transition
            data[ch, post_start : post_start + artifact_length] += transition[::-1]

            gap_start = pre_trig + post_samples + 1
            gap_end = post_trig - pre_samples - 1
            data[ch, gap_start : gap_end + 1] = 15e-6

        info = mne.create_info(
            ch_names=[f"EEG{i + 1:03d}" for i in range(n_channels)],
            sfreq=sfreq,
            ch_types=["eeg"] * n_channels,
        )
        raw = mne.io.RawArray(data, info, verbose=False)

        metadata = ProcessingMetadata(
            triggers=triggers,
            artifact_length=artifact_length,
            pre_trigger_samples=pre_samples,
            post_trigger_samples=post_samples,
            volume_gaps=True,
            artifact_to_trigger_offset=0.0,
        )
        return ProcessingContext(raw=raw, raw_original=raw.copy(), metadata=metadata)

    def test_volume_artifact_correction_requires_artifact_length(self, sample_raw, sample_triggers):
        """Processor should validate artifact length."""
        metadata = ProcessingMetadata(triggers=sample_triggers, artifact_length=None, volume_gaps=True)
        context = ProcessingContext(raw=sample_raw, raw_original=sample_raw.copy(), metadata=metadata)
        processor = VolumeArtifactCorrection()
        with pytest.raises(ProcessorValidationError):
            processor.execute(context)

    def test_volume_artifact_correction_no_volume_gaps_noop(self, sample_context):
        """Processor should no-op when volume_gaps metadata is False."""
        processor = VolumeArtifactCorrection()
        result = processor.execute(sample_context)
        assert result is sample_context

    def test_volume_artifact_correction_execution(self):
        """Volume artifact correction should modify data around detected gaps."""
        context = self._build_volume_gap_context()
        original = context.get_raw()._data.copy()

        processor = VolumeArtifactCorrection()
        result = processor.execute(context)

        corrected = result.get_raw()._data
        assert corrected.shape == original.shape
        assert not np.array_equal(corrected, original)
        assert not np.any(np.isnan(corrected))
        assert not np.any(np.isinf(corrected))

        pre_trig = context.get_triggers()[7]
        post_trig = context.get_triggers()[8]
        pre = context.metadata.pre_trigger_samples
        post = context.metadata.post_trigger_samples
        gap_start = pre_trig + post + 1
        gap_end = post_trig - pre - 1
        assert not np.array_equal(original[:, gap_start : gap_end + 1], corrected[:, gap_start : gap_end + 1])


@pytest.mark.unit
class TestAASWeightingVariants:
    """Tests for additional MATLAB-style AAS weighting variants."""

    def test_weighting_variants_accept_epoch_alpha_scaling(self, temp_dir):
        """All Flex template variants should expose shared epoch alpha scaling."""
        rp_file = temp_dir / "rp_alpha.tsv"
        rp_file.write_text("x\ty\tz\tpitch\troll\tyaw\n0\t0\t0\t0\t0\t0\n", encoding="utf-8")

        processors = [
            CorrespondingSliceCorrection(slices_per_volume=2, apply_epoch_alpha_scaling=True),
            VolumeTriggerCorrection(apply_epoch_alpha_scaling=True),
            SliceTriggerCorrection(apply_epoch_alpha_scaling=True),
            FARMCorrection(apply_epoch_alpha_scaling=True),
            MoosmannCorrection(rp_file=str(rp_file), apply_epoch_alpha_scaling=True),
        ]

        for proc in processors:
            assert proc.apply_epoch_alpha_scaling is True

    def test_corresponding_slice_correction_execution(self, sample_context):
        metadata = sample_context.metadata.copy()
        metadata.slices_per_volume = 2
        context = sample_context.with_metadata(metadata)

        processor = CorrespondingSliceCorrection(window_size=2, realign_after_averaging=False)
        result = processor.execute(context)

        assert result.get_raw() is not None
        assert result.has_estimated_noise()

    def test_volume_trigger_correction_execution(self, sample_context):
        processor = VolumeTriggerCorrection(window_size=4, realign_after_averaging=False)
        result = processor.execute(sample_context)

        assert result.get_raw() is not None
        assert result.has_estimated_noise()

    def test_slice_trigger_correction_execution(self, sample_context):
        processor = SliceTriggerCorrection(window_size=3, realign_after_averaging=False)
        result = processor.execute(sample_context)

        assert result.get_raw() is not None
        assert result.has_estimated_noise()

    def test_slice_trigger_correction_uses_every_second_epochs(self):
        processor = SliceTriggerCorrection(window_size=2, realign_after_averaging=False)
        epochs = np.zeros((12, 10), dtype=float)

        matrix = processor._calc_averaging_matrix(
            epochs=epochs,
            window_size=2,
            rel_window_offset=0.0,
            correlation_threshold=0.0,
        )

        first_row = np.flatnonzero(matrix[0])
        second_row = np.flatnonzero(matrix[1])

        assert np.array_equal(first_row, np.array([1, 3, 5]))
        assert np.array_equal(second_row, np.array([2, 4, 6]))
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)

    def test_moosmann_correction_execution(self, sample_context, temp_dir):
        rp_file = temp_dir / "rp_test.tsv"
        n_rows = max(2, len(sample_context.get_triggers()))
        with rp_file.open("w", encoding="utf-8") as f:
            f.write("x\ty\tz\tpitch\troll\tyaw\n")
            for _ in range(n_rows):
                f.write("0\t0\t0\t0\t0\t0\n")

        processor = MoosmannCorrection(
            rp_file=str(rp_file),
            window_size=3,
            motion_threshold=5.0,
            realign_after_averaging=False,
        )
        result = processor.execute(sample_context)

        assert result.get_raw() is not None
        assert result.has_estimated_noise()
        assert "moosmann" in result.metadata.custom


@pytest.mark.integration
class TestCorrectionPipeline:
    """Integration tests for correction pipeline."""

    def test_aas_anc_pipeline(self, sample_context):
        """Test AAS followed by ANC."""
        from facet.core import Pipeline

        pipeline = Pipeline([AASCorrection(window_size=5), ANCCorrection(filter_order=3, use_c_extension=False)])

        result = pipeline.run(initial_context=sample_context)

        assert result.success is True
        assert result.context.get_raw() is not None

    def test_anc_channel_sequential_matches_serial(self, sample_context_with_noise):
        """ANC should produce the same result in serial and channel-sequential modes."""
        from facet.core import Pipeline

        pipeline = Pipeline([ANCCorrection(filter_order=5, use_c_extension=False)])

        serial = pipeline.run(initial_context=sample_context_with_noise, channel_sequential=False, show_progress=False)
        channel_seq = pipeline.run(
            initial_context=sample_context_with_noise, channel_sequential=True, show_progress=False
        )

        assert serial.success is True
        assert channel_seq.success is True
        np.testing.assert_allclose(
            channel_seq.context.get_data(copy=False),
            serial.context.get_data(copy=False),
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            channel_seq.context.get_estimated_noise(),
            serial.context.get_estimated_noise(),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_full_correction_pipeline(self, sample_edf_file):
        """Test full correction pipeline."""
        from facet.core import Pipeline
        from facet.io import Loader
        from facet.preprocessing import DownSample, TriggerDetector, UpSample

        pipeline = Pipeline(
            [
                Loader(path=str(sample_edf_file), preload=True),
                TriggerDetector(regex=r"\b1\b"),
                UpSample(factor=2),
                AASCorrection(window_size=5),
                DownSample(factor=2),
            ]
        )

        result = pipeline.run()

        assert result.success is True

    def test_aas_pca_pipeline(self, sample_context):
        """Test AAS followed by PCA."""
        from facet.core import Pipeline

        pipeline = Pipeline([AASCorrection(window_size=5), PCACorrection(n_components=2)])

        result = pipeline.run(initial_context=sample_context)

        assert result.success is True
