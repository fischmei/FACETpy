"""
Tests for ProcessingContext and ProcessingMetadata.
"""

import mne
import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata


@pytest.mark.unit
class TestProcessingMetadata:
    """Tests for ProcessingMetadata class."""

    def test_initialization(self):
        """Test metadata initialization."""
        metadata = ProcessingMetadata()

        assert metadata.triggers is None
        assert metadata.artifact_length is None
        assert metadata.upsampling_factor == 10
        assert metadata.artifact_to_trigger_offset == 0.0
        assert metadata.volume_gaps is False
        assert isinstance(metadata.custom, dict)

    def test_copy(self):
        """Test metadata copying."""
        metadata = ProcessingMetadata()
        metadata.triggers = np.array([100, 200, 300])
        metadata.artifact_length = 50
        metadata.custom["test"] = "value"

        # Copy
        metadata_copy = metadata.copy()

        # Check it's a different object
        assert metadata_copy is not metadata

        # Check values are equal
        assert np.array_equal(metadata_copy.triggers, metadata.triggers)
        assert metadata_copy.artifact_length == metadata.artifact_length
        assert metadata_copy.custom["test"] == "value"

        # Modify copy shouldn't affect original
        metadata_copy.artifact_length = 100
        assert metadata.artifact_length == 50

    def test_to_dict(self):
        """Test serialization to dict."""
        metadata = ProcessingMetadata()
        metadata.triggers = np.array([100, 200])
        metadata.artifact_length = 50

        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert "triggers" in data
        assert "artifact_length" in data
        assert data["artifact_length"] == 50

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {"triggers": [100, 200, 300], "artifact_length": 50, "upsampling_factor": 5, "custom": {"key": "value"}}

        metadata = ProcessingMetadata.from_dict(data)

        assert np.array_equal(metadata.triggers, np.array([100, 200, 300]))
        assert metadata.artifact_length == 50
        assert metadata.upsampling_factor == 5
        assert metadata.custom["key"] == "value"


@pytest.mark.unit
class TestProcessingContext:
    """Tests for ProcessingContext class."""

    def test_initialization(self, sample_raw):
        """Test context initialization."""
        context = ProcessingContext(raw=sample_raw)

        assert context.get_raw() is sample_raw
        assert context.get_raw_original() is sample_raw
        assert context.metadata is not None

    def test_initialization_with_metadata(self, sample_raw):
        """Test context initialization with metadata."""
        metadata = ProcessingMetadata()
        metadata.triggers = np.array([100, 200])

        context = ProcessingContext(raw=sample_raw, metadata=metadata)

        assert context.metadata is metadata
        assert np.array_equal(context.get_triggers(), np.array([100, 200]))

    def test_with_raw(self, sample_raw):
        """Test creating new context with different raw."""
        context1 = ProcessingContext(raw=sample_raw)

        # Create modified raw
        raw2 = sample_raw.copy()
        raw2._data *= 2

        # Create new context
        context2 = context1.with_raw(raw2)

        # Check context1 unchanged
        assert context1.get_raw() is sample_raw

        # Check context2 has new raw
        assert context2.get_raw() is raw2
        assert not np.array_equal(context1.get_raw()._data, context2.get_raw()._data)

    def test_with_raw_copies_estimated_noise_by_default(self, sample_context):
        """Ordinary immutable operations keep independent noise arrays."""
        noise = np.ones_like(sample_context.get_raw()._data)
        sample_context.set_estimated_noise(noise)

        context2 = sample_context.with_raw(sample_context.get_raw().copy())

        assert context2.get_estimated_noise() is not noise
        np.testing.assert_array_equal(context2.get_estimated_noise(), noise)
        context2.get_estimated_noise()[0, 0] = 2.0
        assert sample_context.get_estimated_noise()[0, 0] == 1.0

    def test_with_raw_can_transfer_replacement_estimated_noise(self, sample_context):
        """Callers can transfer an owned replacement without a full copy."""
        old_noise = np.zeros_like(sample_context.get_raw()._data)
        replacement = np.ones_like(old_noise)
        sample_context.set_estimated_noise(old_noise)

        context2 = sample_context.with_raw(
            sample_context.get_raw().copy(),
            estimated_noise=replacement,
            copy_estimated_noise=False,
        )

        assert context2.get_estimated_noise() is replacement
        assert sample_context.get_estimated_noise() is old_noise

    def test_with_raw_can_drop_estimated_noise(self, sample_context):
        """Explicit None distinguishes dropping noise from preserving it."""
        sample_context.set_estimated_noise(np.ones_like(sample_context.get_raw()._data))

        context2 = sample_context.with_raw(
            sample_context.get_raw().copy(),
            estimated_noise=None,
        )

        assert not context2.has_estimated_noise()

    def test_with_metadata(self, sample_context):
        """Test creating new context with different metadata."""
        new_metadata = sample_context.metadata.copy()
        new_metadata.artifact_length = 100

        context2 = sample_context.with_metadata(new_metadata)

        # Check original unchanged
        assert sample_context.metadata.artifact_length != 100

        # Check new context has new metadata
        assert context2.metadata.artifact_length == 100

    def test_with_metadata_can_share_estimated_noise_explicitly(self, sample_context):
        """Metadata-only callers can explicitly avoid copying read-only noise."""
        noise = np.ones_like(sample_context.get_raw()._data)
        sample_context.set_estimated_noise(noise)

        context2 = sample_context.with_metadata(
            sample_context.metadata.copy(),
            copy_estimated_noise=False,
        )

        assert context2.get_estimated_noise() is noise

    def test_get_triggers(self, sample_context):
        """Test getting triggers."""
        triggers = sample_context.get_triggers()
        assert triggers is not None
        assert len(triggers) > 0

    def test_has_triggers(self, sample_context):
        """Test checking for triggers."""
        assert sample_context.has_triggers()

        # Context without triggers
        context_no_triggers = ProcessingContext(raw=sample_context.get_raw())
        assert not context_no_triggers.has_triggers()

    def test_get_artifact_length(self, sample_context):
        """Test getting artifact length."""
        length = sample_context.get_artifact_length()
        assert length is not None
        assert length > 0

    def test_estimated_noise(self, sample_context):
        """Test estimated noise handling."""
        # Initially no noise
        assert not sample_context.has_estimated_noise()

        # Set noise
        noise = np.random.randn(*sample_context.get_raw()._data.shape)
        sample_context.set_estimated_noise(noise)

        # Check it's stored
        assert sample_context.has_estimated_noise()
        retrieved_noise = sample_context.get_estimated_noise()
        assert np.array_equal(retrieved_noise, noise)

    def test_accumulate_noise_can_transfer_first_contribution(self, sample_context):
        """An owned first contribution need not be duplicated."""
        noise = np.ones_like(sample_context.get_raw()._data)

        sample_context.accumulate_noise(noise, copy=False)

        assert sample_context.get_estimated_noise() is noise

    def test_accumulate_noise_copies_first_contribution_by_default(self, sample_context):
        """Default accumulation retains its historical independent copy."""
        noise = np.ones_like(sample_context.get_raw()._data)

        sample_context.accumulate_noise(noise)

        assert sample_context.get_estimated_noise() is not noise
        np.testing.assert_array_equal(sample_context.get_estimated_noise(), noise)

    def test_processing_history(self, sample_context):
        """Test processing history tracking."""
        # Initially empty
        history = sample_context.get_history()
        assert len(history) == 0

        # Add entry
        sample_context.add_history_entry(processor_name="test_processor", parameters={"param1": "value1"})

        # Check it's recorded
        history = sample_context.get_history()
        assert len(history) == 1
        assert history[0].name == "test_processor"
        assert history[0].parameters == {"param1": "value1"}
        assert history[0].timestamp > 0

    def test_to_dict(self, sample_context):
        """Test serialization to dict."""
        data = sample_context.to_dict()

        assert isinstance(data, dict)
        assert "raw" not in data
        assert "raw_data" in data
        assert "metadata" in data
        assert "history" in data
        assert data["raw_data"] is sample_context.get_raw()._data

    def test_from_dict(self, sample_context):
        """Test deserialization from dict."""
        # Serialize
        data = sample_context.to_dict()

        # Deserialize
        context2 = ProcessingContext.from_dict(data)

        # Check equality
        assert context2.get_raw().info["sfreq"] == sample_context.get_raw().info["sfreq"]
        assert len(context2.get_history()) == len(sample_context.get_history())

    def test_copy_preserves_original_raw(self, sample_raw):
        """Test that with_raw preserves raw_original."""
        context1 = ProcessingContext(raw=sample_raw)

        # Modify raw
        raw2 = sample_raw.copy()
        raw2._data *= 2

        context2 = context1.with_raw(raw2)

        # Check original is preserved
        assert context2.get_raw_original() is sample_raw
        assert context2.get_raw() is raw2

    def test_immutability(self, sample_context):
        """Test that context is effectively immutable."""
        original_raw = sample_context.get_raw()
        original_triggers = sample_context.get_triggers().copy()

        # Create new context with modifications
        new_raw = original_raw.copy()
        new_raw._data *= 2
        sample_context.with_raw(new_raw)

        # Original should be unchanged
        assert sample_context.get_raw() is original_raw
        assert np.array_equal(sample_context.get_triggers(), original_triggers)

    def test_with_trigger_samples_accepts_tr_seconds(self, sample_raw):
        """tr_seconds should be converted to artifact_length in samples."""
        context = ProcessingContext(raw=sample_raw)
        result = context.with_trigger_samples([10, 30, 50], tr_seconds=0.2)

        assert np.array_equal(result.get_triggers(), np.array([10, 30, 50], dtype=np.int32))
        assert result.get_artifact_length() == int(round(0.2 * sample_raw.info["sfreq"]))

    def test_with_trigger_samples_infers_artifact_length_from_trigger_delta(self, sample_raw):
        """Without explicit length, artifact_length defaults to median trigger delta."""
        context = ProcessingContext(raw=sample_raw)
        result = context.with_trigger_samples([80, 20, 50])  # unsorted on purpose

        assert np.array_equal(result.get_triggers(), np.array([80, 20, 50], dtype=np.int32))
        assert result.get_artifact_length() == 30

    def test_with_trigger_samples_normalizes_absolute_samples(self):
        """Absolute trigger samples should be shifted by raw.first_samp."""
        sfreq = 200.0
        data = np.zeros((1, 300))
        info = mne.create_info(ch_names=["EEG001"], sfreq=sfreq, ch_types=["eeg"])
        raw = mne.io.RawArray(data, info, first_samp=1000, verbose=False)

        context = ProcessingContext(raw=raw)
        result = context.with_trigger_samples([1010, 1020], samples_are_absolute=True)

        assert np.array_equal(result.get_triggers(), np.array([10, 20], dtype=np.int32))

    def test_with_mne_events_by_name(self, sample_raw):
        """Event names should resolve via event_id mapping."""
        context = ProcessingContext(raw=sample_raw)
        events = np.array([[100, 0, 1], [200, 0, 2], [300, 0, 1]])
        event_id = {"R128": 1, "other": 2}

        result = context.with_mne_events(events, event="R128", event_id=event_id, tr_seconds=0.5)

        assert np.array_equal(result.get_triggers(), np.array([100, 300], dtype=np.int32))
        assert result.get_artifact_length() == int(round(0.5 * sample_raw.info["sfreq"]))
        assert result.metadata.custom["event_id"] == event_id
        assert result.metadata.custom["selected_event_name"] == "R128"
        assert result.metadata.custom["selected_event_code"] == 1

    def test_with_mne_events_infers_artifact_length_from_selected_event_deltas(self, sample_raw):
        """with_mne_events should default artifact_length to event sample delta."""
        context = ProcessingContext(raw=sample_raw)
        events = np.array([[100, 0, 1], [130, 0, 1], [170, 0, 2], [160, 0, 1]])

        result = context.with_mne_events(events, event=1)

        assert np.array_equal(result.get_triggers(), np.array([100, 130, 160], dtype=np.int32))
        assert result.get_artifact_length() == 30

    def test_with_mne_events_normalizes_absolute_samples(self):
        """MNE absolute event samples should be shifted by first_samp."""
        sfreq = 200.0
        data = np.zeros((1, 300))
        info = mne.create_info(ch_names=["EEG001"], sfreq=sfreq, ch_types=["eeg"])
        raw = mne.io.RawArray(data, info, first_samp=5000, verbose=False)
        context = ProcessingContext(raw=raw)
        events = np.array([[5010, 0, 1], [5030, 0, 1]])

        result = context.with_mne_events(events, event=1)

        assert np.array_equal(result.get_triggers(), np.array([10, 30], dtype=np.int32))

    def test_with_mne_events_requires_event_selection_for_multiple_codes(self, sample_raw):
        """Multiple event codes require explicit event selection."""
        context = ProcessingContext(raw=sample_raw)
        events = np.array([[100, 0, 1], [200, 0, 2]])

        with pytest.raises(ValueError, match="multiple event codes"):
            context.with_mne_events(events)

    def test_with_mne_events_requires_event_id_for_string_event(self, sample_raw):
        """String event selection requires event_id mapping."""
        context = ProcessingContext(raw=sample_raw)
        events = np.array([[100, 0, 1]])

        with pytest.raises(ValueError, match="event_id is required"):
            context.with_mne_events(events, event="R128")


@pytest.mark.unit
class TestProcessingContextPipeOperator:
    """Tests for ProcessingContext.__or__ (pipe operator)."""

    def test_pipe_with_processor(self, sample_context):
        """ctx | Processor applies the processor and returns new context."""
        from facet.core import Processor

        class DoubleData(Processor):
            name = "double_data"

            def process(self, context):
                raw = context.get_raw().copy()
                raw._data *= 2
                return context.with_raw(raw)

        original_data = sample_context.get_raw()._data.copy()
        result = sample_context | DoubleData()

        assert isinstance(result, ProcessingContext)
        # Original is unchanged
        np.testing.assert_array_equal(sample_context.get_raw()._data, original_data)
        # Result has doubled data
        np.testing.assert_array_almost_equal(result.get_raw()._data, original_data * 2)

    def test_pipe_with_callable(self, sample_context):
        """ctx | callable applies the function and returns new context."""

        def noop(ctx):
            return ctx.with_raw(ctx.get_raw().copy())

        result = sample_context | noop

        assert isinstance(result, ProcessingContext)
        assert result.get_sfreq() == sample_context.get_sfreq()

    def test_pipe_chaining(self, sample_context):
        """Multiple | calls can be chained."""
        from facet.core import Processor

        class AddToCustom(Processor):
            name = "add_custom"
            modifies_raw = False

            def __init__(self, key, value):
                self.key = key
                self.value = value
                super().__init__()

            def process(self, context):
                context.metadata.custom[self.key] = self.value
                return context

        result = sample_context | AddToCustom("step1", 1) | AddToCustom("step2", 2)

        assert result.metadata.custom["step1"] == 1
        assert result.metadata.custom["step2"] == 2

    def test_pipe_with_lambda(self, sample_context):
        """ctx | lambda works and returns context."""
        result = sample_context | (lambda ctx: ctx)
        assert isinstance(result, ProcessingContext)

    def test_pipe_unsupported_type_returns_not_implemented(self, sample_context):
        """ctx | unsupported_type returns NotImplemented."""
        result = sample_context.__or__(42)
        assert result is NotImplemented

    def test_pipe_history_tracked(self, sample_context):
        """Processor applied via | records a history entry."""
        from tests.conftest import create_mock_processor

        proc = create_mock_processor("pipe_proc")
        result = sample_context | proc

        names = [h.name for h in result.get_history()]
        assert "pipe_proc" in names
