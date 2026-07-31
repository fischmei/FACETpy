"""Regression tests for the shared Flex template-correction hierarchy."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable

import numpy as np
import pytest

import facet
from facet.core import Processor, ProcessorValidationError, get_processor
from facet.correction import (
    AASCorrection,
    AveragedArtifactSubtraction,
    AvgArtWghtCorrespondingSliceCorrection,
    AvgArtWghtMoosmannCorrection,
    AvgArtWghtSliceTriggerCorrection,
    AvgArtWghtVolumeTriggerCorrection,
    CorrespondingSliceCorrection,
    FARMArtifactCorrection,
    FARMCorrection,
    Flex,
    FlexCorrection,
    MoosmannCorrection,
    SliceTriggerCorrection,
    VolumeTriggerCorrection,
)

pytestmark = pytest.mark.unit

TEMPLATE_CORRECTION_CLASSES = (
    AASCorrection,
    FARMCorrection,
    CorrespondingSliceCorrection,
    VolumeTriggerCorrection,
    SliceTriggerCorrection,
    MoosmannCorrection,
)


def _scaled_epochs(n_epochs: int = 8) -> np.ndarray:
    """Return non-constant epochs with identical positive correlation."""
    base_epoch = np.array([-3.0, -1.0, 0.0, 2.0, 4.0])
    return np.vstack([(epoch_idx + 1.0) * base_epoch for epoch_idx in range(n_epochs)])


def _averaging_matrix(processor, epochs: np.ndarray) -> np.ndarray:
    """Call the matrix-strategy hook through its shared engine contract."""
    return processor._calc_averaging_matrix(
        epochs=epochs,
        window_size=processor.window_size,
        rel_window_offset=processor.rel_window_position,
        correlation_threshold=processor.correlation_threshold,
    )


class TestTemplateCorrectionHierarchy:
    """Lock the inheritance direction and public discovery contract."""

    def test_flex_directly_inherits_processor(self):
        """Flex should own the engine without depending on a legacy method."""
        assert Flex.__bases__ == (Processor,)

    @pytest.mark.parametrize("correction_class", TEMPLATE_CORRECTION_CLASSES)
    def test_template_variants_directly_inherit_flex(self, correction_class):
        """Legacy matrix strategies should be removable independently of one another."""
        assert Flex in correction_class.__bases__

    @pytest.mark.parametrize(
        ("name", "correction_class"),
        [
            ("flex_correction", Flex),
            ("aas_correction", AASCorrection),
            ("farm_correction", FARMCorrection),
            ("corresponding_slice_correction", CorrespondingSliceCorrection),
            ("volume_trigger_correction", VolumeTriggerCorrection),
            ("slice_trigger_correction", SliceTriggerCorrection),
            ("moosmann_correction", MoosmannCorrection),
        ],
    )
    def test_template_variants_remain_registered(self, name, correction_class):
        """Reparenting must not change processor registry names or classes."""
        assert get_processor(name) is correction_class

    @pytest.mark.parametrize(
        ("public_object", "expected"),
        [
            (facet.Flex, Flex),
            (facet.FlexCorrection, Flex),
            (FlexCorrection, Flex),
            (AveragedArtifactSubtraction, AASCorrection),
            (FARMArtifactCorrection, FARMCorrection),
            (AvgArtWghtCorrespondingSliceCorrection, CorrespondingSliceCorrection),
            (AvgArtWghtVolumeTriggerCorrection, VolumeTriggerCorrection),
            (AvgArtWghtSliceTriggerCorrection, SliceTriggerCorrection),
            (AvgArtWghtMoosmannCorrection, MoosmannCorrection),
        ],
    )
    def test_public_aliases_remain_stable(self, public_object, expected):
        """Existing imports should retain identity after the hierarchy changes."""
        assert public_object is expected

    def test_correction_modules_import_in_a_clean_interpreter(self, tmp_path):
        """The base-first module graph should remain free of circular imports."""
        script = "\n".join(
            [
                "import importlib",
                "importlib.import_module('facet.correction.flex')",
                "importlib.import_module('facet.correction.aas')",
                "importlib.import_module('facet.correction.farm')",
                "importlib.import_module('facet.correction.weighted')",
            ]
        )
        environment = os.environ.copy()
        environment["MNE_DONTWRITE_HOME"] = "true"
        environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr


class TestTemplateCorrectionReconstruction:
    """Ensure multiprocessing can recreate each configured correction class."""

    PROCESSOR_IDS = (
        "flex",
        "aas",
        "farm",
        "corresponding-slice",
        "volume-trigger",
        "slice-trigger",
        "moosmann",
    )

    @staticmethod
    def _processor_factories(rp_file: str) -> tuple[Callable[[], Processor], ...]:
        return (
            lambda: Flex(
                window_size=7,
                threshold=0.81,
                min_accepted=3,
                N_distribution="normal",
                realign_after_averaging=False,
                interpolate_volume_gaps=True,
                apply_epoch_alpha_scaling=True,
            ),
            lambda: AASCorrection(
                window_size=7,
                rel_window_position=0.25,
                correlation_threshold=0.81,
                realign_after_averaging=False,
                interpolate_volume_gaps=True,
                apply_epoch_alpha_scaling=True,
            ),
            lambda: FARMCorrection(
                window_size=7,
                correlation_threshold=0.81,
                search_half_window=11,
                search_half_window_factor=2.5,
                rel_window_position=0.25,
                realign_after_averaging=False,
                interpolate_volume_gaps=True,
                apply_epoch_alpha_scaling=True,
            ),
            lambda: CorrespondingSliceCorrection(
                slices_per_volume=4,
                window_size=7,
                realign_after_averaging=False,
                apply_epoch_alpha_scaling=True,
            ),
            lambda: VolumeTriggerCorrection(
                window_size=7,
                realign_after_averaging=False,
                apply_epoch_alpha_scaling=True,
            ),
            lambda: SliceTriggerCorrection(
                window_size=7,
                realign_after_averaging=False,
                apply_epoch_alpha_scaling=True,
            ),
            lambda: MoosmannCorrection(
                rp_file=rp_file,
                window_size=7,
                motion_threshold=2.5,
                motion_window_size=9,
                realign_after_averaging=False,
                apply_epoch_alpha_scaling=True,
            ),
        )

    @pytest.mark.parametrize(
        "factory_index",
        range(len(PROCESSOR_IDS)),
        ids=PROCESSOR_IDS,
    )
    def test_worker_parameters_reconstruct_every_template_variant(self, tmp_path, factory_index):
        """``_parameters`` must contain exactly constructor-compatible state."""
        rp_file = tmp_path / "rp_reconstruction.txt"
        rp_file.write_text("0 0 0 0 0 0\n", encoding="utf-8")

        factory = self._processor_factories(str(rp_file))[factory_index]
        processor = factory()
        reconstructed = type(processor)(**processor._parameters)

        assert type(reconstructed) is type(processor)
        assert reconstructed._parameters == processor._parameters


class TestLegacyAveragingMatrices:
    """Freeze the numerical matrix strategies while their engine moves to Flex."""

    def test_shared_engine_rejects_nonfinite_data_before_legacy_strategy(self, sample_context):
        """Legacy matrix overrides must not bypass Flex's finite-data guard."""
        sample_context.get_raw()._data[0, 0] = np.nan
        processor = AASCorrection(window_size=3, realign_after_averaging=False)

        with pytest.raises(ProcessorValidationError, match="requires finite epoch data"):
            processor.execute(sample_context)

    def test_aas_matrix_is_unchanged(self):
        """AAS should retain its block-seeded running-average weights."""
        processor = AASCorrection(
            window_size=3,
            correlation_threshold=0.99,
            realign_after_averaging=False,
        )
        matrix = _averaging_matrix(processor, _scaled_epochs())

        expected = np.zeros((8, 8), dtype=float)
        expected[0:3, 0:5] = 1.0 / 5.0
        expected[3:6, 3:8] = 1.0 / 5.0
        expected[6:8, 6:8] = 1.0 / 2.0

        np.testing.assert_allclose(matrix, expected)

    def test_farm_matrix_is_unchanged(self):
        """FARM should retain absolute-correlation ranking within its search window."""
        epochs = np.random.RandomState(47).normal(size=(8, 11))
        processor = FARMCorrection(
            window_size=3,
            correlation_threshold=0.1,
            search_half_window=3,
            realign_after_averaging=False,
        )
        matrix = _averaging_matrix(processor, epochs)

        selected_by_row = (
            (1, 4, 6),
            (3, 5, 6),
            (4, 5, 6),
            (1, 5, 6),
            (1, 3, 5),
            (1, 3, 6),
            (1, 2, 3),
            (2, 3, 6),
        )
        expected = np.zeros((8, 8), dtype=float)
        for row, selected in enumerate(selected_by_row):
            expected[row, selected] = 1.0 / 3.0

        np.testing.assert_allclose(matrix, expected)

    def test_corresponding_slice_matrix_is_unchanged(self):
        """Corresponding-slice templates should retain slice-position parity."""
        processor = CorrespondingSliceCorrection(
            slices_per_volume=2,
            window_size=2,
            realign_after_averaging=False,
        )
        # The execution path resolves this value from processor configuration
        # or context metadata immediately before asking the shared engine for A.
        processor._runtime_slices_per_volume = 2
        matrix = _averaging_matrix(processor, _scaled_epochs())

        expected = np.zeros((8, 8), dtype=float)
        for row in range(8):
            expected[row, row % 2 :: 2] = 1.0 / 4.0

        np.testing.assert_allclose(matrix, expected)

    def test_volume_trigger_matrix_is_unchanged(self):
        """Volume-trigger correction should retain its fixed border window."""
        processor = VolumeTriggerCorrection(
            window_size=4,
            realign_after_averaging=False,
        )
        matrix = _averaging_matrix(processor, _scaled_epochs())

        expected = np.zeros((8, 8), dtype=float)
        expected[:, 1:6] = 1.0 / 5.0

        np.testing.assert_allclose(matrix, expected)

    def test_slice_trigger_matrix_is_unchanged(self):
        """Slice-trigger correction should retain alternating epoch sets."""
        processor = SliceTriggerCorrection(
            window_size=2,
            realign_after_averaging=False,
        )
        matrix = _averaging_matrix(processor, _scaled_epochs())

        expected = np.zeros((8, 8), dtype=float)
        expected[0, [1, 3, 5]] = 1.0 / 3.0
        expected[1:, [2, 4, 6]] = 1.0 / 3.0

        np.testing.assert_allclose(matrix, expected)

    def test_moosmann_matrix_is_unchanged_without_motion(self, tmp_path):
        """Moosmann should retain its normalized centered moving window."""
        rp_file = tmp_path / "rp_no_motion.txt"
        rp_file.write_text("0 0 0 0 0 0\n" * 8, encoding="utf-8")
        processor = MoosmannCorrection(
            rp_file=str(rp_file),
            window_size=2,
            realign_after_averaging=False,
        )
        matrix = _averaging_matrix(processor, _scaled_epochs())

        expected = np.zeros((8, 8), dtype=float)
        for row in range(8):
            start = max(0, row - 2)
            stop = min(8, row + 3)
            expected[row, start:stop] = 1.0 / (stop - start)

        np.testing.assert_allclose(matrix, expected)
