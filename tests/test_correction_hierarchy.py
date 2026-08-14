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


class TestTemplateCorrectionHierarchy:
    """Lock the inheritance direction and public discovery contract."""

    def test_flex_directly_inherits_processor(self):
        """Flex should own the engine without depending on a legacy method."""
        assert Flex.__bases__ == (Processor,)

    @pytest.mark.parametrize("correction_class", TEMPLATE_CORRECTION_CLASSES)
    def test_template_variants_use_flex_engine(self, correction_class):
        """Compatibility names should all delegate to the shared Flex engine."""
        assert issubclass(correction_class, Flex)

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
                "importlib.import_module('facet.correction.presets')",
                "importlib.import_module('facet.correction.legacy_adapters')",
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


class TestPresetBackedCompatibility:
    """Verify that public legacy names now expose named Flex recipes."""

    def test_shared_engine_rejects_nonfinite_data(self, sample_context):
        """Compatibility adapters must retain Flex's finite-data guard."""
        sample_context.get_raw()._data[0, 0] = np.nan
        processor = AASCorrection(window_size=3, realign_after_averaging=False)

        with pytest.raises(ProcessorValidationError, match="requires finite epoch data"):
            processor.execute(sample_context)

    @pytest.mark.parametrize(
        ("processor", "preset"),
        [
            (AASCorrection(), "aas_per_target"),
            (FARMCorrection(), "farm_per_target_k10"),
            (CorrespondingSliceCorrection(slices_per_volume=2), "corresponding_slice"),
            (VolumeTriggerCorrection(), "structural_volume"),
            (SliceTriggerCorrection(), "structural_slice"),
        ],
    )
    def test_legacy_names_expose_flex_decision_manifest(self, processor, preset):
        """Every compatibility adapter should carry reportable recipe provenance."""
        assert processor.flex_preset_name == preset
        assert processor.legacy_algorithm_resemblance
        assert processor.matrix_decisions is not None
        assert set(processor.matrix_decisions.to_dict()) == {
            "motion",
            "quota",
            "sampling",
            "scoring",
            "target_policy",
            "template_size",
            "weighting",
        }

    def test_moosmann_name_exposes_motion_flex_recipe(self, tmp_path):
        """Moosmann compatibility should resolve to the motion-aware recipe."""
        rp_file = tmp_path / "rp_no_motion.txt"
        rp_file.write_text("0 0 0 0 0 0\n", encoding="utf-8")
        processor = MoosmannCorrection(rp_file=str(rp_file))

        assert processor.flex_preset_name == "moosmann_cost"
        assert processor.matrix_decisions.quota.global_mode is True
        assert processor.matrix_decisions.motion.motion_stable_only is True
