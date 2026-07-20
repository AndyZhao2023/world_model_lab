from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_train_visual_dynamics_objective import save_source_checkpoint
from tests.test_visual_windows import make_visual_dataset
from world_model_lab.diagnose_model import sha256_file
from world_model_lab.train_visual_latent_model import (
    load_visual_latent_checkpoint,
)
from world_model_lab.train_visual_object_slot import (
    main,
    run_visual_object_slot_training,
)
from world_model_lab.visual_dataset import save_visual_dataset
from world_model_lab.visual_observation import CAR_COLOR, HEADING_COLOR


def write_object_slot_fixture(root: Path) -> tuple[Path, Path]:
    visual = make_visual_dataset((4,) * 10)
    visual["states"].fill(0.0)
    visual["frames"][:, 31, 31] = CAR_COLOR
    visual["frames"][:, 31, 32] = HEADING_COLOR
    data_path = root / "visual.npz"
    source_path = root / "source.pt"
    save_visual_dataset(visual, data_path)
    save_source_checkpoint(source_path, data_path=data_path)
    return data_path, source_path


class VisualObjectSlotRunnerTest(unittest.TestCase):
    def test_runner_trains_only_slot_and_records_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path, source_path = write_object_slot_fixture(root)
            output_path = root / "candidate.pt"
            preview_path = root / "candidate.png"
            source = load_visual_latent_checkpoint(source_path)

            summary = run_visual_object_slot_training(
                data_path=data_path,
                source_checkpoint_path=source_path,
                output_path=output_path,
                preview_path=preview_path,
                patch_size=11,
                hidden_size=8,
                initial_alpha=0.01,
                epochs=1,
                batch_size=8,
                learning_rate=1e-3,
                foreground_loss_weight=1.0,
                mask_loss_weight=0.01,
                centre_loss_weight=1.0,
                heading_loss_weight=0.1,
                source_probe_ridge=1e-3,
            )
            candidate = load_visual_latent_checkpoint(output_path)
            preview_size = preview_path.stat().st_size
            source_sha256 = sha256_file(source_path)

        self.assertTrue(candidate.autoencoder.object_slot_decoder)
        self.assertEqual(candidate.autoencoder.object_slot_patch_size, 11)
        self.assertEqual(candidate.autoencoder.object_slot_hidden_size, 8)
        self.assertGreater(preview_size, 0)
        config = candidate.training_config
        self.assertTrue(config["autoencoder_encoder_frozen"])
        self.assertTrue(config["autoencoder_base_decoder_frozen"])
        self.assertTrue(config["dynamics_reused_unmodified"])
        self.assertEqual(config["object_slot_patch_size"], 11)
        self.assertEqual(config["object_slot_hidden_size"], 8)
        self.assertEqual(config["object_centre_loss_weight"], 1.0)
        self.assertEqual(config["object_heading_loss_weight"], 0.1)
        self.assertEqual(config["object_slot_source_probe_ridge"], 1e-3)
        self.assertEqual(
            config["source_checkpoint_sha256"],
            source_sha256,
        )
        self.assertIn("source_probe", summary["autoencoder"])
        self.assertIn("slot", summary["autoencoder"])
        self.assertIn("state_gates", summary["decision"])
        self.assertLessEqual(
            summary["autoencoder"]["slot"]["max_support_pixels"],
            121,
        )
        for module_name in ("encoder_convolutions", "decoder_convolutions"):
            source_module = getattr(source.autoencoder, module_name)
            candidate_module = getattr(candidate.autoencoder, module_name)
            for name, expected in source_module.state_dict().items():
                torch.testing.assert_close(
                    candidate_module.state_dict()[name],
                    expected,
                    rtol=0.0,
                    atol=0.0,
                )
        for name, expected in source.dynamics.state_dict().items():
            torch.testing.assert_close(
                candidate.dynamics.state_dict()[name],
                expected,
                rtol=0.0,
                atol=0.0,
            )
        for split_name, expected in source.split_episode_ids.items():
            np.testing.assert_array_equal(
                candidate.split_episode_ids[split_name],
                expected,
            )

    def test_runner_rejects_output_collision_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path, source_path = write_object_slot_fixture(root)
            output_path = root / "existing.pt"
            output_path.write_bytes(b"existing")
            with patch(
                "world_model_lab.train_visual_object_slot."
                "train_object_slot_decoder"
            ) as train:
                with self.assertRaises(FileExistsError):
                    run_visual_object_slot_training(
                        data_path=data_path,
                        source_checkpoint_path=source_path,
                        output_path=output_path,
                        preview_path=root / "preview.png",
                        epochs=1,
                        batch_size=8,
                    )
                train.assert_not_called()

    def test_global_affine_runner_freezes_probe_and_uses_stability_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path, source_path = write_object_slot_fixture(root)
            output_path = root / "candidate-affine.pt"
            preview_path = root / "candidate-affine.png"

            summary = run_visual_object_slot_training(
                data_path=data_path,
                source_checkpoint_path=source_path,
                output_path=output_path,
                preview_path=preview_path,
                locator="global_affine",
                patch_size=11,
                hidden_size=8,
                initial_alpha=0.01,
                epochs=1,
                batch_size=8,
                learning_rate=1e-3,
                foreground_loss_weight=1.0,
                mask_loss_weight=0.01,
                centre_loss_weight=1.0,
                heading_loss_weight=0.1,
                source_probe_ridge=1e-3,
            )
            candidate = load_visual_latent_checkpoint(output_path)

        self.assertEqual(
            candidate.autoencoder.object_slot_locator,
            "global_affine",
        )
        self.assertEqual(
            candidate.training_config["object_slot_locator"],
            "global_affine",
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in candidate.autoencoder.object_center.parameters()
            )
        )
        centre_gate, heading_gate = summary["decision"]["state_gates"]
        self.assertEqual(
            centre_gate["name"],
            "held_out_centre_error_stability",
        )
        self.assertEqual(centre_gate["operator"], "<=")
        self.assertAlmostEqual(
            centre_gate["limit"],
            1.05
            * summary["autoencoder"]["source_probe"]["test"][
                "mean_centre_error_pixels"
            ],
        )
        self.assertEqual(heading_gate["operator"], "<")
        self.assertIn("centre_probe_conversion", summary["autoencoder"])

    def test_cli_help_and_pyproject_register_command(self):
        output = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["world-model-train-visual-object-slot", "--help"],
        ):
            with redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    main()

        self.assertEqual(raised.exception.code, 0)
        for option in (
            "--data",
            "--source-checkpoint",
            "--output",
            "--preview",
            "--locator",
            "--patch-size",
            "--hidden-size",
            "--initial-alpha",
            "--foreground-loss-weight",
            "--mask-loss-weight",
            "--centre-loss-weight",
            "--heading-loss-weight",
            "--source-probe-ridge",
        ):
            self.assertIn(option, output.getvalue())
        pyproject = (
            Path(__file__).resolve().parents[1] / "pyproject.toml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "world-model-train-visual-object-slot = "
            '"world_model_lab.train_visual_object_slot:main"',
            pyproject,
        )


if __name__ == "__main__":
    unittest.main()
