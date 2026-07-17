from __future__ import annotations

import io
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch

from tests.test_train_visual_dynamics_objective import save_source_checkpoint
from tests.test_visual_windows import make_visual_dataset
from world_model_lab.diagnose_model import sha256_file
from world_model_lab.train_visual_dynamics_recursive import (
    main,
    run_recursive_dynamics_training,
)
from world_model_lab.train_visual_latent_model import (
    load_visual_latent_checkpoint,
)
from world_model_lab.visual_dataset import save_visual_dataset


class RecursiveDynamicsRunnerTest(unittest.TestCase):
    def test_cli_help_lists_recursive_protocol(self):
        output = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["world-model-train-visual-dynamics-recursive", "--help"],
        ):
            with redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    main()

        self.assertEqual(raised.exception.code, 0)
        for name in (
            "--data",
            "--source-checkpoint",
            "--output",
            "--preview",
            "--changed-pixel-loss-weight",
            "--rollout-horizon",
            "--rollout-loss-weight",
            "--dynamics-epochs",
            "--dynamics-batch-size",
        ):
            self.assertIn(name, output.getvalue())

    def test_pyproject_registers_recursive_training_command(self):
        project_root = Path(__file__).resolve().parents[1]
        pyproject = (project_root / "pyproject.toml").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "world-model-train-visual-dynamics-recursive = "
            '"world_model_lab.train_visual_dynamics_recursive:main"',
            pyproject,
        )

    def test_runner_reuses_representation_and_records_h5_protocol(self):
        visual = make_visual_dataset((8,) * 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            source_path = root / "source.pt"
            output_path = root / "candidate.pt"
            preview_path = root / "candidate.png"
            save_visual_dataset(visual, data_path)
            save_source_checkpoint(source_path, data_path=data_path)
            source = load_visual_latent_checkpoint(source_path)

            summary = run_recursive_dynamics_training(
                data_path=data_path,
                source_checkpoint_path=source_path,
                output_path=output_path,
                preview_path=preview_path,
                changed_pixel_loss_weight=0.1,
                rollout_horizon=5,
                rollout_loss_weight=1.0,
                dynamics_epochs=1,
                dynamics_batch_size=8,
            )
            candidate = load_visual_latent_checkpoint(output_path)

            self.assertTrue(output_path.is_file())
            self.assertTrue(preview_path.is_file())
            self.assertGreater(preview_path.stat().st_size, 0)
            self.assertEqual(
                summary["one_step_windows"],
                {"train": 40, "validation": 5, "test": 5},
            )
            self.assertEqual(
                summary["rollout_windows"],
                {"train": 8, "validation": 1, "test": 1},
            )
            config = candidate.training_config
            self.assertEqual(config["dynamics_rollout_horizon"], 5)
            self.assertEqual(config["dynamics_rollout_loss_weight"], 1.0)
            self.assertEqual(
                config["dynamics_changed_pixel_loss_weight"],
                0.1,
            )
            self.assertTrue(config["dynamics_reinitialized"])
            self.assertTrue(config["autoencoder_frozen"])
            self.assertEqual(
                config["source_checkpoint_sha256"],
                sha256_file(source_path),
            )
            for name, tensor in source.autoencoder.state_dict().items():
                torch.testing.assert_close(
                    candidate.autoencoder.state_dict()[name],
                    tensor,
                    rtol=0.0,
                    atol=0.0,
                )
            for split_name, expected in source.split_episode_ids.items():
                np.testing.assert_array_equal(
                    candidate.split_episode_ids[split_name],
                    expected,
                )
            for candidate_normalizer, source_normalizer in (
                (
                    candidate.latent_normalizer,
                    source.latent_normalizer,
                ),
                (
                    candidate.action_normalizer,
                    source.action_normalizer,
                ),
            ):
                np.testing.assert_array_equal(
                    candidate_normalizer.mean,
                    source_normalizer.mean,
                )
                np.testing.assert_array_equal(
                    candidate_normalizer.std,
                    source_normalizer.std,
                )

    def test_runner_rejects_invalid_protocol_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                ({"rollout_horizon": 1}, "greater than one"),
                ({"rollout_loss_weight": -1.0}, "rollout_loss_weight"),
                (
                    {"rollout_loss_weight": float("inf")},
                    "rollout_loss_weight",
                ),
            )
            for index, (update, message) in enumerate(cases):
                with self.subTest(update=update):
                    kwargs = {
                        "data_path": root / "missing.npz",
                        "source_checkpoint_path": root / "missing.pt",
                        "output_path": root / f"candidate-{index}.pt",
                        "preview_path": root / f"candidate-{index}.png",
                        "changed_pixel_loss_weight": 0.1,
                        "rollout_horizon": 5,
                        "rollout_loss_weight": 1.0,
                    }
                    kwargs.update(update)
                    with self.assertRaisesRegex(ValueError, message):
                        run_recursive_dynamics_training(**kwargs)

    def test_runner_rejects_dataset_hash_or_non_cnn_source(self):
        visual = make_visual_dataset((8,) * 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_data_path = root / "source-data.npz"
            changed_data_path = root / "changed-data.npz"
            source_path = root / "source.pt"
            convgru_path = root / "convgru.pt"
            save_visual_dataset(visual, source_data_path)
            save_source_checkpoint(source_path, data_path=source_data_path)
            save_source_checkpoint(
                convgru_path,
                data_path=source_data_path,
                architecture="convgru",
            )
            changed = {name: values.copy() for name, values in visual.items()}
            changed["frames"][0, 0, 0, 0] ^= np.uint8(1)
            save_visual_dataset(changed, changed_data_path)

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                run_recursive_dynamics_training(
                    data_path=changed_data_path,
                    source_checkpoint_path=source_path,
                    output_path=root / "hash.pt",
                    preview_path=root / "hash.png",
                )
            with self.assertRaisesRegex(ValueError, "spatial CNN"):
                run_recursive_dynamics_training(
                    data_path=source_data_path,
                    source_checkpoint_path=convgru_path,
                    output_path=root / "convgru-candidate.pt",
                    preview_path=root / "convgru-candidate.png",
                )

    def test_runner_rejects_splits_without_h5_windows(self):
        visual = make_visual_dataset((4,) * 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            source_path = root / "source.pt"
            save_visual_dataset(visual, data_path)
            save_source_checkpoint(source_path, data_path=data_path)

            with self.assertRaisesRegex(ValueError, "train.*H5"):
                run_recursive_dynamics_training(
                    data_path=data_path,
                    source_checkpoint_path=source_path,
                    output_path=root / "candidate.pt",
                    preview_path=root / "candidate.png",
                    rollout_horizon=5,
                )

    def test_preview_failure_rolls_back_candidate_checkpoint(self):
        visual = make_visual_dataset((8,) * 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            source_path = root / "source.pt"
            output_path = root / "candidate.pt"
            save_visual_dataset(visual, data_path)
            save_source_checkpoint(source_path, data_path=data_path)

            with patch(
                "world_model_lab.train_visual_dynamics_recursive."
                "plot_visual_latent_predictions",
                side_effect=RuntimeError("preview failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "preview failed"):
                    run_recursive_dynamics_training(
                        data_path=data_path,
                        source_checkpoint_path=source_path,
                        output_path=output_path,
                        preview_path=root / "candidate.png",
                        changed_pixel_loss_weight=0.1,
                        rollout_horizon=5,
                        rollout_loss_weight=1.0,
                        dynamics_epochs=1,
                        dynamics_batch_size=8,
                    )

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
