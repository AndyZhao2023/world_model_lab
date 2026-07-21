from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tests.test_visual_windows import make_visual_dataset
from world_model_lab.dataset import Normalizer
from world_model_lab.diagnose_model import sha256_file
from world_model_lab.train_visual_dynamics_objective import (
    main,
    run_frozen_decoder_dynamics_training,
)
from world_model_lab.train_visual_latent_model import (
    PhaseTrainingResult,
    load_visual_latent_checkpoint,
    run_visual_latent_training,
    save_visual_latent_checkpoint,
)
from world_model_lab.visual_dataset import save_visual_dataset
from world_model_lab.visual_latent_model import (
    SpatialConvAutoencoder,
    SpatialLatentDynamicsCNN,
    SpatialLatentDynamicsConvGRU,
)


def save_source_checkpoint(
    path: Path,
    *,
    data_path: Path,
    architecture: str = "cnn",
) -> Path:
    autoencoder = SpatialConvAutoencoder(
        latent_channels=2,
        base_channels=2,
    )
    if architecture == "cnn":
        dynamics = SpatialLatentDynamicsCNN(
            latent_channels=2,
            hidden_channels=4,
        )
    elif architecture == "convgru":
        dynamics = SpatialLatentDynamicsConvGRU(
            latent_channels=2,
            hidden_channels=4,
        )
    else:
        raise ValueError("unsupported test architecture")
    autoencoder_result = PhaseTrainingResult(
        model=autoencoder,
        train_losses=[0.4],
        validation_losses=[0.5],
        best_epoch=1,
    )
    dynamics_result = PhaseTrainingResult(
        model=dynamics,
        train_losses=[0.3],
        validation_losses=[0.35],
        best_epoch=1,
    )
    return save_visual_latent_checkpoint(
        path,
        autoencoder_result=autoencoder_result,
        dynamics_result=dynamics_result,
        latent_normalizer=Normalizer(
            np.zeros(autoencoder.latent_dim),
            np.ones(autoencoder.latent_dim),
        ),
        action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
        split_episode_ids={
            "train": np.arange(10, 18, dtype=np.int64),
            "validation": np.asarray([18], dtype=np.int64),
            "test": np.asarray([19], dtype=np.int64),
        },
        training_config={
            "data_path": str(data_path),
            "device": "cpu",
            "latent_layout": "spatial",
            "latent_dim": autoencoder.latent_dim,
            "spatial_latent_channels": 2,
            "spatial_dynamics_architecture": architecture,
            "base_channels": 2,
            "dynamics_hidden_size": 4,
            "autoencoder_epochs": 1,
            "dynamics_epochs": 1,
            "autoencoder_batch_size": 8,
            "dynamics_batch_size": 8,
            "autoencoder_learning_rate": 1e-3,
            "dynamics_learning_rate": 1e-3,
            "motion_loss_weight": 0.0,
            "seed": 3,
            "split_seed": 19,
        },
        dataset_metadata={
            "path": str(data_path.resolve()),
            "sha256": sha256_file(data_path),
            "schema_version": 1,
            "renderer_version": "pillow-raster-v1",
        },
        autoencoder_test_metrics={
            "frames": 5,
            "pixel_mse": 0.1,
            "pixel_mae": 0.2,
            "psnr_db": 10.0,
        },
        dynamics_test_metrics={"windows": 1},
    )


class FrozenDecoderDynamicsTrainingTest(unittest.TestCase):
    def test_cli_help_lists_controlled_experiment_parameters(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["world-model-objective", "--help"]):
            with redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    main()

        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        for name in (
            "--data",
            "--source-checkpoint",
            "--output",
            "--preview",
            "--changed-pixel-loss-weight",
            "--dynamics-epochs",
            "--dynamics-batch-size",
        ):
            self.assertIn(name, help_text)

    def test_pyproject_registers_objective_training_command(self):
        project_root = Path(__file__).resolve().parents[1]
        pyproject = (project_root / "pyproject.toml").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "world-model-train-visual-dynamics-objective = "
            '"world_model_lab.train_visual_dynamics_objective:main"',
            pyproject,
        )

    def test_runner_reuses_source_autoencoder_and_metadata(self):
        visual = make_visual_dataset((4,) * 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            source_path = root / "source.pt"
            output_path = root / "candidate.pt"
            preview_path = root / "candidate.png"
            save_visual_dataset(visual, data_path)
            save_source_checkpoint(source_path, data_path=data_path)
            source = load_visual_latent_checkpoint(source_path)

            summary = run_frozen_decoder_dynamics_training(
                data_path=data_path,
                source_checkpoint_path=source_path,
                output_path=output_path,
                preview_path=preview_path,
                changed_pixel_loss_weight=0.1,
                dynamics_epochs=1,
                dynamics_batch_size=8,
            )
            loaded = load_visual_latent_checkpoint(output_path)

            self.assertTrue(output_path.is_file())
            self.assertTrue(preview_path.is_file())
            self.assertGreater(preview_path.stat().st_size, 0)
            self.assertEqual(
                loaded.training_config["source_checkpoint_sha256"],
                sha256_file(source_path),
            )
            self.assertTrue(loaded.training_config["autoencoder_frozen"])
            self.assertEqual(
                loaded.training_config[
                    "dynamics_changed_pixel_loss_weight"
                ],
                0.1,
            )
            self.assertEqual(
                summary["split_windows"],
                {"train": 8, "validation": 1, "test": 1},
            )
            for name, tensor in source.autoencoder.state_dict().items():
                torch.testing.assert_close(
                    loaded.autoencoder.state_dict()[name],
                    tensor,
                    rtol=0.0,
                    atol=0.0,
                )
            for split_name, expected in source.split_episode_ids.items():
                np.testing.assert_array_equal(
                    loaded.split_episode_ids[split_name],
                    expected,
                )
            np.testing.assert_array_equal(
                loaded.latent_normalizer.mean,
                source.latent_normalizer.mean,
            )
            np.testing.assert_array_equal(
                loaded.latent_normalizer.std,
                source.latent_normalizer.std,
            )
            np.testing.assert_array_equal(
                loaded.action_normalizer.mean,
                source.action_normalizer.mean,
            )
            np.testing.assert_array_equal(
                loaded.action_normalizer.std,
                source.action_normalizer.std,
            )

    def test_runner_rejects_dataset_hash_mismatch(self):
        visual = make_visual_dataset((4,) * 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_data_path = root / "source-data.npz"
            changed_data_path = root / "changed-data.npz"
            source_path = root / "source.pt"
            save_visual_dataset(visual, source_data_path)
            save_source_checkpoint(source_path, data_path=source_data_path)
            changed = {name: values.copy() for name, values in visual.items()}
            changed["frames"][0, 0, 0, 0] ^= np.uint8(1)
            save_visual_dataset(changed, changed_data_path)

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                run_frozen_decoder_dynamics_training(
                    data_path=changed_data_path,
                    source_checkpoint_path=source_path,
                    output_path=root / "candidate.pt",
                    preview_path=root / "candidate.png",
                    changed_pixel_loss_weight=0.1,
                    dynamics_epochs=1,
                    dynamics_batch_size=8,
                )

    def test_runner_rejects_non_cnn_spatial_source(self):
        visual = make_visual_dataset((4,) * 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            source_path = root / "convgru.pt"
            save_visual_dataset(visual, data_path)
            save_source_checkpoint(
                source_path,
                data_path=data_path,
                architecture="convgru",
            )

            with self.assertRaisesRegex(ValueError, "spatial CNN"):
                run_frozen_decoder_dynamics_training(
                    data_path=data_path,
                    source_checkpoint_path=source_path,
                    output_path=root / "candidate.pt",
                    preview_path=root / "candidate.png",
                    changed_pixel_loss_weight=0.1,
                    dynamics_epochs=1,
                    dynamics_batch_size=8,
                )

    def test_runner_rejects_global_source(self):
        visual = make_visual_dataset((4,) * 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            source_path = root / "global.pt"
            save_visual_dataset(visual, data_path)
            run_visual_latent_training(
                data_path=data_path,
                output_path=source_path,
                preview_path=root / "global.png",
                latent_dim=4,
                base_channels=2,
                dynamics_hidden_size=4,
                autoencoder_epochs=1,
                dynamics_epochs=1,
                autoencoder_batch_size=8,
                dynamics_batch_size=8,
                seed=3,
                split_seed=19,
            )

            with self.assertRaisesRegex(ValueError, "spatial CNN"):
                run_frozen_decoder_dynamics_training(
                    data_path=data_path,
                    source_checkpoint_path=source_path,
                    output_path=root / "candidate.pt",
                    preview_path=root / "candidate.png",
                    changed_pixel_loss_weight=0.1,
                    dynamics_epochs=1,
                    dynamics_batch_size=8,
                )

    def test_preview_failure_rolls_back_candidate_checkpoint(self):
        visual = make_visual_dataset((4,) * 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            source_path = root / "source.pt"
            output_path = root / "candidate.pt"
            save_visual_dataset(visual, data_path)
            save_source_checkpoint(source_path, data_path=data_path)

            with patch(
                "world_model_lab.train_visual_dynamics_objective."
                "plot_visual_latent_predictions",
                side_effect=RuntimeError("preview failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "preview failed"):
                    run_frozen_decoder_dynamics_training(
                        data_path=data_path,
                        source_checkpoint_path=source_path,
                        output_path=output_path,
                        preview_path=root / "candidate.png",
                        changed_pixel_loss_weight=0.1,
                        dynamics_epochs=1,
                        dynamics_batch_size=8,
                    )

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
