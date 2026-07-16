from __future__ import annotations

import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tests.test_visual_windows import make_visual_dataset
from world_model_lab.dataset import Normalizer, split_episode_ids
from world_model_lab.train_visual_latent_model import (
    PhaseTrainingResult,
    _motion_weighted_mse,
    evaluate_autoencoder,
    evaluate_latent_dynamics,
    load_visual_latent_checkpoint,
    main,
    run_visual_latent_training,
    save_visual_latent_checkpoint,
    train_autoencoder,
    train_latent_dynamics,
)
from world_model_lab.visual_dataset import save_visual_dataset
from world_model_lab.visual_latent_data import LatentWindowArrays
from world_model_lab.visual_latent_model import (
    ConvAutoencoder,
    LatentDynamicsMLP,
)


def make_tiny_visual_dataset() -> dict[str, np.ndarray]:
    return make_visual_dataset((4,) * 10)


def make_latent_arrays(
    *,
    count: int = 64,
    latent_dim: int = 3,
    seed: int = 7,
) -> LatentWindowArrays:
    rng = np.random.default_rng(seed)
    context = rng.normal(size=(count, 4, latent_dim)).astype(np.float32)
    history_actions = rng.normal(size=(count, 3, 2))
    current_actions = rng.normal(size=(count, 2))
    target = context[:, -1].copy()
    target[:, 0] += 0.15 * current_actions[:, 0]
    if latent_dim > 1:
        target[:, 1] -= 0.2 * current_actions[:, 1]
    return LatentWindowArrays(
        context_latents=context,
        history_actions=history_actions,
        current_actions=current_actions,
        target_latents=target.astype(np.float32),
        last_frame_indices=np.arange(count, dtype=np.int64),
        target_frame_indices=np.arange(count, dtype=np.int64),
        episode_ids=np.zeros(count, dtype=np.int64),
        step_ids=np.arange(count, dtype=np.int64) + 3,
    )


class VisualAutoencoderTrainingTest(unittest.TestCase):
    def test_motion_weighted_mse_matches_plain_mse_and_weighted_formula(self):
        target = torch.zeros((1, 3, 1, 2), dtype=torch.float32)
        prediction = torch.tensor(
            [[[[1.0, 0.5]], [[1.0, 0.5]], [[1.0, 0.5]]]],
            dtype=torch.float32,
        )
        mask = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)

        plain = _motion_weighted_mse(
            prediction,
            target,
            mask,
            motion_loss_weight=0.0,
        )
        weighted = _motion_weighted_mse(
            prediction,
            target,
            mask,
            motion_loss_weight=3.0,
        )

        torch.testing.assert_close(
            plain,
            torch.mean(torch.square(prediction - target)),
        )
        torch.testing.assert_close(weighted, torch.tensor(12.75 / 15.0))

    def test_motion_weighted_mse_rejects_invalid_inputs(self):
        images = torch.zeros((1, 3, 2, 2))
        masks = torch.zeros((1, 1, 2, 2))
        cases = (
            (images[:, :2], images, masks, 1.0),
            (images, images, masks[:, :, :1], 1.0),
            (images, images, masks, -1.0),
            (images, images, masks, float("nan")),
            (images, images, masks, float("inf")),
        )
        for prediction, target, mask, weight in cases:
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    _motion_weighted_mse(
                        prediction,
                        target,
                        mask,
                        motion_loss_weight=weight,
                    )

    def test_autoencoder_training_records_histories_and_test_metrics(self):
        visual = make_tiny_visual_dataset()
        splits = split_episode_ids(visual["episode_ids"], seed=19)

        result = train_autoencoder(
            visual,
            split_episode_ids=splits,
            latent_dim=4,
            base_channels=2,
            epochs=2,
            batch_size=8,
            learning_rate=1e-3,
            seed=3,
        )
        metrics = evaluate_autoencoder(
            result.model,
            visual,
            selected_episode_ids=splits["test"],
            batch_size=8,
        )

        self.assertIsInstance(result, PhaseTrainingResult)
        self.assertEqual(len(result.train_losses), 2)
        self.assertEqual(len(result.validation_losses), 2)
        self.assertTrue(np.all(np.isfinite(result.train_losses)))
        self.assertTrue(np.all(np.isfinite(result.validation_losses)))
        self.assertGreaterEqual(result.best_epoch, 1)
        self.assertLessEqual(result.best_epoch, 2)
        self.assertFalse(result.model.training)
        self.assertEqual(metrics["frames"], 5)
        for name in ("pixel_mse", "pixel_mae", "psnr_db"):
            self.assertTrue(math.isfinite(float(metrics[name])))

    def test_autoencoder_training_rejects_invalid_hyperparameters(self):
        visual = make_tiny_visual_dataset()
        splits = split_episode_ids(visual["episode_ids"], seed=19)
        invalid = (
            {"epochs": 0},
            {"batch_size": 0},
            {"learning_rate": 0.0},
            {"motion_loss_weight": -1.0},
            {"motion_loss_weight": float("nan")},
            {"motion_loss_weight": float("inf")},
            {"seed": -1},
        )
        for values in invalid:
            with self.subTest(values=values):
                config = {
                    "epochs": 1,
                    "batch_size": 8,
                    "learning_rate": 1e-3,
                    "seed": 3,
                }
                config.update(values)
                with self.assertRaises(ValueError):
                    train_autoencoder(
                        visual,
                        split_episode_ids=splits,
                        latent_dim=4,
                        base_channels=2,
                        **config,
                    )


class VisualLatentDynamicsTrainingTest(unittest.TestCase):
    def test_dynamics_training_reduces_a_deterministic_residual(self):
        arrays = make_latent_arrays()
        train = LatentWindowArrays(
            **{
                name: np.asarray(getattr(arrays, name))[:48]
                for name in arrays.__dataclass_fields__
            }
        )
        validation = LatentWindowArrays(
            **{
                name: np.asarray(getattr(arrays, name))[48:]
                for name in arrays.__dataclass_fields__
            }
        )
        latent_normalizer = Normalizer(
            mean=np.zeros(3),
            std=np.ones(3),
        )
        action_normalizer = Normalizer(
            mean=np.zeros(2),
            std=np.ones(2),
        )

        result = train_latent_dynamics(
            train,
            validation,
            latent_normalizer=latent_normalizer,
            action_normalizer=action_normalizer,
            hidden_size=16,
            epochs=60,
            batch_size=16,
            learning_rate=5e-3,
            seed=5,
        )

        self.assertEqual(len(result.train_losses), 60)
        self.assertTrue(np.all(np.isfinite(result.train_losses)))
        self.assertLess(result.train_losses[-1], result.train_losses[0])
        self.assertGreaterEqual(result.best_epoch, 1)
        self.assertLessEqual(result.best_epoch, 60)
        self.assertFalse(result.model.training)

    def test_decoded_metrics_include_changed_pixels_and_copy_last(self):
        visual = make_tiny_visual_dataset()
        visual["frames"][3].fill(0)
        visual["frames"][4].fill(255)
        arrays = LatentWindowArrays(
            context_latents=np.zeros((1, 4, 4), dtype=np.float32),
            history_actions=np.zeros((1, 3, 2), dtype=np.float64),
            current_actions=np.zeros((1, 2), dtype=np.float64),
            target_latents=np.zeros((1, 4), dtype=np.float32),
            last_frame_indices=np.asarray([3], dtype=np.int64),
            target_frame_indices=np.asarray([4], dtype=np.int64),
            episode_ids=np.asarray([10], dtype=np.int64),
            step_ids=np.asarray([3], dtype=np.int64),
        )
        autoencoder = ConvAutoencoder(latent_dim=4, base_channels=2)
        dynamics = LatentDynamicsMLP(latent_dim=4, hidden_size=8)
        for parameter in autoencoder.parameters():
            torch.nn.init.zeros_(parameter)
        for parameter in dynamics.parameters():
            torch.nn.init.zeros_(parameter)

        metrics = evaluate_latent_dynamics(
            dynamics,
            autoencoder,
            visual,
            arrays,
            latent_normalizer=Normalizer(np.zeros(4), np.ones(4)),
            action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
            batch_size=1,
        )

        self.assertEqual(metrics["windows"], 1)
        self.assertEqual(metrics["changed_pixel_count"], 64 * 64)
        self.assertAlmostEqual(metrics["normalized_latent_mse"], 0.0)
        self.assertAlmostEqual(metrics["pixel_mse"], 0.25)
        self.assertAlmostEqual(metrics["pixel_mae"], 0.5)
        self.assertAlmostEqual(metrics["changed_pixel_mae"], 0.5)
        self.assertAlmostEqual(
            metrics["oracle_reconstruction_pixel_mse"],
            0.25,
        )
        self.assertAlmostEqual(
            metrics["oracle_reconstruction_pixel_mae"],
            0.5,
        )
        self.assertAlmostEqual(
            metrics["oracle_reconstruction_changed_pixel_mae"],
            0.5,
        )
        self.assertAlmostEqual(metrics["copy_last_pixel_mse"], 1.0)
        self.assertAlmostEqual(metrics["copy_last_pixel_mae"], 1.0)
        self.assertAlmostEqual(metrics["copy_last_changed_pixel_mae"], 1.0)

    def test_oracle_reconstruction_decodes_the_target_latent(self):
        class ScalarDecoder(torch.nn.Module):
            def decode(self, latents: torch.Tensor) -> torch.Tensor:
                return latents[:, :1, None, None].expand(
                    -1,
                    3,
                    64,
                    64,
                )

        visual = make_tiny_visual_dataset()
        visual["frames"][3].fill(0)
        visual["frames"][4].fill(255)
        arrays = LatentWindowArrays(
            context_latents=np.zeros((1, 4, 1), dtype=np.float32),
            history_actions=np.zeros((1, 3, 2), dtype=np.float64),
            current_actions=np.zeros((1, 2), dtype=np.float64),
            target_latents=np.ones((1, 1), dtype=np.float32),
            last_frame_indices=np.asarray([3], dtype=np.int64),
            target_frame_indices=np.asarray([4], dtype=np.int64),
            episode_ids=np.asarray([10], dtype=np.int64),
            step_ids=np.asarray([3], dtype=np.int64),
        )
        dynamics = LatentDynamicsMLP(latent_dim=1, hidden_size=4)
        for parameter in dynamics.parameters():
            torch.nn.init.zeros_(parameter)

        metrics = evaluate_latent_dynamics(
            dynamics,
            ScalarDecoder(),
            visual,
            arrays,
            latent_normalizer=Normalizer(np.zeros(1), np.ones(1)),
            action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
            batch_size=1,
        )

        self.assertAlmostEqual(metrics["pixel_mse"], 1.0)
        self.assertAlmostEqual(
            metrics["oracle_reconstruction_pixel_mse"],
            0.0,
        )


class VisualLatentCheckpointTest(unittest.TestCase):
    def _make_phase_results(
        self,
    ) -> tuple[PhaseTrainingResult, PhaseTrainingResult]:
        autoencoder = ConvAutoencoder(latent_dim=4, base_channels=2)
        dynamics = LatentDynamicsMLP(latent_dim=4, hidden_size=8)
        return (
            PhaseTrainingResult(
                model=autoencoder,
                train_losses=[0.4, 0.2],
                validation_losses=[0.5, 0.25],
                best_epoch=2,
            ),
            PhaseTrainingResult(
                model=dynamics,
                train_losses=[0.3, 0.1],
                validation_losses=[0.35, 0.15],
                best_epoch=2,
            ),
        )

    def test_checkpoint_round_trip_preserves_models_and_metadata(self):
        autoencoder_result, dynamics_result = self._make_phase_results()
        split_ids = {
            "train": np.arange(8, dtype=np.int64),
            "validation": np.asarray([8], dtype=np.int64),
            "test": np.asarray([9], dtype=np.int64),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual-latent.pt"
            save_visual_latent_checkpoint(
                path,
                autoencoder_result=autoencoder_result,
                dynamics_result=dynamics_result,
                latent_normalizer=Normalizer(np.zeros(4), np.ones(4)),
                action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
                split_episode_ids=split_ids,
                training_config={"seed": 3, "split_seed": 19},
                dataset_metadata={
                    "path": "/tmp/visual.npz",
                    "sha256": "a" * 64,
                    "schema_version": 1,
                    "renderer_version": "pillow-raster-v1",
                },
                autoencoder_test_metrics={
                    "frames": 5,
                    "pixel_mse": 0.1,
                    "pixel_mae": 0.2,
                    "psnr_db": 10.0,
                },
                dynamics_test_metrics={
                    "windows": 1,
                    "normalized_latent_mse": 0.2,
                    "pixel_mse": 0.3,
                    "pixel_mae": 0.4,
                    "psnr_db": 5.0,
                    "changed_pixel_mae": 0.5,
                    "changed_pixel_count": 10,
                    "copy_last_pixel_mse": 0.6,
                    "copy_last_pixel_mae": 0.7,
                    "copy_last_changed_pixel_mae": 0.8,
                },
            )
            loaded = load_visual_latent_checkpoint(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(payload["kind"], "visual_latent_world_model")
        self.assertEqual(loaded.autoencoder.latent_dim, 4)
        self.assertEqual(loaded.autoencoder.base_channels, 2)
        self.assertEqual(loaded.dynamics.hidden_size, 8)
        self.assertEqual(loaded.training_config["split_seed"], 19)
        self.assertEqual(loaded.dataset_metadata["sha256"], "a" * 64)
        self.assertEqual(loaded.autoencoder_history["best_epoch"], 2)
        self.assertEqual(loaded.dynamics_history["best_epoch"], 2)
        for name, expected in split_ids.items():
            np.testing.assert_array_equal(
                loaded.split_episode_ids[name],
                expected,
            )
        context = torch.zeros((2, 4, 4))
        history = torch.zeros((2, 3, 2))
        current = torch.zeros((2, 2))
        with torch.no_grad():
            expected = dynamics_result.model(context, history, current)
            actual = loaded.dynamics(context, history, current)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_checkpoint_refuses_to_overwrite(self):
        autoencoder_result, dynamics_result = self._make_phase_results()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual-latent.pt"
            path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                save_visual_latent_checkpoint(
                    path,
                    autoencoder_result=autoencoder_result,
                    dynamics_result=dynamics_result,
                    latent_normalizer=Normalizer(np.zeros(4), np.ones(4)),
                    action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
                    split_episode_ids={
                        "train": np.arange(8),
                        "validation": np.asarray([8]),
                        "test": np.asarray([9]),
                    },
                    training_config={},
                    dataset_metadata={
                        "path": "/tmp/visual.npz",
                        "sha256": "a" * 64,
                        "schema_version": 1,
                        "renderer_version": "pillow-raster-v1",
                    },
                    autoencoder_test_metrics={"frames": 1},
                    dynamics_test_metrics={"windows": 1},
                )
            self.assertEqual(path.read_bytes(), b"existing")

    def test_loader_rejects_wrong_format_kind_and_normalizer(self):
        autoencoder_result, dynamics_result = self._make_phase_results()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.pt"
            save_visual_latent_checkpoint(
                valid,
                autoencoder_result=autoencoder_result,
                dynamics_result=dynamics_result,
                latent_normalizer=Normalizer(np.zeros(4), np.ones(4)),
                action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
                split_episode_ids={
                    "train": np.arange(8),
                    "validation": np.asarray([8]),
                    "test": np.asarray([9]),
                },
                training_config={},
                dataset_metadata={
                    "path": "/tmp/visual.npz",
                    "sha256": "a" * 64,
                    "schema_version": 1,
                    "renderer_version": "pillow-raster-v1",
                },
                autoencoder_test_metrics={"frames": 1},
                dynamics_test_metrics={"windows": 1},
            )
            payload = torch.load(valid, map_location="cpu", weights_only=True)
            cases = (
                ("format", {"format_version": 2}, "format"),
                ("kind", {"kind": "other"}, "kind"),
                (
                    "latent std",
                    {"latent_std": torch.zeros(4)},
                    "latent_normalizer",
                ),
                (
                    "action shape",
                    {"action_mean": torch.zeros(3)},
                    "action_normalizer",
                ),
            )
            for name, update, message in cases:
                with self.subTest(name=name):
                    changed = dict(payload)
                    changed.update(update)
                    path = root / f"{name}.pt"
                    torch.save(changed, path)
                    with self.assertRaisesRegex(ValueError, message):
                        load_visual_latent_checkpoint(path)


class VisualLatentEndToEndTest(unittest.TestCase):
    def test_tiny_end_to_end_training_writes_checkpoint_and_preview(self):
        visual = make_tiny_visual_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            checkpoint_path = root / "visual-latent.pt"
            preview_path = root / "preview.png"
            save_visual_dataset(visual, data_path)

            summary = run_visual_latent_training(
                data_path=data_path,
                output_path=checkpoint_path,
                preview_path=preview_path,
                latent_dim=4,
                base_channels=2,
                dynamics_hidden_size=8,
                autoencoder_epochs=1,
                dynamics_epochs=1,
                autoencoder_batch_size=8,
                dynamics_batch_size=8,
                autoencoder_learning_rate=1e-3,
                dynamics_learning_rate=1e-3,
                motion_loss_weight=3.0,
                seed=3,
                split_seed=19,
            )
            loaded = load_visual_latent_checkpoint(checkpoint_path)

            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue(preview_path.is_file())
            self.assertGreater(preview_path.stat().st_size, 0)
            self.assertEqual(
                summary["split_episodes"],
                {"train": 8, "validation": 1, "test": 1},
            )
            self.assertEqual(
                sum(summary["split_windows"].values()),
                10,
            )
            self.assertIn(
                "copy_last_pixel_mse",
                summary["dynamics"]["test"],
            )
            self.assertEqual(loaded.training_config["split_seed"], 19)
            self.assertEqual(
                loaded.training_config["motion_loss_weight"],
                3.0,
            )

    def test_output_collisions_fail_before_training(self):
        visual = make_tiny_visual_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            save_visual_dataset(visual, data_path)
            cases = (
                (root / "existing.pt", root / "preview.png"),
                (root / "checkpoint.pt", root / "existing.png"),
            )
            for output, preview in cases:
                with self.subTest(output=output, preview=preview):
                    if "existing" in output.name:
                        output.write_bytes(b"existing")
                    if "existing" in preview.name:
                        preview.write_bytes(b"existing")
                    with patch(
                        "world_model_lab.train_visual_latent_model."
                        "train_autoencoder"
                    ) as train:
                        with self.assertRaises(FileExistsError):
                            run_visual_latent_training(
                                data_path=data_path,
                                output_path=output,
                                preview_path=preview,
                                latent_dim=4,
                                base_channels=2,
                                dynamics_hidden_size=8,
                                autoencoder_epochs=1,
                                dynamics_epochs=1,
                                autoencoder_batch_size=8,
                                dynamics_batch_size=8,
                                seed=3,
                                split_seed=19,
                            )
                        train.assert_not_called()

    def test_cli_help_and_pyproject_expose_visual_training_command(self):
        standard_output = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["world-model-train-visual-latent", "--help"],
        ):
            with redirect_stdout(standard_output):
                with self.assertRaises(SystemExit) as context:
                    main()

        self.assertEqual(context.exception.code, 0)
        help_text = standard_output.getvalue()
        for option in (
            "--autoencoder-epochs",
            "--dynamics-epochs",
            "--latent-dim",
            "--motion-loss-weight",
            "--split-seed",
        ):
            self.assertIn(option, help_text)
        pyproject = (
            Path(__file__).resolve().parents[1] / "pyproject.toml"
        ).read_text(encoding="utf-8")
        self.assertIn("world-model-train-visual-latent", pyproject)

    def test_readme_describes_visual_latent_scope_and_baseline(self):
        readme = (
            Path(__file__).resolve().parents[1] / "README.md"
        ).read_text(encoding="utf-8")
        for phrase in (
            "ConvAutoencoder",
            "LatentDynamicsMLP",
            "copy-last",
            "oracle reconstruction",
            "不读取 `states`",
            "暂不接入 MPC",
        ):
            self.assertIn(phrase, readme)

    def test_cli_prints_strict_sorted_json(self):
        expected = {"z": 1, "a": {"value": 2}}
        standard_output = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["world-model-train-visual-latent"],
        ):
            with patch(
                "world_model_lab.train_visual_latent_model."
                "run_visual_latent_training",
                return_value=expected,
            ):
                with redirect_stdout(standard_output):
                    main()

        self.assertEqual(json.loads(standard_output.getvalue()), expected)
        self.assertLess(
            standard_output.getvalue().find('"a"'),
            standard_output.getvalue().find('"z"'),
        )


if __name__ == "__main__":
    unittest.main()
