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
    _changed_pixel_mae_loss,
    _motion_weighted_mse,
    _object_balanced_mse,
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
    SpatialConvAutoencoder,
    SpatialLatentDynamicsCNN,
    SpatialLatentDynamicsConvGRU,
)
from world_model_lab.visual_observation import CAR_COLOR, HEADING_COLOR


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

    def test_object_balanced_mse_normalizes_object_and_background_separately(
        self,
    ):
        target = torch.zeros((1, 3, 1, 2), dtype=torch.float32)
        prediction = torch.tensor(
            [[[[1.0, 0.5]], [[1.0, 0.5]], [[1.0, 0.5]]]],
            dtype=torch.float32,
        )
        mask = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)

        plain = _object_balanced_mse(
            prediction,
            target,
            mask,
            object_loss_weight=0.0,
        )
        balanced = _object_balanced_mse(
            prediction,
            target,
            mask,
            object_loss_weight=1.0,
        )

        torch.testing.assert_close(
            plain,
            torch.mean(torch.square(prediction - target)),
        )
        torch.testing.assert_close(balanced, torch.tensor(1.25))

    def test_object_balanced_mse_rejects_invalid_inputs(self):
        images = torch.zeros((1, 3, 2, 2))
        masks = torch.zeros((1, 1, 2, 2))
        cases = (
            (images[:, :2], images, masks, 1.0),
            (images, images, masks[:, :, :1], 1.0),
            (images, images, torch.full_like(masks, 0.5), 1.0),
            (images, images, masks, -1.0),
            (images, images, masks, float("nan")),
            (images, images, masks, float("inf")),
        )
        for prediction, target, mask, weight in cases:
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    _object_balanced_mse(
                        prediction,
                        target,
                        mask,
                        object_loss_weight=weight,
                    )
        with self.assertRaisesRegex(ValueError, "object and background"):
            _object_balanced_mse(
                images,
                images,
                masks,
                object_loss_weight=1.0,
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
        for name in (
            "object_pixels",
            "background_pixels",
            "object_pixel_mse",
            "object_pixel_mae",
            "background_pixel_mse",
            "background_pixel_mae",
        ):
            self.assertIn(name, metrics)

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
            {"object_loss_weight": -1.0},
            {"object_loss_weight": float("nan")},
            {"object_loss_weight": float("inf")},
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
        with self.assertRaisesRegex(ValueError, "cannot both be positive"):
            train_autoencoder(
                visual,
                split_episode_ids=splits,
                latent_dim=4,
                base_channels=2,
                epochs=1,
                batch_size=8,
                learning_rate=1e-3,
                motion_loss_weight=1.0,
                object_loss_weight=1.0,
                seed=3,
            )


class VisualLatentDynamicsTrainingTest(unittest.TestCase):
    def test_changed_pixel_mae_loss_uses_only_masked_rgb_values(self):
        target = torch.zeros((1, 3, 1, 2), dtype=torch.float32)
        prediction = torch.tensor(
            [[[[1.0, 0.5]], [[1.0, 0.5]], [[1.0, 0.5]]]],
            dtype=torch.float32,
        )
        mask = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)

        loss = _changed_pixel_mae_loss(prediction, target, mask)

        torch.testing.assert_close(loss, torch.tensor(1.0))

    def test_changed_pixel_mae_loss_returns_differentiable_zero_for_empty_mask(
        self,
    ):
        prediction = torch.ones(
            (1, 3, 2, 2),
            dtype=torch.float32,
            requires_grad=True,
        )

        loss = _changed_pixel_mae_loss(
            prediction,
            torch.zeros_like(prediction),
            torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        )
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        torch.testing.assert_close(
            prediction.grad,
            torch.zeros_like(prediction),
        )

    def test_changed_pixel_mae_loss_rejects_invalid_inputs(self):
        prediction = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        target = torch.zeros_like(prediction)
        mask = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        cases = (
            (prediction[:, :2], target, mask),
            (prediction, target, mask[:, :, :1]),
            (prediction, target, torch.full_like(mask, 0.5)),
        )
        for candidate, candidate_target, candidate_mask in cases:
            with self.subTest(
                prediction_shape=tuple(candidate.shape),
                mask_shape=tuple(candidate_mask.shape),
            ):
                with self.assertRaises(ValueError):
                    _changed_pixel_mae_loss(
                        candidate,
                        candidate_target,
                        candidate_mask,
                    )

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

    def test_convgru_dynamics_training_uses_requested_architecture(self):
        arrays = make_latent_arrays(count=12, latent_dim=2 * 8 * 8)
        train = LatentWindowArrays(
            **{
                name: np.asarray(getattr(arrays, name))[:8]
                for name in arrays.__dataclass_fields__
            }
        )
        validation = LatentWindowArrays(
            **{
                name: np.asarray(getattr(arrays, name))[8:]
                for name in arrays.__dataclass_fields__
            }
        )

        result = train_latent_dynamics(
            train,
            validation,
            latent_normalizer=Normalizer(
                np.zeros(2 * 8 * 8),
                np.ones(2 * 8 * 8),
            ),
            action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
            latent_layout="spatial",
            spatial_latent_channels=2,
            spatial_dynamics_architecture="convgru",
            hidden_size=4,
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            seed=5,
        )

        self.assertIsInstance(result.model, SpatialLatentDynamicsConvGRU)
        self.assertFalse(result.model.training)

    def test_changed_pixel_objective_freezes_decoder_parameters(self):
        visual = make_tiny_visual_dataset()
        arrays = make_latent_arrays(count=12, latent_dim=2 * 8 * 8)
        arrays = LatentWindowArrays(
            **{
                name: (
                    np.arange(1, 13, dtype=np.int64)
                    if name == "target_frame_indices"
                    else np.asarray(getattr(arrays, name))
                )
                for name in arrays.__dataclass_fields__
            }
        )
        train = LatentWindowArrays(
            **{
                name: np.asarray(getattr(arrays, name))[:8]
                for name in arrays.__dataclass_fields__
            }
        )
        validation = LatentWindowArrays(
            **{
                name: np.asarray(getattr(arrays, name))[8:]
                for name in arrays.__dataclass_fields__
            }
        )
        decoder = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
        )
        original_parameters = {
            name: value.detach().clone()
            for name, value in decoder.state_dict().items()
        }

        result = train_latent_dynamics(
            train,
            validation,
            latent_normalizer=Normalizer(
                np.zeros(2 * 8 * 8),
                np.ones(2 * 8 * 8),
            ),
            action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
            latent_layout="spatial",
            spatial_latent_channels=2,
            hidden_size=4,
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            seed=5,
            decoder=decoder,
            visual_dataset=visual,
            changed_pixel_loss_weight=0.1,
        )

        self.assertIsInstance(result.model, SpatialLatentDynamicsCNN)
        self.assertFalse(decoder.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in decoder.parameters()
            )
        )
        for name, expected in original_parameters.items():
            torch.testing.assert_close(
                decoder.state_dict()[name],
                expected,
                rtol=0.0,
                atol=0.0,
            )

    def test_changed_pixel_objective_rejects_invalid_configuration(self):
        arrays = make_latent_arrays(count=4)
        kwargs = {
            "latent_normalizer": Normalizer(np.zeros(3), np.ones(3)),
            "action_normalizer": Normalizer(np.zeros(2), np.ones(2)),
            "hidden_size": 4,
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "seed": 5,
        }
        for weight in (-1.0, float("nan"), float("inf")):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    train_latent_dynamics(
                        arrays,
                        arrays,
                        changed_pixel_loss_weight=weight,
                        **kwargs,
                    )
        with self.assertRaisesRegex(ValueError, "decoder and visual_dataset"):
            train_latent_dynamics(
                arrays,
                arrays,
                changed_pixel_loss_weight=0.1,
                **kwargs,
            )

    def test_zero_changed_pixel_weight_preserves_legacy_training_exactly(self):
        arrays = make_latent_arrays(count=12)
        train = LatentWindowArrays(
            **{
                name: np.asarray(getattr(arrays, name))[:8]
                for name in arrays.__dataclass_fields__
            }
        )
        validation = LatentWindowArrays(
            **{
                name: np.asarray(getattr(arrays, name))[8:]
                for name in arrays.__dataclass_fields__
            }
        )
        kwargs = {
            "latent_normalizer": Normalizer(np.zeros(3), np.ones(3)),
            "action_normalizer": Normalizer(np.zeros(2), np.ones(2)),
            "hidden_size": 4,
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "seed": 5,
        }

        legacy = train_latent_dynamics(
            train,
            validation,
            **kwargs,
        )
        explicit_zero = train_latent_dynamics(
            train,
            validation,
            decoder=ConvAutoencoder(latent_dim=3, base_channels=2),
            visual_dataset=make_tiny_visual_dataset(),
            changed_pixel_loss_weight=0.0,
            **kwargs,
        )

        self.assertEqual(legacy.train_losses, explicit_zero.train_losses)
        self.assertEqual(
            legacy.validation_losses,
            explicit_zero.validation_losses,
        )
        self.assertEqual(legacy.best_epoch, explicit_zero.best_epoch)
        for name, expected in legacy.model.state_dict().items():
            torch.testing.assert_close(
                explicit_zero.model.state_dict()[name],
                expected,
                rtol=0.0,
                atol=0.0,
            )

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
        self.assertAlmostEqual(
            metrics["decoded_last_latent_pixel_mse"],
            0.25,
        )
        self.assertAlmostEqual(
            metrics["decoded_last_latent_pixel_mae"],
            0.5,
        )
        self.assertAlmostEqual(
            metrics["decoded_last_latent_changed_pixel_mae"],
            0.5,
        )

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

    def test_action_ablations_are_deterministic_and_change_sensitive_model(self):
        class CurrentActionDynamics(torch.nn.Module):
            def forward(
                self,
                context: torch.Tensor,
                history: torch.Tensor,
                current: torch.Tensor,
            ) -> torch.Tensor:
                del history
                return context[:, -1] + current[:, :1]

        class ScalarDecoder(torch.nn.Module):
            def decode(self, latents: torch.Tensor) -> torch.Tensor:
                return latents[:, :1, None, None].expand(
                    -1,
                    3,
                    64,
                    64,
                )

        visual = make_tiny_visual_dataset()
        current_actions = np.asarray(
            [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]],
            dtype=np.float64,
        )
        arrays = LatentWindowArrays(
            context_latents=np.zeros((4, 4, 1), dtype=np.float32),
            history_actions=np.zeros((4, 3, 2), dtype=np.float64),
            current_actions=current_actions,
            target_latents=current_actions[:, :1].astype(np.float32),
            last_frame_indices=np.asarray([3, 8, 13, 18], dtype=np.int64),
            target_frame_indices=np.asarray([4, 9, 14, 19], dtype=np.int64),
            episode_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
            step_ids=np.full(4, 3, dtype=np.int64),
        )
        kwargs = {
            "latent_normalizer": Normalizer(np.zeros(1), np.ones(1)),
            "action_normalizer": Normalizer(np.zeros(2), np.ones(2)),
            "batch_size": 2,
            "action_shuffle_seed": 7,
        }

        first = evaluate_latent_dynamics(
            CurrentActionDynamics(),
            ScalarDecoder(),
            visual,
            arrays,
            **kwargs,
        )
        second = evaluate_latent_dynamics(
            CurrentActionDynamics(),
            ScalarDecoder(),
            visual,
            arrays,
            **kwargs,
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["normalized_latent_mse"], 0.0)
        self.assertGreater(
            first["mean_action_ablation_normalized_latent_mse"],
            0.0,
        )
        self.assertGreater(
            first["shuffled_action_ablation_normalized_latent_mse"],
            0.0,
        )
        for prefix in ("mean_action_ablation", "shuffled_action_ablation"):
            for suffix in (
                "normalized_latent_mse",
                "pixel_mse",
                "changed_pixel_mae",
            ):
                self.assertTrue(math.isfinite(float(first[f"{prefix}_{suffix}"])))

        with self.assertRaises(ValueError):
            evaluate_latent_dynamics(
                CurrentActionDynamics(),
                ScalarDecoder(),
                visual,
                arrays,
                latent_normalizer=kwargs["latent_normalizer"],
                action_normalizer=kwargs["action_normalizer"],
                batch_size=2,
                action_shuffle_seed=-1,
            )

    def test_context_ablations_expose_history_sensitive_dynamics(self):
        class HistoryDynamics(torch.nn.Module):
            def forward(
                self,
                context: torch.Tensor,
                history: torch.Tensor,
                current: torch.Tensor,
            ) -> torch.Tensor:
                del history, current
                return context[:, -1] + context[:, -2] - context[:, -3]

        class ScalarDecoder(torch.nn.Module):
            def decode(self, latents: torch.Tensor) -> torch.Tensor:
                return latents[:, :1, None, None].expand(
                    -1,
                    3,
                    64,
                    64,
                )

        visual = make_tiny_visual_dataset()
        context = np.tile(
            np.asarray([0.1, 0.2, 0.4, 0.7], dtype=np.float32)[
                None,
                :,
                None,
            ],
            (4, 1, 1),
        )
        arrays = LatentWindowArrays(
            context_latents=context,
            history_actions=np.zeros((4, 3, 2), dtype=np.float64),
            current_actions=np.zeros((4, 2), dtype=np.float64),
            target_latents=np.full((4, 1), 0.9, dtype=np.float32),
            last_frame_indices=np.asarray([3, 8, 13, 18], dtype=np.int64),
            target_frame_indices=np.asarray([4, 9, 14, 19], dtype=np.int64),
            episode_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
            step_ids=np.full(4, 3, dtype=np.int64),
        )

        metrics = evaluate_latent_dynamics(
            HistoryDynamics(),
            ScalarDecoder(),
            visual,
            arrays,
            latent_normalizer=Normalizer(np.zeros(1), np.ones(1)),
            action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
            batch_size=2,
        )

        self.assertAlmostEqual(metrics["normalized_latent_mse"], 0.0)
        self.assertAlmostEqual(
            metrics["repeat_last_context_normalized_latent_mse"],
            0.04,
        )
        self.assertAlmostEqual(
            metrics["reverse_history_context_normalized_latent_mse"],
            0.09,
        )
        for prefix in ("repeat_last_context", "reverse_history_context"):
            for suffix in (
                "normalized_latent_mse",
                "pixel_mse",
                "changed_pixel_mae",
            ):
                self.assertTrue(
                    math.isfinite(float(metrics[f"{prefix}_{suffix}"]))
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

    def test_spatial_checkpoint_round_trip_preserves_grid_model(self):
        autoencoder = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
        )
        dynamics = SpatialLatentDynamicsCNN(
            latent_channels=2,
            hidden_channels=4,
        )
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
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spatial.pt"
            save_visual_latent_checkpoint(
                path,
                autoencoder_result=autoencoder_result,
                dynamics_result=dynamics_result,
                latent_normalizer=Normalizer(
                    np.zeros(autoencoder.latent_dim),
                    np.ones(autoencoder.latent_dim),
                ),
                action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
                split_episode_ids={
                    "train": np.arange(8, dtype=np.int64),
                    "validation": np.asarray([8], dtype=np.int64),
                    "test": np.asarray([9], dtype=np.int64),
                },
                training_config={"latent_layout": "spatial"},
                dataset_metadata={
                    "path": "/tmp/visual.npz",
                    "sha256": "a" * 64,
                    "schema_version": 1,
                    "renderer_version": "pillow-raster-v1",
                },
                autoencoder_test_metrics={"frames": 1},
                dynamics_test_metrics={"windows": 1},
            )
            loaded = load_visual_latent_checkpoint(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            legacy_payload = dict(payload)
            legacy_model_config = dict(legacy_payload["model_config"])
            legacy_model_config.pop("spatial_dynamics_architecture")
            legacy_model_config.pop("object_residual_decoder", None)
            legacy_model_config.pop("object_head_channels", None)
            legacy_model_config.pop("object_slot_decoder", None)
            legacy_model_config.pop("object_slot_patch_size", None)
            legacy_model_config.pop("object_slot_hidden_size", None)
            legacy_payload["model_config"] = legacy_model_config
            legacy_path = Path(directory) / "legacy-spatial.pt"
            torch.save(legacy_payload, legacy_path)
            legacy_loaded = load_visual_latent_checkpoint(legacy_path)

        self.assertIsInstance(loaded.autoencoder, SpatialConvAutoencoder)
        self.assertIsInstance(loaded.dynamics, SpatialLatentDynamicsCNN)
        self.assertIsInstance(
            legacy_loaded.dynamics,
            SpatialLatentDynamicsCNN,
        )
        self.assertEqual(payload["model_config"]["latent_layout"], "spatial")
        self.assertEqual(
            payload["model_config"]["spatial_dynamics_architecture"],
            "cnn",
        )
        self.assertFalse(
            payload["model_config"]["object_residual_decoder"]
        )
        self.assertEqual(payload["model_config"]["object_head_channels"], 0)
        self.assertFalse(legacy_loaded.autoencoder.object_residual_decoder)
        self.assertFalse(legacy_loaded.autoencoder.object_slot_decoder)
        self.assertEqual(loaded.autoencoder.latent_channels, 2)
        context = torch.zeros((2, 4, autoencoder.latent_dim))
        history = torch.zeros((2, 3, 2))
        current = torch.zeros((2, 2))
        with torch.no_grad():
            expected = dynamics(context, history, current)
            actual = loaded.dynamics(context, history, current)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_object_residual_checkpoint_round_trip_preserves_predictions(
        self,
    ):
        autoencoder = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
            object_residual_decoder=True,
            object_head_channels=3,
        )
        dynamics = SpatialLatentDynamicsCNN(
            latent_channels=2,
            hidden_channels=4,
        )
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
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object-residual.pt"
            save_visual_latent_checkpoint(
                path,
                autoencoder_result=autoencoder_result,
                dynamics_result=dynamics_result,
                latent_normalizer=Normalizer(
                    np.zeros(autoencoder.latent_dim),
                    np.ones(autoencoder.latent_dim),
                ),
                action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
                split_episode_ids={
                    "train": np.arange(8, dtype=np.int64),
                    "validation": np.asarray([8], dtype=np.int64),
                    "test": np.asarray([9], dtype=np.int64),
                },
                training_config={"latent_layout": "spatial"},
                dataset_metadata={
                    "path": "/tmp/visual.npz",
                    "sha256": "a" * 64,
                    "schema_version": 1,
                    "renderer_version": "pillow-raster-v1",
                },
                autoencoder_test_metrics={"frames": 1},
                dynamics_test_metrics={"windows": 1},
            )
            loaded = load_visual_latent_checkpoint(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)

        self.assertTrue(loaded.autoencoder.object_residual_decoder)
        self.assertEqual(loaded.autoencoder.object_head_channels, 3)
        self.assertTrue(payload["model_config"]["object_residual_decoder"])
        self.assertEqual(payload["model_config"]["object_head_channels"], 3)
        images = torch.randn((2, 3, 64, 64))
        with torch.no_grad():
            expected = autoencoder(images)
            actual = loaded.autoencoder(images)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        for name, expected_tensor in autoencoder.state_dict().items():
            torch.testing.assert_close(
                loaded.autoencoder.state_dict()[name],
                expected_tensor,
                rtol=0.0,
                atol=0.0,
            )

    def test_object_slot_checkpoint_round_trip_preserves_predictions(self):
        autoencoder = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
            object_slot_decoder=True,
            object_slot_patch_size=11,
            object_slot_hidden_size=8,
        )
        dynamics = SpatialLatentDynamicsCNN(
            latent_channels=2,
            hidden_channels=4,
        )
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
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object-slot.pt"
            save_visual_latent_checkpoint(
                path,
                autoencoder_result=autoencoder_result,
                dynamics_result=dynamics_result,
                latent_normalizer=Normalizer(
                    np.zeros(autoencoder.latent_dim),
                    np.ones(autoencoder.latent_dim),
                ),
                action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
                split_episode_ids={
                    "train": np.arange(8, dtype=np.int64),
                    "validation": np.asarray([8], dtype=np.int64),
                    "test": np.asarray([9], dtype=np.int64),
                },
                training_config={"latent_layout": "spatial"},
                dataset_metadata={
                    "path": "/tmp/visual.npz",
                    "sha256": "a" * 64,
                    "schema_version": 1,
                    "renderer_version": "pillow-raster-v1",
                },
                autoencoder_test_metrics={"frames": 1},
                dynamics_test_metrics={"windows": 1},
            )
            loaded = load_visual_latent_checkpoint(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)

        self.assertTrue(loaded.autoencoder.object_slot_decoder)
        self.assertEqual(loaded.autoencoder.object_slot_patch_size, 11)
        self.assertEqual(loaded.autoencoder.object_slot_hidden_size, 8)
        self.assertTrue(payload["model_config"]["object_slot_decoder"])
        self.assertEqual(payload["model_config"]["object_slot_patch_size"], 11)
        self.assertEqual(payload["model_config"]["object_slot_hidden_size"], 8)
        images = torch.randn((2, 3, 64, 64))
        with torch.no_grad():
            expected = autoencoder(images)
            actual = loaded.autoencoder(images)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        for name, expected_tensor in autoencoder.state_dict().items():
            torch.testing.assert_close(
                loaded.autoencoder.state_dict()[name],
                expected_tensor,
                rtol=0.0,
                atol=0.0,
            )

    def test_convgru_checkpoint_round_trip_preserves_predictions(self):
        autoencoder = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
        )
        dynamics = SpatialLatentDynamicsConvGRU(
            latent_channels=2,
            hidden_channels=4,
        )
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
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spatial-convgru.pt"
            save_visual_latent_checkpoint(
                path,
                autoencoder_result=autoencoder_result,
                dynamics_result=dynamics_result,
                latent_normalizer=Normalizer(
                    np.zeros(autoencoder.latent_dim),
                    np.ones(autoencoder.latent_dim),
                ),
                action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
                split_episode_ids={
                    "train": np.arange(8, dtype=np.int64),
                    "validation": np.asarray([8], dtype=np.int64),
                    "test": np.asarray([9], dtype=np.int64),
                },
                training_config={"latent_layout": "spatial"},
                dataset_metadata={
                    "path": "/tmp/visual.npz",
                    "sha256": "a" * 64,
                    "schema_version": 1,
                    "renderer_version": "pillow-raster-v1",
                },
                autoencoder_test_metrics={"frames": 1},
                dynamics_test_metrics={"windows": 1},
            )
            loaded = load_visual_latent_checkpoint(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)

        self.assertIsInstance(
            loaded.dynamics,
            SpatialLatentDynamicsConvGRU,
        )
        self.assertEqual(
            payload["model_config"]["spatial_dynamics_architecture"],
            "convgru",
        )
        context = torch.randn((2, 4, autoencoder.latent_dim))
        history = torch.randn((2, 3, 2))
        current = torch.randn((2, 2))
        with torch.no_grad():
            expected = dynamics(context, history, current)
            actual = loaded.dynamics(context, history, current)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_checkpoint_writer_rejects_an_unloadable_model_pair(self):
        autoencoder_result, _ = self._make_phase_results()
        incompatible_dynamics_result = PhaseTrainingResult(
            model=LatentDynamicsMLP(latent_dim=5, hidden_size=8),
            train_losses=[0.3],
            validation_losses=[0.35],
            best_epoch=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unloadable.pt"

            with self.assertRaisesRegex(ValueError, "unloadable"):
                save_visual_latent_checkpoint(
                    path,
                    autoencoder_result=autoencoder_result,
                    dynamics_result=incompatible_dynamics_result,
                    latent_normalizer=Normalizer(
                        np.zeros(4),
                        np.ones(4),
                    ),
                    action_normalizer=Normalizer(
                        np.zeros(2),
                        np.ones(2),
                    ),
                    split_episode_ids={
                        "train": np.arange(8, dtype=np.int64),
                        "validation": np.asarray([8], dtype=np.int64),
                        "test": np.asarray([9], dtype=np.int64),
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

            self.assertFalse(path.exists())

    def test_checkpoint_validation_does_not_advance_torch_rng(self):
        autoencoder_result, dynamics_result = self._make_phase_results()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual-latent.pt"
            torch.manual_seed(17)
            state_before = torch.random.get_rng_state().clone()

            save_visual_latent_checkpoint(
                path,
                autoencoder_result=autoencoder_result,
                dynamics_result=dynamics_result,
                latent_normalizer=Normalizer(np.zeros(4), np.ones(4)),
                action_normalizer=Normalizer(np.zeros(2), np.ones(2)),
                split_episode_ids={
                    "train": np.arange(8, dtype=np.int64),
                    "validation": np.asarray([8], dtype=np.int64),
                    "test": np.asarray([9], dtype=np.int64),
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

            torch.testing.assert_close(
                torch.random.get_rng_state(),
                state_before,
                rtol=0.0,
                atol=0.0,
            )

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

    def test_tiny_spatial_training_writes_reloadable_models(self):
        visual = make_tiny_visual_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            checkpoint_path = root / "spatial.pt"
            preview_path = root / "spatial.png"
            save_visual_dataset(visual, data_path)

            summary = run_visual_latent_training(
                data_path=data_path,
                output_path=checkpoint_path,
                preview_path=preview_path,
                latent_layout="spatial",
                spatial_latent_channels=2,
                base_channels=2,
                dynamics_hidden_size=4,
                autoencoder_epochs=1,
                dynamics_epochs=1,
                autoencoder_batch_size=8,
                dynamics_batch_size=8,
                seed=3,
                split_seed=19,
            )
            loaded = load_visual_latent_checkpoint(checkpoint_path)

        self.assertIsInstance(loaded.autoencoder, SpatialConvAutoencoder)
        self.assertIsInstance(loaded.dynamics, SpatialLatentDynamicsCNN)
        self.assertEqual(loaded.training_config["latent_layout"], "spatial")
        self.assertEqual(loaded.training_config["spatial_latent_channels"], 2)
        self.assertIn("oracle_reconstruction_pixel_mse", summary["dynamics"]["test"])

    def test_object_aligned_training_records_weight_and_region_metrics(self):
        visual = make_tiny_visual_dataset()
        visual["frames"][:, 2, 3] = CAR_COLOR
        visual["frames"][:, 4, 5] = HEADING_COLOR
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            checkpoint_path = root / "object.pt"
            preview_path = root / "object.png"
            save_visual_dataset(visual, data_path)

            summary = run_visual_latent_training(
                data_path=data_path,
                output_path=checkpoint_path,
                preview_path=preview_path,
                latent_layout="spatial",
                spatial_latent_channels=2,
                base_channels=2,
                dynamics_hidden_size=4,
                autoencoder_epochs=1,
                dynamics_epochs=1,
                autoencoder_batch_size=8,
                dynamics_batch_size=8,
                object_loss_weight=1.0,
                seed=3,
                split_seed=19,
            )
            loaded = load_visual_latent_checkpoint(checkpoint_path)

        self.assertEqual(loaded.training_config["object_loss_weight"], 1.0)
        self.assertEqual(summary["autoencoder"]["test"]["object_pixels"], 10)
        self.assertGreater(
            summary["autoencoder"]["test"]["background_pixels"],
            0,
        )

    def test_tiny_spatial_convgru_training_writes_reloadable_models(self):
        visual = make_tiny_visual_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            checkpoint_path = root / "spatial-convgru.pt"
            preview_path = root / "spatial-convgru.png"
            save_visual_dataset(visual, data_path)

            run_visual_latent_training(
                data_path=data_path,
                output_path=checkpoint_path,
                preview_path=preview_path,
                latent_layout="spatial",
                spatial_latent_channels=2,
                spatial_dynamics_architecture="convgru",
                base_channels=2,
                dynamics_hidden_size=4,
                autoencoder_epochs=1,
                dynamics_epochs=1,
                autoencoder_batch_size=8,
                dynamics_batch_size=8,
                seed=3,
                split_seed=19,
            )
            loaded = load_visual_latent_checkpoint(checkpoint_path)

        self.assertIsInstance(
            loaded.dynamics,
            SpatialLatentDynamicsConvGRU,
        )
        self.assertEqual(
            loaded.training_config["spatial_dynamics_architecture"],
            "convgru",
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

    def test_output_paths_cannot_be_ancestors(self):
        visual = make_tiny_visual_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            save_visual_dataset(visual, data_path)
            cases = (
                (
                    root / "checkpoint-parent",
                    root / "checkpoint-parent" / "preview.png",
                ),
                (
                    root / "preview-parent" / "checkpoint.pt",
                    root / "preview-parent",
                ),
            )

            for output, preview in cases:
                with self.subTest(output=output, preview=preview):
                    with patch(
                        "world_model_lab.train_visual_latent_model."
                        "train_autoencoder"
                    ) as train:
                        with self.assertRaisesRegex(ValueError, "ancestors"):
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
                    self.assertFalse(output.exists())
                    self.assertFalse(preview.exists())

    def test_preview_failure_rolls_back_the_new_checkpoint(self):
        visual = make_tiny_visual_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            checkpoint_path = root / "visual-latent.pt"
            preview_path = root / "preview.png"
            save_visual_dataset(visual, data_path)

            with patch(
                "world_model_lab.train_visual_latent_model."
                "plot_visual_latent_predictions",
                side_effect=RuntimeError("preview failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "preview failed"):
                    run_visual_latent_training(
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
                        seed=3,
                        split_seed=19,
                    )

            self.assertFalse(checkpoint_path.exists())
            self.assertFalse(preview_path.exists())

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
            "--latent-layout",
            "--spatial-latent-channels",
            "--spatial-dynamics-architecture",
            "--motion-loss-weight",
            "--object-loss-weight",
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
            "SpatialConvAutoencoder",
            "SpatialLatentDynamicsCNN",
            "SpatialLatentDynamicsConvGRU",
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
