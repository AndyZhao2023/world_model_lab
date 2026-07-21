from __future__ import annotations

import unittest

import numpy as np
import torch

from tests.test_visual_windows import make_visual_dataset
from world_model_lab.dataset import Normalizer
from world_model_lab.visual_latent_data import (
    build_latent_rollout_arrays,
    build_latent_window_arrays,
)
from world_model_lab.visual_latent_model import SpatialConvAutoencoder
from world_model_lab.train_visual_latent_model import _dynamics_batch_loss
from world_model_lab.visual_object_position import (
    LinearObjectPositionProbe,
)
from world_model_lab.visual_recursive_training import (
    recursive_normalized_latents,
    recursive_rollout_objective,
    train_recursive_latent_dynamics,
)
from world_model_lab.visual_windows import build_visual_window_index


class CurrentActionDynamics(torch.nn.Module):
    def forward(
        self,
        context: torch.Tensor,
        history: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        del history
        return context[:, -1] + current[:, :1]


class ScaledActionDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        context: torch.Tensor,
        history: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        del history
        return context[:, -1] + self.scale * current[:, :1]


class LatentToPixelDecoder(torch.nn.Module):
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents[:, :1, None, None].expand(-1, 3, 2, 2)


class RecursiveLatentPredictionTest(unittest.TestCase):
    def setUp(self):
        self.context = torch.tensor(
            [[[-2.0], [-1.0], [0.0], [1.0]]],
            dtype=torch.float32,
        )
        self.history = torch.zeros((1, 3, 2), dtype=torch.float32)
        self.actions = torch.tensor(
            [[[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]],
            dtype=torch.float32,
        )

    def test_recursion_shifts_predicted_latents_and_actions(self):
        predictions = recursive_normalized_latents(
            CurrentActionDynamics(),
            context_latents=self.context,
            history_actions=self.history,
            rollout_actions=self.actions,
        )

        torch.testing.assert_close(
            predictions,
            torch.tensor([[[3.0], [6.0], [10.0]]]),
            rtol=0.0,
            atol=0.0,
        )

    def test_recursion_rejects_misaligned_tensors(self):
        with self.assertRaises(ValueError):
            recursive_normalized_latents(
                CurrentActionDynamics(),
                context_latents=self.context,
                history_actions=self.history,
                rollout_actions=self.actions[:, :, :1],
            )


class RecursiveRolloutObjectiveTest(unittest.TestCase):
    def setUp(self):
        self.context = torch.tensor(
            [[[-2.0], [-1.0], [0.0], [1.0]]],
            dtype=torch.float32,
        )
        self.history = torch.zeros((1, 3, 2), dtype=torch.float32)
        self.actions = torch.tensor(
            [[[2.0, 0.0], [3.0, 0.0]]],
            dtype=torch.float32,
        )
        self.identity_latent = Normalizer(np.zeros(1), np.ones(1))
        self.position_probe = LinearObjectPositionProbe(
            weight=np.asarray([[1.0], [1.0]], dtype=np.float32),
            bias=np.zeros(2, dtype=np.float32),
        )

    def test_one_step_position_term_adds_to_existing_objective(self):
        batch = (
            self.context,
            self.history,
            self.actions[:, 0],
            torch.tensor([[3.0]], dtype=torch.float32),
        )

        loss = _dynamics_batch_loss(
            CurrentActionDynamics(),
            batch,
            latent_normalizer=self.identity_latent,
            decoder=None,
            changed_pixel_loss_weight=0.0,
            position_probe=self.position_probe,
            target_positions=torch.tensor(
                [[2.5, 2.5]],
                dtype=torch.float32,
            ),
            object_position_loss_weight=2.0,
        )

        self.assertAlmostEqual(float(loss), 0.5)

    def test_zero_and_positive_latent_only_objectives_are_exact(self):
        matching_targets = torch.tensor([[[3.0], [6.0]]])

        zero = recursive_rollout_objective(
            CurrentActionDynamics(),
            context_latents=self.context,
            history_actions=self.history,
            rollout_actions=self.actions,
            target_latents=matching_targets,
            latent_normalizer=self.identity_latent,
            changed_pixel_loss_weight=0.0,
        )
        changed_targets = matching_targets.clone()
        changed_targets[:, 1] += 1.0
        positive = recursive_rollout_objective(
            CurrentActionDynamics(),
            context_latents=self.context,
            history_actions=self.history,
            rollout_actions=self.actions,
            target_latents=changed_targets,
            latent_normalizer=self.identity_latent,
            changed_pixel_loss_weight=0.0,
        )

        self.assertEqual(float(zero), 0.0)
        self.assertEqual(float(positive), 0.5)

    def test_image_term_uses_every_rollout_step(self):
        target_frames = torch.zeros(
            (1, 2, 2, 2, 3),
            dtype=torch.uint8,
        )
        changed_masks = torch.ones(
            (1, 2, 2, 2),
            dtype=torch.bool,
        )
        targets = torch.tensor([[[3.0], [6.0]]])

        loss = recursive_rollout_objective(
            CurrentActionDynamics(),
            context_latents=self.context,
            history_actions=self.history,
            rollout_actions=self.actions,
            target_latents=targets,
            latent_normalizer=self.identity_latent,
            decoder=LatentToPixelDecoder(),
            target_frames_uint8=target_frames,
            changed_masks=changed_masks,
            changed_pixel_loss_weight=0.1,
        )

        self.assertAlmostEqual(float(loss), 0.45)

    def test_position_term_uses_every_rollout_step(self):
        targets = torch.tensor([[[3.0], [6.0]]])
        target_positions = torch.tensor(
            [[[2.0, 2.0], [4.0, 4.0]]],
            dtype=torch.float32,
        )

        loss = recursive_rollout_objective(
            CurrentActionDynamics(),
            context_latents=self.context,
            history_actions=self.history,
            rollout_actions=self.actions,
            target_latents=targets,
            latent_normalizer=self.identity_latent,
            changed_pixel_loss_weight=0.0,
            position_probe=self.position_probe,
            target_positions=target_positions,
            object_position_loss_weight=0.5,
        )

        self.assertAlmostEqual(float(loss), 1.25)

    def test_gradient_flows_through_later_recursive_steps(self):
        model = ScaledActionDynamics()
        targets = torch.zeros((1, 2, 1), dtype=torch.float32)

        loss = recursive_rollout_objective(
            model,
            context_latents=self.context,
            history_actions=self.history,
            rollout_actions=self.actions,
            target_latents=targets,
            latent_normalizer=self.identity_latent,
            changed_pixel_loss_weight=0.0,
            position_probe=self.position_probe,
            target_positions=torch.zeros(
                (1, 2, 2),
                dtype=torch.float32,
            ),
            object_position_loss_weight=1.0,
        )
        loss.backward()

        self.assertIsNotNone(model.scale.grad)
        self.assertTrue(torch.isfinite(model.scale.grad))
        self.assertNotEqual(float(model.scale.grad), 0.0)
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in self.position_probe.parameters()
            )
        )

    def test_positive_image_weight_requires_complete_supervision(self):
        with self.assertRaises(ValueError):
            recursive_rollout_objective(
                CurrentActionDynamics(),
                context_latents=self.context,
                history_actions=self.history,
                rollout_actions=self.actions,
                target_latents=torch.zeros((1, 2, 1)),
                latent_normalizer=self.identity_latent,
                changed_pixel_loss_weight=0.1,
            )

    def test_positive_position_weight_requires_complete_supervision(self):
        with self.assertRaises(ValueError):
            recursive_rollout_objective(
                CurrentActionDynamics(),
                context_latents=self.context,
                history_actions=self.history,
                rollout_actions=self.actions,
                target_latents=torch.zeros((1, 2, 1)),
                latent_normalizer=self.identity_latent,
                changed_pixel_loss_weight=0.0,
                object_position_loss_weight=1.0,
            )


class RecursiveDynamicsTrainingTest(unittest.TestCase):
    def setUp(self):
        self.visual = make_visual_dataset((8,) * 4)
        self.autoencoder = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
        )
        rng = np.random.default_rng(7)
        self.latents = rng.normal(
            size=(
                self.visual["frames"].shape[0],
                self.autoencoder.latent_dim,
            )
        ).astype(np.float32)
        self.latent_normalizer = Normalizer(
            np.zeros(self.autoencoder.latent_dim),
            np.ones(self.autoencoder.latent_dim),
        )
        self.action_normalizer = Normalizer(np.zeros(2), np.ones(2))
        self.train_ids = np.asarray([10, 11], dtype=np.int64)
        self.validation_ids = np.asarray([12, 13], dtype=np.int64)

    def _arrays(self, ids: np.ndarray, *, horizon: int):
        one_step = build_latent_window_arrays(
            self.visual,
            build_visual_window_index(self.visual, ids),
            self.latents,
        )
        rollout = build_latent_rollout_arrays(
            self.visual,
            ids,
            self.latents,
            horizon=horizon,
        )
        return one_step, rollout

    def test_training_is_deterministic_and_records_component_histories(self):
        train_one_step, train_rollout = self._arrays(
            self.train_ids,
            horizon=5,
        )
        validation_one_step, validation_rollout = self._arrays(
            self.validation_ids,
            horizon=5,
        )
        results = [
            train_recursive_latent_dynamics(
                train_one_step,
                validation_one_step,
                train_rollout,
                validation_rollout,
                latent_normalizer=self.latent_normalizer,
                action_normalizer=self.action_normalizer,
                spatial_latent_channels=2,
                hidden_size=4,
                epochs=2,
                batch_size=4,
                learning_rate=1e-3,
                seed=3,
                decoder=self.autoencoder,
                visual_dataset=self.visual,
                changed_pixel_loss_weight=0.0,
                rollout_loss_weight=1.0,
            )
            for _ in range(2)
        ]

        first, second = results
        for name in (
            "train_losses",
            "validation_losses",
            "train_one_step_losses",
            "train_rollout_losses",
            "validation_one_step_losses",
            "validation_rollout_losses",
        ):
            first_values = np.asarray(getattr(first, name))
            second_values = np.asarray(getattr(second, name))
            self.assertEqual(first_values.shape, (2,))
            self.assertTrue(np.all(np.isfinite(first_values)))
            np.testing.assert_array_equal(first_values, second_values)
        self.assertGreaterEqual(first.best_epoch, 1)
        self.assertLessEqual(first.best_epoch, 2)
        for name, tensor in first.model.state_dict().items():
            torch.testing.assert_close(
                tensor,
                second.model.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )

    def test_training_rejects_horizon_one_or_mismatch(self):
        train_one_step, train_h1 = self._arrays(
            self.train_ids,
            horizon=1,
        )
        validation_one_step, validation_h1 = self._arrays(
            self.validation_ids,
            horizon=1,
        )
        with self.assertRaisesRegex(ValueError, "greater than one"):
            train_recursive_latent_dynamics(
                train_one_step,
                validation_one_step,
                train_h1,
                validation_h1,
                latent_normalizer=self.latent_normalizer,
                action_normalizer=self.action_normalizer,
                spatial_latent_channels=2,
                hidden_size=4,
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                seed=3,
                decoder=self.autoencoder,
                visual_dataset=self.visual,
                changed_pixel_loss_weight=0.0,
                rollout_loss_weight=1.0,
            )

        train_h5 = build_latent_rollout_arrays(
            self.visual,
            self.train_ids,
            self.latents,
            horizon=5,
        )
        with self.assertRaisesRegex(ValueError, "horizons must match"):
            train_recursive_latent_dynamics(
                train_one_step,
                validation_one_step,
                train_h5,
                validation_h1,
                latent_normalizer=self.latent_normalizer,
                action_normalizer=self.action_normalizer,
                spatial_latent_channels=2,
                hidden_size=4,
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                seed=3,
                decoder=self.autoencoder,
                visual_dataset=self.visual,
                changed_pixel_loss_weight=0.0,
                rollout_loss_weight=1.0,
            )


if __name__ == "__main__":
    unittest.main()
