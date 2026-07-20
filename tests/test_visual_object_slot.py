from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from tests.test_visual_windows import make_visual_dataset
from world_model_lab.dataset import split_episode_ids
from world_model_lab.visual_latent_model import SpatialConvAutoencoder
from world_model_lab.visual_object_slot import (
    evaluate_object_slot_decoder,
    initialize_object_slot_autoencoder,
    normalize_object_slot_targets,
    object_slot_objective,
    train_object_slot_decoder,
)
from world_model_lab.visual_observation import CAR_COLOR, HEADING_COLOR


def make_object_slot_visual_dataset() -> dict[str, np.ndarray]:
    visual = make_visual_dataset((4,) * 10)
    visual["states"].fill(0.0)
    visual["frames"][:, 31, 31] = CAR_COLOR
    visual["frames"][:, 31, 32] = HEADING_COLOR
    return visual


class ObjectSlotGeometryTest(unittest.TestCase):
    def test_targets_use_letterboxed_image_coordinates_and_wrapped_heading(
        self,
    ):
        states = np.asarray(
            [
                [-5.0, -4.0, 0.0, 0.0],
                [5.0, 4.0, math.pi / 2.0, 0.0],
                [0.0, 0.0, 2.0 * math.pi, 0.0],
            ],
            dtype=np.float64,
        )

        targets = normalize_object_slot_targets(
            states,
            np.asarray([-5.0, 5.0, -4.0, 4.0], dtype=np.float64),
        )

        expected_y_extent = 50.4 / 63.0
        np.testing.assert_allclose(
            targets,
            np.asarray(
                [
                    [-1.0, expected_y_extent, 0.0, 1.0],
                    [1.0, -expected_y_extent, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
        )

    def test_local_alpha_is_exactly_zero_outside_eleven_pixel_support(self):
        model = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
            object_slot_decoder=True,
            object_slot_patch_size=11,
            object_slot_hidden_size=8,
        )
        latents = torch.randn((2, 2, 8, 8), dtype=torch.float32)

        components = model.decode_object_slot_components(latents)

        self.assertEqual(tuple(components.slot.shape), (2, 4))
        self.assertEqual(tuple(components.foreground.shape), (2, 3, 64, 64))
        self.assertEqual(tuple(components.alpha.shape), (2, 1, 64, 64))
        self.assertEqual(tuple(components.composite.shape), (2, 3, 64, 64))
        self.assertTrue(bool(torch.all(torch.isfinite(components.composite))))
        for item in range(2):
            support = components.support[item, 0]
            alpha = components.alpha[item, 0]
            self.assertLessEqual(int(torch.sum(support)), 121)
            self.assertTrue(bool(torch.all(alpha[~support] == 0.0)))

    def test_slot_and_residual_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            SpatialConvAutoencoder(
                latent_channels=2,
                base_channels=2,
                object_residual_decoder=True,
                object_head_channels=2,
                object_slot_decoder=True,
                object_slot_patch_size=11,
                object_slot_hidden_size=8,
            )


class ObjectSlotTrainingTest(unittest.TestCase):
    def test_objective_matches_all_five_declared_terms(self):
        model = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
            object_slot_decoder=True,
            object_slot_patch_size=11,
            object_slot_hidden_size=8,
        )
        images = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        images[:, :, 31, 31] = 1.0
        masks = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        masks[:, :, 31, 31] = 1.0
        targets = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        loss, components = object_slot_objective(
            model,
            images,
            masks,
            targets,
            foreground_loss_weight=1.0,
            mask_loss_weight=0.01,
            centre_loss_weight=1.0,
            heading_loss_weight=0.1,
        )

        expected = (
            components["full_mse"]
            + components["foreground_object_mse"]
            + 0.01 * components["balanced_alpha_bce"]
            + components["centre_mse"]
            + 0.1 * components["heading_mse"]
        )
        torch.testing.assert_close(loss, expected)

    def test_initialized_candidate_freezes_source_and_trains_only_slot(self):
        source = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
        )
        source_state = {
            name: tensor.detach().clone()
            for name, tensor in source.state_dict().items()
        }
        candidate = initialize_object_slot_autoencoder(
            source,
            patch_size=11,
            hidden_size=8,
            initial_alpha=0.01,
            seed=7,
        )
        images = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
        images[:, :, 31, 31] = 1.0
        masks = torch.zeros((2, 1, 64, 64), dtype=torch.float32)
        masks[:, :, 31, 31] = 1.0
        targets = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        loss, _ = object_slot_objective(
            candidate,
            images,
            masks,
            targets,
            foreground_loss_weight=1.0,
            mask_loss_weight=0.01,
            centre_loss_weight=1.0,
            heading_loss_weight=0.1,
        )
        loss.backward()

        for prefix in ("encoder_convolutions", "decoder_convolutions"):
            parameters = [
                parameter
                for name, parameter in candidate.named_parameters()
                if name.startswith(prefix)
            ]
            self.assertTrue(parameters)
            self.assertTrue(
                all(not parameter.requires_grad for parameter in parameters)
            )
            self.assertTrue(all(parameter.grad is None for parameter in parameters))
        slot_parameters = [
            parameter
            for name, parameter in candidate.named_parameters()
            if name.startswith(
                ("object_attention", "object_heading", "object_patch_decoder")
            )
        ]
        self.assertTrue(all(parameter.requires_grad for parameter in slot_parameters))
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.any(parameter.grad != 0))
                for parameter in slot_parameters
            )
        )
        for name, expected in source_state.items():
            torch.testing.assert_close(
                candidate.state_dict()[name],
                expected,
                rtol=0.0,
                atol=0.0,
            )

    def test_training_is_deterministic_and_reports_state_and_alpha_metrics(
        self,
    ):
        visual = make_object_slot_visual_dataset()
        splits = split_episode_ids(visual["episode_ids"], seed=19)
        source = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
        )
        kwargs = {
            "split_episode_ids": splits,
            "patch_size": 11,
            "hidden_size": 8,
            "initial_alpha": 0.01,
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "foreground_loss_weight": 1.0,
            "mask_loss_weight": 0.01,
            "centre_loss_weight": 1.0,
            "heading_loss_weight": 0.1,
            "seed": 5,
        }

        first = train_object_slot_decoder(source, visual, **kwargs)
        second = train_object_slot_decoder(source, visual, **kwargs)
        metrics = evaluate_object_slot_decoder(
            first.model,
            visual,
            selected_episode_ids=splits["test"],
            batch_size=8,
        )

        self.assertEqual(first.train_losses, second.train_losses)
        self.assertEqual(first.validation_losses, second.validation_losses)
        for name, expected in first.model.state_dict().items():
            torch.testing.assert_close(
                second.model.state_dict()[name],
                expected,
                rtol=0.0,
                atol=0.0,
            )
        for name in (
            "mean_centre_error_pixels",
            "mean_heading_error_degrees",
            "object_mask_iou",
            "object_mask_precision",
            "object_mask_recall",
            "mean_object_alpha",
            "mean_background_alpha",
        ):
            self.assertTrue(np.isfinite(metrics[name]))
        self.assertLessEqual(metrics["max_support_pixels"], 121)

    def test_objective_rejects_invalid_masks_targets_and_weights(self):
        model = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
            object_slot_decoder=True,
            object_slot_patch_size=11,
            object_slot_hidden_size=8,
        )
        images = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        masks = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        targets = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "object and background"):
            object_slot_objective(
                model,
                images,
                masks,
                targets,
                foreground_loss_weight=1.0,
                mask_loss_weight=0.01,
                centre_loss_weight=1.0,
                heading_loss_weight=0.1,
            )
        masks[:, :, 31, 31] = 1.0
        for name in (
            "foreground_loss_weight",
            "mask_loss_weight",
            "centre_loss_weight",
            "heading_loss_weight",
        ):
            values = {
                "foreground_loss_weight": 1.0,
                "mask_loss_weight": 0.01,
                "centre_loss_weight": 1.0,
                "heading_loss_weight": 0.1,
            }
            values[name] = -1.0
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    object_slot_objective(
                        model,
                        images,
                        masks,
                        targets,
                        **values,
                    )


if __name__ == "__main__":
    unittest.main()
