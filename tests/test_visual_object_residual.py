from __future__ import annotations

import unittest

import numpy as np
import torch

from tests.test_visual_windows import make_visual_dataset
from world_model_lab.dataset import split_episode_ids
from world_model_lab.visual_latent_model import SpatialConvAutoencoder
from world_model_lab.visual_object_residual import (
    evaluate_object_residual_mask,
    initialize_object_residual_autoencoder,
    object_residual_objective,
    train_object_residual_decoder,
)
from world_model_lab.visual_observation import CAR_COLOR, HEADING_COLOR


def make_object_visual_dataset() -> dict[str, np.ndarray]:
    visual = make_visual_dataset((4,) * 10)
    visual["frames"][:, 2, 3] = CAR_COLOR
    visual["frames"][:, 4, 5] = HEADING_COLOR
    return visual


class ObjectResidualObjectiveTest(unittest.TestCase):
    def test_objective_matches_declared_three_terms(self):
        model = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
            object_residual_decoder=True,
            object_head_channels=3,
        )
        images = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        images[:, :, 2, 3] = 1.0
        masks = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        masks[:, :, 2, 3] = 1.0

        loss, components = object_residual_objective(
            model,
            images,
            masks,
            foreground_loss_weight=1.0,
            mask_loss_weight=0.01,
        )
        latents = model.encode(images)
        _, foreground, mask_logits, composite = model.decode_components(
            latents
        )
        expanded = masks.expand_as(images)
        full_mse = torch.mean(torch.square(composite - images))
        object_mse = torch.sum(
            torch.square(foreground - images) * expanded
        ) / torch.sum(expanded)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            mask_logits,
            masks,
            reduction="none",
        )
        mask_bce = (
            torch.sum(bce * masks) / torch.sum(masks)
            + torch.sum(bce * (1.0 - masks))
            / torch.sum(1.0 - masks)
        )

        torch.testing.assert_close(
            loss,
            full_mse + object_mse + 0.01 * mask_bce,
        )
        torch.testing.assert_close(components["full_mse"], full_mse)
        torch.testing.assert_close(
            components["foreground_object_mse"],
            object_mse,
        )
        torch.testing.assert_close(
            components["balanced_mask_bce"],
            mask_bce,
        )

    def test_initialized_candidate_freezes_base_and_backpropagates_to_head(
        self,
    ):
        source = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
        )
        source_state = {
            name: tensor.detach().clone()
            for name, tensor in source.state_dict().items()
        }
        candidate = initialize_object_residual_autoencoder(
            source,
            head_channels=3,
            initial_alpha=0.01,
            seed=7,
        )
        images = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
        images[:, :, 2, 3] = 1.0
        masks = torch.zeros((2, 1, 64, 64), dtype=torch.float32)
        masks[:, :, 2, 3] = 1.0

        loss, _ = object_residual_objective(
            candidate,
            images,
            masks,
            foreground_loss_weight=1.0,
            mask_loss_weight=0.01,
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
        head_parameters = [
            parameter
            for name, parameter in candidate.named_parameters()
            if name.startswith("object_decoder_convolutions")
        ]
        self.assertTrue(all(parameter.requires_grad for parameter in head_parameters))
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.any(parameter.grad != 0))
                for parameter in head_parameters
            )
        )
        for name, expected in source_state.items():
            torch.testing.assert_close(
                source.state_dict()[name],
                expected,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                candidate.state_dict()[name],
                expected,
                rtol=0.0,
                atol=0.0,
            )

    def test_training_is_deterministic_and_reports_finite_mask_metrics(self):
        visual = make_object_visual_dataset()
        splits = split_episode_ids(visual["episode_ids"], seed=19)
        source = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
        )

        first = train_object_residual_decoder(
            source,
            visual,
            split_episode_ids=splits,
            head_channels=3,
            initial_alpha=0.01,
            epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            foreground_loss_weight=1.0,
            mask_loss_weight=0.01,
            seed=5,
        )
        second = train_object_residual_decoder(
            source,
            visual,
            split_episode_ids=splits,
            head_channels=3,
            initial_alpha=0.01,
            epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            foreground_loss_weight=1.0,
            mask_loss_weight=0.01,
            seed=5,
        )
        metrics = evaluate_object_residual_mask(
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
            "object_mask_iou",
            "object_mask_precision",
            "object_mask_recall",
            "mean_object_alpha",
            "mean_background_alpha",
        ):
            self.assertTrue(np.isfinite(metrics[name]))

    def test_objective_rejects_empty_regions_and_invalid_weights(self):
        model = SpatialConvAutoencoder(
            latent_channels=2,
            base_channels=2,
            object_residual_decoder=True,
            object_head_channels=3,
        )
        images = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        empty = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        for foreground_weight, mask_weight in (
            (-1.0, 0.01),
            (1.0, -0.01),
            (float("nan"), 0.01),
            (1.0, float("inf")),
        ):
            with self.subTest(
                foreground_weight=foreground_weight,
                mask_weight=mask_weight,
            ):
                with self.assertRaises(ValueError):
                    object_residual_objective(
                        model,
                        images,
                        empty,
                        foreground_loss_weight=foreground_weight,
                        mask_loss_weight=mask_weight,
                    )
        with self.assertRaisesRegex(ValueError, "object and background"):
            object_residual_objective(
                model,
                images,
                empty,
                foreground_loss_weight=1.0,
                mask_loss_weight=0.01,
            )


if __name__ == "__main__":
    unittest.main()
