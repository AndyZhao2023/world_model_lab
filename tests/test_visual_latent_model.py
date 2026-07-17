from __future__ import annotations

import unittest

import torch

from world_model_lab.visual_latent_model import (
    ConvAutoencoder,
    LatentDynamicsMLP,
    SpatialConvAutoencoder,
    SpatialLatentDynamicsCNN,
    SpatialLatentDynamicsConvGRU,
)


class ConvAutoencoderTest(unittest.TestCase):
    def test_encode_decode_and_forward_shapes_are_exact(self):
        model = ConvAutoencoder(latent_dim=12, base_channels=4)
        images = torch.zeros((5, 3, 64, 64), dtype=torch.float32)

        latents = model.encode(images)
        reconstructions = model.decode(latents)
        forwarded = model(images)

        self.assertEqual(tuple(latents.shape), (5, 12))
        self.assertEqual(tuple(reconstructions.shape), (5, 3, 64, 64))
        self.assertEqual(tuple(forwarded.shape), (5, 3, 64, 64))
        self.assertTrue(torch.all(reconstructions >= 0.0))
        self.assertTrue(torch.all(reconstructions <= 1.0))

    def test_invalid_shapes_and_configuration_are_rejected(self):
        for kwargs in (
            {"latent_dim": 0},
            {"base_channels": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ConvAutoencoder(**kwargs)

        model = ConvAutoencoder(latent_dim=8, base_channels=4)
        with self.assertRaisesRegex(ValueError, r"\[B, 3, 64, 64\]"):
            model.encode(torch.zeros((2, 3, 32, 32)))
        with self.assertRaisesRegex(ValueError, r"\[B, 8\]"):
            model.decode(torch.zeros((2, 7)))


class LatentDynamicsMLPTest(unittest.TestCase):
    def test_predicts_one_next_latent_from_exact_context(self):
        model = LatentDynamicsMLP(
            latent_dim=8,
            hidden_size=16,
            context_frames=4,
        )

        output = model(
            torch.zeros((6, 4, 8)),
            torch.zeros((6, 3, 2)),
            torch.zeros((6, 2)),
        )

        self.assertEqual(tuple(output.shape), (6, 8))

    def test_zero_network_residual_returns_last_context_latent(self):
        model = LatentDynamicsMLP(
            latent_dim=4,
            hidden_size=8,
            context_frames=4,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        context = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)

        output = model(
            context,
            torch.zeros((2, 3, 2)),
            torch.zeros((2, 2)),
        )

        torch.testing.assert_close(output, context[:, -1])

    def test_dynamics_rejects_invalid_configuration(self):
        for kwargs in (
            {"latent_dim": 0},
            {"hidden_size": 0},
            {"context_frames": 1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LatentDynamicsMLP(**kwargs)

    def test_dynamics_rejects_misaligned_inputs(self):
        model = LatentDynamicsMLP(latent_dim=4, hidden_size=8)
        cases = (
            (
                torch.zeros((2, 3, 4)),
                torch.zeros((2, 3, 2)),
                torch.zeros((2, 2)),
            ),
            (
                torch.zeros((2, 4, 4)),
                torch.zeros((2, 2, 2)),
                torch.zeros((2, 2)),
            ),
            (
                torch.zeros((2, 4, 4)),
                torch.zeros((2, 3, 2)),
                torch.zeros((3, 2)),
            ),
        )
        for context, history, current in cases:
            with self.subTest(
                shapes=(context.shape, history.shape, current.shape)
            ):
                with self.assertRaises(ValueError):
                    model(context, history, current)


class SpatialConvAutoencoderTest(unittest.TestCase):
    def test_encode_preserves_an_exact_spatial_grid(self):
        model = SpatialConvAutoencoder(
            latent_channels=3,
            base_channels=4,
        )
        images = torch.zeros((2, 3, 64, 64), dtype=torch.float32)

        latents = model.encode(images)
        decoded_grid = model.decode(latents)
        decoded_flat = model.decode(latents.flatten(start_dim=1))

        self.assertEqual(tuple(latents.shape), (2, 3, 8, 8))
        self.assertEqual(model.latent_dim, 3 * 8 * 8)
        self.assertEqual(tuple(decoded_grid.shape), (2, 3, 64, 64))
        torch.testing.assert_close(decoded_flat, decoded_grid)

    def test_invalid_configuration_and_shapes_are_rejected(self):
        for kwargs in (
            {"latent_channels": 0},
            {"base_channels": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SpatialConvAutoencoder(**kwargs)

        model = SpatialConvAutoencoder(latent_channels=2, base_channels=4)
        with self.assertRaisesRegex(ValueError, r"\[B, 3, 64, 64\]"):
            model.encode(torch.zeros((2, 3, 32, 32)))
        for latents in (
            torch.zeros((2, 2, 7, 8)),
            torch.zeros((2, 127)),
        ):
            with self.subTest(shape=latents.shape):
                with self.assertRaises(ValueError):
                    model.decode(latents)


class SpatialLatentDynamicsCNNTest(unittest.TestCase):
    def test_predicts_a_flattened_next_spatial_latent(self):
        model = SpatialLatentDynamicsCNN(
            latent_channels=3,
            hidden_channels=8,
            context_frames=4,
        )

        output = model(
            torch.zeros((5, 4, 3 * 8 * 8)),
            torch.zeros((5, 3, 2)),
            torch.zeros((5, 2)),
        )

        self.assertEqual(tuple(output.shape), (5, 3 * 8 * 8))

    def test_zero_network_residual_returns_last_context_grid(self):
        model = SpatialLatentDynamicsCNN(
            latent_channels=2,
            hidden_channels=4,
            context_frames=4,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        context = torch.arange(
            2 * 4 * model.latent_dim,
            dtype=torch.float32,
        ).reshape(2, 4, model.latent_dim)

        output = model(
            context,
            torch.zeros((2, 3, 2)),
            torch.zeros((2, 2)),
        )

        torch.testing.assert_close(output, context[:, -1])

    def test_invalid_configuration_and_inputs_are_rejected(self):
        for kwargs in (
            {"latent_channels": 0},
            {"hidden_channels": 0},
            {"context_frames": 1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SpatialLatentDynamicsCNN(**kwargs)

        model = SpatialLatentDynamicsCNN(
            latent_channels=2,
            hidden_channels=4,
        )
        cases = (
            (
                torch.zeros((2, 4, 127)),
                torch.zeros((2, 3, 2)),
                torch.zeros((2, 2)),
            ),
            (
                torch.zeros((2, 4, 128)),
                torch.zeros((2, 2, 2)),
                torch.zeros((2, 2)),
            ),
            (
                torch.zeros((2, 4, 128)),
                torch.zeros((2, 3, 2)),
                torch.zeros((3, 2)),
            ),
        )
        for context, history, current in cases:
            with self.subTest(
                shapes=(context.shape, history.shape, current.shape)
            ):
                with self.assertRaises(ValueError):
                    model(context, history, current)


class SpatialLatentDynamicsConvGRUTest(unittest.TestCase):
    def test_predicts_a_flattened_next_spatial_latent(self):
        model = SpatialLatentDynamicsConvGRU(
            latent_channels=3,
            hidden_channels=8,
            context_frames=4,
        )

        output = model(
            torch.zeros((5, 4, 3 * 8 * 8)),
            torch.zeros((5, 3, 2)),
            torch.zeros((5, 2)),
        )

        self.assertEqual(tuple(output.shape), (5, 3 * 8 * 8))

    def test_zero_network_residual_returns_last_context_grid(self):
        model = SpatialLatentDynamicsConvGRU(
            latent_channels=2,
            hidden_channels=4,
            context_frames=4,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        context = torch.arange(
            2 * 4 * model.latent_dim,
            dtype=torch.float32,
        ).reshape(2, 4, model.latent_dim)

        output = model(
            context,
            torch.zeros((2, 3, 2)),
            torch.zeros((2, 2)),
        )

        torch.testing.assert_close(output, context[:, -1])

    def test_invalid_configuration_and_inputs_are_rejected(self):
        for kwargs in (
            {"latent_channels": 0},
            {"hidden_channels": 0},
            {"context_frames": 1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SpatialLatentDynamicsConvGRU(**kwargs)

        model = SpatialLatentDynamicsConvGRU(
            latent_channels=2,
            hidden_channels=4,
        )
        cases = (
            (
                torch.zeros((2, 4, 127)),
                torch.zeros((2, 3, 2)),
                torch.zeros((2, 2)),
            ),
            (
                torch.zeros((2, 4, 128)),
                torch.zeros((2, 2, 2)),
                torch.zeros((2, 2)),
            ),
            (
                torch.zeros((2, 4, 128)),
                torch.zeros((2, 3, 2)),
                torch.zeros((3, 2)),
            ),
        )
        for context, history, current in cases:
            with self.subTest(
                shapes=(context.shape, history.shape, current.shape)
            ):
                with self.assertRaises(ValueError):
                    model(context, history, current)


if __name__ == "__main__":
    unittest.main()
