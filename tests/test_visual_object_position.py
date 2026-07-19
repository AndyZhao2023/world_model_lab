from __future__ import annotations

import unittest

import numpy as np
import torch

from world_model_lab.visual_object_position import (
    fit_linear_object_position_probe,
    normalize_object_positions,
    object_position_probe_sha256,
    world_position_errors,
)


class ObjectPositionNormalizationTest(unittest.TestCase):
    def setUp(self):
        self.world_bounds = np.asarray(
            [0.0, 10.0, 0.0, 8.0],
            dtype=np.float64,
        )

    def test_world_positions_map_independently_to_minus_one_and_one(self):
        states = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [5.0, 4.0, 0.0, 0.0],
                [10.0, 8.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        positions = normalize_object_positions(
            states,
            self.world_bounds,
        )

        np.testing.assert_array_equal(
            positions,
            np.asarray(
                [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],
                dtype=np.float32,
            ),
        )

    def test_world_error_restores_axis_scales_before_euclidean_norm(self):
        predicted = np.asarray([[0.2, 0.5]], dtype=np.float32)
        target = np.zeros((1, 2), dtype=np.float32)

        errors = world_position_errors(
            predicted,
            target,
            self.world_bounds,
        )

        np.testing.assert_allclose(errors, [np.sqrt(5.0)], rtol=1e-7)

    def test_normalization_rejects_malformed_or_nonfinite_values(self):
        cases = (
            np.zeros((2, 2), dtype=np.float64),
            np.asarray([[0.0, 0.0, np.nan, 0.0]]),
        )
        for states in cases:
            with self.subTest(shape=states.shape):
                with self.assertRaises(ValueError):
                    normalize_object_positions(states, self.world_bounds)
        with self.assertRaises(ValueError):
            normalize_object_positions(
                np.zeros((2, 4), dtype=np.float64),
                np.asarray([0.0, 0.0, 0.0, 8.0]),
            )


class LinearObjectPositionProbeTest(unittest.TestCase):
    def setUp(self):
        self.latents = np.asarray(
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.positions = np.stack(
            (
                0.25 + 0.5 * self.latents[:, 0],
                -0.1 + 0.25 * self.latents[:, 1],
            ),
            axis=1,
        ).astype(np.float32)

    def test_probe_recovers_affine_mapping_and_is_deterministic(self):
        probes = [
            fit_linear_object_position_probe(
                self.latents,
                self.positions,
                ridge=0.0,
            )
            for _ in range(2)
        ]

        predictions = probes[0](
            torch.as_tensor(self.latents)
        ).detach().numpy()
        np.testing.assert_allclose(
            predictions,
            self.positions,
            rtol=0.0,
            atol=1e-6,
        )
        self.assertEqual(
            object_position_probe_sha256(probes[0]),
            object_position_probe_sha256(probes[1]),
        )
        for first, second in zip(
            probes[0].state_dict().values(),
            probes[1].state_dict().values(),
            strict=True,
        ):
            torch.testing.assert_close(
                first,
                second,
                rtol=0.0,
                atol=0.0,
            )

    def test_probe_is_frozen_but_preserves_input_gradients(self):
        probe = fit_linear_object_position_probe(
            self.latents,
            self.positions,
            ridge=1e-3,
        )
        values = torch.tensor(
            [[0.3, -0.2]],
            dtype=torch.float32,
            requires_grad=True,
        )

        loss = torch.sum(torch.square(probe(values)))
        loss.backward()

        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in probe.parameters()
            )
        )
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.all(torch.isfinite(values.grad)))
        self.assertNotEqual(float(torch.sum(torch.abs(values.grad))), 0.0)

    def test_probe_rejects_invalid_fit_and_forward_inputs(self):
        invalid_fits = (
            (self.latents[:0], self.positions[:0], 1e-3),
            (self.latents[:, :1], self.positions[:4], 1e-3),
            (self.latents, self.positions[:, :1], 1e-3),
            (self.latents, self.positions, -1.0),
            (self.latents, self.positions, float("inf")),
        )
        for latents, positions, ridge in invalid_fits:
            with self.subTest(
                latent_shape=latents.shape,
                position_shape=positions.shape,
                ridge=ridge,
            ):
                with self.assertRaises(ValueError):
                    fit_linear_object_position_probe(
                        latents,
                        positions,
                        ridge=ridge,
                    )

        probe = fit_linear_object_position_probe(
            self.latents,
            self.positions,
            ridge=1e-3,
        )
        with self.assertRaises(ValueError):
            probe(torch.zeros((2, 3), dtype=torch.float32))
        with self.assertRaises(ValueError):
            probe(
                torch.tensor(
                    [[float("nan"), 0.0]],
                    dtype=torch.float32,
                )
            )


if __name__ == "__main__":
    unittest.main()
