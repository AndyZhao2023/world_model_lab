import math
import unittest

import numpy as np

from world_model_lab.dataset import (
    build_model_arrays,
    build_model_inputs,
    fit_normalizer,
    split_episode_ids,
    wrap_angle,
)


class DatasetTest(unittest.TestCase):
    def test_build_model_inputs_encodes_heading_without_targets(self):
        states = np.asarray([[1.0, 2.0, math.pi / 2.0, 0.5]])
        actions = np.asarray([[0.1, -0.2]])

        inputs = build_model_inputs(states, actions)

        np.testing.assert_allclose(
            inputs,
            [[1.0, 2.0, 1.0, 0.0, 0.5, 0.1, -0.2]],
            atol=1e-12,
        )

    def test_wrap_angle_maps_values_to_half_open_pi_interval(self):
        values = np.asarray(
            [-3 * math.pi, -math.pi, -0.2, math.pi, 3 * math.pi],
            dtype=np.float64,
        )

        wrapped = wrap_angle(values)

        np.testing.assert_allclose(
            wrapped,
            [-math.pi, -math.pi, -0.2, -math.pi, -math.pi],
            atol=1e-12,
        )
        self.assertTrue(np.all(wrapped >= -math.pi))
        self.assertTrue(np.all(wrapped < math.pi))

    def test_build_model_arrays_uses_continuous_heading_and_wrapped_delta(self):
        states = np.asarray(
            [
                [1.0, 2.0, math.radians(179.0), 0.5],
                [3.0, 4.0, math.radians(-30.0), 1.5],
            ]
        )
        actions = np.asarray([[0.1, 0.2], [-0.2, -0.4]])
        next_states = np.asarray(
            [
                [1.1, 2.2, math.radians(-179.0), 0.52],
                [3.2, 3.8, math.radians(-25.0), 1.46],
            ]
        )

        inputs, targets = build_model_arrays(states, actions, next_states)

        self.assertEqual(inputs.shape, (2, 7))
        self.assertEqual(targets.shape, (2, 4))
        np.testing.assert_allclose(
            inputs[0],
            [
                1.0,
                2.0,
                math.sin(math.radians(179.0)),
                math.cos(math.radians(179.0)),
                0.5,
                0.1,
                0.2,
            ],
        )
        np.testing.assert_allclose(
            targets[0],
            [0.1, 0.2, math.radians(2.0), 0.02],
            atol=1e-12,
        )

    def test_split_episode_ids_is_deterministic_and_disjoint(self):
        transition_episode_ids = np.repeat(np.arange(10), [2, 3, 1, 4, 2, 2, 3, 1, 2, 5])

        first = split_episode_ids(transition_episode_ids, seed=19)
        second = split_episode_ids(transition_episode_ids, seed=19)

        for split_name in ("train", "validation", "test"):
            np.testing.assert_array_equal(first[split_name], second[split_name])
        self.assertEqual(first["train"].size, 8)
        self.assertEqual(first["validation"].size, 1)
        self.assertEqual(first["test"].size, 1)
        self.assertEqual(
            set(first["train"]) | set(first["validation"]) | set(first["test"]),
            set(range(10)),
        )
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["validation"]) & set(first["test"]))

    def test_split_requires_at_least_three_episodes(self):
        with self.assertRaisesRegex(ValueError, "at least three episodes"):
            split_episode_ids(np.asarray([0, 0, 1, 1]), seed=0)

    def test_fit_normalizer_round_trips_values(self):
        values = np.asarray([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])

        normalizer = fit_normalizer(values)

        normalized = normalizer.normalize(values)
        np.testing.assert_allclose(normalized.mean(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(normalized.std(axis=0), [1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(normalizer.denormalize(normalized), values)

    def test_fit_normalizer_rejects_constant_dimensions(self):
        values = np.asarray([[1.0, 2.0], [1.0, 3.0]])

        with self.assertRaisesRegex(ValueError, "zero variance"):
            fit_normalizer(values)


if __name__ == "__main__":
    unittest.main()
