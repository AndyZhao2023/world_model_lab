import unittest

import numpy as np

from world_model_lab.bootstrap import (
    expand_episode_transition_indices,
    sample_episode_bootstrap,
)


class EpisodeBootstrapTest(unittest.TestCase):
    def test_sampling_is_deterministic_and_keeps_zero_count_episodes(self):
        result = sample_episode_bootstrap(
            np.asarray([10, 11, 12, 13]),
            seed=3,
        )

        np.testing.assert_array_equal(
            result.drawn_episode_ids,
            [13, 10, 10, 10],
        )
        self.assertEqual(
            result.episode_counts,
            {10: 3, 11: 0, 12: 0, 13: 1},
        )
        self.assertEqual(result.draw_count, 4)
        self.assertEqual(result.unique_count, 2)

        repeated = sample_episode_bootstrap(
            np.asarray([10, 11, 12, 13]),
            seed=3,
        )
        np.testing.assert_array_equal(
            repeated.drawn_episode_ids,
            result.drawn_episode_ids,
        )

    def test_transition_expansion_repeats_complete_episode_groups(self):
        indices = expand_episode_transition_indices(
            np.asarray([10, 11, 10, 12, 13, 10]),
            np.asarray([13, 10, 10]),
        )

        np.testing.assert_array_equal(indices, [4, 0, 2, 5, 0, 2, 5])

    def test_sampling_rejects_invalid_source_ids_and_seeds(self):
        invalid_ids = (
            np.asarray([]),
            np.asarray([[1, 2]]),
            np.asarray([1.0, 2.0]),
            np.asarray([True, False]),
            np.asarray([-1, 2]),
            np.asarray([1, 1]),
        )
        for values in invalid_ids:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    sample_episode_bootstrap(values, seed=0)

        for seed in (True, -1, 1.5):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "bootstrap seed"):
                    sample_episode_bootstrap(np.asarray([1, 2]), seed=seed)

    def test_transition_expansion_rejects_missing_drawn_episode(self):
        with self.assertRaisesRegex(ValueError, "episode 3.*missing"):
            expand_episode_transition_indices(
                np.asarray([1, 1, 2]),
                np.asarray([1, 3]),
            )
