import tempfile
import unittest
from pathlib import Path

import numpy as np

from world_model_lab.collect_data import (
    collect_transitions,
    save_dataset,
    summarize_dataset,
)


class CollectDataTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "episodes": 4,
            "max_steps": 12,
            "action_hold_steps": 3,
            "seed": 17,
        }

    def test_collection_returns_aligned_training_arrays(self):
        dataset = collect_transitions(**self.config)
        count = dataset["states"].shape[0]

        self.assertGreater(count, 0)
        self.assertLessEqual(count, 48)
        self.assertEqual(dataset["states"].shape, (count, 4))
        self.assertEqual(dataset["actions"].shape, (count, 2))
        self.assertEqual(dataset["next_states"].shape, (count, 4))
        for name in (
            "rewards",
            "dones",
            "episode_ids",
            "step_ids",
            "terminal_reasons",
        ):
            self.assertEqual(dataset[name].shape, (count,))

    def test_same_seed_is_exactly_reproducible(self):
        first = collect_transitions(**self.config)
        second = collect_transitions(**self.config)

        self.assertEqual(first.keys(), second.keys())
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_actions_are_applied_values_within_limits(self):
        dataset = collect_transitions(**self.config)

        self.assertTrue(np.all(np.abs(dataset["actions"][:, 0]) <= 0.5))
        self.assertTrue(np.all(np.abs(dataset["actions"][:, 1]) <= 1.0))

    def test_each_episode_has_contiguous_steps_and_one_terminal(self):
        dataset = collect_transitions(**self.config)

        np.testing.assert_array_equal(np.unique(dataset["episode_ids"]), np.arange(4))
        for episode_id in range(4):
            mask = dataset["episode_ids"] == episode_id
            steps = dataset["step_ids"][mask]
            dones = dataset["dones"][mask]
            reasons = dataset["terminal_reasons"][mask]

            np.testing.assert_array_equal(steps, np.arange(steps.size))
            self.assertEqual(np.count_nonzero(dones), 1)
            self.assertTrue(dones[-1])
            self.assertTrue(reasons[-1])
            self.assertTrue(np.all(reasons[:-1] == ""))

    def test_initial_states_are_inside_safe_free_space(self):
        dataset = collect_transitions(**self.config)

        for episode_id in range(4):
            state = dataset["states"][dataset["episode_ids"] == episode_id][0]
            x, y = state[:2]
            self.assertGreaterEqual(x - 0.2, 0.0)
            self.assertLessEqual(x + 0.2, 10.0)
            self.assertGreaterEqual(y - 0.2, 0.0)
            self.assertLessEqual(y + 0.2, 8.0)
            self.assertGreater(np.linalg.norm(state[:2] - [5.0, 4.0]), 1.2)
            self.assertGreater(np.linalg.norm(state[:2] - [9.0, 7.0]), 0.5)

    def test_dataset_round_trips_through_npz(self):
        dataset = collect_transitions(**self.config)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transitions.npz"
            save_dataset(dataset, path)
            with np.load(path) as loaded:
                self.assertEqual(set(dataset), set(loaded.files))
                for name, expected in dataset.items():
                    np.testing.assert_array_equal(loaded[name], expected)

    def test_summary_counts_transitions_and_terminal_reasons(self):
        dataset = collect_transitions(**self.config)

        summary = summarize_dataset(dataset)

        self.assertEqual(summary["transitions"], dataset["states"].shape[0])
        self.assertEqual(summary["episodes"], 4)
        self.assertEqual(
            summary["terminal_total"], np.count_nonzero(dataset["dones"])
        )

    def test_invalid_collection_configuration_is_rejected(self):
        for overrides in (
            {"episodes": 0},
            {"max_steps": 0},
            {"action_hold_steps": 0},
        ):
            config = self.config | overrides
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                collect_transitions(**config)


if __name__ == "__main__":
    unittest.main()
