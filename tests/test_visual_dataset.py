import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.visual_fixtures import clone_arrays, make_transition_source
from world_model_lab.visual_dataset import (
    load_transition_dataset,
    reconstruct_episodes,
)


class VisualSourceValidationTest(unittest.TestCase):
    def setUp(self):
        self.source = make_transition_source()

    def test_reconstructs_episodes_in_canonical_id_and_step_order(self):
        episodes = reconstruct_episodes(self.source)

        self.assertEqual([episode.episode_id for episode in episodes], [3, 7])
        np.testing.assert_array_equal(episodes[0].step_ids, [0, 1])
        np.testing.assert_array_equal(episodes[1].step_ids, [0, 1, 2, 3])
        self.assertEqual(episodes[0].states.shape, (2, 4))
        self.assertEqual(episodes[1].actions.shape, (4, 2))

    def test_shuffled_rows_reconstruct_identically(self):
        permutation = np.asarray([5, 1, 4, 0, 3, 2])
        shuffled = {
            name: values[permutation]
            for name, values in self.source.items()
        }

        expected = reconstruct_episodes(self.source)
        actual = reconstruct_episodes(shuffled)

        for expected_episode, actual_episode in zip(
            expected,
            actual,
            strict=True,
        ):
            self.assertEqual(actual_episode.episode_id, expected_episode.episode_id)
            for field in (
                "states",
                "actions",
                "next_states",
                "rewards",
                "dones",
                "step_ids",
                "terminal_reasons",
            ):
                np.testing.assert_array_equal(
                    getattr(actual_episode, field),
                    getattr(expected_episode, field),
                )

    def test_loader_uses_allow_pickle_false(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object-source.npz"
            unsafe = clone_arrays(self.source)
            unsafe["states"] = np.asarray([object()], dtype=object)
            np.savez_compressed(path, **unsafe)

            with self.assertRaisesRegex(ValueError, "Object arrays"):
                load_transition_dataset(path)

    def test_loader_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.npz"
            with self.assertRaisesRegex(FileNotFoundError, "regular file"):
                load_transition_dataset(path)

    def test_missing_required_array_names_the_array(self):
        source = clone_arrays(self.source)
        source.pop("actions")

        with self.assertRaisesRegex(ValueError, "missing arrays: actions"):
            reconstruct_episodes(source)

    def test_non_finite_numeric_value_names_the_array(self):
        source = clone_arrays(self.source)
        source["states"][0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "states.*finite"):
            reconstruct_episodes(source)

    def test_invalid_action_shape_names_the_array(self):
        source = clone_arrays(self.source)
        source["actions"] = source["actions"][:, :1]

        with self.assertRaisesRegex(ValueError, r"actions.*\[6, 2\]"):
            reconstruct_episodes(source)

    def test_float_step_ids_are_rejected(self):
        source = clone_arrays(self.source)
        source["step_ids"] = source["step_ids"].astype(np.float64)

        with self.assertRaisesRegex(ValueError, "step_ids.*integer"):
            reconstruct_episodes(source)

    def test_duplicate_missing_or_negative_steps_are_rejected(self):
        for replacement in (
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([-1, 0], dtype=np.int64),
        ):
            with self.subTest(replacement=replacement):
                source = clone_arrays(self.source)
                mask = source["episode_ids"] == 3
                source["step_ids"][mask] = replacement
                with self.assertRaisesRegex(
                    ValueError,
                    "episode 3 step_ids.*contiguous",
                ):
                    reconstruct_episodes(source)

    def test_transition_discontinuity_names_episode_and_step(self):
        source = clone_arrays(self.source)
        first_row = np.flatnonzero(
            (source["episode_ids"] == 7) & (source["step_ids"] == 0)
        )[0]
        source["next_states"][first_row, 0] += 1e-3

        with self.assertRaisesRegex(
            ValueError,
            "episode 7 is discontinuous after step 0",
        ):
            reconstruct_episodes(source)

    def test_early_terminal_names_episode_and_step(self):
        source = clone_arrays(self.source)
        first_row = np.flatnonzero(
            (source["episode_ids"] == 7) & (source["step_ids"] == 0)
        )[0]
        source["dones"][first_row] = True

        with self.assertRaisesRegex(
            ValueError,
            "episode 7 terminates.*step 0",
        ):
            reconstruct_episodes(source)

    def test_non_terminal_final_row_names_episode(self):
        source = clone_arrays(self.source)
        final_row = np.flatnonzero(
            (source["episode_ids"] == 3) & (source["step_ids"] == 1)
        )[0]
        source["dones"][final_row] = False
        source["terminal_reasons"][final_row] = ""

        with self.assertRaisesRegex(ValueError, "episode 3 final row"):
            reconstruct_episodes(source)

    def test_invalid_terminal_reason_names_episode(self):
        source = clone_arrays(self.source)
        final_row = np.flatnonzero(
            (source["episode_ids"] == 3) & (source["step_ids"] == 1)
        )[0]
        source["terminal_reasons"][final_row] = "unknown"

        with self.assertRaisesRegex(
            ValueError,
            "episode 3 final terminal reason is invalid",
        ):
            reconstruct_episodes(source)
