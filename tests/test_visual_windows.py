from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from tests.visual_fixtures import make_transition_source
from world_model_lab.visual_dataset import (
    build_visual_dataset,
    validate_visual_dataset,
)
from world_model_lab.visual_windows import (
    build_visual_window_dataset,
    build_visual_window_index,
    build_visual_window_splits,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_visual_dataset(
    transition_lengths: tuple[int, ...],
) -> dict[str, np.ndarray]:
    template = build_visual_dataset(make_transition_source())
    episode_count = len(transition_lengths)
    episode_ids = np.arange(10, 10 + episode_count, dtype=np.int64)
    transition_offsets = np.zeros(episode_count + 1, dtype=np.int64)
    frame_offsets = np.zeros(episode_count + 1, dtype=np.int64)
    for index, length in enumerate(transition_lengths):
        if length < 1:
            raise ValueError("fixture episodes require at least one transition")
        transition_offsets[index + 1] = transition_offsets[index] + length
        frame_offsets[index + 1] = frame_offsets[index] + length + 1

    transition_count = int(transition_offsets[-1])
    frame_count = int(frame_offsets[-1])
    frames = np.empty((frame_count, 64, 64, 3), dtype=np.uint8)
    states = np.zeros((frame_count, 4), dtype=np.float64)
    actions = np.empty((transition_count, 2), dtype=np.float64)
    rewards = np.zeros(transition_count, dtype=np.float64)
    dones = np.zeros(transition_count, dtype=np.bool_)
    terminal_reasons = np.full(
        transition_count,
        "",
        dtype=template["terminal_reasons"].dtype,
    )

    for episode_index, episode_id in enumerate(episode_ids):
        frame_start = int(frame_offsets[episode_index])
        frame_stop = int(frame_offsets[episode_index + 1])
        action_start = int(transition_offsets[episode_index])
        action_stop = int(transition_offsets[episode_index + 1])
        for local_frame, global_frame in enumerate(range(frame_start, frame_stop)):
            frames[global_frame].fill(episode_index * 23 + local_frame)
            states[global_frame] = [episode_id, local_frame, 0.0, 0.0]
        for local_step, global_step in enumerate(range(action_start, action_stop)):
            actions[global_step] = [float(episode_id), float(local_step)]
        dones[action_stop - 1] = True
        terminal_reasons[action_stop - 1] = "time_limit"

    dataset = {name: values.copy() for name, values in template.items()}
    dataset.update(
        {
            "frames": frames,
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "terminal_reasons": terminal_reasons,
            "episode_ids": episode_ids,
            "frame_offsets": frame_offsets,
            "transition_offsets": transition_offsets,
        }
    )
    validate_visual_dataset(dataset)
    return dataset


class VisualWindowIndexTest(unittest.TestCase):
    def test_index_follows_selected_episode_and_step_order(self):
        dataset = make_visual_dataset((5, 4, 2))

        index = build_visual_window_index(
            dataset,
            np.asarray([11, 10, 12], dtype=np.int64),
        )

        np.testing.assert_array_equal(index.episode_indices, [1, 0, 0])
        np.testing.assert_array_equal(index.step_ids, [3, 3, 4])
        np.testing.assert_array_equal(index.selected_episode_ids, [11, 10, 12])
        np.testing.assert_array_equal(index.eligible_episode_ids, [11, 10])
        np.testing.assert_array_equal(index.skipped_episode_ids, [12])
        self.assertEqual(index.count, 3)

    def test_all_short_episodes_form_a_valid_empty_index(self):
        dataset = make_visual_dataset((1, 3, 2))

        index = build_visual_window_index(dataset, dataset["episode_ids"])

        self.assertEqual(index.count, 0)
        self.assertEqual(index.episode_indices.dtype, np.dtype(np.int64))
        self.assertEqual(index.step_ids.dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(index.eligible_episode_ids, [])
        np.testing.assert_array_equal(index.skipped_episode_ids, [10, 11, 12])

    def test_selected_episode_ids_are_strictly_validated(self):
        dataset = make_visual_dataset((4, 4, 4))
        invalid_cases = (
            (np.asarray([], dtype=np.int64), "must be non-empty"),
            (np.asarray([[10]], dtype=np.int64), "one-dimensional"),
            (np.asarray([True]), "integer array"),
            (np.asarray([10.0]), "integer array"),
            (np.asarray([10, 10]), "must not contain duplicates"),
            (np.asarray([99]), "missing from the visual dataset"),
        )
        for selected, message in invalid_cases:
            with self.subTest(selected=selected, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_visual_window_index(dataset, selected)


class VisualWindowDatasetTest(unittest.TestCase):
    def setUp(self):
        self.visual = make_visual_dataset((5, 4, 2))
        self.windows = build_visual_window_dataset(
            self.visual,
            np.asarray([10, 11, 12], dtype=np.int64),
        )

    def test_first_and_last_samples_have_exact_temporal_alignment(self):
        first = self.windows[0]
        np.testing.assert_array_equal(
            first["context_frames"],
            self.visual["frames"][0:4],
        )
        np.testing.assert_array_equal(
            first["history_actions"],
            self.visual["actions"][0:3],
        )
        np.testing.assert_array_equal(
            first["current_action"],
            self.visual["actions"][3],
        )
        np.testing.assert_array_equal(
            first["target_frame"],
            self.visual["frames"][4],
        )
        self.assertEqual((first["episode_id"], first["step_id"]), (10, 3))

        last_of_first_episode = self.windows[1]
        np.testing.assert_array_equal(
            last_of_first_episode["target_frame"],
            self.visual["frames"][5],
        )
        self.assertEqual(last_of_first_episode["step_id"], 4)

    def test_next_episode_uses_its_own_offsets(self):
        sample = self.windows[2]
        frame_start = int(self.visual["frame_offsets"][1])
        action_start = int(self.visual["transition_offsets"][1])
        np.testing.assert_array_equal(
            sample["context_frames"],
            self.visual["frames"][frame_start : frame_start + 4],
        )
        np.testing.assert_array_equal(
            sample["history_actions"],
            self.visual["actions"][action_start : action_start + 3],
        )
        self.assertEqual((sample["episode_id"], sample["step_id"]), (11, 3))

    def test_sample_contract_and_copy_isolation(self):
        sample = self.windows[0]
        self.assertEqual(
            set(sample),
            {
                "context_frames",
                "history_actions",
                "current_action",
                "target_frame",
                "episode_id",
                "step_id",
            },
        )
        self.assertEqual(sample["context_frames"].shape, (4, 64, 64, 3))
        self.assertEqual(sample["context_frames"].dtype, np.dtype(np.uint8))
        self.assertEqual(sample["history_actions"].shape, (3, 2))
        self.assertEqual(sample["history_actions"].dtype, np.dtype(np.float64))
        self.assertEqual(sample["current_action"].shape, (2,))
        self.assertEqual(sample["target_frame"].shape, (64, 64, 3))
        self.assertIs(type(sample["episode_id"]), int)
        self.assertIs(type(sample["step_id"]), int)

        original_pixel = int(self.visual["frames"][0, 0, 0, 0])
        original_action = float(self.visual["actions"][0, 0])
        sample["context_frames"][0, 0, 0, 0] = 255
        sample["history_actions"][0, 0] = -999.0
        self.assertEqual(int(self.visual["frames"][0, 0, 0, 0]), original_pixel)
        self.assertEqual(float(self.visual["actions"][0, 0]), original_action)

    def test_scalar_index_rules_match_python_sequences(self):
        self.assertEqual(self.windows[-1]["episode_id"], 11)
        self.assertEqual(self.windows[np.int64(0)]["step_id"], 3)
        for invalid in (
            slice(None),
            np.asarray([0]),
            0.5,
            True,
            np.bool_(False),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    self.windows[invalid]
        for invalid in (len(self.windows), -len(self.windows) - 1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(IndexError):
                    self.windows[invalid]


class VisualWindowSplitTest(unittest.TestCase):
    def test_splits_are_episode_disjoint_exhaustive_and_windowed_afterward(self):
        dataset = make_visual_dataset((4,) * 10)

        splits = build_visual_window_splits(dataset, seed=19)

        self.assertEqual(set(splits), {"train", "validation", "test"})
        selected = {
            name: set(window.index.selected_episode_ids.tolist())
            for name, window in splits.items()
        }
        self.assertEqual(
            selected["train"] | selected["validation"] | selected["test"],
            set(dataset["episode_ids"].tolist()),
        )
        self.assertFalse(selected["train"] & selected["validation"])
        self.assertFalse(selected["train"] & selected["test"])
        self.assertFalse(selected["validation"] & selected["test"])
        self.assertEqual(
            {name: len(window) for name, window in splits.items()},
            {"train": 8, "validation": 1, "test": 1},
        )

    def test_fixed_seed_repeats_split_ids_and_window_order(self):
        dataset = make_visual_dataset((5,) * 10)

        first = build_visual_window_splits(dataset, seed=23)
        second = build_visual_window_splits(dataset, seed=23)

        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(
                first[name].index.selected_episode_ids,
                second[name].index.selected_episode_ids,
            )
            np.testing.assert_array_equal(
                first[name].index.episode_indices,
                second[name].index.episode_indices,
            )
            np.testing.assert_array_equal(
                first[name].index.step_ids,
                second[name].index.step_ids,
            )


class VisualWindowDocumentationTest(unittest.TestCase):
    def test_readme_documents_lazy_split_before_window_usage(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("build_visual_window_splits", readme)
        self.assertIn('sample["context_frames"]', readme)
        self.assertIn('sample["history_actions"]', readme)
        self.assertIn('sample["current_action"]', readme)
        self.assertIn('sample["target_frame"]', readme)
        self.assertIn("先按完整 episode", readme)
        self.assertIn("uint8 NHWC", readme)


if __name__ == "__main__":
    unittest.main()
