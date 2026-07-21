from __future__ import annotations

import unittest

import numpy as np
import torch

from tests.test_visual_windows import make_visual_dataset
from world_model_lab.visual_latent_data import (
    LatentRolloutArrays,
    LatentWindowArrays,
    VisualFrameDataset,
    VisualMotionFrameDataset,
    VisualObjectFrameDataset,
    build_latent_rollout_arrays,
    build_latent_window_arrays,
    encode_all_frames,
    fit_safe_normalizer,
    frame_indices_for_episode_ids,
    frames_to_tensor,
    renderer_object_masks,
    transition_indices_for_episode_ids,
)
from world_model_lab.visual_latent_model import (
    ConvAutoencoder,
    SpatialConvAutoencoder,
)
from world_model_lab.visual_observation import CAR_COLOR, HEADING_COLOR
from world_model_lab.visual_windows import build_visual_window_index


class VisualFrameAdapterTest(unittest.TestCase):
    def setUp(self):
        self.visual = make_visual_dataset((5, 4, 2))

    def test_frames_to_tensor_preserves_values_and_reorders_channels(self):
        frames = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        frames[0, 1, 2] = [0, 127, 255]

        tensor = frames_to_tensor(frames)
        single = frames_to_tensor(frames[0])

        self.assertEqual(tuple(tensor.shape), (2, 3, 64, 64))
        self.assertEqual(tuple(single.shape), (3, 64, 64))
        self.assertEqual(tensor.dtype, torch.float32)
        torch.testing.assert_close(
            tensor[0, :, 1, 2],
            torch.tensor([0.0, 127.0 / 255.0, 1.0]),
        )
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_frames_to_tensor_rejects_wrong_dtype_or_shape(self):
        invalid = (
            np.zeros((2, 64, 64, 3), dtype=np.float32),
            np.zeros((2, 32, 32, 3), dtype=np.uint8),
            np.zeros((2, 64, 64, 1), dtype=np.uint8),
            np.zeros((64, 64), dtype=np.uint8),
        )
        for values in invalid:
            with self.subTest(shape=values.shape, dtype=values.dtype):
                with self.assertRaises(ValueError):
                    frames_to_tensor(values)

    def test_frame_and_transition_indices_follow_selected_episode_order(self):
        frame_indices = frame_indices_for_episode_ids(
            self.visual,
            np.asarray([11, 10], dtype=np.int64),
        )
        transition_indices = transition_indices_for_episode_ids(
            self.visual,
            np.asarray([11, 10], dtype=np.int64),
        )

        np.testing.assert_array_equal(
            frame_indices,
            np.concatenate((np.arange(6, 11), np.arange(0, 6))),
        )
        np.testing.assert_array_equal(
            transition_indices,
            np.concatenate((np.arange(5, 9), np.arange(0, 5))),
        )

    def test_selected_episode_ids_are_validated(self):
        cases = (
            np.asarray([], dtype=np.int64),
            np.asarray([[10]], dtype=np.int64),
            np.asarray([True]),
            np.asarray([10.0]),
            np.asarray([10, 10]),
            np.asarray([99]),
        )
        for selected in cases:
            with self.subTest(selected=selected):
                with self.assertRaises(ValueError):
                    frame_indices_for_episode_ids(self.visual, selected)

    def test_frame_dataset_returns_owned_float_tensor_and_python_indexing(self):
        dataset = VisualFrameDataset(
            self.visual,
            np.asarray([11, 10], dtype=np.int64),
        )

        first = dataset[0]
        last = dataset[-1]

        self.assertEqual(len(dataset), 11)
        self.assertEqual(tuple(first.shape), (3, 64, 64))
        self.assertEqual(first.dtype, torch.float32)
        torch.testing.assert_close(
            first,
            frames_to_tensor(self.visual["frames"][6]),
        )
        torch.testing.assert_close(
            last,
            frames_to_tensor(self.visual["frames"][5]),
        )
        before = int(self.visual["frames"][6, 0, 0, 0])
        first[0, 0, 0] = 1.0
        self.assertEqual(int(self.visual["frames"][6, 0, 0, 0]), before)
        for invalid in (slice(None), np.asarray([0]), 0.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    dataset[invalid]

    def test_motion_frame_dataset_is_episode_local_and_spatial(self):
        second_episode_start = int(self.visual["frame_offsets"][1])
        self.visual["frames"][second_episode_start + 1] = self.visual[
            "frames"
        ][second_episode_start]
        self.visual["frames"][second_episode_start + 1, 2, 3, 1] += 1
        dataset = VisualMotionFrameDataset(
            self.visual,
            np.asarray([11, 10], dtype=np.int64),
        )

        first_image, first_mask = dataset[0]
        second_image, second_mask = dataset[1]
        first_of_next_episode = dataset[5]

        self.assertEqual(len(dataset), 11)
        self.assertEqual(tuple(first_image.shape), (3, 64, 64))
        self.assertEqual(tuple(first_mask.shape), (1, 64, 64))
        self.assertEqual(first_mask.dtype, torch.float32)
        self.assertEqual(int(torch.count_nonzero(first_mask)), 0)
        self.assertEqual(int(torch.count_nonzero(second_mask)), 1)
        self.assertEqual(float(second_mask[0, 2, 3]), 1.0)
        self.assertEqual(
            int(torch.count_nonzero(first_of_next_episode[1])),
            0,
        )
        torch.testing.assert_close(
            second_image,
            frames_to_tensor(
                self.visual["frames"][second_episode_start + 1]
            ),
        )

        second_mask[0, 2, 3] = 0.0
        self.assertEqual(
            int(self.visual["frames"][second_episode_start + 1, 2, 3, 1]),
            int(self.visual["frames"][second_episode_start, 2, 3, 1]) + 1,
        )
        for invalid in (slice(None), np.asarray([0]), 0.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    dataset[invalid]

    def test_renderer_object_masks_match_exact_car_and_heading_colours(self):
        frames = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        frames[0, 2, 3] = CAR_COLOR
        frames[0, 4, 5] = HEADING_COLOR
        frames[0, 6, 7] = np.asarray(CAR_COLOR) + [0, 0, 1]

        masks = renderer_object_masks(frames)
        single = renderer_object_masks(frames[0])

        self.assertEqual(tuple(masks.shape), (2, 1, 64, 64))
        self.assertEqual(tuple(single.shape), (1, 64, 64))
        self.assertEqual(masks.dtype, torch.float32)
        self.assertEqual(int(torch.count_nonzero(masks[0])), 2)
        self.assertEqual(float(masks[0, 0, 2, 3]), 1.0)
        self.assertEqual(float(masks[0, 0, 4, 5]), 1.0)
        self.assertEqual(float(masks[0, 0, 6, 7]), 0.0)
        self.assertEqual(int(torch.count_nonzero(masks[1])), 0)

    def test_object_frame_dataset_preserves_order_and_returns_owned_tensors(
        self,
    ):
        first_selected_index = int(self.visual["frame_offsets"][1])
        self.visual["frames"][first_selected_index, 2, 3] = CAR_COLOR
        self.visual["frames"][first_selected_index, 4, 5] = HEADING_COLOR
        dataset = VisualObjectFrameDataset(
            self.visual,
            np.asarray([11, 10], dtype=np.int64),
        )

        image, mask = dataset[0]

        self.assertEqual(len(dataset), 11)
        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertEqual(tuple(mask.shape), (1, 64, 64))
        self.assertEqual(int(torch.count_nonzero(mask)), 2)
        torch.testing.assert_close(
            image,
            frames_to_tensor(self.visual["frames"][first_selected_index]),
        )
        image[:, 2, 3] = 0.0
        mask[:, 2, 3] = 0.0
        self.assertEqual(
            tuple(self.visual["frames"][first_selected_index, 2, 3]),
            CAR_COLOR,
        )
        for invalid in (slice(None), np.asarray([0]), 0.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    dataset[invalid]


class VisualLatentArrayTest(unittest.TestCase):
    def setUp(self):
        self.visual = make_visual_dataset((5, 4, 2))

    def test_safe_normalizer_round_trips_and_handles_constant_features(self):
        values = np.asarray([[1.0, 2.0], [1.0, 4.0]], dtype=np.float64)

        normalizer = fit_safe_normalizer(values)
        normalized = normalizer.normalize(values)

        np.testing.assert_array_equal(normalizer.mean, [1.0, 3.0])
        np.testing.assert_array_equal(normalizer.std, [1.0, 1.0])
        self.assertTrue(np.all(np.isfinite(normalized)))
        np.testing.assert_allclose(normalizer.denormalize(normalized), values)

    def test_safe_normalizer_rejects_invalid_values(self):
        cases = (
            np.empty((0, 2)),
            np.zeros((2, 2, 1)),
            np.asarray([[np.nan, 1.0]]),
        )
        for values in cases:
            with self.subTest(shape=values.shape):
                with self.assertRaises(ValueError):
                    fit_safe_normalizer(values)
        with self.assertRaises(ValueError):
            fit_safe_normalizer(np.zeros((2, 2)), minimum_std=0.0)

    def test_latent_windows_follow_exact_frame_and_action_offsets(self):
        latent_frames = np.arange(
            self.visual["frames"].shape[0] * 3,
            dtype=np.float32,
        ).reshape(-1, 3)
        index = build_visual_window_index(
            self.visual,
            np.asarray([10, 11], dtype=np.int64),
        )

        arrays = build_latent_window_arrays(
            self.visual,
            index,
            latent_frames,
        )

        self.assertIsInstance(arrays, LatentWindowArrays)
        self.assertEqual(arrays.count, 3)
        np.testing.assert_array_equal(
            arrays.context_latents[0],
            latent_frames[0:4],
        )
        np.testing.assert_array_equal(
            arrays.target_latents[0],
            latent_frames[4],
        )
        np.testing.assert_array_equal(
            arrays.history_actions[0],
            self.visual["actions"][0:3],
        )
        np.testing.assert_array_equal(
            arrays.current_actions[0],
            self.visual["actions"][3],
        )
        self.assertEqual(
            (
                int(arrays.last_frame_indices[0]),
                int(arrays.target_frame_indices[0]),
                int(arrays.episode_ids[0]),
                int(arrays.step_ids[0]),
            ),
            (3, 4, 10, 3),
        )

        second_episode = arrays.count - 1
        frame_start = int(self.visual["frame_offsets"][1])
        action_start = int(self.visual["transition_offsets"][1])
        np.testing.assert_array_equal(
            arrays.context_latents[second_episode],
            latent_frames[frame_start : frame_start + 4],
        )
        np.testing.assert_array_equal(
            arrays.history_actions[second_episode],
            self.visual["actions"][action_start : action_start + 3],
        )
        self.assertEqual(int(arrays.episode_ids[second_episode]), 11)

    def test_latent_windows_reject_wrong_or_non_finite_latents(self):
        index = build_visual_window_index(
            self.visual,
            np.asarray([10, 11], dtype=np.int64),
        )
        count = self.visual["frames"].shape[0]
        invalid = (
            np.zeros((count - 1, 3), dtype=np.float32),
            np.zeros((count, 3, 1), dtype=np.float32),
            np.full((count, 3), np.nan, dtype=np.float32),
        )
        for latent_frames in invalid:
            with self.subTest(shape=latent_frames.shape):
                with self.assertRaises(ValueError):
                    build_latent_window_arrays(
                        self.visual,
                        index,
                        latent_frames,
                    )

    def test_encode_all_frames_returns_one_finite_latent_per_frame(self):
        model = ConvAutoencoder(latent_dim=5, base_channels=2)

        latents = encode_all_frames(
            model,
            self.visual["frames"],
            batch_size=4,
        )

        self.assertEqual(
            latents.shape,
            (self.visual["frames"].shape[0], 5),
        )
        self.assertEqual(latents.dtype, np.dtype(np.float32))
        self.assertTrue(np.all(np.isfinite(latents)))
        self.assertFalse(model.training)

    def test_encode_all_frames_reversibly_flattens_a_spatial_grid(self):
        model = SpatialConvAutoencoder(
            latent_channels=3,
            base_channels=2,
        )
        frames = self.visual["frames"][:2]

        latents = encode_all_frames(model, frames, batch_size=2)
        with torch.no_grad():
            expected = model.encode(frames_to_tensor(frames)).flatten(
                start_dim=1
            )

        self.assertEqual(latents.shape, (2, 3 * 8 * 8))
        np.testing.assert_allclose(latents, expected.cpu().numpy())


class VisualLatentRolloutArrayTest(unittest.TestCase):
    def setUp(self):
        self.visual = make_visual_dataset((10, 8, 7))
        self.latent_frames = np.arange(
            self.visual["frames"].shape[0] * 3,
            dtype=np.float32,
        ).reshape(-1, 3)

    def test_rollouts_follow_exact_episode_frame_and_action_offsets(self):
        arrays = build_latent_rollout_arrays(
            self.visual,
            np.asarray([10, 11, 12], dtype=np.int64),
            self.latent_frames,
            horizon=5,
        )

        self.assertIsInstance(arrays, LatentRolloutArrays)
        self.assertEqual(arrays.count, 4)
        self.assertEqual(arrays.horizon, 5)
        self.assertEqual(arrays.context_latents.shape, (4, 4, 3))
        self.assertEqual(arrays.history_actions.shape, (4, 3, 2))
        self.assertEqual(arrays.rollout_actions.shape, (4, 5, 2))
        self.assertEqual(arrays.target_latents.shape, (4, 5, 3))
        self.assertEqual(arrays.target_frame_indices.shape, (4, 5))
        np.testing.assert_array_equal(arrays.episode_ids, [10, 10, 10, 11])
        np.testing.assert_array_equal(arrays.start_step_ids, [3, 4, 5, 3])

        np.testing.assert_array_equal(
            arrays.context_latents[0],
            self.latent_frames[0:4],
        )
        np.testing.assert_array_equal(
            arrays.history_actions[0],
            self.visual["actions"][0:3],
        )
        np.testing.assert_array_equal(
            arrays.rollout_actions[0],
            self.visual["actions"][3:8],
        )
        np.testing.assert_array_equal(
            arrays.target_latents[0],
            self.latent_frames[4:9],
        )
        np.testing.assert_array_equal(
            arrays.target_frame_indices[0],
            np.arange(4, 9, dtype=np.int64),
        )
        self.assertEqual(int(arrays.initial_frame_indices[0]), 3)

        second_episode_index = 3
        frame_start = int(self.visual["frame_offsets"][1])
        action_start = int(self.visual["transition_offsets"][1])
        np.testing.assert_array_equal(
            arrays.context_latents[second_episode_index],
            self.latent_frames[frame_start : frame_start + 4],
        )
        np.testing.assert_array_equal(
            arrays.rollout_actions[second_episode_index],
            self.visual["actions"][action_start + 3 : action_start + 8],
        )
        for values in arrays.__dict__.values():
            self.assertFalse(np.asarray(values).flags.writeable)

    def test_rollouts_reject_invalid_protocol_or_latents(self):
        count = self.visual["frames"].shape[0]
        cases = (
            {
                "selected_episode_ids": np.asarray([], dtype=np.int64),
            },
            {
                "selected_episode_ids": np.asarray([[10]], dtype=np.int64),
            },
            {
                "selected_episode_ids": np.asarray([10.0]),
            },
            {
                "selected_episode_ids": np.asarray([10, 10], dtype=np.int64),
            },
            {
                "selected_episode_ids": np.asarray([99], dtype=np.int64),
            },
            {"horizon": 0},
            {
                "latent_frames": np.zeros(
                    (count - 1, 3),
                    dtype=np.float32,
                ),
            },
            {
                "latent_frames": np.full(
                    (count, 3),
                    np.nan,
                    dtype=np.float32,
                ),
            },
        )
        for update in cases:
            with self.subTest(update=update):
                kwargs = {
                    "dataset": self.visual,
                    "selected_episode_ids": np.asarray(
                        [10, 11, 12],
                        dtype=np.int64,
                    ),
                    "latent_frames": self.latent_frames,
                    "horizon": 5,
                }
                kwargs.update(update)
                with self.assertRaises(ValueError):
                    build_latent_rollout_arrays(**kwargs)

        with self.assertRaisesRegex(ValueError, "long enough"):
            build_latent_rollout_arrays(
                self.visual,
                np.asarray([12], dtype=np.int64),
                self.latent_frames,
                horizon=5,
            )


if __name__ == "__main__":
    unittest.main()
