from __future__ import annotations

import unittest

import numpy as np

from tests.test_visual_windows import make_visual_dataset
from world_model_lab.vjepa_probe import (
    build_probe_clip_batch,
    select_evenly_spaced_positions,
    state_probe_metrics,
    state_to_probe_targets,
)
from world_model_lab.visual_windows import build_visual_window_index


class ProbeClipBatchTest(unittest.TestCase):
    def setUp(self):
        self.visual = make_visual_dataset((5, 4, 2))
        self.index = build_visual_window_index(
            self.visual,
            np.asarray([10, 11, 12], dtype=np.int64),
        )

    def test_recorded_clips_follow_exact_window_frame_and_state_offsets(self):
        positions = np.asarray([0, 2], dtype=np.int64)

        batch = build_probe_clip_batch(
            self.visual,
            self.index,
            positions,
            order="recorded",
        )

        self.assertEqual(batch.frames.shape, (2, 4, 64, 64, 3))
        self.assertEqual(batch.frames.dtype, np.uint8)
        self.assertEqual(batch.states.shape, (2, 4))
        self.assertEqual(batch.states.dtype, np.float64)
        np.testing.assert_array_equal(
            batch.frames[0],
            self.visual["frames"][0:4],
        )
        second_start = int(self.visual["frame_offsets"][1])
        np.testing.assert_array_equal(
            batch.frames[1],
            self.visual["frames"][second_start : second_start + 4],
        )
        np.testing.assert_array_equal(
            batch.states,
            self.visual["states"][[3, second_start + 3]],
        )
        np.testing.assert_array_equal(batch.episode_ids, [10, 11])
        np.testing.assert_array_equal(batch.step_ids, [3, 3])
        with self.assertRaises(ValueError):
            batch.frames[0, 0, 0, 0, 0] = 1

    def test_temporal_ablations_change_frames_but_keep_recorded_targets(self):
        position = np.asarray([0], dtype=np.int64)
        recorded = build_probe_clip_batch(
            self.visual,
            self.index,
            position,
            order="recorded",
        )
        reversed_batch = build_probe_clip_batch(
            self.visual,
            self.index,
            position,
            order="reversed",
        )
        repeated = build_probe_clip_batch(
            self.visual,
            self.index,
            position,
            order="repeat_last",
        )

        np.testing.assert_array_equal(
            reversed_batch.frames[0],
            recorded.frames[0, ::-1],
        )
        np.testing.assert_array_equal(
            repeated.frames[0],
            np.repeat(recorded.frames[0, -1:], 4, axis=0),
        )
        np.testing.assert_array_equal(reversed_batch.states, recorded.states)
        np.testing.assert_array_equal(repeated.states, recorded.states)

    def test_invalid_positions_and_orders_are_rejected(self):
        invalid_positions = (
            np.asarray([], dtype=np.int64),
            np.asarray([[0]], dtype=np.int64),
            np.asarray([0.0]),
            np.asarray([True]),
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([-1], dtype=np.int64),
            np.asarray([self.index.count], dtype=np.int64),
        )
        for positions in invalid_positions:
            with self.subTest(positions=positions):
                with self.assertRaises(ValueError):
                    build_probe_clip_batch(
                        self.visual,
                        self.index,
                        positions,
                        order="recorded",
                    )
        with self.assertRaises(ValueError):
            build_probe_clip_batch(
                self.visual,
                self.index,
                np.asarray([0], dtype=np.int64),
                order="shuffled",
            )


class ProbeSelectionTest(unittest.TestCase):
    def test_even_spacing_is_deterministic_unique_and_includes_boundaries(self):
        first = select_evenly_spaced_positions(10, limit=4)
        second = select_evenly_spaced_positions(10, limit=4)

        np.testing.assert_array_equal(first, [0, 3, 6, 9])
        np.testing.assert_array_equal(second, first)
        np.testing.assert_array_equal(
            select_evenly_spaced_positions(4, limit=8),
            [0, 1, 2, 3],
        )

    def test_even_spacing_rejects_non_positive_counts_or_limits(self):
        for count, limit in ((0, 1), (-1, 1), (1, 0), (1, -1)):
            with self.subTest(count=count, limit=limit):
                with self.assertRaises(ValueError):
                    select_evenly_spaced_positions(count, limit=limit)


class StateProbeMetricTest(unittest.TestCase):
    def test_state_targets_encode_heading_without_angle_discontinuity(self):
        states = np.asarray(
            [
                [1.0, 2.0, 0.0, 0.5],
                [3.0, 4.0, np.pi / 2.0, 1.5],
            ],
            dtype=np.float64,
        )

        targets = state_to_probe_targets(states)

        np.testing.assert_allclose(
            targets,
            [
                [1.0, 2.0, 0.0, 1.0, 0.5],
                [3.0, 4.0, 1.0, 0.0, 1.5],
            ],
            rtol=0.0,
            atol=1e-7,
        )
        self.assertEqual(targets.dtype, np.float32)

    def test_metrics_restore_pixels_circular_heading_and_velocity(self):
        target = np.asarray(
            [[5.0, 4.0, 0.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        predicted = np.asarray(
            [[6.0, 4.0, 1.0, 0.0, 1.5]],
            dtype=np.float32,
        )

        metrics = state_probe_metrics(
            predicted,
            target,
            world_bounds=np.asarray([0.0, 10.0, 0.0, 8.0]),
        )

        self.assertEqual(metrics["samples"], 1)
        self.assertAlmostEqual(metrics["mean_centre_error_pixels"], 6.3)
        self.assertAlmostEqual(metrics["p95_centre_error_pixels"], 6.3)
        self.assertAlmostEqual(metrics["mean_heading_error_degrees"], 90.0)
        self.assertAlmostEqual(metrics["p95_heading_error_degrees"], 90.0)
        self.assertAlmostEqual(metrics["mean_velocity_error"], 0.5)
        self.assertAlmostEqual(metrics["p95_velocity_error"], 0.5)

    def test_state_and_metric_validation_fail_closed(self):
        with self.assertRaises(ValueError):
            state_to_probe_targets(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            state_to_probe_targets(
                np.asarray([[0.0, 0.0, np.nan, 0.0]])
            )
        target = np.zeros((1, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            state_probe_metrics(
                np.zeros((2, 5), dtype=np.float32),
                target,
                world_bounds=np.asarray([0.0, 10.0, 0.0, 8.0]),
            )
        with self.assertRaises(ValueError):
            state_probe_metrics(
                target,
                target,
                world_bounds=np.asarray([0.0, 0.0, 0.0, 8.0]),
            )


if __name__ == "__main__":
    unittest.main()
