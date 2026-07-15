import math
import unittest
from pathlib import Path

import numpy as np

from world_model_lab.car_env import CarEnv
from world_model_lab.visual_observation import (
    BOUNDARY_COLOR,
    CAR_COLOR,
    GOAL_COLOR,
    HEADING_COLOR,
    IMAGE_SIZE,
    OBSTACLE_COLOR,
    build_pixel_transform,
    render_observation,
    scene_from_env,
    world_to_pixel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VisualObservationTest(unittest.TestCase):
    def setUp(self):
        self.env = CarEnv()
        self.scene = scene_from_env(self.env)

    def test_pillow_is_an_explicit_dependency(self):
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"Pillow>=12.0"', pyproject)

    def test_default_transform_preserves_aspect_ratio_and_centres_letterbox(self):
        transform = build_pixel_transform(self.scene)

        self.assertEqual(IMAGE_SIZE, 64)
        self.assertAlmostEqual(transform.scale, 6.3)
        self.assertAlmostEqual(transform.offset_x, 0.0)
        self.assertAlmostEqual(transform.offset_y, 6.3)
        self.assertEqual(
            world_to_pixel((0.0, 8.0), scene=self.scene),
            (0, 6),
        )
        self.assertEqual(
            world_to_pixel((10.0, 0.0), scene=self.scene),
            (63, 57),
        )
        self.assertEqual(
            world_to_pixel((5.0, 4.0), scene=self.scene),
            (32, 32),
        )
        self.assertEqual(
            world_to_pixel((9.0, 7.0), scene=self.scene),
            (57, 13),
        )

    def test_world_axes_map_rightward_and_upward(self):
        left = world_to_pixel((2.0, 2.0), scene=self.scene)
        right = world_to_pixel((3.0, 2.0), scene=self.scene)
        low = world_to_pixel((2.0, 2.0), scene=self.scene)
        high = world_to_pixel((2.0, 3.0), scene=self.scene)

        self.assertGreater(right[0], left[0])
        self.assertLess(high[1], low[1])

    def test_render_returns_exact_rgb_shape_and_is_repeatable(self):
        state = np.asarray([2.0, 2.0, 0.25, 1.5], dtype=np.float64)

        first = render_observation(state, scene=self.scene)
        second = render_observation(state, scene=self.scene)

        self.assertEqual(first.shape, (64, 64, 3))
        self.assertEqual(first.dtype, np.uint8)
        np.testing.assert_array_equal(first, second)

    def test_default_scene_layers_occupy_expected_pixels(self):
        state = np.asarray([2.0, 2.0, 0.0, 1.0], dtype=np.float64)
        frame = render_observation(state, scene=self.scene)
        obstacle = world_to_pixel(self.scene.obstacle, scene=self.scene)
        goal = world_to_pixel(self.scene.goal, scene=self.scene)
        car = world_to_pixel(state[:2], scene=self.scene)

        self.assertEqual(tuple(frame[6, 0]), BOUNDARY_COLOR)
        self.assertEqual(tuple(frame[obstacle[1], obstacle[0]]), OBSTACLE_COLOR)
        self.assertEqual(tuple(frame[goal[1], goal[0]]), GOAL_COLOR)
        self.assertEqual(tuple(frame[car[1], car[0] - 1]), CAR_COLOR)
        self.assertEqual(tuple(frame[car[1], car[0] + 5]), HEADING_COLOR)

    def test_heading_rotates_marker_without_moving_car(self):
        east = render_observation((2.0, 2.0, 0.0, 1.0), scene=self.scene)
        north = render_observation(
            (2.0, 2.0, math.pi / 2.0, 1.0),
            scene=self.scene,
        )
        car = world_to_pixel((2.0, 2.0), scene=self.scene)

        self.assertEqual(tuple(east[car[1], car[0] + 5]), HEADING_COLOR)
        self.assertEqual(tuple(north[car[1] - 5, car[0]]), HEADING_COLOR)
        self.assertEqual(tuple(east[car[1], car[0] - 1]), CAR_COLOR)
        self.assertEqual(tuple(north[car[1], car[0] - 1]), CAR_COLOR)

    def test_velocity_only_change_is_pixel_identical(self):
        stopped = render_observation((2.0, 2.0, 0.3, 0.0), scene=self.scene)
        moving = render_observation((2.0, 2.0, 0.3, 3.0), scene=self.scene)

        np.testing.assert_array_equal(stopped, moving)

    def test_world_space_obstacle_is_circular_in_pixels(self):
        frame = render_observation((1.0, 1.0, 0.0, 0.0), scene=self.scene)
        mask = np.all(frame == np.asarray(OBSTACLE_COLOR), axis=2)
        rows, columns = np.where(mask)

        self.assertLessEqual(
            abs((columns.max() - columns.min()) - (rows.max() - rows.min())),
            1,
        )

    def test_near_boundary_car_is_clipped_without_state_clamping(self):
        state = np.asarray([-0.1, 1.0, 0.0, 1.0], dtype=np.float64)
        original = state.copy()

        frame = render_observation(state, scene=self.scene)

        np.testing.assert_array_equal(state, original)
        self.assertTrue(np.any(np.all(frame == np.asarray(CAR_COLOR), axis=2)))

    def test_large_finite_offscreen_state_does_not_overflow(self):
        frame = render_observation(
            (np.finfo(np.float64).max, 1.0, 0.0, 1.0),
            scene=self.scene,
        )

        self.assertEqual(frame.shape, (64, 64, 3))
        self.assertFalse(np.any(np.all(frame == np.asarray(CAR_COLOR), axis=2)))
        self.assertFalse(np.any(np.all(frame == np.asarray(HEADING_COLOR), axis=2)))

    def test_invalid_state_shape_and_non_finite_state_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"state must have shape \[4\]"):
            render_observation((1.0, 2.0, 0.0), scene=self.scene)
        with self.assertRaisesRegex(ValueError, "state.*finite"):
            render_observation(
                (1.0, 2.0, np.nan, 0.0),
                scene=self.scene,
            )

    def test_rendering_and_scene_snapshot_do_not_mutate_environment(self):
        self.env.step(0.0, 0.0)
        before_state = self.env.state
        before_trajectory = self.env.trajectory
        before_steps = self.env.steps
        before_done = self.env.done

        scene = scene_from_env(self.env)
        render_observation((3.0, 2.0, 0.4, 2.0), scene=scene)

        np.testing.assert_array_equal(self.env.state, before_state)
        self.assertEqual(self.env.steps, before_steps)
        self.assertEqual(self.env.done, before_done)
        self.assertEqual(len(self.env.trajectory), len(before_trajectory))
        for actual, expected in zip(self.env.trajectory, before_trajectory, strict=True):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
