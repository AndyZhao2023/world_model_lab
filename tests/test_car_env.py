import math
import unittest

import numpy as np

from world_model_lab.car_env import CarEnv


class CarEnvTest(unittest.TestCase):
    def test_reset_restores_initial_state_and_history(self):
        env = CarEnv(initial_state=(1.0, 2.0, 0.25, 1.0))
        env.step(steering=0.0, acceleration=0.0)

        state = env.reset()

        np.testing.assert_allclose(state, [1.0, 2.0, 0.25, 1.0])
        self.assertEqual(len(env.trajectory), 1)
        np.testing.assert_allclose(env.trajectory[0], state)

    def test_zero_steering_moves_straight(self):
        env = CarEnv(initial_state=(1.0, 1.0, 0.0, 2.0), dt=0.5)

        state, _, done, _ = env.step(steering=0.0, acceleration=0.0)

        np.testing.assert_allclose(state, [2.0, 1.0, 0.0, 2.0])
        self.assertFalse(done)

    def test_positive_steering_increases_heading(self):
        env = CarEnv(initial_state=(1.0, 1.0, 0.0, 2.0), dt=0.5)

        state, _, _, _ = env.step(steering=0.2, acceleration=0.0)

        self.assertGreater(state[2], 0.0)

    def test_action_and_velocity_are_clipped(self):
        env = CarEnv(
            initial_state=(1.0, 1.0, 0.0, 1.9),
            dt=1.0,
            max_speed=2.0,
            max_steering=0.3,
            max_acceleration=0.5,
        )

        state, _, _, info = env.step(steering=10.0, acceleration=10.0)

        self.assertAlmostEqual(state[3], 2.0)
        self.assertAlmostEqual(info["applied_steering"], 0.3)
        self.assertAlmostEqual(info["applied_acceleration"], 0.5)

    def test_goal_terminates_with_bonus(self):
        env = CarEnv(
            initial_state=(8.6, 7.0, 0.0, 1.0),
            dt=0.4,
            goal=(9.0, 7.0),
            goal_radius=0.5,
        )

        _, reward, done, info = env.step(steering=0.0, acceleration=0.0)

        self.assertTrue(done)
        self.assertTrue(info["reached_goal"])
        self.assertGreater(reward, 9.0)

    def test_obstacle_collision_terminates_with_penalty(self):
        env = CarEnv(
            initial_state=(3.8, 4.0, 0.0, 1.0),
            dt=0.2,
            obstacle=(5.0, 4.0),
            obstacle_radius=1.0,
            car_radius=0.2,
        )

        _, reward, done, info = env.step(steering=0.0, acceleration=0.0)

        self.assertTrue(done)
        self.assertTrue(info["collision"])
        self.assertLess(reward, -9.0)

    def test_out_of_bounds_terminates_with_penalty(self):
        env = CarEnv(initial_state=(9.7, 1.0, 0.0, 1.0), dt=0.2, car_radius=0.2)

        _, reward, done, info = env.step(steering=0.0, acceleration=0.0)

        self.assertTrue(done)
        self.assertTrue(info["out_of_bounds"])
        self.assertLess(reward, -9.0)

    def test_max_steps_terminates(self):
        env = CarEnv(initial_state=(1.0, 1.0, 0.0, 0.0), max_steps=2)

        _, _, first_done, _ = env.step(0.0, 0.0)
        _, _, second_done, info = env.step(0.0, 0.0)

        self.assertFalse(first_done)
        self.assertTrue(second_done)
        self.assertTrue(info["time_limit"])

    def test_step_after_termination_raises(self):
        env = CarEnv(initial_state=(9.7, 1.0, 0.0, 1.0), dt=0.2, car_radius=0.2)
        env.step(0.0, 0.0)

        with self.assertRaises(RuntimeError):
            env.step(0.0, 0.0)

    def test_returned_values_cannot_mutate_internal_state(self):
        env = CarEnv()
        state = env.reset()
        state[:] = 99.0
        history = env.trajectory
        history[0][:] = -99.0

        np.testing.assert_allclose(env.state, [1.0, 1.0, 0.0, 0.0])

    def test_heading_is_wrapped_to_minus_pi_pi(self):
        env = CarEnv(initial_state=(1.0, 1.0, math.pi - 0.01, 2.0), dt=1.0)

        state, _, _, _ = env.step(steering=0.4, acceleration=0.0)

        self.assertGreaterEqual(state[2], -math.pi)
        self.assertLess(state[2], math.pi)


if __name__ == "__main__":
    unittest.main()
