import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world_model_lab.car_env import CarEnv
from world_model_lab.visualize import draw_environment


class VisualizeTest(unittest.TestCase):
    def test_draw_environment_returns_figure_without_mutating_state(self):
        env = CarEnv(initial_state=(1.0, 1.0, 0.0, 1.0))
        env.step(0.0, 0.0)
        before = env.state

        figure, axes = draw_environment(env)

        self.assertIs(figure, axes.figure)
        np.testing.assert_allclose(env.state, before)
        self.assertGreaterEqual(len(axes.patches), 3)
        self.assertGreaterEqual(len(axes.lines), 1)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
