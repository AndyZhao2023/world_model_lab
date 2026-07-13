import math
import unittest

import torch
from torch import nn

from world_model_lab.rollout_training import (
    rollout_state_loss,
    rollout_states,
    wrapped_state_errors,
)


class ConstantDeltaModel(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = nn.Parameter(torch.as_tensor(delta, dtype=torch.float32))

    def forward(self, inputs):
        return self.delta.expand(inputs.shape[0], -1)


class RolloutTrainingTest(unittest.TestCase):
    def test_rollout_recursively_updates_predicted_states(self):
        model = ConstantDeltaModel([1.0, 0.0, 0.0, 0.5])

        predictions = rollout_states(
            model,
            torch.zeros((1, 4)),
            torch.zeros((1, 3, 2)),
            input_mean=torch.zeros(7),
            input_std=torch.ones(7),
            target_mean=torch.zeros(4),
            target_std=torch.ones(4),
        )

        torch.testing.assert_close(
            predictions[0],
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.5],
                    [2.0, 0.0, 0.0, 1.0],
                    [3.0, 0.0, 0.0, 1.5],
                ]
            ),
        )

    def test_later_rollout_errors_backpropagate_to_model_parameters(self):
        model = ConstantDeltaModel([0.5, 0.0, 0.0, 0.0])
        true_states = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]]
        )

        loss = rollout_state_loss(
            model,
            initial_states=torch.zeros((1, 4)),
            actions=torch.zeros((1, 2, 2)),
            true_next_states=true_states,
            input_mean=torch.zeros(7),
            input_std=torch.ones(7),
            target_mean=torch.zeros(4),
            target_std=torch.ones(4),
        )
        loss.backward()

        self.assertIsNotNone(model.delta.grad)
        self.assertNotEqual(float(model.delta.grad[0]), 0.0)

    def test_state_error_wraps_heading_at_pi_boundary(self):
        predicted = torch.tensor(
            [[[0.0, 0.0, math.radians(179.0), 0.0]]]
        )
        true = torch.tensor(
            [[[0.0, 0.0, math.radians(-179.0), 0.0]]]
        )

        errors = wrapped_state_errors(predicted, true)

        self.assertAlmostEqual(
            abs(math.degrees(float(errors[0, 0, 2]))),
            2.0,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
