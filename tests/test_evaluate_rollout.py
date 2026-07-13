import math
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

from world_model_lab.dataset import Normalizer, wrap_angle
from world_model_lab.evaluate_rollout import (
    build_episode_rollouts,
    plot_episode_rollout,
    rollout_episode,
    run_rollout_evaluation,
    summarize_horizons,
)
from world_model_lab.model import WorldModelMLP
from world_model_lab.train_world_model import (
    LoadedWorldModel,
    TrainingResult,
    save_checkpoint,
)


def make_constant_delta_world_model(delta: np.ndarray) -> LoadedWorldModel:
    model = WorldModelMLP(hidden_size=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.copy_(
            torch.as_tensor(delta, dtype=torch.float32)
        )
    model.eval()
    return LoadedWorldModel(
        model=model,
        input_normalizer=Normalizer(mean=np.zeros(7), std=np.ones(7)),
        target_normalizer=Normalizer(mean=np.zeros(4), std=np.ones(4)),
        split_episode_ids={
            "train": np.asarray([0]),
            "validation": np.asarray([1]),
            "test": np.asarray([2, 3]),
        },
        training_config={},
        train_losses=[],
        validation_losses=[],
        best_epoch=0,
        test_metrics={},
    )


def make_state_sequence(initial_state: np.ndarray, delta: np.ndarray, steps: int):
    sequence = [np.asarray(initial_state, dtype=np.float64)]
    for _ in range(steps):
        next_state = sequence[-1] + delta
        next_state[2] = wrap_angle(next_state[2])
        sequence.append(next_state)
    return np.asarray(sequence)


class EvaluateRolloutTest(unittest.TestCase):
    def test_rollout_recursively_feeds_predictions_back_into_model(self):
        delta = np.asarray([1.0, 0.0, 0.2, 0.5])
        world_model = make_constant_delta_world_model(delta)
        initial_state = np.asarray([0.0, 0.0, math.pi - 0.1, 0.0])
        actions = np.zeros((2, 2), dtype=np.float64)

        predicted_states = rollout_episode(world_model, initial_state, actions)

        expected = make_state_sequence(initial_state, delta, steps=2)
        self.assertEqual(predicted_states.shape, (3, 4))
        np.testing.assert_allclose(predicted_states, expected, atol=1e-6)
        self.assertTrue(np.all(predicted_states[:, 2] >= -math.pi))
        self.assertTrue(np.all(predicted_states[:, 2] < math.pi))

    def test_horizon_summary_uses_only_episodes_long_enough_for_each_horizon(self):
        delta = np.asarray([1.0, 0.0, 0.2, 0.5])
        world_model = make_constant_delta_world_model(delta)
        episode_2 = make_state_sequence(
            np.asarray([0.0, 0.0, 0.0, 0.0]), delta, steps=3
        )
        episode_3 = make_state_sequence(
            np.asarray([10.0, 1.0, -1.0, 2.0]), delta, steps=2
        )
        states = np.vstack((episode_2[:-1], episode_3[:-1]))
        next_states = np.vstack((episode_2[1:], episode_3[1:]))
        actions = np.zeros((5, 2), dtype=np.float64)
        episode_ids = np.asarray([2, 2, 2, 3, 3])
        step_ids = np.asarray([0, 1, 2, 0, 1])

        rollouts = build_episode_rollouts(
            world_model,
            states=states,
            actions=actions,
            next_states=next_states,
            episode_ids=episode_ids,
            step_ids=step_ids,
            selected_episode_ids=np.asarray([2, 3]),
        )
        summary = summarize_horizons(rollouts, horizons=(1, 2, 3))

        self.assertEqual(set(rollouts), {2, 3})
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["horizons"]["1"]["episodes"], 2)
        self.assertEqual(summary["horizons"]["2"]["episodes"], 2)
        self.assertEqual(summary["horizons"]["3"]["episodes"], 1)
        for metrics in summary["horizons"].values():
            self.assertLess(metrics["mean_position_error"], 1e-5)
            self.assertLess(metrics["mean_heading_error_degrees"], 1e-5)
            self.assertLess(metrics["mean_velocity_error"], 1e-5)

    def test_plot_episode_rollout_saves_png(self):
        delta = np.asarray([1.0, 0.0, 0.2, 0.5])
        world_model = make_constant_delta_world_model(delta)
        true_states = make_state_sequence(
            np.asarray([0.0, 0.0, 0.0, 0.0]), delta, steps=3
        )
        rollouts = build_episode_rollouts(
            world_model,
            states=true_states[:-1],
            actions=np.zeros((3, 2)),
            next_states=true_states[1:],
            episode_ids=np.asarray([2, 2, 2]),
            step_ids=np.asarray([0, 1, 2]),
            selected_episode_ids=np.asarray([2]),
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rollout.png"
            returned = plot_episode_rollout(rollouts[2], output)
            signature = output.read_bytes()[:8]

        self.assertEqual(returned, output)
        self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")

    def test_run_rollout_evaluation_uses_checkpoint_test_episodes(self):
        delta = np.asarray([1.0, 0.0, 0.2, 0.5])
        world_model = make_constant_delta_world_model(delta)
        episode_2 = make_state_sequence(
            np.asarray([0.0, 0.0, 0.0, 0.0]), delta, steps=3
        )
        episode_3 = make_state_sequence(
            np.asarray([10.0, 1.0, -1.0, 2.0]), delta, steps=2
        )
        states = np.vstack((episode_2[:-1], episode_3[:-1]))
        next_states = np.vstack((episode_2[1:], episode_3[1:]))
        actions = np.zeros((5, 2), dtype=np.float64)
        episode_ids = np.asarray([2, 2, 2, 3, 3])
        step_ids = np.asarray([0, 1, 2, 0, 1])
        training_result = TrainingResult(
            model=world_model.model,
            input_normalizer=world_model.input_normalizer,
            target_normalizer=world_model.target_normalizer,
            train_losses=[1.0],
            validation_losses=[1.0],
            best_epoch=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            data_path = directory_path / "transitions.npz"
            checkpoint_path = directory_path / "world_model.pt"
            plot_path = directory_path / "rollout.png"
            np.savez_compressed(
                data_path,
                states=states,
                actions=actions,
                next_states=next_states,
                episode_ids=episode_ids,
                step_ids=step_ids,
            )
            save_checkpoint(
                checkpoint_path,
                training_result,
                split_episode_ids=world_model.split_episode_ids,
                training_config={},
                test_metrics={},
            )

            summary = run_rollout_evaluation(
                data_path=data_path,
                checkpoint_path=checkpoint_path,
                plot_path=plot_path,
                horizons=(1, 2, 3),
            )
            plot_exists = plot_path.exists()

        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["representative_episode"], 2)
        self.assertEqual(summary["horizons"]["3"]["episodes"], 1)
        self.assertTrue(plot_exists)


if __name__ == "__main__":
    unittest.main()
