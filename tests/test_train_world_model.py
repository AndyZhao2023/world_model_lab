import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from world_model_lab.model import WorldModelMLP
from world_model_lab.train_world_model import (
    evaluate_model,
    load_checkpoint,
    predict_deltas,
    run_training,
    save_checkpoint,
    train_model,
)


def make_linear_dynamics(count: int = 256) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(23)
    inputs = rng.normal(size=(count, 7))
    weights = rng.normal(size=(7, 4))
    targets = inputs @ weights + np.asarray([0.2, -0.1, 0.05, 0.3])
    return inputs, targets


class TrainWorldModelTest(unittest.TestCase):
    def test_predict_deltas_returns_denormalized_physical_values(self):
        inputs, targets = make_linear_dynamics(count=64)
        result = train_model(
            inputs[:48],
            targets[:48],
            validation_inputs=inputs[48:],
            validation_targets=targets[48:],
            hidden_size=16,
            epochs=2,
            batch_size=32,
            learning_rate=1e-3,
            seed=3,
        )

        predictions = predict_deltas(result, inputs[:5])

        self.assertEqual(predictions.shape, (5, 4))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_mlp_maps_each_input_row_to_four_deltas(self):
        model = WorldModelMLP(hidden_size=16)

        output = model(torch.zeros((5, 7), dtype=torch.float32))

        self.assertEqual(tuple(output.shape), (5, 4))

    def test_training_reduces_normalized_loss_on_deterministic_dynamics(self):
        inputs, targets = make_linear_dynamics()
        train_inputs, validation_inputs = inputs[:192], inputs[192:]
        train_targets, validation_targets = targets[:192], targets[192:]

        result = train_model(
            train_inputs,
            train_targets,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            hidden_size=32,
            epochs=100,
            batch_size=64,
            learning_rate=1e-2,
            seed=11,
        )

        self.assertEqual(len(result.train_losses), 100)
        self.assertEqual(len(result.validation_losses), 100)
        self.assertLess(result.train_losses[-1], result.train_losses[0] * 0.1)
        self.assertGreaterEqual(result.best_epoch, 1)
        self.assertLessEqual(result.best_epoch, 100)
        validation_metrics = evaluate_model(
            result, validation_inputs, validation_targets
        )
        self.assertAlmostEqual(
            validation_metrics["normalized_mse"],
            min(result.validation_losses),
            places=6,
        )
        metrics = evaluate_model(result, inputs, targets)
        self.assertLess(metrics["mae_x"], 0.12)
        self.assertLess(metrics["mae_y"], 0.12)
        self.assertLess(metrics["mae_heading_radians"], 0.12)
        self.assertLess(metrics["mae_velocity"], 0.12)
        self.assertAlmostEqual(
            metrics["mae_heading_degrees"],
            np.degrees(metrics["mae_heading_radians"]),
        )

    def test_checkpoint_round_trip_preserves_predictions_and_metadata(self):
        inputs, targets = make_linear_dynamics(count=64)
        result = train_model(
            inputs[:48],
            targets[:48],
            validation_inputs=inputs[48:],
            validation_targets=targets[48:],
            hidden_size=12,
            epochs=2,
            batch_size=32,
            learning_rate=1e-3,
            seed=5,
        )
        split_ids = {
            "train": np.asarray([0, 1, 2]),
            "validation": np.asarray([3]),
            "test": np.asarray([4]),
        }
        test_metrics = {
            "normalized_mse": 0.25,
            "mae_x": 0.1,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world_model.pt"
            save_checkpoint(
                path,
                result,
                split_episode_ids=split_ids,
                training_config={"epochs": 2, "seed": 5},
                test_metrics=test_metrics,
            )
            loaded = load_checkpoint(path)

        self.assertEqual(loaded.model.hidden_size, 12)
        self.assertEqual(loaded.training_config, {"epochs": 2, "seed": 5})
        self.assertEqual(loaded.train_losses, result.train_losses)
        self.assertEqual(loaded.validation_losses, result.validation_losses)
        self.assertEqual(loaded.best_epoch, result.best_epoch)
        self.assertEqual(loaded.test_metrics, test_metrics)
        for name in split_ids:
            np.testing.assert_array_equal(loaded.split_episode_ids[name], split_ids[name])
        np.testing.assert_allclose(loaded.input_normalizer.mean, result.input_normalizer.mean)
        np.testing.assert_allclose(loaded.target_normalizer.std, result.target_normalizer.std)

        normalized = result.input_normalizer.normalize(inputs[:8])
        tensor = torch.as_tensor(normalized, dtype=torch.float32)
        with torch.no_grad():
            expected = result.model(tensor).numpy()
            actual = loaded.model(tensor).numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_run_training_splits_npz_by_episode_and_saves_checkpoint(self):
        rng = np.random.default_rng(41)
        episode_ids = np.repeat(np.arange(10), 8)
        count = episode_ids.size
        states = np.column_stack(
            (
                rng.uniform(0.5, 9.5, count),
                rng.uniform(0.5, 7.5, count),
                rng.uniform(-np.pi, np.pi, count),
                rng.uniform(0.1, 2.9, count),
            )
        )
        actions = rng.uniform([-0.5, -1.0], [0.5, 1.0], size=(count, 2))
        deltas = np.column_stack(
            (
                0.03 * states[:, 3] + 0.01 * actions[:, 0],
                -0.02 * states[:, 3] + 0.01 * actions[:, 1],
                0.04 * actions[:, 0] + 0.01 * states[:, 3],
                0.05 * actions[:, 1] - 0.01 * states[:, 3],
            )
        )
        next_states = states + deltas

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            dataset_path = directory_path / "transitions.npz"
            checkpoint_path = directory_path / "world_model.pt"
            np.savez_compressed(
                dataset_path,
                states=states,
                actions=actions,
                next_states=next_states,
                episode_ids=episode_ids,
            )

            summary = run_training(
                data_path=dataset_path,
                output_path=checkpoint_path,
                hidden_size=16,
                epochs=3,
                batch_size=32,
                learning_rate=1e-3,
                seed=7,
            )
            loaded = load_checkpoint(checkpoint_path)

        self.assertEqual(summary["transitions"], count)
        self.assertEqual(summary["split_episodes"], {"train": 8, "validation": 1, "test": 1})
        self.assertEqual(loaded.training_config["epochs"], 3)
        self.assertEqual(len(loaded.train_losses), 3)
        self.assertEqual(len(loaded.validation_losses), 3)
        self.assertEqual(loaded.test_metrics, summary["test"])
        self.assertEqual(summary["best_epoch"], loaded.best_epoch)
        self.assertAlmostEqual(
            summary["best_validation_loss"], min(loaded.validation_losses)
        )
        self.assertTrue(np.isfinite(summary["validation"]["normalized_mse"]))
        self.assertTrue(np.isfinite(summary["test"]["normalized_mse"]))


if __name__ == "__main__":
    unittest.main()
