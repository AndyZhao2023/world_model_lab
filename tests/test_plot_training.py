import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")

from world_model_lab.plot_training import plot_training_history
from world_model_lab.train_world_model import save_checkpoint, train_model


class PlotTrainingTest(unittest.TestCase):
    def test_plot_training_history_saves_png_from_checkpoint(self):
        rng = np.random.default_rng(13)
        inputs = rng.normal(size=(48, 7))
        weights = rng.normal(size=(7, 4))
        targets = inputs @ weights
        result = train_model(
            inputs[:36],
            targets[:36],
            validation_inputs=inputs[36:],
            validation_targets=targets[36:],
            hidden_size=8,
            epochs=3,
            batch_size=12,
            seed=2,
        )

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            checkpoint = directory_path / "world_model.pt"
            output = directory_path / "training_loss.png"
            save_checkpoint(
                checkpoint,
                result,
                split_episode_ids={
                    "train": np.asarray([0]),
                    "validation": np.asarray([1]),
                    "test": np.asarray([2]),
                },
                training_config={"epochs": 3},
                test_metrics={"normalized_mse": 0.1},
            )

            returned_path = plot_training_history(checkpoint, output)
            signature = output.read_bytes()[:8]

        self.assertEqual(returned_path, output)
        self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")

    def test_plot_training_history_includes_multistep_components(self):
        rng = np.random.default_rng(17)
        inputs = rng.normal(size=(48, 7))
        weights = rng.normal(size=(7, 4))
        targets = inputs @ weights
        result = train_model(
            inputs[:36],
            targets[:36],
            validation_inputs=inputs[36:],
            validation_targets=targets[36:],
            hidden_size=8,
            epochs=3,
            batch_size=12,
            seed=3,
        )
        result.train_rollout_losses = [0.3, 0.2, 0.1]
        result.validation_rollout_losses = [0.4, 0.3, 0.2]

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            checkpoint = directory_path / "world_model.pt"
            output = directory_path / "training_loss.png"
            save_checkpoint(
                checkpoint,
                result,
                split_episode_ids={
                    "train": np.asarray([0]),
                    "validation": np.asarray([1]),
                    "test": np.asarray([2]),
                },
                training_config={"epochs": 3, "rollout_horizon": 3},
                test_metrics={"normalized_mse": 0.1},
            )
            labels = []
            original_plot = matplotlib.axes.Axes.plot

            def record_plot(axis, *args, **kwargs):
                labels.append(kwargs.get("label"))
                return original_plot(axis, *args, **kwargs)

            with patch("matplotlib.axes.Axes.plot", new=record_plot):
                returned_path = plot_training_history(checkpoint, output)
            signature = output.read_bytes()[:8]

        self.assertEqual(returned_path, output)
        self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(
            set(labels),
            {
                "Train total",
                "Validation total",
                "Train one-step",
                "Validation one-step",
                "Train rollout",
                "Validation rollout",
            },
        )


if __name__ == "__main__":
    unittest.main()
