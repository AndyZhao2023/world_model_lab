import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
