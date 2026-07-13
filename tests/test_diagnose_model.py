import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

from world_model_lab.dataset import Normalizer
from world_model_lab.diagnose_model import run_diagnostics, sha256_file
from world_model_lab.model import WorldModelMLP
from world_model_lab.train_world_model import TrainingResult, save_checkpoint


def make_state_sequence(initial_x: float, steps: int) -> np.ndarray:
    sequence = [np.asarray([initial_x, 0.0, 0.0, 0.0])]
    for _ in range(steps):
        sequence.append(sequence[-1] + np.asarray([1.0, 0.0, 0.0, 0.0]))
    return np.asarray(sequence)


def save_constant_delta_checkpoint(path: Path) -> None:
    model = WorldModelMLP(hidden_size=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.copy_(
            torch.as_tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        )
    result = TrainingResult(
        model=model,
        input_normalizer=Normalizer(mean=np.zeros(7), std=np.ones(7)),
        target_normalizer=Normalizer(mean=np.zeros(4), std=np.ones(4)),
        train_losses=[1.0],
        validation_losses=[1.0],
        best_epoch=1,
    )
    save_checkpoint(
        path,
        result,
        split_episode_ids={
            "train": np.asarray([0]),
            "validation": np.asarray([1]),
            "test": np.asarray([2]),
        },
        training_config={"epochs": 1, "seed": 7},
        test_metrics={},
    )


def save_dataset(path: Path, *, test_steps: int = 3) -> None:
    train = make_state_sequence(-5.0, steps=3)
    test = make_state_sequence(0.0, steps=test_steps)
    states = np.vstack((train[:-1], test[:-1]))
    next_states = np.vstack((train[1:], test[1:]))
    actions = np.zeros((states.shape[0], 2), dtype=np.float64)
    np.savez_compressed(
        path,
        states=states,
        actions=actions,
        next_states=next_states,
        episode_ids=np.asarray([0] * 3 + [2] * test_steps),
        step_ids=np.asarray(list(range(3)) + list(range(test_steps))),
    )


class DiagnoseModelTest(unittest.TestCase):
    def test_sha256_file_is_content_based(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"world-model")

            digest = sha256_file(path)

        self.assertEqual(
            digest,
            "f6fafe16f3d1018f98141e97f8c60ad98233b537e62e6d2383a53704d28016df",
        )

    def test_run_diagnostics_writes_reproducible_output_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "transitions.npz"
            checkpoint_path = root / "world_model.pt"
            output_dir = root / "diagnostics"
            save_dataset(data_path)
            save_constant_delta_checkpoint(checkpoint_path)

            summary = run_diagnostics(
                data_path=data_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                horizons=(1, 2),
                windows_per_episode=2,
                xy_bins=2,
                feature_bins=2,
                min_bin_count=1,
            )

            output_names = {
                path.name for path in output_dir.iterdir() if path.is_file()
            }
            metrics = json.loads((output_dir / "metrics.json").read_text())
            manifest = json.loads((output_dir / "manifest.json").read_text())
            dataset_hash = sha256_file(data_path)

        self.assertEqual(
            output_names,
            {"metrics.json", "manifest.json", "overview.png", "rollout_errors.png"},
        )
        self.assertEqual(metrics["schema_version"], 2)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["dataset"]["sha256"], dataset_hash)
        self.assertEqual(manifest["checkpoint"]["hidden_size"], 4)
        self.assertEqual(manifest["checkpoint"]["test_episode_ids"], [2])
        self.assertEqual(manifest["diagnostics"]["horizons"], [1, 2])
        self.assertEqual(summary["longest_horizon"], 2)

    def test_run_diagnostics_rejects_an_impossible_maximum_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "transitions.npz"
            checkpoint_path = root / "world_model.pt"
            save_dataset(data_path, test_steps=1)
            save_constant_delta_checkpoint(checkpoint_path)

            with self.assertRaisesRegex(ValueError, "not long enough"):
                run_diagnostics(
                    data_path=data_path,
                    checkpoint_path=checkpoint_path,
                    output_dir=root / "diagnostics",
                    horizons=(1, 2),
                )


if __name__ == "__main__":
    unittest.main()
