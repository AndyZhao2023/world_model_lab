from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tests.test_visual_windows import make_visual_dataset
from world_model_lab import run_vjepa_probe
from world_model_lab.run_vjepa_probe import run_probe
from world_model_lab.visual_dataset import save_visual_dataset


class FakeEncoder:
    def __init__(self):
        self.calls: list[int] = []
        self.metadata = {
            "model_id": "fake/vjepa",
            "requested_revision": "requested",
            "resolved_revision": "resolved",
            "device": "cpu",
            "tubelet_size": 2,
            "crop_size": 256,
            "patch_size": 16,
            "hidden_size": 3,
            "feature_dim": 6,
            "pooling": "last_tubelet_mean_plus_last_minus_first",
        }

    def encode(self, clips: np.ndarray) -> np.ndarray:
        self.calls.append(int(clips.shape[0]))
        numeric = clips.astype(np.float32) / 255.0
        first = np.mean(numeric[:, 0], axis=(1, 2))
        last = np.mean(numeric[:, -1], axis=(1, 2))
        return np.concatenate((last, last - first), axis=1)


class FakeEncoderFactory:
    def __init__(self):
        self.encoder = FakeEncoder()
        self.arguments: dict[str, str] | None = None

    def __call__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str,
    ) -> FakeEncoder:
        self.arguments = {
            "model_id": model_id,
            "revision": revision,
            "device": device,
        }
        return self.encoder


class RunVJEPAProbeTest(unittest.TestCase):
    def setUp(self):
        self.visual = make_visual_dataset((5,) * 10)

    def test_runner_writes_complete_feature_and_result_artifacts(self):
        factory = FakeEncoderFactory()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            feature_path = root / "features.npz"
            result_path = root / "result.json"
            save_visual_dataset(self.visual, data_path)

            summary = run_probe(
                data_path=data_path,
                feature_path=feature_path,
                result_path=result_path,
                model_id="fake/vjepa",
                model_revision="requested",
                device="cpu",
                batch_size=3,
                max_train=8,
                max_validation=2,
                max_test=2,
                ridge=1e-3,
                split_seed=42,
                encoder_factory=factory,
            )

            self.assertTrue(feature_path.is_file())
            self.assertTrue(result_path.is_file())
            self.assertEqual(
                factory.arguments,
                {
                    "model_id": "fake/vjepa",
                    "revision": "requested",
                    "device": "cpu",
                },
            )
            self.assertEqual(summary["schema_version"], 1)
            self.assertEqual(summary["samples"], {"train": 8, "validation": 2, "test": 2})
            self.assertIn("mean_target", summary["baseline"])
            self.assertIn("recorded", summary["test"])
            self.assertIn("reversed", summary["test"])
            self.assertIn("repeat_last", summary["test"])
            self.assertEqual(set(summary["decision"]["gates"]), {
                "centre_mean_le_3px",
                "heading_mean_lt_45deg",
                "velocity_beats_reversed_5pct",
                "velocity_beats_repeat_last_5pct",
            })
            with np.load(feature_path, allow_pickle=False) as cache:
                self.assertEqual(int(cache["schema_version"]), 1)
                self.assertEqual(cache["train_recorded_features"].shape, (8, 6))
                self.assertEqual(cache["test_reversed_features"].shape, (2, 6))
                self.assertEqual(cache["test_repeat_last_features"].shape, (2, 6))
                train_ids = cache["train_episode_ids"]
                validation_ids = cache["validation_episode_ids"]
                test_ids = cache["test_episode_ids"]
                self.assertTrue(set(train_ids).isdisjoint(validation_ids))
                self.assertTrue(set(train_ids).isdisjoint(test_ids))
                self.assertTrue(set(validation_ids).isdisjoint(test_ids))
            persisted = json.loads(result_path.read_text())
            self.assertEqual(persisted, summary)

    def test_existing_output_fails_before_loading_data_or_model(self):
        factory = mock.Mock(side_effect=AssertionError("must not load model"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "features.npz"
            existing.write_bytes(b"owned")

            with self.assertRaises(FileExistsError):
                run_probe(
                    data_path=root / "missing.npz",
                    feature_path=existing,
                    result_path=root / "result.json",
                    encoder_factory=factory,
                )

            factory.assert_not_called()
            self.assertEqual(existing.read_bytes(), b"owned")

    def test_runner_rejects_invalid_protocol_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                {"batch_size": 0},
                {"max_train": 0},
                {"max_validation": 0},
                {"max_test": 0},
                {"ridge": -1.0},
                {"split_seed": -1},
                {"model_id": ""},
                {"model_revision": ""},
            )
            for overrides in cases:
                arguments = {
                    "data_path": root / "missing.npz",
                    "feature_path": root / "features.npz",
                    "result_path": root / "result.json",
                    "encoder_factory": FakeEncoderFactory(),
                }
                arguments.update(overrides)
                with self.subTest(overrides=overrides):
                    with self.assertRaises(ValueError):
                        run_probe(**arguments)

    def test_cli_help_and_pyproject_register_the_probe(self):
        pyproject = Path("pyproject.toml").read_text()
        self.assertIn(
            'world-model-vjepa-probe = "world_model_lab.run_vjepa_probe:main"',
            pyproject,
        )
        self.assertIn('"transformers>=5.14,<6"', pyproject)
        self.assertIn('"torchvision>=0.21,<0.29"', pyproject)
        with mock.patch(
            "sys.argv",
            ["world-model-vjepa-probe", "--help"],
        ):
            with self.assertRaises(SystemExit) as raised:
                run_vjepa_probe.main()
        self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
