from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_train_visual_dynamics_objective import save_source_checkpoint
from tests.test_visual_windows import make_visual_dataset
from world_model_lab.diagnose_visual_autoencoder import (
    _build_representation_decision,
    main,
    run_visual_autoencoder_diagnostics,
    summarize_autoencoder_reconstructions,
)
from world_model_lab.visual_dataset import save_visual_dataset
from world_model_lab.visual_observation import CAR_COLOR, HEADING_COLOR


class AutoencoderReconstructionMetricTest(unittest.TestCase):
    def test_metrics_average_windows_inside_each_episode_first(self):
        episode_ids = np.asarray([10, 10, 11], dtype=np.int64)
        initial = np.zeros((3, 3, 1, 2), dtype=np.uint8)
        targets = np.full((3, 1, 3, 1, 2), 255, dtype=np.uint8)
        reconstructions = np.asarray(
            [
                np.zeros((1, 3, 1, 2), dtype=np.float32),
                np.zeros((1, 3, 1, 2), dtype=np.float32),
                np.full((1, 3, 1, 2), -2.0, dtype=np.float32),
            ]
        )
        object_masks = np.zeros((3, 1, 1, 1, 2), dtype=np.float32)
        object_masks[..., 0] = 1.0

        metrics = summarize_autoencoder_reconstructions(
            reconstructed_frames=reconstructions,
            true_initial_frames=initial,
            true_target_frames=targets,
            object_masks=object_masks,
            episode_ids=episode_ids,
        )

        self.assertEqual(set(metrics), {"1"})
        self.assertEqual(metrics["1"]["episodes"], 2)
        self.assertEqual(metrics["1"]["windows"], 3)
        for name, expected in (
            ("pixel_mse", 5.0),
            ("object_pixel_mse", 5.0),
            ("background_pixel_mse", 5.0),
            ("object_pixel_mae", 2.0),
            ("cumulative_changed_pixel_mae", 2.0),
        ):
            self.assertAlmostEqual(metrics["1"][name], expected)

    def test_representation_decision_uses_strict_improvement_and_stability(
        self,
    ):
        source_static = {
            "object_pixel_mse": 0.2,
            "pixel_mse": 0.01,
            "background_pixel_mse": 0.005,
        }
        candidate_static = {
            "object_pixel_mse": 0.19,
            "pixel_mse": 0.011,
            "background_pixel_mse": 0.0055,
        }
        source_steps = {"5": {"cumulative_changed_pixel_mae": 0.3}}
        candidate_steps = {"5": {"cumulative_changed_pixel_mae": 0.29}}

        passed = _build_representation_decision(
            source_static=source_static,
            candidate_static=candidate_static,
            source_steps=source_steps,
            candidate_steps=candidate_steps,
            decision_horizon=5,
        )
        equal = _build_representation_decision(
            source_static=source_static,
            candidate_static={
                **candidate_static,
                "object_pixel_mse": 0.2,
            },
            source_steps=source_steps,
            candidate_steps=source_steps,
            decision_horizon=5,
        )

        self.assertTrue(passed["passed"])
        self.assertEqual(len(passed["gates"]), 4)
        self.assertTrue(all(gate["passed"] for gate in passed["gates"]))
        self.assertFalse(equal["passed"])
        failed = {
            gate["name"]
            for gate in equal["gates"]
            if not gate["passed"]
        }
        self.assertEqual(
            failed,
            {
                "held_out_object_mse_improvement",
                "horizon_cumulative_changed_pixel_mae_improvement",
            },
        )


class VisualAutoencoderDiagnosticRunnerTest(unittest.TestCase):
    def _write_fixture(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path]:
        visual = make_visual_dataset((7,) * 10)
        visual["frames"][:, 2, 3] = CAR_COLOR
        visual["frames"][:, 4, 5] = HEADING_COLOR
        data_path = root / "visual.npz"
        source_path = root / "source.pt"
        candidate_path = root / "candidate.pt"
        save_visual_dataset(visual, data_path)
        save_source_checkpoint(source_path, data_path=data_path)
        payload = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
        )
        first_name = next(iter(payload["autoencoder_state_dict"]))
        payload["autoencoder_state_dict"][first_name].add_(0.01)
        torch.save(payload, candidate_path)
        return data_path, source_path, candidate_path

    def test_runner_allows_different_autoencoders_and_writes_atomic_bundle(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path, source_path, candidate_path = self._write_fixture(root)
            output = root / "diagnostics"

            summary = run_visual_autoencoder_diagnostics(
                data_path=data_path,
                source_checkpoint_path=source_path,
                candidate_checkpoint_path=candidate_path,
                output_dir=output,
                horizons=(1, 2),
                windows_per_episode=2,
                decision_horizon=2,
                batch_size=8,
            )

            self.assertEqual(summary["horizons"], [1, 2])
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "manifest.json",
                    "metrics.json",
                    "visual_autoencoder_comparison.png",
                },
            )
            metrics = json.loads(
                (output / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(metrics["models"]), {"source", "candidate"})
            self.assertEqual(set(metrics["models"]["source"]["steps"]), {"1", "2"})
            self.assertEqual(len(metrics["decision"]["gates"]), 4)
            self.assertGreater(
                (output / "visual_autoencoder_comparison.png").stat().st_size,
                0,
            )

    def test_runner_rejects_test_split_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path, source_path, candidate_path = self._write_fixture(root)
            payload = torch.load(
                candidate_path,
                map_location="cpu",
                weights_only=True,
            )
            payload["split_episode_ids"]["test"] = torch.as_tensor(
                [9],
                dtype=torch.int64,
            )
            torch.save(payload, candidate_path)

            with self.assertRaisesRegex(ValueError, "test split"):
                run_visual_autoencoder_diagnostics(
                    data_path=data_path,
                    source_checkpoint_path=source_path,
                    candidate_checkpoint_path=candidate_path,
                    output_dir=root / "diagnostics",
                    horizons=(1, 2),
                    windows_per_episode=1,
                    decision_horizon=2,
                    batch_size=8,
                )

    def test_cli_help_and_pyproject_register_command(self):
        output = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["world-model-diagnose-visual-autoencoder", "--help"],
        ):
            with redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    main()

        self.assertEqual(raised.exception.code, 0)
        for option in (
            "--data",
            "--source-checkpoint",
            "--candidate-checkpoint",
            "--output-dir",
            "--horizons",
            "--windows-per-episode",
            "--decision-horizon",
        ):
            self.assertIn(option, output.getvalue())
        pyproject = (
            Path(__file__).resolve().parents[1] / "pyproject.toml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "world-model-diagnose-visual-autoencoder = "
            '"world_model_lab.diagnose_visual_autoencoder:main"',
            pyproject,
        )


if __name__ == "__main__":
    unittest.main()
