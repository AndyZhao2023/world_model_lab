from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_train_visual_dynamics_objective import save_source_checkpoint
from tests.test_visual_counterfactual_data import (
    make_physical_visual_dataset,
)
from world_model_lab.diagnose_visual_counterfactual import (
    _build_counterfactual_comparison,
    _build_preregistered_decision,
    aggregate_counterfactual_seed_records,
    main,
    run_visual_counterfactual_diagnostics,
    summarize_matched_counterfactual_predictions,
)
from world_model_lab.visual_object_position import (
    LinearObjectPositionProbe,
)
from world_model_lab.visual_dataset import save_visual_dataset


class MatchedCounterfactualMetricTest(unittest.TestCase):
    def setUp(self):
        self.episode_ids = np.asarray([10, 10, 11], dtype=np.int64)
        self.valid = np.asarray(
            [[True, True], [True, False], [True, True]],
            dtype=np.bool_,
        )
        self.true_counterfactual_latents = np.zeros(
            (3, 2, 1),
            dtype=np.float32,
        )
        self.predicted_counterfactual_latents = np.asarray(
            [[[1.0], [2.0]], [[3.0], [999.0]], [[5.0], [6.0]]],
            dtype=np.float32,
        )
        self.true_factual_latents = np.full(
            (3, 2, 1),
            -1.0,
            dtype=np.float32,
        )
        self.predicted_factual_latents = (
            self.predicted_counterfactual_latents - 1.0
        )
        self.true_counterfactual_frames = np.full(
            (3, 2, 3, 2, 2),
            255,
            dtype=np.uint8,
        )
        self.predicted_counterfactual_frames = np.zeros(
            (3, 2, 3, 2, 2),
            dtype=np.float32,
        )
        self.true_factual_frames = np.zeros(
            (3, 2, 3, 2, 2),
            dtype=np.uint8,
        )
        self.predicted_factual_frames = np.full(
            (3, 2, 3, 2, 2),
            -1.0,
            dtype=np.float32,
        )
        self.initial_frames = np.zeros(
            (3, 3, 2, 2),
            dtype=np.uint8,
        )
        self.position_probe = LinearObjectPositionProbe(
            weight=np.asarray([[1.0], [1.0]], dtype=np.float32),
            bias=np.zeros(2, dtype=np.float32),
        )
        self.true_counterfactual_positions = np.zeros(
            (3, 2, 2),
            dtype=np.float32,
        )
        self.true_factual_positions = np.full(
            (3, 2, 2),
            -1.0,
            dtype=np.float32,
        )
        self.world_bounds = np.asarray(
            [0.0, 2.0, 0.0, 2.0],
            dtype=np.float64,
        )

    def _summarize(self):
        return summarize_matched_counterfactual_predictions(
            predicted_counterfactual_normalized_latents=(
                self.predicted_counterfactual_latents
            ),
            true_counterfactual_normalized_latents=(
                self.true_counterfactual_latents
            ),
            predicted_factual_normalized_latents=(
                self.predicted_factual_latents
            ),
            true_factual_normalized_latents=self.true_factual_latents,
            predicted_counterfactual_frames=(
                self.predicted_counterfactual_frames
            ),
            true_counterfactual_frames=(
                self.true_counterfactual_frames
            ),
            predicted_factual_frames=self.predicted_factual_frames,
            true_factual_frames=self.true_factual_frames,
            true_initial_frames=self.initial_frames,
            position_probe=self.position_probe,
            true_counterfactual_normalized_positions=(
                self.true_counterfactual_positions
            ),
            true_factual_normalized_positions=(
                self.true_factual_positions
            ),
            world_bounds=self.world_bounds,
            valid_steps=self.valid,
            episode_ids=self.episode_ids,
        )

    def test_masked_metrics_weight_episodes_equally(self):
        record = self._summarize()

        self.assertEqual(set(record), {"1", "2"})
        self.assertEqual(record["1"]["episodes"], 2)
        self.assertEqual(record["1"]["valid_windows"], 3)
        self.assertEqual(record["2"]["episodes"], 2)
        self.assertEqual(record["2"]["valid_windows"], 2)
        self.assertAlmostEqual(
            record["1"]["normalized_latent_mse"],
            15.0,
        )
        self.assertAlmostEqual(
            record["2"]["normalized_latent_mse"],
            20.0,
        )
        self.assertAlmostEqual(
            record["1"]["normalized_position_mse"],
            15.0,
        )
        self.assertAlmostEqual(
            record["2"]["normalized_position_mse"],
            20.0,
        )
        self.assertAlmostEqual(
            record["1"]["world_position_error"],
            3.5 * np.sqrt(2.0),
        )
        self.assertAlmostEqual(
            record["2"]["world_position_error"],
            4.0 * np.sqrt(2.0),
        )
        for step in ("1", "2"):
            self.assertAlmostEqual(record[step]["pixel_mse"], 1.0)
            self.assertAlmostEqual(
                record[step]["cumulative_changed_pixel_mae"],
                1.0,
            )
        self.assertAlmostEqual(
            record["1"]["transition_changed_pixel_mae"],
            1.0,
        )
        self.assertAlmostEqual(
            record["2"]["transition_changed_pixel_mae"],
            0.0,
        )

    def test_action_effect_can_be_right_when_direct_prediction_is_wrong(self):
        record = self._summarize()

        for step in ("1", "2"):
            self.assertEqual(
                record[step]["normalized_latent_effect_mse"],
                0.0,
            )
            self.assertEqual(record[step]["pixel_effect_mse"], 0.0)
            self.assertEqual(
                record[step]["normalized_position_effect_mse"],
                0.0,
            )

    def test_seed_aggregation_comparison_and_gates_are_exact(self):
        seed_record = self._summarize()
        source = aggregate_counterfactual_seed_records(
            (seed_record, seed_record),
            seeds=(0, 1),
        )
        candidate = aggregate_counterfactual_seed_records(
            (seed_record, seed_record),
            seeds=(0, 1),
        )
        comparison = _build_counterfactual_comparison(
            source=source,
            candidate=candidate,
        )
        decision = _build_preregistered_decision(
            source=source,
            candidate=candidate,
            decision_horizon=2,
        )

        self.assertEqual(source["seeds"], [0, 1])
        self.assertEqual(
            source["steps"]["1"]["normalized_latent_mse"],
            {"mean": 15.0, "sample_std": 0.0},
        )
        self.assertEqual(
            comparison["1"]["normalized_latent_mse"]["absolute"],
            0.0,
        )
        self.assertFalse(decision["candidate_passes"])
        self.assertEqual(len(decision["gates"]), 6)
        self.assertEqual(
            sum(gate["passed"] for gate in decision["gates"]),
            2,
        )
        self.assertEqual(
            decision["gates"][3]["name"],
            "horizon_direct_position_improvement",
        )

    def test_metrics_reject_misaligned_arrays(self):
        with self.assertRaises(ValueError):
            summarize_matched_counterfactual_predictions(
                predicted_counterfactual_normalized_latents=(
                    self.predicted_counterfactual_latents[:, :1]
                ),
                true_counterfactual_normalized_latents=(
                    self.true_counterfactual_latents
                ),
                predicted_factual_normalized_latents=(
                    self.predicted_factual_latents
                ),
                true_factual_normalized_latents=self.true_factual_latents,
                predicted_counterfactual_frames=(
                    self.predicted_counterfactual_frames
                ),
                true_counterfactual_frames=(
                    self.true_counterfactual_frames
                ),
                predicted_factual_frames=self.predicted_factual_frames,
                true_factual_frames=self.true_factual_frames,
                true_initial_frames=self.initial_frames,
                position_probe=self.position_probe,
                true_counterfactual_normalized_positions=(
                    self.true_counterfactual_positions
                ),
                true_factual_normalized_positions=(
                    self.true_factual_positions
                ),
                world_bounds=self.world_bounds,
                valid_steps=self.valid,
                episode_ids=self.episode_ids,
            )


class VisualCounterfactualDiagnosticsRunnerTest(unittest.TestCase):
    def test_cli_help_lists_matched_protocol(self):
        output = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["world-model-diagnose-visual-counterfactual", "--help"],
        ):
            with redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    main()

        self.assertEqual(raised.exception.code, 0)
        for name in (
            "--data",
            "--source-checkpoint",
            "--candidate-checkpoint",
            "--output-dir",
            "--horizons",
            "--windows-per-episode",
            "--counterfactual-seeds",
            "--decision-horizon",
        ):
            self.assertIn(name, output.getvalue())

    def test_pyproject_registers_matched_counterfactual_command(self):
        project_root = Path(__file__).resolve().parents[1]
        pyproject = (project_root / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "world-model-diagnose-visual-counterfactual = "
            '"world_model_lab.diagnose_visual_counterfactual:main"',
            pyproject,
        )

    def test_runner_publishes_complete_deterministic_bundle(self):
        visual = make_physical_visual_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            source_path = root / "source.pt"
            candidate_path = root / "candidate.pt"
            first_output = root / "first"
            second_output = root / "second"
            save_visual_dataset(visual, data_path)
            save_source_checkpoint(source_path, data_path=data_path)
            shutil.copyfile(source_path, candidate_path)

            summaries = [
                run_visual_counterfactual_diagnostics(
                    data_path=data_path,
                    source_checkpoint_path=source_path,
                    candidate_checkpoint_path=candidate_path,
                    output_dir=output,
                    horizons=(1, 2),
                    windows_per_episode=2,
                    counterfactual_seeds=(0, 1),
                    decision_horizon=2,
                    encode_batch_size=8,
                    decode_batch_size=8,
                )
                for output in (first_output, second_output)
            ]

            self.assertEqual(summaries[0]["horizons"], [1, 2])
            expected_files = {
                "manifest.json",
                "metrics.json",
                "matched_counterfactual_comparison.png",
            }
            self.assertEqual(
                {path.name for path in first_output.iterdir()},
                expected_files,
            )
            metrics = json.loads(
                (first_output / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (first_output / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metrics["schema_version"], 1)
            self.assertEqual(
                set(metrics["models"]),
                {"source", "candidate"},
            )
            self.assertEqual(set(metrics["snapshots"]), {"1", "2"})
            self.assertIn("oracle", metrics)
            self.assertIn("coverage", metrics)
            self.assertIn("object_position_probe", metrics)
            self.assertFalse(metrics["decision"]["candidate_passes"])
            self.assertIn(
                "normalized_position_mse",
                metrics["snapshots"]["2"]["source"],
            )
            self.assertEqual(
                manifest["protocol"]["object_position_target"],
                "normalized_xy",
            )
            self.assertEqual(
                metrics["comparison"]["candidate_minus_source"]["2"][
                    "normalized_latent_mse"
                ]["absolute"],
                0.0,
            )
            for permutation in manifest["branches"]["donor_permutations"]:
                self.assertTrue(
                    all(
                        donor != recipient
                        for recipient, donor in enumerate(permutation)
                    )
                )
            for filename in expected_files:
                self.assertEqual(
                    (first_output / filename).read_bytes(),
                    (second_output / filename).read_bytes(),
                )

    def test_runner_rejects_bad_protocol_or_partial_publication(self):
        visual = make_physical_visual_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            source_path = root / "source.pt"
            candidate_path = root / "candidate.pt"
            save_visual_dataset(visual, data_path)
            save_source_checkpoint(source_path, data_path=data_path)
            shutil.copyfile(source_path, candidate_path)

            with self.assertRaisesRegex(ValueError, "decision_horizon"):
                run_visual_counterfactual_diagnostics(
                    data_path=data_path,
                    source_checkpoint_path=source_path,
                    candidate_checkpoint_path=candidate_path,
                    output_dir=root / "missing-horizon",
                    horizons=(1, 2),
                    decision_horizon=5,
                    counterfactual_seeds=(0, 1),
                )
            existing = root / "existing"
            existing.mkdir()
            (existing / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                run_visual_counterfactual_diagnostics(
                    data_path=data_path,
                    source_checkpoint_path=source_path,
                    candidate_checkpoint_path=candidate_path,
                    output_dir=existing,
                    horizons=(1, 2),
                    decision_horizon=2,
                    counterfactual_seeds=(0, 1),
                )
            failed = root / "failed"
            with patch(
                "world_model_lab.diagnose_visual_counterfactual."
                "plot_matched_counterfactual_comparison",
                side_effect=RuntimeError("plot failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "plot failed"):
                    run_visual_counterfactual_diagnostics(
                        data_path=data_path,
                        source_checkpoint_path=source_path,
                        candidate_checkpoint_path=candidate_path,
                        output_dir=failed,
                        horizons=(1, 2),
                        windows_per_episode=2,
                        counterfactual_seeds=(0, 1),
                        decision_horizon=2,
                        encode_batch_size=8,
                        decode_batch_size=8,
                    )
            self.assertFalse(failed.exists())
            self.assertEqual(
                list(root.glob(".failed.tmp-*")),
                [],
            )

    def test_runner_rejects_incompatible_candidate(self):
        visual = make_physical_visual_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
            payload["latent_mean"].add_(1.0)
            torch.save(payload, candidate_path)

            with self.assertRaisesRegex(ValueError, "latent normalizer"):
                run_visual_counterfactual_diagnostics(
                    data_path=data_path,
                    source_checkpoint_path=source_path,
                    candidate_checkpoint_path=candidate_path,
                    output_dir=root / "diagnostics",
                    horizons=(1, 2),
                    counterfactual_seeds=(0, 1),
                    decision_horizon=2,
                )


if __name__ == "__main__":
    unittest.main()
