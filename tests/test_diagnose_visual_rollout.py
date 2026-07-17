from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch

from tests.test_train_visual_dynamics_objective import save_source_checkpoint
from tests.test_visual_windows import make_visual_dataset
from world_model_lab.dataset import Normalizer
from world_model_lab.diagnose_visual_rollout import (
    VisualRolloutWindow,
    _sattolo_permutation,
    evaluate_counterfactual_sensitivity,
    main,
    rollout_normalized_latents,
    run_visual_rollout_diagnostics,
    select_visual_rollout_windows,
    summarize_visual_predictions,
    teacher_forced_normalized_latents,
)
from world_model_lab.visual_dataset import save_visual_dataset


class VisualRolloutWindowTest(unittest.TestCase):
    def test_windows_are_evenly_spaced_aligned_and_episode_safe(self):
        visual = make_visual_dataset((15, 8, 7))
        frame_count = int(visual["frames"].shape[0])
        latent_dim = 2
        latents = np.arange(
            frame_count * latent_dim,
            dtype=np.float32,
        ).reshape(frame_count, latent_dim)

        selection = select_visual_rollout_windows(
            dataset=visual,
            latent_frames=latents,
            selected_episode_ids=np.asarray([10, 11, 12], dtype=np.int64),
            max_horizon=5,
            windows_per_episode=2,
        )

        self.assertEqual(
            [
                (window.episode_id, window.start_step)
                for window in selection.windows
            ],
            [(10, 3), (10, 10), (11, 3)],
        )
        np.testing.assert_array_equal(
            selection.eligible_episode_ids,
            [10, 11],
        )
        np.testing.assert_array_equal(
            selection.skipped_episode_ids,
            [12],
        )
        for window in selection.windows:
            self.assertEqual(window.context_latents.shape, (4, latent_dim))
            self.assertEqual(window.history_actions.shape, (3, 2))
            self.assertEqual(window.future_actions.shape, (5, 2))
            self.assertEqual(window.target_latents.shape, (5, latent_dim))
            self.assertEqual(window.target_frame_indices.shape, (5,))

        first = selection.windows[0]
        np.testing.assert_array_equal(first.context_latents, latents[0:4])
        np.testing.assert_array_equal(
            first.history_actions,
            visual["actions"][0:3],
        )
        np.testing.assert_array_equal(
            first.future_actions,
            visual["actions"][3:8],
        )
        np.testing.assert_array_equal(first.target_latents, latents[4:9])
        np.testing.assert_array_equal(
            first.target_frame_indices,
            np.arange(4, 9, dtype=np.int64),
        )
        self.assertEqual(first.initial_frame_index, 3)

        last = selection.windows[-1]
        frame_start = int(visual["frame_offsets"][1])
        action_start = int(visual["transition_offsets"][1])
        np.testing.assert_array_equal(
            last.context_latents,
            latents[frame_start : frame_start + 4],
        )
        np.testing.assert_array_equal(
            last.future_actions,
            visual["actions"][action_start + 3 : action_start + 8],
        )

    def test_window_selection_rejects_invalid_protocol(self):
        visual = make_visual_dataset((8, 8, 8))
        latents = np.zeros((visual["frames"].shape[0], 2), dtype=np.float32)
        invalid = (
            {"max_horizon": 0},
            {"windows_per_episode": 0},
            {
                "selected_episode_ids": np.asarray(
                    [],
                    dtype=np.int64,
                )
            },
            {
                "selected_episode_ids": np.asarray(
                    [10, 10],
                    dtype=np.int64,
                )
            },
        )
        for update in invalid:
            with self.subTest(update=update):
                kwargs = {
                    "dataset": visual,
                    "latent_frames": latents,
                    "selected_episode_ids": np.asarray(
                        [10, 11, 12],
                        dtype=np.int64,
                    ),
                    "max_horizon": 5,
                    "windows_per_episode": 2,
                }
                kwargs.update(update)
                with self.assertRaises(ValueError):
                    select_visual_rollout_windows(**kwargs)


class VisualLatentRolloutTest(unittest.TestCase):
    class CurrentActionDynamics(torch.nn.Module):
        def forward(
            self,
            context: torch.Tensor,
            history: torch.Tensor,
            current: torch.Tensor,
        ) -> torch.Tensor:
            del history
            return context[:, -1] + current[:, :1]

    def setUp(self):
        self.context = np.asarray(
            [[[-2.0], [-1.0], [0.0], [1.0]]],
            dtype=np.float32,
        )
        self.targets = np.asarray(
            [[[10.0], [20.0], [30.0]]],
            dtype=np.float32,
        )
        self.history = np.zeros((1, 3, 2), dtype=np.float64)
        self.future = np.asarray(
            [[[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]],
            dtype=np.float64,
        )
        self.latent_normalizer = Normalizer(
            mean=np.zeros(1),
            std=np.ones(1),
        )
        self.action_normalizer = Normalizer(
            mean=np.zeros(2),
            std=np.ones(2),
        )

    def test_free_rollout_recursively_shifts_predictions_and_actions(self):
        predictions = rollout_normalized_latents(
            self.CurrentActionDynamics(),
            context_latents=self.context,
            history_actions=self.history,
            future_actions=self.future,
            latent_normalizer=self.latent_normalizer,
            action_normalizer=self.action_normalizer,
        )

        np.testing.assert_array_equal(
            predictions,
            np.asarray([[[3.0], [6.0], [10.0]]], dtype=np.float32),
        )

    def test_teacher_forcing_uses_true_rolling_contexts(self):
        predictions = teacher_forced_normalized_latents(
            self.CurrentActionDynamics(),
            context_latents=self.context,
            target_latents=self.targets,
            history_actions=self.history,
            future_actions=self.future,
            latent_normalizer=self.latent_normalizer,
            action_normalizer=self.action_normalizer,
        )

        np.testing.assert_array_equal(
            predictions,
            np.asarray([[[3.0], [13.0], [24.0]]], dtype=np.float32),
        )

    def test_rollout_rejects_misaligned_arrays(self):
        with self.assertRaises(ValueError):
            rollout_normalized_latents(
                self.CurrentActionDynamics(),
                context_latents=self.context,
                history_actions=self.history,
                future_actions=self.future[:, :, :1],
                latent_normalizer=self.latent_normalizer,
                action_normalizer=self.action_normalizer,
            )
        with self.assertRaises(ValueError):
            teacher_forced_normalized_latents(
                self.CurrentActionDynamics(),
                context_latents=self.context,
                target_latents=self.targets[:, :2],
                history_actions=self.history,
                future_actions=self.future,
                latent_normalizer=self.latent_normalizer,
                action_normalizer=self.action_normalizer,
            )


class CounterfactualPermutationTest(unittest.TestCase):
    def test_sattolo_is_deterministic_complete_and_has_no_fixed_points(self):
        first = _sattolo_permutation(12, seed=7)
        second = _sattolo_permutation(12, seed=7)

        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first != np.arange(12)))
        self.assertEqual(sorted(first.tolist()), list(range(12)))

    def test_sattolo_rejects_invalid_count_or_seed(self):
        with self.assertRaises(ValueError):
            _sattolo_permutation(1, seed=0)
        with self.assertRaises(ValueError):
            _sattolo_permutation(2, seed=-1)

    def test_counterfactual_replaces_complete_future_action_rows(self):
        class LatentToPixelDecoder(torch.nn.Module):
            def decode(self, latents: torch.Tensor) -> torch.Tensor:
                return latents[:, :1, None, None].expand(-1, 3, 2, 2)

        windows = tuple(
            VisualRolloutWindow(
                episode_id=episode_id,
                start_step=3,
                context_latents=np.zeros((4, 1), dtype=np.float32),
                history_actions=np.zeros((3, 2), dtype=np.float64),
                future_actions=np.full(
                    (2, 2),
                    [action, 0.0],
                    dtype=np.float64,
                ),
                target_latents=np.zeros((2, 1), dtype=np.float32),
                initial_frame_index=index,
                target_frame_indices=np.asarray(
                    [index + 1, index + 2],
                    dtype=np.int64,
                ),
            )
            for index, (episode_id, action) in enumerate(
                ((10, 1.0), (11, 3.0))
            )
        )
        identity_latent = Normalizer(np.zeros(1), np.ones(1))
        identity_action = Normalizer(np.zeros(2), np.ones(2))

        sensitivity = evaluate_counterfactual_sensitivity(
            dynamics=VisualLatentRolloutTest.CurrentActionDynamics(),
            autoencoder=LatentToPixelDecoder(),
            latent_normalizer=identity_latent,
            action_normalizer=identity_action,
            windows=windows,
            seeds=(0, 1, 2),
        )

        self.assertEqual(sensitivity["seeds"], [0, 1, 2])
        for step, latent_rms, pixel_mse, pixel_mae in (
            ("1", 2.0, 4.0, 2.0),
            ("2", 4.0, 16.0, 4.0),
        ):
            record = sensitivity["steps"][step]
            self.assertAlmostEqual(
                record["normalized_latent_rms"]["mean"],
                latent_rms,
            )
            self.assertAlmostEqual(
                record["decoded_pixel_mse"]["mean"],
                pixel_mse,
            )
            self.assertAlmostEqual(
                record["decoded_pixel_mae"]["mean"],
                pixel_mae,
            )
            for metric in record.values():
                self.assertEqual(metric["sample_std"], 0.0)


class RolloutMetricAggregationTest(unittest.TestCase):
    def test_metrics_average_windows_inside_each_episode_first(self):
        episode_ids = np.asarray([10, 10, 11], dtype=np.int64)
        target_latents = np.zeros((3, 1, 1), dtype=np.float32)
        predicted_latents = np.asarray(
            [[[1.0]], [[1.0]], [[3.0]]],
            dtype=np.float32,
        )
        initial_frames = np.zeros((3, 3, 2, 2), dtype=np.uint8)
        target_frames = np.full((3, 1, 3, 2, 2), 255, dtype=np.uint8)
        predicted_frames = np.asarray(
            [
                np.full((1, 3, 2, 2), 0.0, dtype=np.float32),
                np.full((1, 3, 2, 2), 0.0, dtype=np.float32),
                np.full((1, 3, 2, 2), -2.0, dtype=np.float32),
            ]
        )

        metrics = summarize_visual_predictions(
            predicted_normalized_latents=predicted_latents,
            target_normalized_latents=target_latents,
            predicted_frames=predicted_frames,
            true_initial_frames=initial_frames,
            true_target_frames=target_frames,
            episode_ids=episode_ids,
        )

        self.assertEqual(set(metrics), {"1"})
        self.assertEqual(metrics["1"]["episodes"], 2)
        self.assertEqual(metrics["1"]["windows"], 3)
        self.assertAlmostEqual(
            metrics["1"]["normalized_latent_mse"],
            5.0,
        )
        self.assertAlmostEqual(metrics["1"]["pixel_mse"], 5.0)
        self.assertAlmostEqual(
            metrics["1"]["transition_changed_pixel_mae"],
            2.0,
        )
        self.assertAlmostEqual(
            metrics["1"]["cumulative_changed_pixel_mae"],
            2.0,
        )

    def test_metrics_reject_misaligned_prediction_arrays(self):
        with self.assertRaises(ValueError):
            summarize_visual_predictions(
                predicted_normalized_latents=np.zeros(
                    (2, 1, 1),
                    dtype=np.float32,
                ),
                target_normalized_latents=np.zeros(
                    (2, 2, 1),
                    dtype=np.float32,
                ),
                predicted_frames=np.zeros(
                    (2, 1, 3, 2, 2),
                    dtype=np.float32,
                ),
                true_initial_frames=np.zeros(
                    (2, 3, 2, 2),
                    dtype=np.uint8,
                ),
                true_target_frames=np.zeros(
                    (2, 1, 3, 2, 2),
                    dtype=np.uint8,
                ),
                episode_ids=np.asarray([10, 11], dtype=np.int64),
            )


class VisualRolloutDiagnosticsRunnerTest(unittest.TestCase):
    def test_cli_help_lists_visual_rollout_protocol(self):
        output = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["world-model-diagnose-visual-rollout", "--help"],
        ):
            with redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    main()

        self.assertEqual(raised.exception.code, 0)
        for name in (
            "--data",
            "--baseline-checkpoint",
            "--aligned-checkpoint",
            "--output-dir",
            "--horizons",
            "--windows-per-episode",
            "--counterfactual-seeds",
        ):
            self.assertIn(name, output.getvalue())

    def test_pyproject_registers_visual_rollout_command(self):
        project_root = Path(__file__).resolve().parents[1]
        pyproject = (project_root / "pyproject.toml").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "world-model-diagnose-visual-rollout = "
            '"world_model_lab.diagnose_visual_rollout:main"',
            pyproject,
        )

    def test_runner_writes_complete_bundle_for_compatible_checkpoints(self):
        visual = make_visual_dataset((6,) * 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            baseline_path = root / "baseline.pt"
            aligned_path = root / "aligned.pt"
            output_dir = root / "diagnostics"
            save_visual_dataset(visual, data_path)
            save_source_checkpoint(baseline_path, data_path=data_path)
            shutil.copyfile(baseline_path, aligned_path)

            summary = run_visual_rollout_diagnostics(
                data_path=data_path,
                baseline_checkpoint_path=baseline_path,
                aligned_checkpoint_path=aligned_path,
                output_dir=output_dir,
                horizons=(1, 2),
                windows_per_episode=2,
                counterfactual_seeds=(0, 1),
            )

            self.assertEqual(summary["horizons"], [1, 2])
            self.assertEqual(
                set(path.name for path in output_dir.iterdir()),
                {
                    "manifest.json",
                    "metrics.json",
                    "visual_rollout_comparison.png",
                },
            )
            self.assertGreater(
                (output_dir / "visual_rollout_comparison.png").stat().st_size,
                0,
            )
            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["schema_version"], 1)
            self.assertEqual(set(metrics["models"]), {"baseline", "aligned"})
            self.assertEqual(
                set(metrics["models"]["baseline"]["steps"]),
                {"1", "2"},
            )
            self.assertEqual(set(metrics["snapshots"]), {"1", "2"})
            self.assertEqual(
                metrics["comparison"]["aligned_minus_baseline"]["1"][
                    "free_rollout"
                ]["normalized_latent_mse"]["absolute"],
                0.0,
            )

    def test_runner_rejects_incompatible_checkpoint_metadata(self):
        visual = make_visual_dataset((6,) * 10)
        mutations = (
            (
                "autoencoder weights",
                lambda payload: payload["autoencoder_state_dict"][
                    next(iter(payload["autoencoder_state_dict"]))
                ].add_(1.0),
            ),
            (
                "latent normalizer",
                lambda payload: payload["latent_mean"].add_(1.0),
            ),
            (
                "action normalizer",
                lambda payload: payload["action_mean"].add_(1.0),
            ),
            (
                "test split",
                lambda payload: payload["split_episode_ids"].__setitem__(
                    "test",
                    torch.as_tensor([99], dtype=torch.int64),
                ),
            ),
            (
                "dataset SHA-256",
                lambda payload: payload["dataset"].__setitem__(
                    "sha256",
                    "0" * 64,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "visual.npz"
            baseline_path = root / "baseline.pt"
            save_visual_dataset(visual, data_path)
            save_source_checkpoint(baseline_path, data_path=data_path)
            for index, (message, mutate) in enumerate(mutations):
                with self.subTest(message=message):
                    candidate_path = root / f"candidate-{index}.pt"
                    payload = torch.load(
                        baseline_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    mutate(payload)
                    torch.save(payload, candidate_path)
                    with self.assertRaisesRegex(ValueError, message):
                        run_visual_rollout_diagnostics(
                            data_path=data_path,
                            baseline_checkpoint_path=baseline_path,
                            aligned_checkpoint_path=candidate_path,
                            output_dir=root / f"diagnostics-{index}",
                            horizons=(1, 2),
                            windows_per_episode=1,
                            counterfactual_seeds=(0, 1),
                        )

            convgru_path = root / "convgru.pt"
            save_source_checkpoint(
                convgru_path,
                data_path=data_path,
                architecture="convgru",
            )
            with self.assertRaisesRegex(ValueError, "spatial CNN"):
                run_visual_rollout_diagnostics(
                    data_path=data_path,
                    baseline_checkpoint_path=baseline_path,
                    aligned_checkpoint_path=convgru_path,
                    output_dir=root / "diagnostics-convgru",
                    horizons=(1, 2),
                    windows_per_episode=1,
                    counterfactual_seeds=(0, 1),
                )


if __name__ == "__main__":
    unittest.main()
