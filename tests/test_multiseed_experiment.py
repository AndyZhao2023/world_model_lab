import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from world_model_lab.diagnose_model import sha256_file
from world_model_lab import multiseed_experiment
from world_model_lab.multiseed_experiment import (
    _validate_dataset_hash_invariant,
    _validate_split_invariant,
    build_experiment_summary,
    plot_multiseed_comparison,
    run_multiseed_experiment,
    write_summary_csv,
)
from world_model_lab.train_world_model import load_checkpoint


def make_metrics(position, heading, velocity, total):
    return {
        "schema_version": 2,
        "rollout": {
            "curves": {
                "steps": [1, 2],
                "free_rollout": {
                    "physical": {
                        "position": position,
                        "heading_degrees": heading,
                        "velocity": velocity,
                    },
                    "normalized_mse": {"total": total},
                },
            }
        },
    }


def make_records():
    return [
        {
            "seed": 0,
            "h1_metrics": make_metrics(
                [1.0, 2.0],
                [1.0, 2.0],
                [0.4, 0.6],
                [0.8, 1.0],
            ),
            "h10_metrics": make_metrics(
                [0.8, 2.5],
                [0.5, 2.5],
                [0.3, 0.5],
                [0.7, 0.9],
            ),
            "h1_metrics_path": "runs/seed_0/h1/diagnostics/metrics.json",
            "h10_metrics_path": "runs/seed_0/h10/diagnostics/metrics.json",
        },
        {
            "seed": 1,
            "h1_metrics": make_metrics(
                [1.2, 2.2],
                [1.0, 2.0],
                [0.5, 0.7],
                [0.9, 1.1],
            ),
            "h10_metrics": make_metrics(
                [1.0, 2.3],
                [0.5, 1.5],
                [0.4, 0.6],
                [0.8, 1.0],
            ),
            "h1_metrics_path": "runs/seed_1/h1/diagnostics/metrics.json",
            "h10_metrics_path": "runs/seed_1/h10/diagnostics/metrics.json",
        },
    ]


def save_sequence_dynamics(path: Path) -> None:
    states = []
    actions = []
    next_states = []
    episode_ids = []
    step_ids = []
    for episode_id in range(10):
        state = np.asarray(
            [float(episode_id), 0.2 * episode_id, 0.03 * episode_id, 0.2]
        )
        for step in range(12):
            action = np.asarray(
                [0.03 * episode_id + 0.01 * step, 0.05 + 0.02 * step]
            )
            delta = np.asarray(
                [
                    0.1 + 0.02 * state[3],
                    0.01 + 0.005 * state[0],
                    0.01 * action[0] + 0.002 * state[3],
                    0.01 + 0.02 * action[1],
                ]
            )
            next_state = state + delta
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            episode_ids.append(episode_id)
            step_ids.append(step)
            state = next_state
    np.savez_compressed(
        path,
        states=np.asarray(states),
        actions=np.asarray(actions),
        next_states=np.asarray(next_states),
        episode_ids=np.asarray(episode_ids),
        step_ids=np.asarray(step_ids),
    )


class MultiseedExperimentTest(unittest.TestCase):
    def test_pyproject_registers_multiseed_command(self):
        pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        _, scripts_and_after = pyproject.split("[project.scripts]", maxsplit=1)
        scripts = scripts_and_after.split("\n[", maxsplit=1)[0]

        self.assertIn(
            'world-model-multiseed = "world_model_lab.multiseed_experiment:main"',
            scripts.splitlines(),
        )

    def test_cli_help_lists_every_experiment_parameter(self):
        entrypoint = getattr(multiseed_experiment, "main", None)
        self.assertTrue(callable(entrypoint))
        standard_output = io.StringIO()
        with patch.object(sys, "argv", ["world-model-multiseed", "--help"]):
            with redirect_stdout(standard_output):
                with self.assertRaises(SystemExit) as context:
                    entrypoint()

        self.assertEqual(context.exception.code, 0)
        help_text = standard_output.getvalue()
        for flag in (
            "--data",
            "--output-dir",
            "--seeds",
            "--split-seed",
            "--hidden-size",
            "--epochs",
            "--batch-size",
            "--learning-rate",
            "--rollout-loss-weight",
            "--diagnostic-horizons",
            "--windows-per-episode",
            "--xy-bins",
            "--feature-bins",
            "--min-bin-count",
        ):
            self.assertIn(flag, help_text)

    def test_cli_reports_missing_dataset_as_argument_error(self):
        entrypoint = getattr(multiseed_experiment, "main", None)
        self.assertTrue(callable(entrypoint))
        standard_error = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                sys,
                "argv",
                [
                    "world-model-multiseed",
                    "--data",
                    str(Path(directory) / "missing.npz"),
                ],
            ):
                with redirect_stderr(standard_error):
                    with self.assertRaises(SystemExit) as context:
                        entrypoint()

        self.assertEqual(context.exception.code, 2)
        self.assertIn("dataset is not a regular file", standard_error.getvalue())

    def test_cli_prints_returned_contract_as_sorted_indented_json(self):
        entrypoint = getattr(multiseed_experiment, "main", None)
        self.assertTrue(callable(entrypoint))
        standard_output = io.StringIO()
        with patch.object(
            multiseed_experiment,
            "run_multiseed_experiment",
            return_value={"z": 1, "a": 2},
        ):
            with patch.object(sys, "argv", ["world-model-multiseed"]):
                with redirect_stdout(standard_output):
                    entrypoint()

        self.assertEqual(standard_output.getvalue(), '{\n  "a": 2,\n  "z": 1\n}\n')

    def test_run_multiseed_experiment_rejects_fewer_than_two_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "transitions.npz"
            save_sequence_dynamics(data_path)

            with self.assertRaisesRegex(ValueError, "at least two"):
                run_multiseed_experiment(
                    data_path=data_path,
                    output_dir=Path(directory) / "output",
                    seeds=(0,),
                )

    def test_run_multiseed_experiment_rejects_duplicate_or_negative_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "transitions.npz"
            save_sequence_dynamics(data_path)
            for seeds, message in (((0, 0), "unique"), ((-1, 0), "non-negative")):
                with self.subTest(seeds=seeds):
                    with self.assertRaisesRegex(ValueError, message):
                        run_multiseed_experiment(
                            data_path=data_path,
                            output_dir=Path(directory) / "output",
                            seeds=seeds,
                        )

    def test_run_multiseed_experiment_rejects_missing_dataset_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "dataset"):
                run_multiseed_experiment(
                    data_path=root / "missing.npz",
                    output_dir=root / "output",
                    seeds=(0, 1),
                )

    def test_run_multiseed_experiment_rejects_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "transitions.npz"
            output_dir = root / "output"
            save_sequence_dynamics(data_path)
            output_dir.mkdir()
            (output_dir / "existing.txt").write_text("old", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "empty"):
                run_multiseed_experiment(
                    data_path=data_path,
                    output_dir=output_dir,
                    seeds=(0, 1),
                )

    def test_run_multiseed_experiment_rejects_invalid_protocol_values(self):
        invalid_values = (
            ({"split_seed": -1}, "split_seed"),
            ({"hidden_size": 0}, "hidden_size"),
            ({"epochs": 0}, "epochs"),
            ({"batch_size": 0}, "batch_size"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"rollout_loss_weight": -1.0}, "rollout_loss_weight"),
            ({"rollout_loss_weight": float("inf")}, "rollout_loss_weight"),
            ({"diagnostic_horizons": ()}, "diagnostic_horizons"),
            ({"diagnostic_horizons": (1, 1)}, "diagnostic_horizons"),
            ({"diagnostic_horizons": (2, 1)}, "diagnostic_horizons"),
            ({"diagnostic_horizons": (0, 1)}, "diagnostic_horizons"),
            ({"windows_per_episode": 0}, "windows_per_episode"),
            ({"xy_bins": 0}, "xy_bins"),
            ({"feature_bins": 0}, "feature_bins"),
            ({"min_bin_count": 0}, "min_bin_count"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "transitions.npz"
            save_sequence_dynamics(data_path)
            for arguments, message in invalid_values:
                with self.subTest(arguments=arguments):
                    with self.assertRaisesRegex(ValueError, message):
                        run_multiseed_experiment(
                            data_path=data_path,
                            output_dir=root / "output",
                            seeds=(0, 1),
                            **arguments,
                        )

    def test_validate_split_invariant_rejects_any_different_split_ids(self):
        reference = SimpleNamespace(
            split_episode_ids={
                "train": np.asarray([0, 1]),
                "validation": np.asarray([2]),
                "test": np.asarray([3]),
            }
        )
        for split_name in ("train", "validation", "test"):
            changed_ids = {
                name: values.copy()
                for name, values in reference.split_episode_ids.items()
            }
            changed_ids[split_name] = np.asarray([99])
            with self.subTest(split=split_name):
                with self.assertRaisesRegex(ValueError, "split episode IDs"):
                    _validate_split_invariant(
                        [reference, SimpleNamespace(split_episode_ids=changed_ids)]
                    )

    def test_validate_dataset_hash_invariant_rejects_different_hashes(self):
        manifests = [
            {"dataset": {"sha256": "a" * 64}},
            {"dataset": {"sha256": "b" * 64}},
        ]

        with self.assertRaisesRegex(ValueError, "dataset SHA-256"):
            _validate_dataset_hash_invariant(manifests)

    def test_build_experiment_summary_aggregates_paired_curves(self):
        summary = build_experiment_summary(
            make_records(),
            snapshot_horizons=(1, 2),
        )

        position = summary["metrics"]["position"]
        np.testing.assert_allclose(position["h1"]["mean"], [1.1, 2.1])
        np.testing.assert_allclose(
            position["h1"]["std"],
            [np.sqrt(0.02), np.sqrt(0.02)],
        )
        np.testing.assert_allclose(position["paired_delta"]["mean"], [-0.2, 0.3])
        np.testing.assert_allclose(
            position["paired_delta"]["std"],
            [0.0, np.sqrt(0.08)],
            atol=1e-15,
        )
        self.assertEqual(position["paired_delta"]["improved_seed_count"][1], 0)
        self.assertEqual(position["paired_delta"]["worse_seed_count"][1], 2)
        self.assertEqual(
            summary["per_seed"]["0"]["first_heading_regression_step"],
            2,
        )
        self.assertIsNone(
            summary["per_seed"]["1"]["first_heading_regression_step"]
        )

    def test_build_experiment_summary_accepts_schema_v2_step_curves(self):
        records = make_records()
        for record in records:
            for model_name in ("h1", "h10"):
                rollout = record[f"{model_name}_metrics"]["rollout"]
                rollout["step_curves"] = rollout.pop("curves")

        summary = build_experiment_summary(records, snapshot_horizons=(1, 2))

        np.testing.assert_allclose(
            summary["metrics"]["position"]["paired_delta"]["mean"],
            [-0.2, 0.3],
        )

    def test_build_experiment_summary_rejects_duplicate_or_invalid_seeds(self):
        records = make_records()
        records[1]["seed"] = 0
        with self.assertRaisesRegex(ValueError, "unique"):
            build_experiment_summary(records, snapshot_horizons=(1, 2))

        records = make_records()
        records[0]["seed"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_experiment_summary(records, snapshot_horizons=(1, 2))

        with self.assertRaisesRegex(ValueError, "at least two"):
            build_experiment_summary(records[:1], snapshot_horizons=(1, 2))

    def test_build_experiment_summary_rejects_inconsistent_steps(self):
        records = make_records()
        records[1]["h10_metrics"]["rollout"]["curves"]["steps"] = [1, 3]

        with self.assertRaisesRegex(ValueError, "identical steps"):
            build_experiment_summary(records, snapshot_horizons=(1, 2))

    def test_summary_outputs_csv_and_png(self):
        summary = build_experiment_summary(
            make_records(),
            snapshot_horizons=(1, 2),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "summary.csv"
            plot_path = root / "comparison.png"

            self.assertEqual(write_summary_csv(summary, csv_path), csv_path)
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertEqual(
                csv_text.splitlines()[0],
                "seed,horizon,metric,h1,h10,delta_h10_minus_h1",
            )
            rows = list(csv.DictReader(io.StringIO(csv_text)))
            expected_values = {
                0: {
                    "position": ([1.0, 2.0], [0.8, 2.5]),
                    "heading_degrees": ([1.0, 2.0], [0.5, 2.5]),
                    "velocity": ([0.4, 0.6], [0.3, 0.5]),
                    "normalized_total": ([0.8, 1.0], [0.7, 0.9]),
                },
                1: {
                    "position": ([1.2, 2.2], [1.0, 2.3]),
                    "heading_degrees": ([1.0, 2.0], [0.5, 1.5]),
                    "velocity": ([0.5, 0.7], [0.4, 0.6]),
                    "normalized_total": ([0.9, 1.1], [0.8, 1.0]),
                },
            }
            expected_order = [
                (seed, horizon, metric)
                for seed in (0, 1)
                for horizon in (1, 2)
                for metric in (
                    "position",
                    "heading_degrees",
                    "velocity",
                    "normalized_total",
                )
            ]
            self.assertEqual(
                [
                    (int(row["seed"]), int(row["horizon"]), row["metric"])
                    for row in rows
                ],
                expected_order,
            )
            for row, (seed, horizon, metric) in zip(
                rows, expected_order, strict=True
            ):
                h1_values, h10_values = expected_values[seed][metric]
                h1_value = h1_values[horizon - 1]
                h10_value = h10_values[horizon - 1]
                self.assertEqual(float(row["h1"]), h1_value)
                self.assertEqual(float(row["h10"]), h10_value)
                self.assertEqual(
                    float(row["delta_h10_minus_h1"]),
                    h10_value - h1_value,
                )

            real_subplots = multiseed_experiment.plt.subplots
            captured_plot = {}

            def capture_subplots(*args, **kwargs):
                figure, axes = real_subplots(*args, **kwargs)
                captured_plot["figure"] = figure
                captured_plot["axes"] = axes
                return figure, axes

            with patch.object(
                multiseed_experiment.plt,
                "subplots",
                side_effect=capture_subplots,
            ) as subplots_spy:
                self.assertEqual(
                    plot_multiseed_comparison(summary, plot_path),
                    plot_path,
                )
            subplots_spy.assert_called_once_with(2, 2, figsize=(11, 8))
            self.assertEqual(captured_plot["axes"].shape, (2, 2))
            self.assertEqual(len(captured_plot["figure"].axes), 4)
            self.assertEqual(plot_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_run_multiseed_experiment_sorts_seeds_and_writes_complete_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "transitions.npz"
            output_dir = root / "experiment"
            save_sequence_dynamics(data_path)

            result = run_multiseed_experiment(
                data_path=data_path,
                output_dir=output_dir,
                seeds=(1, 0),
                split_seed=0,
                rollout_loss_weight=1.0,
                hidden_size=8,
                epochs=1,
                batch_size=32,
                diagnostic_horizons=(1, 2),
                windows_per_episode=2,
                xy_bins=2,
                feature_bins=2,
                min_bin_count=1,
            )

            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "experiment_manifest.json",
                    "summary.json",
                    "summary.csv",
                    "multiseed_comparison.png",
                    "runs",
                },
            )
            expected_diagnostics = {
                "metrics.json",
                "manifest.json",
                "overview.png",
                "rollout_errors.png",
                "rollout_loss_components.png",
            }
            checkpoints = []
            for seed in (0, 1):
                for model in ("h1", "h10"):
                    model_dir = output_dir / "runs" / f"seed_{seed}" / model
                    self.assertEqual(
                        {path.name for path in model_dir.iterdir()},
                        {"world_model.pt", "diagnostics"},
                    )
                    self.assertEqual(
                        {
                            path.name
                            for path in (model_dir / "diagnostics").iterdir()
                        },
                        expected_diagnostics,
                    )
                    checkpoints.append(load_checkpoint(model_dir / "world_model.pt"))

            for split_name in ("train", "validation", "test"):
                for checkpoint in checkpoints[1:]:
                    np.testing.assert_array_equal(
                        checkpoints[0].split_episode_ids[split_name],
                        checkpoint.split_episode_ids[split_name],
                    )

            manifest = json.loads(
                (output_dir / "experiment_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            experiment_summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(experiment_summary["seeds"], [0, 1])
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(
                manifest["dataset"],
                {"path": str(data_path.resolve()), "sha256": sha256_file(data_path)},
            )
            self.assertEqual(
                manifest["training"],
                {
                    "seeds": [0, 1],
                    "split_seed": 0,
                    "hidden_size": 8,
                    "epochs": 1,
                    "batch_size": 32,
                    "learning_rate": 1e-3,
                    "models": {
                        "h1": {
                            "rollout_horizon": 1,
                            "rollout_loss_weight": 0.0,
                        },
                        "h10": {
                            "rollout_horizon": 10,
                            "rollout_loss_weight": 1.0,
                        },
                    },
                },
            )
            self.assertEqual(
                manifest["diagnostics"],
                {
                    "horizons": [1, 2],
                    "windows_per_episode": 2,
                    "xy_bins": 2,
                    "feature_bins": 2,
                    "min_bin_count": 1,
                },
            )
            self.assertEqual(
                manifest["runs"],
                [
                    {
                        "seed": seed,
                        "model": model,
                        "checkpoint": (
                            f"runs/seed_{seed}/{model}/world_model.pt"
                        ),
                        "metrics": (
                            f"runs/seed_{seed}/{model}/diagnostics/metrics.json"
                        ),
                        "manifest": (
                            f"runs/seed_{seed}/{model}/diagnostics/manifest.json"
                        ),
                    }
                    for seed in (0, 1)
                    for model in ("h1", "h10")
                ],
            )
            self.assertEqual(result["seeds"], [0, 1])
            self.assertEqual(result["split_seed"], 0)
            self.assertEqual(result["longest_horizon"], 2)
            self.assertEqual(
                set(result["first_heading_regression_steps"]), {"0", "1"}
            )
            self.assertEqual(result["output_dir"], str(output_dir))
            self.assertEqual(
                result["manifest"], str(output_dir / "experiment_manifest.json")
            )


if __name__ == "__main__":
    unittest.main()
