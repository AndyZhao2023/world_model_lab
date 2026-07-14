import copy
import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from world_model_lab import bootstrap_experiment
from world_model_lab.bootstrap_experiment import (
    build_bootstrap_comparison,
    plot_bootstrap_comparison,
    run_bootstrap_experiment,
    write_comparison_csv,
)
from world_model_lab.train_world_model import load_checkpoint, run_training
from world_model_lab.ensemble import WorldModelEnsemble, load_ensemble


METRICS = (
    "position",
    "heading_degrees",
    "velocity",
    "normalized_total",
)


def make_ensemble_metrics(
    *,
    error: float,
    correlation: float | None,
    disagreement: float = 0.2,
):
    one_step = {
        name: {
            "ensemble_error": {"mean": error},
            "pearson_correlation": correlation,
        }
        for name in METRICS
    }
    rollout_metrics = {
        name: {
            "ensemble_error_mean": [error, error + 1.0],
            "disagreement_mean": [disagreement, disagreement * 2.0],
            "pearson_correlation": [correlation, correlation],
        }
        for name in METRICS
    }
    return {
        "schema_version": 1,
        "one_step": {"metrics": one_step},
        "rollout": {
            "steps": [1, 2],
            "metrics": rollout_metrics,
            "horizons": {
                str(horizon): {
                    "metrics": {
                        name: {
                            "ensemble_error_mean": error + horizon - 1,
                            "disagreement_mean": disagreement * horizon,
                            "pearson_correlation": correlation,
                        }
                        for name in METRICS
                    }
                }
                for horizon in (1, 2)
            },
        },
    }


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


class BootstrapExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_directory = tempfile.TemporaryDirectory()
        cls.fixture_root = Path(cls.fixture_directory.name)
        cls.data_path = cls.fixture_root / "transitions.npz"
        save_sequence_dynamics(cls.data_path)
        cls.baseline_paths = []
        for seed in (0, 1):
            checkpoint_path = cls.fixture_root / f"baseline-seed-{seed}.pt"
            run_training(
                data_path=cls.data_path,
                output_path=checkpoint_path,
                hidden_size=8,
                epochs=1,
                batch_size=32,
                learning_rate=1e-3,
                seed=seed,
                split_seed=0,
                rollout_horizon=10,
                rollout_loss_weight=1.0,
            )
            cls.baseline_paths.append(checkpoint_path)

    @classmethod
    def tearDownClass(cls):
        cls.fixture_directory.cleanup()

    def test_pyproject_registers_bootstrap_ensemble_command(self):
        pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        _, scripts_and_after = pyproject.split("[project.scripts]", maxsplit=1)
        scripts = scripts_and_after.split("\n[", maxsplit=1)[0]

        self.assertIn(
            "world-model-bootstrap-ensemble = "
            '"world_model_lab.bootstrap_experiment:main"',
            scripts.splitlines(),
        )

    def test_cli_help_lists_every_experiment_parameter(self):
        entrypoint = getattr(bootstrap_experiment, "main", None)
        self.assertTrue(callable(entrypoint))
        standard_output = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["world-model-bootstrap-ensemble", "--help"],
        ):
            with redirect_stdout(standard_output):
                with self.assertRaises(SystemExit) as context:
                    entrypoint()

        self.assertEqual(context.exception.code, 0)
        help_text = standard_output.getvalue()
        for flag in (
            "--data",
            "--baseline-checkpoints",
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
            "--calibration-bins",
        ):
            self.assertIn(flag, help_text)

    def test_cli_reports_missing_dataset_as_argument_error(self):
        entrypoint = getattr(bootstrap_experiment, "main", None)
        self.assertTrue(callable(entrypoint))
        standard_error = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                sys,
                "argv",
                [
                    "world-model-bootstrap-ensemble",
                    "--data",
                    str(root / "missing.npz"),
                    "--baseline-checkpoints",
                    str(root / "missing.pt"),
                ],
            ):
                with redirect_stderr(standard_error):
                    with self.assertRaises(SystemExit) as context:
                        entrypoint()

        self.assertEqual(context.exception.code, 2)
        self.assertIn("dataset is not a regular file", standard_error.getvalue())

    def test_two_member_end_to_end_writes_complete_artifact_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "experiment"
            result = run_bootstrap_experiment(
                data_path=self.data_path,
                baseline_checkpoint_paths=(
                    self.baseline_paths[1],
                    self.baseline_paths[0],
                ),
                output_dir=output_dir,
                seeds=(1, 0),
                split_seed=0,
                hidden_size=8,
                epochs=1,
                batch_size=32,
                learning_rate=1e-3,
                rollout_loss_weight=1.0,
                diagnostic_horizons=(1, 2),
                windows_per_episode=2,
                calibration_bins=2,
            )

            self.assertEqual(
                set(result),
                {
                    "experiment_manifest.json",
                    "comparison.json",
                    "comparison.csv",
                    "comparison.png",
                    "baseline_diagnostics",
                    "bootstrap_diagnostics",
                    "runs",
                },
            )
            for seed in (0, 1):
                checkpoint_path = (
                    output_dir / "runs" / f"seed_{seed}" / "world_model.pt"
                )
                checkpoint = load_checkpoint(checkpoint_path)
                self.assertEqual(checkpoint.training_config["seed"], seed)
                self.assertEqual(
                    checkpoint.training_config["bootstrap_seed"], seed
                )
                self.assertEqual(
                    result["runs"][str(seed)], str(checkpoint_path.resolve())
                )

            diagnostic_files = {
                "manifest.json",
                "metrics.json",
                "one_step_calibration.csv",
                "one_step_calibration.png",
                "rollout_uncertainty.png",
            }
            for name in ("baseline_diagnostics", "bootstrap_diagnostics"):
                self.assertEqual(
                    {path.name for path in (output_dir / name).iterdir()},
                    diagnostic_files,
                )
                self.assertEqual(
                    Path(result[name]["output_dir"]),
                    (output_dir / name).resolve(),
                )

            manifest = json.loads(
                (output_dir / "experiment_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            comparison = json.loads(
                (output_dir / "comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [record["seed"] for record in manifest["baseline_checkpoints"]],
                [0, 1],
            )
            self.assertEqual(
                [record["seed"] for record in manifest["bootstrap_checkpoints"]],
                [0, 1],
            )
            json.dumps(manifest, allow_nan=False)
            json.dumps(comparison, allow_nan=False)
            for name in (
                "experiment_manifest.json",
                "comparison.json",
                "comparison.csv",
                "comparison.png",
            ):
                self.assertEqual(result[name], str((output_dir / name).resolve()))

    def test_runner_rejects_baseline_protocol_mismatches_before_training(self):
        mismatched_data = self.fixture_root / "same-data-different-path.npz"
        shutil.copyfile(self.data_path, mismatched_data)
        cases = (
            ("training seeds", {"seeds": (0, 2)}),
            ("hidden_size", {"hidden_size": 4}),
            ("split_seed", {"split_seed": 1}),
            ("data_path", {"data_path": mismatched_data}),
        )
        for index, (message, overrides) in enumerate(cases):
            with self.subTest(message=message):
                output = self.fixture_root / f"mismatch-output-{index}"
                arguments = {
                    "data_path": self.data_path,
                    "baseline_checkpoint_paths": self.baseline_paths,
                    "output_dir": output,
                    "seeds": (0, 1),
                    "split_seed": 0,
                    "hidden_size": 8,
                    "epochs": 1,
                    "batch_size": 32,
                    "learning_rate": 1e-3,
                    "rollout_loss_weight": 1.0,
                    "diagnostic_horizons": (1, 2),
                    "windows_per_episode": 2,
                    "calibration_bins": 2,
                }
                arguments.update(overrides)
                with self.assertRaisesRegex(ValueError, message):
                    run_bootstrap_experiment(**arguments)
                self.assertFalse(output.exists())

    def test_cross_ensemble_validation_rejects_data_path_mismatch(self):
        baseline = load_ensemble(self.baseline_paths)
        bootstrap_members = copy.deepcopy(baseline.members)
        for seed, member in zip(baseline.seeds, bootstrap_members, strict=True):
            member.training_config.update(
                {
                    "bootstrap_seed": seed,
                    "bootstrap_episode_draws": 6,
                    "bootstrap_unique_episodes": 4,
                    "bootstrap_episode_counts": {"0": 1},
                }
            )
        bootstrap_members[1].training_config["data_path"] = str(
            self.fixture_root / "other.npz"
        )
        bootstrap = WorldModelEnsemble(
            members=tuple(bootstrap_members),
            seeds=baseline.seeds,
            target_std=baseline.target_std.copy(),
            checkpoint_paths=baseline.checkpoint_paths,
        )

        with self.assertRaisesRegex(ValueError, "data_path"):
            bootstrap_experiment._validate_comparable_ensembles(
                baseline,
                bootstrap,
            )

    def test_dataset_preflight_rejects_missing_training_arrays(self):
        malformed = self.fixture_root / "missing-training-arrays.npz"
        np.savez_compressed(malformed, episode_ids=np.arange(10))
        baseline = load_ensemble(self.baseline_paths)

        with self.assertRaisesRegex(ValueError, "dataset is missing arrays"):
            bootstrap_experiment._validate_dataset_split_coverage(
                malformed,
                baseline,
            )

    def test_runner_rejects_invalid_protocol_and_nonempty_output(self):
        invalid_cases = (
            ("training seeds", {"seeds": (0, 0)}),
            ("training seeds", {"seeds": (0, -1)}),
            ("hidden_size", {"hidden_size": 0}),
            ("learning_rate", {"learning_rate": float("nan")}),
            ("rollout_loss_weight", {"rollout_loss_weight": float("inf")}),
            ("diagnostic_horizons", {"diagnostic_horizons": (2, 1)}),
            ("windows_per_episode", {"windows_per_episode": 0}),
            ("calibration_bins", {"calibration_bins": 0}),
        )
        for index, (message, overrides) in enumerate(invalid_cases):
            with self.subTest(message=message, overrides=overrides):
                output = self.fixture_root / f"invalid-output-{index}"
                arguments = {
                    "data_path": self.data_path,
                    "baseline_checkpoint_paths": self.baseline_paths,
                    "output_dir": output,
                    "seeds": (0, 1),
                    "split_seed": 0,
                    "hidden_size": 8,
                    "epochs": 1,
                    "batch_size": 32,
                    "learning_rate": 1e-3,
                    "rollout_loss_weight": 1.0,
                    "diagnostic_horizons": (1, 2),
                    "windows_per_episode": 2,
                    "calibration_bins": 2,
                }
                arguments.update(overrides)
                with self.assertRaisesRegex(ValueError, message):
                    run_bootstrap_experiment(**arguments)
                self.assertFalse(output.exists())

        nonempty_output = self.fixture_root / "nonempty-output"
        nonempty_output.mkdir()
        (nonempty_output / "stale.txt").write_text("stale", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "output directory"):
            run_bootstrap_experiment(
                data_path=self.data_path,
                baseline_checkpoint_paths=self.baseline_paths,
                output_dir=nonempty_output,
                seeds=(0, 1),
                split_seed=0,
                hidden_size=8,
                epochs=1,
                batch_size=32,
                diagnostic_horizons=(1, 2),
                windows_per_episode=2,
                calibration_bins=2,
            )
        self.assertEqual(
            (nonempty_output / "stale.txt").read_text(encoding="utf-8"),
            "stale",
        )

    def test_training_failure_preserves_partial_run_without_final_artifacts(self):
        real_run_training = bootstrap_experiment.run_training
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "experiment"

            def fail_second_seed(**kwargs):
                if kwargs["seed"] == 1:
                    raise RuntimeError("second seed failed")
                return real_run_training(**kwargs)

            with patch.object(
                bootstrap_experiment,
                "run_training",
                side_effect=fail_second_seed,
            ):
                with self.assertRaisesRegex(RuntimeError, "second seed failed"):
                    run_bootstrap_experiment(
                        data_path=self.data_path,
                        baseline_checkpoint_paths=self.baseline_paths,
                        output_dir=output_dir,
                        seeds=(0, 1),
                        split_seed=0,
                        hidden_size=8,
                        epochs=1,
                        batch_size=32,
                        diagnostic_horizons=(1, 2),
                        windows_per_episode=2,
                        calibration_bins=2,
                    )

            self.assertTrue(
                (output_dir / "runs" / "seed_0" / "world_model.pt").is_file()
            )
            self.assertFalse((output_dir / "comparison.json").exists())
            self.assertFalse((output_dir / "experiment_manifest.json").exists())

    def test_diagnostic_failure_preserves_partial_outputs_without_final_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "experiment"
            call_count = 0

            def fail_second_diagnostic(**kwargs):
                nonlocal call_count
                call_count += 1
                diagnostic_output = Path(kwargs["output_dir"])
                diagnostic_output.mkdir(parents=True)
                (diagnostic_output / "partial.txt").write_text(
                    f"call {call_count}", encoding="utf-8"
                )
                if call_count == 2:
                    raise RuntimeError("bootstrap diagnostic failed")
                return {"output_dir": str(diagnostic_output)}

            with patch.object(
                bootstrap_experiment,
                "run_ensemble_diagnostics",
                side_effect=fail_second_diagnostic,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "bootstrap diagnostic failed"
                ):
                    run_bootstrap_experiment(
                        data_path=self.data_path,
                        baseline_checkpoint_paths=self.baseline_paths,
                        output_dir=output_dir,
                        seeds=(0, 1),
                        split_seed=0,
                        hidden_size=8,
                        epochs=1,
                        batch_size=32,
                        diagnostic_horizons=(1, 2),
                        windows_per_episode=2,
                        calibration_bins=2,
                    )

            self.assertTrue(
                (output_dir / "baseline_diagnostics" / "partial.txt").is_file()
            )
            self.assertTrue(
                (output_dir / "bootstrap_diagnostics" / "partial.txt").is_file()
            )
            self.assertFalse((output_dir / "comparison.json").exists())
            self.assertFalse((output_dir / "experiment_manifest.json").exists())

    def test_comparison_csv_has_exact_header_and_stable_row_order(self):
        comparison = build_bootstrap_comparison(
            make_ensemble_metrics(error=2.0, correlation=0.1),
            make_ensemble_metrics(error=1.5, correlation=0.4),
            horizons=(1, 2),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = write_comparison_csv(
                comparison,
                Path(directory) / "nested" / "comparison.csv",
            )
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                handle.seek(0)
                fieldnames = next(csv.reader(handle))

        self.assertEqual(
            fieldnames,
            [
                "evaluation",
                "horizon",
                "metric",
                "baseline_error",
                "bootstrap_error",
                "error_delta",
                "baseline_disagreement",
                "bootstrap_disagreement",
                "disagreement_delta",
                "baseline_correlation",
                "bootstrap_correlation",
                "correlation_delta",
            ],
        )
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            [
                (row["evaluation"], row["horizon"], row["metric"])
                for row in rows
            ],
            [
                ("one_step", "", metric)
                for metric in METRICS
            ]
            + [
                ("rollout", str(horizon), metric)
                for horizon in (1, 2)
                for metric in METRICS
            ],
        )

    def test_comparison_csv_leaves_inapplicable_and_null_cells_empty(self):
        comparison = build_bootstrap_comparison(
            make_ensemble_metrics(error=2.0, correlation=None),
            make_ensemble_metrics(error=1.5, correlation=0.4),
            horizons=(1, 2),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = write_comparison_csv(
                comparison,
                Path(directory) / "comparison.csv",
            )
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        for row in rows[:4]:
            self.assertEqual(row["horizon"], "")
            self.assertEqual(row["baseline_disagreement"], "")
            self.assertEqual(row["bootstrap_disagreement"], "")
            self.assertEqual(row["disagreement_delta"], "")
        for row in rows:
            self.assertEqual(row["baseline_correlation"], "")
            self.assertEqual(row["correlation_delta"], "")
            self.assertNotIn("None", row.values())
            self.assertNotIn("nan", row.values())

    def test_comparison_plot_is_four_panels_with_null_correlation_gaps(self):
        comparison = build_bootstrap_comparison(
            make_ensemble_metrics(error=2.0, correlation=None),
            make_ensemble_metrics(error=1.5, correlation=0.4),
            horizons=(1, 2),
        )
        real_subplots = bootstrap_experiment.plt.subplots
        captured = {}

        def capture_subplots(*args, **kwargs):
            figure, axes = real_subplots(*args, **kwargs)
            captured["figure"] = figure
            captured["axes"] = axes
            return figure, axes

        figure_numbers_before = set(bootstrap_experiment.plt.get_fignums())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "comparison.png"
            with patch.object(
                bootstrap_experiment.plt,
                "subplots",
                side_effect=capture_subplots,
            ) as subplots_spy:
                path = plot_bootstrap_comparison(comparison, output)
            contents = path.read_bytes()

        subplots_spy.assert_called_once_with(2, 2, figsize=(11, 8))
        self.assertEqual(captured["axes"].shape, (2, 2))
        self.assertEqual(len(captured["figure"].axes), 8)
        for correlation_axis in captured["figure"].axes[4:]:
            self.assertEqual(len(correlation_axis.lines), 2)
            self.assertTrue(
                np.isnan(correlation_axis.lines[0].get_ydata()).all()
            )
            np.testing.assert_allclose(
                correlation_axis.lines[1].get_ydata(),
                [0.4, 0.4],
            )
        self.assertEqual(contents[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(contents), 1000)
        self.assertEqual(
            set(bootstrap_experiment.plt.get_fignums()),
            figure_numbers_before,
        )

    def test_comparison_uses_bootstrap_minus_baseline_finite_deltas(self):
        comparison = build_bootstrap_comparison(
            make_ensemble_metrics(
                error=2.0,
                correlation=0.1,
                disagreement=0.2,
            ),
            make_ensemble_metrics(
                error=1.5,
                correlation=0.4,
                disagreement=0.3,
            ),
            horizons=(1, 2),
        )

        self.assertEqual(comparison["schema_version"], 1)
        self.assertEqual(
            comparison["delta_definition"],
            "bootstrap minus baseline",
        )
        self.assertEqual(comparison["horizons"], [1, 2])

        one_step = comparison["one_step"]["position"]
        self.assertEqual(one_step["baseline_error"], 2.0)
        self.assertEqual(one_step["bootstrap_error"], 1.5)
        self.assertEqual(one_step["error_delta"], -0.5)
        self.assertAlmostEqual(one_step["correlation_delta"], 0.3)

        rollout = comparison["rollout"]["2"]["position"]
        self.assertEqual(rollout["baseline_error"], 3.0)
        self.assertEqual(rollout["bootstrap_error"], 2.5)
        self.assertEqual(rollout["error_delta"], -0.5)
        self.assertAlmostEqual(rollout["disagreement_delta"], 0.2)
        self.assertAlmostEqual(rollout["correlation_delta"], 0.3)
        json.dumps(comparison, allow_nan=False)

    def test_comparison_keeps_null_correlation_delta_json_safe(self):
        cases = (
            (None, 0.4),
            (0.1, None),
        )
        for baseline_correlation, bootstrap_correlation in cases:
            with self.subTest(
                baseline=baseline_correlation,
                bootstrap=bootstrap_correlation,
            ):
                comparison = build_bootstrap_comparison(
                    make_ensemble_metrics(
                        error=2.0,
                        correlation=baseline_correlation,
                    ),
                    make_ensemble_metrics(
                        error=1.5,
                        correlation=bootstrap_correlation,
                    ),
                    horizons=(1, 2),
                )

                self.assertIsNone(
                    comparison["one_step"]["position"][
                        "correlation_delta"
                    ]
                )
                self.assertIsNone(
                    comparison["rollout"]["1"]["position"][
                        "correlation_delta"
                    ]
                )
                json.dumps(comparison, allow_nan=False)

    def test_comparison_requires_schema_version_one(self):
        for side in ("baseline", "bootstrap"):
            with self.subTest(side=side):
                baseline = make_ensemble_metrics(error=2.0, correlation=0.1)
                bootstrap = make_ensemble_metrics(error=1.5, correlation=0.4)
                target = baseline if side == "baseline" else bootstrap
                target["schema_version"] = 2

                with self.assertRaisesRegex(
                    ValueError,
                    rf"{side}\.schema_version",
                ):
                    build_bootstrap_comparison(
                        baseline,
                        bootstrap,
                        horizons=(1, 2),
                    )

    def test_comparison_requires_strictly_increasing_positive_unique_horizons(
        self,
    ):
        invalid_horizons = (
            (),
            (1, 1),
            (2, 1),
            (0, 1),
            (-1, 1),
            (True,),
            (1.5,),
        )
        for horizons in invalid_horizons:
            with self.subTest(horizons=horizons):
                with self.assertRaisesRegex(ValueError, "horizons"):
                    build_bootstrap_comparison(
                        make_ensemble_metrics(error=2.0, correlation=0.1),
                        make_ensemble_metrics(error=1.5, correlation=0.4),
                        horizons=horizons,
                    )

    def test_comparison_rejects_missing_metric_with_field_name(self):
        cases = (
            (
                "baseline.one_step.metrics.position",
                lambda baseline, bootstrap: baseline["one_step"]["metrics"].pop(
                    "position"
                ),
            ),
            (
                "bootstrap.rollout.horizons.1.metrics.position",
                lambda baseline, bootstrap: bootstrap["rollout"]["horizons"][
                    "1"
                ]["metrics"].pop("position"),
            ),
        )
        for expected_field, mutate in cases:
            with self.subTest(field=expected_field):
                baseline = make_ensemble_metrics(error=2.0, correlation=0.1)
                bootstrap = make_ensemble_metrics(error=1.5, correlation=0.4)
                mutate(baseline, bootstrap)

                with self.assertRaisesRegex(
                    ValueError,
                    expected_field.replace(".", r"\."),
                ):
                    build_bootstrap_comparison(
                        baseline,
                        bootstrap,
                        horizons=(1, 2),
                    )

    def test_comparison_rejects_non_finite_numeric_fields(self):
        cases = (
            (
                "baseline.one_step.metrics.position.ensemble_error.mean",
                lambda baseline, bootstrap: baseline["one_step"]["metrics"][
                    "position"
                ]["ensemble_error"].__setitem__("mean", float("nan")),
            ),
            (
                "bootstrap.one_step.metrics.position.pearson_correlation",
                lambda baseline, bootstrap: bootstrap["one_step"]["metrics"][
                    "position"
                ].__setitem__("pearson_correlation", float("inf")),
            ),
            (
                "baseline.rollout.horizons.1.metrics.position.ensemble_error_mean",
                lambda baseline, bootstrap: baseline["rollout"]["horizons"][
                    "1"
                ]["metrics"]["position"].__setitem__(
                    "ensemble_error_mean",
                    float("-inf"),
                ),
            ),
            (
                "bootstrap.rollout.horizons.1.metrics.position.disagreement_mean",
                lambda baseline, bootstrap: bootstrap["rollout"]["horizons"][
                    "1"
                ]["metrics"]["position"].__setitem__(
                    "disagreement_mean",
                    float("nan"),
                ),
            ),
            (
                "baseline.rollout.horizons.1.metrics.position.pearson_correlation",
                lambda baseline, bootstrap: baseline["rollout"]["horizons"][
                    "1"
                ]["metrics"]["position"].__setitem__(
                    "pearson_correlation",
                    float("inf"),
                ),
            ),
        )
        for expected_field, mutate in cases:
            with self.subTest(field=expected_field):
                baseline = make_ensemble_metrics(error=2.0, correlation=0.1)
                bootstrap = make_ensemble_metrics(error=1.5, correlation=0.4)
                mutate(baseline, bootstrap)

                with self.assertRaisesRegex(
                    ValueError,
                    expected_field.replace(".", r"\."),
                ):
                    build_bootstrap_comparison(
                        baseline,
                        bootstrap,
                        horizons=(1, 2),
                    )

    def test_comparison_rejects_non_finite_delta(self):
        baseline = make_ensemble_metrics(error=-1e308, correlation=0.1)
        bootstrap = make_ensemble_metrics(error=1e308, correlation=0.4)

        with self.assertRaisesRegex(
            ValueError,
            r"one_step\.position\.error_delta",
        ):
            build_bootstrap_comparison(
                baseline,
                bootstrap,
                horizons=(1, 2),
            )

    def test_comparison_rejects_missing_horizon_snapshot_with_field_name(self):
        baseline = make_ensemble_metrics(error=2.0, correlation=0.1)
        bootstrap = make_ensemble_metrics(error=1.5, correlation=0.4)
        del bootstrap["rollout"]["horizons"]["2"]

        with self.assertRaisesRegex(
            ValueError,
            r"bootstrap\.rollout\.horizons\.2",
        ):
            build_bootstrap_comparison(
                baseline,
                bootstrap,
                horizons=(1, 2),
            )
