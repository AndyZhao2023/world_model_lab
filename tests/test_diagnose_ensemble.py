import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from tests.test_ensemble import make_member, save_member_checkpoint
from world_model_lab import diagnose_ensemble
from world_model_lab.diagnose_ensemble import (
    build_calibration_bins,
    evaluate_one_step_calibration,
    evaluate_ensemble_rollouts,
    pearson_correlation,
    plot_one_step_calibration,
    plot_rollout_uncertainty,
    run_ensemble_diagnostics,
)
from world_model_lab.diagnose_model import sha256_file
from world_model_lab.diagnostics import RolloutWindow
from world_model_lab.ensemble import build_ensemble


PLOT_METRIC_NAMES = (
    "position",
    "heading_degrees",
    "velocity",
    "normalized_total",
)


def make_one_step_metrics():
    return {
        "metrics": {
            name: {
                "calibration_bins": [
                    {
                        "count": 2,
                        "disagreement_mean": 0.25,
                        "error_mean": 0.5,
                    },
                    {
                        "count": 2,
                        "disagreement_mean": 0.75,
                        "error_mean": 1.5,
                    },
                ]
            }
            for name in PLOT_METRIC_NAMES
        }
    }


def make_rollout_metrics():
    return {
        "steps": [1, 2],
        "metrics": {
            name: {
                "ensemble_error_mean": [0.5, 1.0],
                "mean_member_error_mean": [0.75, 1.5],
                "min_member_error_mean": [0.25, 0.5],
                "max_member_error_mean": [1.0, 2.0],
                "disagreement_mean": [0.2, 0.4],
                "pearson_correlation": [0.1, 0.2],
            }
            for name in PLOT_METRIC_NAMES
        },
    }


def save_diagnostic_dataset(
    path: Path,
    *,
    episode_steps: dict[int, int] | None = None,
    missing_array: str | None = None,
) -> Path:
    steps_by_episode = episode_steps or {0: 3, 1: 3, 2: 3, 3: 3}
    states = []
    actions = []
    next_states = []
    episode_ids = []
    step_ids = []
    for episode_id, step_count in steps_by_episode.items():
        state = np.asarray([10.0 * episode_id, 0.0, 0.0, 0.0])
        for step in range(step_count):
            next_state = state + np.asarray([1.0, 0.0, 0.0, 0.0])
            states.append(state)
            actions.append(np.zeros(2))
            next_states.append(next_state)
            episode_ids.append(episode_id)
            step_ids.append(step)
            state = next_state
    arrays = {
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "next_states": np.asarray(next_states),
        "episode_ids": np.asarray(episode_ids),
        "step_ids": np.asarray(step_ids),
    }
    if missing_array is not None:
        del arrays[missing_array]
    np.savez_compressed(path, **arrays)
    return path


def save_diagnostic_checkpoints(root: Path) -> tuple[Path, Path]:
    seed_0_path = save_member_checkpoint(
        root / "seed-0.pt",
        make_member(0, np.asarray([0.75, 0.0, 0.0, 0.0])),
    )
    seed_1_path = save_member_checkpoint(
        root / "seed-1.pt",
        make_member(1, np.asarray([1.25, 0.0, 0.0, 0.0])),
    )
    return seed_0_path, seed_1_path


def replace_dataset_array(path: Path, name: str, values: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {field: loaded[field] for field in loaded.files}
    arrays[name] = values
    np.savez_compressed(path, **arrays)


class DiagnoseEnsembleTest(unittest.TestCase):
    def test_pyproject_registers_ensemble_diagnostic_command(self):
        pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'world-model-diagnose-ensemble = "world_model_lab.diagnose_ensemble:main"',
            pyproject,
        )

    def test_cli_help_lists_all_protocol_parameters(self):
        entrypoint = getattr(diagnose_ensemble, "main", None)
        self.assertTrue(callable(entrypoint))
        standard_output = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["world-model-diagnose-ensemble", "--help"],
        ):
            with redirect_stdout(standard_output):
                with self.assertRaises(SystemExit) as context:
                    entrypoint()

        self.assertEqual(context.exception.code, 0)
        help_text = standard_output.getvalue()
        for flag in (
            "--data",
            "--checkpoints",
            "--output-dir",
            "--horizons",
            "--windows-per-episode",
            "--calibration-bins",
        ):
            self.assertIn(flag, help_text)

    def test_cli_reports_missing_data_and_checkpoint_as_argument_errors(self):
        entrypoint = getattr(diagnose_ensemble, "main", None)
        self.assertTrue(callable(entrypoint))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_dataset = save_diagnostic_dataset(root / "valid.npz")
            cases = (
                (
                    [
                        "--data",
                        str(root / "missing.npz"),
                        "--checkpoints",
                        str(root / "missing-0.pt"),
                        str(root / "missing-1.pt"),
                    ],
                    "dataset is not a regular file",
                ),
                (
                    [
                        "--data",
                        str(valid_dataset),
                        "--checkpoints",
                        str(root / "missing-0.pt"),
                        str(root / "missing-1.pt"),
                    ],
                    "checkpoint is not a regular file",
                ),
            )
            for arguments, message in cases:
                with self.subTest(message=message):
                    standard_error = io.StringIO()
                    with patch.object(
                        sys,
                        "argv",
                        ["world-model-diagnose-ensemble", *arguments],
                    ):
                        with redirect_stderr(standard_error):
                            with self.assertRaises(SystemExit) as context:
                                entrypoint()

                    self.assertEqual(context.exception.code, 2)
                    self.assertIn(message, standard_error.getvalue())

    def test_cli_prints_returned_contract_as_sorted_indented_json(self):
        entrypoint = getattr(diagnose_ensemble, "main", None)
        self.assertTrue(callable(entrypoint))
        standard_output = io.StringIO()
        with patch.object(
            diagnose_ensemble,
            "run_ensemble_diagnostics",
            return_value={"z": 1, "a": 2},
            create=True,
        ):
            with patch.object(
                sys,
                "argv",
                [
                    "world-model-diagnose-ensemble",
                    "--checkpoints",
                    "seed-0.pt",
                    "seed-1.pt",
                ],
            ):
                with redirect_stdout(standard_output):
                    entrypoint()

        self.assertEqual(
            standard_output.getvalue(),
            json.dumps({"z": 1, "a": 2}, indent=2, sort_keys=True) + "\n",
        )

    def test_run_writes_complete_atomic_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = save_diagnostic_dataset(root / "transitions.npz")
            seed_0_path, seed_1_path = save_diagnostic_checkpoints(root)
            output_dir = root / "diagnostics"
            output_dir.mkdir()

            result = run_ensemble_diagnostics(
                data_path=data_path,
                checkpoint_paths=(seed_1_path, seed_0_path),
                output_dir=output_dir,
                horizons=(1, 2),
                windows_per_episode=2,
                calibration_bins=2,
            )

            expected_names = {
                "manifest.json",
                "metrics.json",
                "one_step_calibration.csv",
                "one_step_calibration.png",
                "rollout_uncertainty.png",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                expected_names,
            )
            for path in output_dir.iterdir():
                self.assertGreater(path.stat().st_size, 0)

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            json.dumps(manifest, allow_nan=False)
            json.dumps(metrics, allow_nan=False)
            self.assertEqual(manifest["member_seeds"], [0, 1])
            self.assertEqual(
                [item["seed"] for item in manifest["checkpoints"]],
                [0, 1],
            )
            self.assertEqual(
                [item["sha256"] for item in manifest["checkpoints"]],
                [sha256_file(seed_0_path), sha256_file(seed_1_path)],
            )
            self.assertEqual(manifest["dataset"]["sha256"], sha256_file(data_path))
            self.assertEqual(metrics["one_step"]["samples"], 3)
            self.assertEqual(metrics["rollout"]["steps"], [1, 2])
            self.assertEqual(
                set(metrics["rollout"]["horizons"]),
                {"1", "2"},
            )

            with (output_dir / "one_step_calibration.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(
                reader.fieldnames,
                [
                    "metric",
                    "bin_index",
                    "count",
                    "disagreement_mean",
                    "error_mean",
                ],
            )
            self.assertEqual(len(rows), 8)

            expected_paths = {
                "manifest": "manifest.json",
                "metrics": "metrics.json",
                "calibration_csv": "one_step_calibration.csv",
                "calibration_plot": "one_step_calibration.png",
                "rollout_plot": "rollout_uncertainty.png",
            }
            self.assertEqual(Path(result["output_dir"]), output_dir.resolve())
            for key, filename in expected_paths.items():
                self.assertEqual(Path(result[key]), output_dir.resolve() / filename)

    def test_run_persists_rollout_eligible_and_skipped_episode_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = save_diagnostic_dataset(
                root / "transitions.npz",
                episode_steps={0: 3, 1: 3, 2: 3, 3: 1},
            )
            members = (
                make_member(0, np.asarray([0.75, 0.0, 0.0, 0.0])),
                make_member(1, np.asarray([1.25, 0.0, 0.0, 0.0])),
            )
            checkpoint_paths = []
            for member in members:
                member.split_episode_ids = {
                    "train": np.asarray([0]),
                    "validation": np.asarray([1]),
                    "test": np.asarray([2, 3]),
                }
                checkpoint_paths.append(
                    save_member_checkpoint(
                        root / f"seed-{member.training_config['seed']}.pt",
                        member,
                    )
                )

            output_dir = root / "diagnostics"
            run_ensemble_diagnostics(
                data_path=data_path,
                checkpoint_paths=checkpoint_paths,
                output_dir=output_dir,
                horizons=(1, 2),
                windows_per_episode=2,
                calibration_bins=2,
            )

            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["rollout"]["eligible_episode_ids"], [2])
            self.assertEqual(metrics["rollout"]["skipped_episode_ids"], [3])

    def test_run_rejects_nonempty_output_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "diagnostics"
            output_dir.mkdir()
            (output_dir / "existing.txt").write_text("old", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "absent or empty"):
                run_ensemble_diagnostics(
                    data_path=root / "missing.npz",
                    checkpoint_paths=(),
                    output_dir=output_dir,
                )

    def test_run_rejects_missing_dataset_arrays_before_loading_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = save_diagnostic_dataset(
                root / "transitions.npz",
                missing_array="step_ids",
            )

            with self.assertRaisesRegex(ValueError, "missing arrays: step_ids"):
                run_ensemble_diagnostics(
                    data_path=data_path,
                    checkpoint_paths=(root / "missing-0.pt", root / "missing-1.pt"),
                    output_dir=root / "diagnostics",
                    horizons=(1,),
                )

    def test_run_rejects_malformed_episode_id_shape_as_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = save_diagnostic_dataset(root / "transitions.npz")
            with np.load(data_path, allow_pickle=False) as loaded:
                episode_ids = loaded["episode_ids"].copy()
            replace_dataset_array(
                data_path,
                "episode_ids",
                episode_ids[:, None],
            )
            seed_0_path, seed_1_path = save_diagnostic_checkpoints(root)

            with self.assertRaisesRegex(
                ValueError,
                r"episode_ids must have shape \[N\]",
            ):
                run_ensemble_diagnostics(
                    data_path=data_path,
                    checkpoint_paths=(seed_0_path, seed_1_path),
                    output_dir=root / "diagnostics",
                    horizons=(1, 2),
                )

    def test_run_validates_dataset_arrays_before_loading_checkpoints(self):
        invalid_arrays = (
            ("states", lambda values: values[:, :3], r"states.*\[N, 4\]"),
            ("actions", lambda values: values[:, :1], r"actions.*\[N, 2\]"),
            (
                "next_states",
                lambda values: values[:, :3],
                r"next_states.*\[N, 4\]",
            ),
            (
                "states",
                lambda values: np.where(
                    np.arange(values.size).reshape(values.shape) == 0,
                    np.nan,
                    values,
                ),
                r"states.*finite",
            ),
            (
                "actions",
                lambda values: np.where(
                    np.arange(values.size).reshape(values.shape) == 0,
                    np.inf,
                    values,
                ),
                r"actions.*finite",
            ),
            (
                "next_states",
                lambda values: np.where(
                    np.arange(values.size).reshape(values.shape) == 0,
                    np.nan,
                    values,
                ),
                r"next_states.*finite",
            ),
            (
                "episode_ids",
                lambda values: values.astype(np.float64),
                r"episode_ids.*integer dtype",
            ),
            (
                "step_ids",
                lambda values: values.astype(np.float64),
                r"step_ids.*integer dtype",
            ),
            (
                "episode_ids",
                lambda values: np.where(values == values[0], np.nan, values),
                r"episode_ids.*finite",
            ),
            (
                "step_ids",
                lambda values: np.where(values == values[0], np.inf, values),
                r"step_ids.*finite",
            ),
            (
                "episode_ids",
                lambda values: np.where(values == values[0], -1, values),
                r"episode_ids.*non-negative",
            ),
            (
                "step_ids",
                lambda values: np.where(values == values[0], -1, values),
                r"step_ids.*non-negative",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (name, mutate, message) in enumerate(invalid_arrays):
                with self.subTest(name=name, message=message):
                    data_path = save_diagnostic_dataset(root / f"data-{index}.npz")
                    with np.load(data_path, allow_pickle=False) as loaded:
                        values = loaded[name].copy()
                    replace_dataset_array(data_path, name, mutate(values))

                    with self.assertRaisesRegex(ValueError, message):
                        run_ensemble_diagnostics(
                            data_path=data_path,
                            checkpoint_paths=(
                                root / "missing-0.pt",
                                root / "missing-1.pt",
                            ),
                            output_dir=root / "diagnostics",
                            horizons=(1,),
                        )

    def test_run_rejects_empty_dataset_before_loading_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "empty.npz"
            np.savez_compressed(
                data_path,
                states=np.empty((0, 4)),
                actions=np.empty((0, 2)),
                next_states=np.empty((0, 4)),
                episode_ids=np.empty(0, dtype=np.int64),
                step_ids=np.empty(0, dtype=np.int64),
            )

            with self.assertRaisesRegex(ValueError, "dataset arrays must not be empty"):
                run_ensemble_diagnostics(
                    data_path=data_path,
                    checkpoint_paths=(root / "missing-0.pt", root / "missing-1.pt"),
                    output_dir=root / "diagnostics",
                    horizons=(1,),
                )

    def test_run_rejects_invalid_horizons_before_reading_inputs(self):
        invalid_horizons = (
            ((), "non-empty"),
            ((0, 1), "positive integers"),
            ((1, 1), "unique"),
            ((2, 1), "strictly increasing"),
            ((1, 2.0), "positive integers"),
            ((True, 2), "positive integers"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for horizons, message in invalid_horizons:
                with self.subTest(horizons=horizons):
                    with self.assertRaisesRegex(ValueError, message):
                        run_ensemble_diagnostics(
                            data_path=root / "missing.npz",
                            checkpoint_paths=(),
                            output_dir=root / "diagnostics",
                            horizons=horizons,
                        )

    def test_run_rejects_dataset_missing_checkpoint_test_episode_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = save_diagnostic_dataset(
                root / "transitions.npz",
                episode_steps={0: 2, 1: 2, 2: 2},
            )
            seed_0_path, seed_1_path = save_diagnostic_checkpoints(root)

            with self.assertRaisesRegex(ValueError, "test episode IDs.*missing"):
                run_ensemble_diagnostics(
                    data_path=data_path,
                    checkpoint_paths=(seed_0_path, seed_1_path),
                    output_dir=root / "diagnostics",
                    horizons=(1,),
                )

    def test_run_rejects_horizon_longer_than_every_test_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = save_diagnostic_dataset(
                root / "transitions.npz",
                episode_steps={0: 2, 1: 2, 2: 2, 3: 1},
            )
            seed_0_path, seed_1_path = save_diagnostic_checkpoints(root)

            with self.assertRaisesRegex(ValueError, "not long enough"):
                run_ensemble_diagnostics(
                    data_path=data_path,
                    checkpoint_paths=(seed_0_path, seed_1_path),
                    output_dir=root / "diagnostics",
                    horizons=(1, 2),
                )

    def test_run_removes_staging_bundle_when_plotting_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = save_diagnostic_dataset(root / "transitions.npz")
            seed_0_path, seed_1_path = save_diagnostic_checkpoints(root)
            output_dir = root / "diagnostics"

            with patch.object(
                diagnose_ensemble,
                "plot_rollout_uncertainty",
                side_effect=RuntimeError("plot failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "plot failed"):
                    run_ensemble_diagnostics(
                        data_path=data_path,
                        checkpoint_paths=(seed_0_path, seed_1_path),
                        output_dir=output_dir,
                        horizons=(1, 2),
                        windows_per_episode=2,
                        calibration_bins=2,
                    )

            self.assertFalse(output_dir.exists())
            self.assertEqual(
                list(root.glob(f".{output_dir.name}.tmp-*")),
                [],
            )

    def test_pearson_returns_none_for_constant_values(self):
        self.assertIsNone(
            pearson_correlation(np.ones(3), np.asarray([1.0, 2.0, 3.0]))
        )

    def test_pearson_returns_none_when_finite_subnormals_produce_nan(self):
        subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
        values = np.asarray([0.0, subnormal])

        self.assertTrue(np.all(np.isfinite(values)))
        self.assertIsNone(pearson_correlation(values, values))

    def test_pearson_rejects_invalid_shape_or_non_finite_values(self):
        for left, right, message in (
            (np.ones((1, 3)), np.ones((1, 3)), "matching non-empty vectors"),
            (np.ones(2), np.ones(3), "matching non-empty vectors"),
            (np.asarray([]), np.asarray([]), "matching non-empty vectors"),
            (
                np.asarray([1.0, np.nan]),
                np.asarray([1.0, 2.0]),
                "finite",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    pearson_correlation(left, right)

    def test_calibration_bins_are_equal_count_and_sorted_by_disagreement(self):
        bins = build_calibration_bins(
            np.asarray([4.0, 1.0, 3.0, 2.0]),
            np.asarray([40.0, 10.0, 30.0, 20.0]),
            bin_count=2,
        )
        self.assertEqual([item["count"] for item in bins], [2, 2])
        self.assertEqual(
            [item["disagreement_mean"] for item in bins],
            [1.5, 3.5],
        )
        self.assertEqual([item["error_mean"] for item in bins], [15.0, 35.0])

    def test_calibration_bins_reject_invalid_inputs(self):
        for disagreement, errors, bin_count, message in (
            (np.ones((1, 2)), np.ones((1, 2)), 1, "matching non-empty vectors"),
            (np.ones(2), np.ones(3), 1, "matching non-empty vectors"),
            (np.asarray([]), np.asarray([]), 1, "matching non-empty vectors"),
            (np.asarray([1.0, np.inf]), np.ones(2), 1, "finite"),
            (np.ones(2), np.asarray([1.0, np.nan]), 1, "finite"),
            (np.ones(2), np.ones(2), 0, "positive"),
            (np.ones(2), np.ones(2), -1, "positive"),
        ):
            with self.subTest(message=message, bin_count=bin_count):
                with self.assertRaisesRegex(ValueError, message):
                    build_calibration_bins(
                        disagreement,
                        errors,
                        bin_count=bin_count,
                    )

    def test_calibration_bins_cap_count_to_sample_count(self):
        bins = build_calibration_bins(
            np.asarray([3.0, 1.0, 2.0]),
            np.asarray([30.0, 10.0, 20.0]),
            bin_count=5,
        )

        self.assertEqual([item["count"] for item in bins], [1, 1, 1])
        self.assertEqual(
            [item["disagreement_mean"] for item in bins],
            [1.0, 2.0, 3.0],
        )

    def test_one_step_reports_ensemble_gain_and_calibration(self):
        ensemble = build_ensemble(
            (
                make_member(0, np.asarray([0.0, 0.0, 0.0, 0.0])),
                make_member(1, np.asarray([2.0, 0.0, 0.0, 0.0])),
            )
        )
        result = evaluate_one_step_calibration(
            ensemble,
            states=np.zeros((2, 4)),
            actions=np.zeros((2, 2)),
            true_next_states=np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0, 0.0],
                ]
            ),
            calibration_bins=2,
        )

        position = result["metrics"]["position"]
        self.assertEqual(result["samples"], 2)
        self.assertEqual(position["ensemble_error"]["mean"], 1.0)
        self.assertEqual(position["mean_member_error"]["mean"], 1.5)
        self.assertEqual(position["ensemble_gain_mean"], 0.5)
        self.assertIsNone(position["pearson_correlation"])

    def test_rollout_metrics_weight_episodes_equally_and_keep_member_divergence(self):
        ensemble = build_ensemble(
            (
                make_member(0, np.asarray([1.0, 0.0, 0.0, 0.0])),
                make_member(1, np.asarray([3.0, 0.0, 0.0, 0.0])),
            )
        )
        windows = (
            RolloutWindow(
                10,
                0,
                np.asarray(
                    [[0, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]],
                    dtype=float,
                ),
                np.zeros((2, 2)),
            ),
            RolloutWindow(
                10,
                1,
                np.asarray(
                    [[0, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]],
                    dtype=float,
                ),
                np.zeros((2, 2)),
            ),
            RolloutWindow(
                20,
                0,
                np.asarray(
                    [[0, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0]],
                    dtype=float,
                ),
                np.zeros((2, 2)),
            ),
        )

        result = evaluate_ensemble_rollouts(
            ensemble,
            windows=windows,
            eligible_episode_ids=np.asarray([10, 20]),
        )

        self.assertEqual(result["steps"], [1, 2])
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["windows"], 3)
        self.assertEqual(
            result["metrics"]["position"]["ensemble_error_mean"],
            [1.0, 2.0],
        )
        self.assertEqual(
            result["metrics"]["position"]["mean_member_error_mean"],
            [1.5, 3.0],
        )
        self.assertEqual(
            result["metrics"]["position"]["min_member_error_mean"],
            [1.0, 2.0],
        )
        self.assertEqual(
            result["metrics"]["position"]["max_member_error_mean"],
            [2.0, 4.0],
        )
        self.assertEqual(
            result["metrics"]["position"]["disagreement_mean"],
            [1.0, 2.0],
        )

    def test_rollout_metrics_keep_heading_circular_across_wrap_boundary(self):
        ensemble = build_ensemble(
            (
                make_member(
                    0,
                    np.asarray([0.0, 0.0, np.deg2rad(179.0), 0.0]),
                ),
                make_member(
                    1,
                    np.asarray([0.0, 0.0, np.deg2rad(-179.0), 0.0]),
                ),
            )
        )
        window = RolloutWindow(
            10,
            0,
            np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, np.pi, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            np.zeros((2, 2)),
        )

        result = evaluate_ensemble_rollouts(
            ensemble,
            windows=(window,),
            eligible_episode_ids=np.asarray([10]),
        )

        heading = result["metrics"]["heading_degrees"]
        np.testing.assert_allclose(
            heading["ensemble_error_mean"],
            [0.0, 0.0],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            heading["disagreement_mean"],
            [1.0, 2.0],
            atol=1e-5,
        )

    def test_rollout_metrics_use_json_null_for_constant_correlations(self):
        ensemble = build_ensemble(
            (
                make_member(0, np.asarray([1.0, 0.0, 0.0, 0.0])),
                make_member(1, np.asarray([3.0, 0.0, 0.0, 0.0])),
            )
        )
        windows = tuple(
            RolloutWindow(
                episode_id,
                0,
                np.asarray(
                    [[0, 0, 0, 0], [target, 0, 0, 0]],
                    dtype=float,
                ),
                np.zeros((1, 2)),
            )
            for episode_id, target in ((10, 2.0), (20, 4.0))
        )

        result = evaluate_ensemble_rollouts(
            ensemble,
            windows=windows,
            eligible_episode_ids=np.asarray([10, 20]),
        )

        for metric in result["metrics"].values():
            self.assertEqual(metric["pearson_correlation"], [None])

    def test_ensemble_plots_save_png_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = plot_one_step_calibration(
                make_one_step_metrics(),
                root / "calibration.png",
            )
            rollout_path = plot_rollout_uncertainty(
                make_rollout_metrics(),
                root / "rollout.png",
            )

            self.assertGreater(calibration_path.stat().st_size, 0)
            self.assertGreater(rollout_path.stat().st_size, 0)

    def test_ensemble_plots_label_all_metric_panels_and_secondary_axes(self):
        labels = (
            ("Position", "error (m)", "disagreement (m)"),
            ("Heading", "error (degrees)", "disagreement (degrees)"),
            ("Velocity", "error (m/s)", "disagreement (m/s)"),
            (
                "Normalized total",
                "normalized MSE",
                "normalized RMS disagreement",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_figure, calibration_axes = plt.subplots(2, 2)
            with patch.object(
                plt,
                "subplots",
                return_value=(calibration_figure, calibration_axes),
            ):
                plot_one_step_calibration(
                    make_one_step_metrics(),
                    root / "calibration.png",
                )

            rollout_figure, rollout_axes = plt.subplots(2, 2)
            with patch.object(
                plt,
                "subplots",
                return_value=(rollout_figure, rollout_axes),
            ):
                plot_rollout_uncertainty(
                    make_rollout_metrics(),
                    root / "rollout.png",
                )

        for axis, (title, error_label, disagreement_label) in zip(
            calibration_axes.flat,
            labels,
        ):
            self.assertEqual(axis.get_title(), title)
            self.assertEqual(axis.get_xlabel(), disagreement_label)
            self.assertEqual(axis.get_ylabel(), error_label)
        for axis, (title, error_label, _) in zip(rollout_axes.flat, labels):
            self.assertEqual(axis.get_title(), title)
            self.assertEqual(axis.get_xlabel(), "Rollout step")
            self.assertEqual(axis.get_ylabel(), error_label)
        self.assertEqual(
            [axis.get_ylabel() for axis in rollout_figure.axes[4:]],
            [item[2] for item in labels],
        )

    def test_ensemble_plots_reject_missing_fields_by_name(self):
        one_step = make_one_step_metrics()
        del one_step["metrics"]["position"]["calibration_bins"]
        rollout = make_rollout_metrics()
        del rollout["metrics"]["velocity"]["disagreement_mean"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                ValueError,
                "position.*calibration_bins",
            ):
                plot_one_step_calibration(one_step, root / "calibration.png")
            with self.assertRaisesRegex(
                ValueError,
                "velocity.*disagreement_mean",
            ):
                plot_rollout_uncertainty(rollout, root / "rollout.png")


if __name__ == "__main__":
    unittest.main()
